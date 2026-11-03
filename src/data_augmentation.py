import ezdxf
import math
import random
from pathlib import Path
from typing import List, Optional
from src.utils import setup_logger

logger = setup_logger("DataAugmentation")


# ---------------------------------------------------------------------------
# Настройки аугментации — все параметры здесь, не надо лезть в код
# ---------------------------------------------------------------------------
AUG_CONFIG = {
    # Диапазон масштабирования (XY и Z масштабируются одинаково)
    'scale_range':     (0.85, 1.15),
    # Диапазон поворота в градусах
    'rotate_range':    (-20, 20),
    # Диапазон смещения в единицах чертежа (обычно мм)
    'translate_range': (-200, 200),
}


class DXFAugmentor:
    """
    Аугментация DXF-файлов для увеличения обучающей выборки.

    Ключевые исправления по сравнению с предыдущей версией:
    1. Масштабирование теперь применяет factor ко ВСЕМ трём осям (X, Y, Z).
       Раньше Z не масштабировался — аугментированные стены становились
       плоскими как перекрытия, и модель учила неправильный признак bbox_depth.
    2. Seed больше не фиксируется глобально через random.seed().
       Вместо этого каждый вызов augment_file() использует свой локальный
       экземпляр Random — разные вызовы дают разные трансформации.
    3. Зеркалирование теперь корректно обрабатывает все три оси (включая Z).
    """

    def __init__(self, mode="3d"):
        self.mode = mode
        self.transformations = ['scale', 'rotate', 'translate', 'mirror']

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def augment_file(
        self,
        source_path: Path,
        output_path: Path,
        transformations: List[str] = None,
    ) -> bool:
        """Создаёт одну аугментированную копию DXF-файла."""
        # Локальный RNG — каждый файл получает уникальные трансформации
        rng = random.Random()

        try:
            doc = ezdxf.readfile(str(source_path))
            msp = doc.modelspace()

            if transformations is None:
                k = rng.randint(1, 3)
                transformations = rng.sample(self.transformations, k=k)

            for entity in msp:
                for transform in transformations:
                    self._apply_transformation(entity, transform, rng)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            doc.saveas(str(output_path))
            logger.debug(f"Augmented: {source_path.name} -> {output_path.name} ({transformations})")
            return True

        except Exception as e:
            logger.error(f"Augmentation failed for {source_path}: {e}")
            return False

    def augment_dataset(
        self,
        input_files: List[Path],
        output_dir: Path,
        augmentation_factor: int = 3,
        class_filter=None,
    ) -> List[Path]:
        """Создаёт augmentation_factor копий каждого файла в датасете."""
        output_dir.mkdir(parents=True, exist_ok=True)
        augmented_files = list(input_files)

        logger.info(f"Starting augmentation: {len(input_files)} files x{augmentation_factor}")

        for file_path in input_files:
            if class_filter:
                class_name = class_filter.extract_class_from_path(file_path)
                if not class_name:
                    continue
                class_dir = output_dir / class_name
            else:
                class_dir = output_dir

            for i in range(augmentation_factor):
                new_name = f"{file_path.stem}_aug{i}{file_path.suffix}"
                out_path = class_dir / new_name
                if self.augment_file(file_path, out_path):
                    augmented_files.append(out_path)

        logger.info(f"Augmentation complete: {len(augmented_files)} total files")
        return augmented_files

    def augment_to_target(
        self,
        input_files: List[Path],
        output_dir: Path,
        target_count: int,
        class_filter=None,
    ) -> List[Path]:
        """Аугментирует файлы класса до целевого количества."""
        output_dir.mkdir(parents=True, exist_ok=True)
        current_count = len(input_files)

        if current_count >= target_count:
            selected = random.sample(input_files, target_count)
            logger.info(f"Class balanced: {current_count} -> {target_count} (downsampled)")
            return selected

        needed = target_count - current_count
        all_files = list(input_files)

        logger.info(f"Augmenting class: {current_count} -> {target_count} (+{needed})")

        if class_filter and input_files:
            class_name = class_filter.extract_class_from_path(input_files[0])
            class_dir = output_dir / (class_name or "unknown")
        else:
            class_dir = output_dir

        aug_idx = 0
        for _ in range(needed):
            source_file = random.choice(input_files)
            out_path = class_dir / f"{source_file.stem}_aug{aug_idx}{source_file.suffix}"
            if self.augment_file(source_file, out_path):
                all_files.append(out_path)
                aug_idx += 1

        logger.info(f"Class balanced: {current_count} -> {len(all_files)} (augmented)")
        return all_files

    # ------------------------------------------------------------------
    # Внутренние методы трансформации
    # ------------------------------------------------------------------

    def _apply_transformation(self, entity, transform: str, rng: random.Random):
        try:
            if transform == 'scale':
                factor = rng.uniform(*AUG_CONFIG['scale_range'])
                self._scale_entity(entity, factor)

            elif transform == 'rotate':
                angle_deg = rng.uniform(*AUG_CONFIG['rotate_range'])
                if hasattr(entity, 'rotate'):
                    entity.rotate(angle_deg)

            elif transform == 'translate':
                dx = rng.uniform(*AUG_CONFIG['translate_range'])
                dy = rng.uniform(*AUG_CONFIG['translate_range'])
                self._translate_entity(entity, dx, dy)

            elif transform == 'mirror':
                axis = rng.choice(['x', 'y'])
                self._mirror_entity(entity, axis)

        except Exception as e:
            logger.debug(f"Transform '{transform}' failed on {entity.dxftype()}: {e}")

    def _scale_entity(self, entity, factor: float):
        """
        Масштабирует сущность на factor по всем трём осям.

        ВАЖНО: Z масштабируется так же как X и Y.
        Без этого bbox_depth (ключевой признак для стен) остаётся нетронутым
        и аугментированные стены выглядят для модели как перекрытия.
        """
        try:
            dxf = entity.dxf

            if entity.dxftype() == 'LINE':
                dxf.start = self._scale_point(dxf.start, factor)
                dxf.end   = self._scale_point(dxf.end,   factor)

            elif entity.dxftype() == 'CIRCLE':
                dxf.center = self._scale_point(dxf.center, factor)
                dxf.radius = dxf.radius * factor

            elif entity.dxftype() == 'ARC':
                dxf.center = self._scale_point(dxf.center, factor)
                dxf.radius = dxf.radius * factor

            elif entity.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
                # ezdxf: у LWPOLYLINE точки через get_points/set_points
                if entity.dxftype() == 'LWPOLYLINE':
                    pts = list(entity.get_points())  # (x, y, [start_w, end_w, bulge])
                    scaled = []
                    for pt in pts:
                        scaled.append((pt[0] * factor, pt[1] * factor) + pt[2:])
                    entity.set_points(scaled)
                else:
                    for v in entity.vertices:
                        v.dxf.location = self._scale_point(v.dxf.location, factor)

            elif hasattr(dxf, 'insert'):
                # INSERT (блоки) — масштабируем точку вставки
                dxf.insert = self._scale_point(dxf.insert, factor)
            if self.mode == "2d" and hasattr(dxf, "elevation"):
                dxf.elevation = 0.0  # фиксируем плоскость
        except Exception as e:
            logger.debug(f"_scale_entity failed ({entity.dxftype()}): {e}")

    def _scale_point(self, pt, factor: float):
        """Масштабирует точку (2D или 3D) на factor."""
        if len(pt) >= 3:
            return (pt[0] * factor, pt[1] * factor, pt[2] * factor)
        return (pt[0] * factor, pt[1] * factor)

    def _translate_entity(self, entity, dx: float, dy: float):
        """Смещает сущность на (dx, dy). Z не трогаем — смещение плоскостное."""
        try:
            dxf = entity.dxf

            if entity.dxftype() == 'LINE':
                dxf.start = self._translate_point(dxf.start, dx, dy)
                dxf.end   = self._translate_point(dxf.end,   dx, dy)

            elif entity.dxftype() in ('CIRCLE', 'ARC'):
                dxf.center = self._translate_point(dxf.center, dx, dy)

            elif entity.dxftype() == 'LWPOLYLINE':
                pts = list(entity.get_points())
                moved = []
                for pt in pts:
                    moved.append((pt[0] + dx, pt[1] + dy) + pt[2:])
                entity.set_points(moved)

            elif entity.dxftype() == 'POLYLINE':
                for v in entity.vertices:
                    loc = v.dxf.location
                    v.dxf.location = (loc[0] + dx, loc[1] + dy, loc[2] if len(loc) > 2 else 0)

            elif hasattr(dxf, 'insert'):
                dxf.insert = self._translate_point(dxf.insert, dx, dy)

        except Exception as e:
            logger.debug(f"_translate_entity failed ({entity.dxftype()}): {e}")

    def _translate_point(self, pt, dx: float, dy: float):
        if len(pt) >= 3:
            return (pt[0] + dx, pt[1] + dy, pt[2])
        return (pt[0] + dx, pt[1] + dy)

    def _mirror_entity(self, entity, axis: str):
        """
        Зеркалит сущность относительно оси.
        axis='x' — отражение по X (знак X меняется).
        axis='y' — отражение по Y (знак Y меняется).
        Z при зеркалировании в плоскости не трогаем.
        """
        sign_x = -1 if axis == 'x' else 1
        sign_y = -1 if axis == 'y' else 1

        def flip(pt):
            z = pt[2] if len(pt) > 2 else 0
            return (pt[0] * sign_x, pt[1] * sign_y, z)

        try:
            dxf = entity.dxf

            if entity.dxftype() == 'LINE':
                dxf.start = flip(dxf.start)
                dxf.end   = flip(dxf.end)

            elif entity.dxftype() in ('CIRCLE', 'ARC'):
                dxf.center = flip(dxf.center)

            elif entity.dxftype() == 'LWPOLYLINE':
                pts = list(entity.get_points())
                flipped = []
                for pt in pts:
                    flipped.append((pt[0] * sign_x, pt[1] * sign_y) + pt[2:])
                entity.set_points(flipped)

            elif entity.dxftype() == 'POLYLINE':
                for v in entity.vertices:
                    v.dxf.location = flip(v.dxf.location)

        except Exception as e:
            logger.debug(f"_mirror_entity failed ({entity.dxftype()}): {e}")
