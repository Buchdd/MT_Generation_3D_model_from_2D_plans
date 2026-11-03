from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional
from src.utils import setup_logger, extract_class_from_filename

logger = setup_logger("ClassFilter")

class RevitClassFilter:
    # 🚫 Папки, файлы из которых ПОЛНОСТЬЮ игнорируются
    EXCLUDE_FOLDERS = [
        'Для России', 'для россии', 'Россия', 'россия',
        'Common', 'common', 'Общие', 'общие',
        'Templates', 'templates', 'Шаблоны', 'шаблоны',
        'System', 'system', 'Системные', 'системные',
        'Archive', 'archive', 'Архив', 'архив',
        'Backup', 'backup', 'Резервные', 'резервные',
        'Old', 'old', 'Старые', 'старые',
        'Test', 'test', 'Тест', 'тест',
        'Example', 'example', 'Пример', 'пример',
        'Sample', 'sample', 'Образец', 'образец',
    ]

    # ⚠️ Папки, которые НЕ должны быть классами (но файлы обрабатываются)
    IGNORED_FOLDER_NAMES = [
        'Train', 'train', 'Training', 'training',
        'DXF', 'dxf', 'Data', 'data',
        'Raw', 'raw', 'Source', 'source',
        'Input', 'input', 'Output', 'output',
        'Processed', 'processed', 'Temp', 'temp',
        'Russia', 'Russian', 'Libraries', 'RVT 2024',
        'ГОСТ', 'GOST', 'AISC', 'ASTM', 'ISO',
        'Стандарт', 'стандарт', 'Норматив', 'норматив',
    ]

    def __init__(
        self,
        base_library_path: str = None,
        exclude_folders: List[str] = None,
        ignored_folder_names: List[str] = None,
        path_class_mapping: dict = None,
    ):
        self.base_library_path = Path(base_library_path) if base_library_path else None
        self.exclude_folders = exclude_folders or self.EXCLUDE_FOLDERS
        self.ignored_folder_names = ignored_folder_names or self.IGNORED_FOLDER_NAMES
        # Явный маппинг подстрок пути -> имя класса (из config.path_class_mapping)
        self.path_class_mapping = path_class_mapping or {}
        self.class_mapping = {}
    
    def should_exclude_file(self, file_path: Path) -> bool:
        """Проверяет, должен ли файл быть полностью исключён"""
        file_path_str = str(file_path).lower()
        
        for exclude_folder in self.exclude_folders:
            if exclude_folder.lower() in file_path_str:
                return True
        return False

    def extract_class_from_path(self, file_path: Path) -> Optional[str]:
        """
        Извлекает класс из пути к файлу.

        Порядок проверок:
        1. Исключение — файл в запрещённой папке -> None
        2. path_class_mapping — явный маппинг подстрок пути -> класс
           (более длинные ключи первыми — они более специфичны)
        3. Родительская папка — если не в ignored_folder_names
        4. Имя файла как запасной вариант
        """
        if self.should_exclude_file(file_path):
            logger.debug(f"Excluded: {file_path}")
            return None

        # Шаг 2: явный маппинг по подстроке пути
        if self.path_class_mapping:
            path_lower = str(file_path).lower().replace('\\', '/')
            for key in sorted(self.path_class_mapping, key=len, reverse=True):
                if key in path_lower:
                    return self.path_class_mapping[key]

        # Шаг 3: имя родительской папки
        parent_folder = file_path.parent.name
        if parent_folder and parent_folder not in self.ignored_folder_names:
            if parent_folder not in self.exclude_folders:
                return parent_folder

        # Шаг 4: из имени файла
        return extract_class_from_filename(file_path.name)

    def group_files_by_class(self, dxf_files: List[Path]) -> Dict[str, List[Path]]:
        """Группирует файлы по классам"""
        groups = {}
        excluded_count = 0
        
        for file_path in dxf_files:
            class_name = self.extract_class_from_path(file_path)
            if class_name is None:
                excluded_count += 1
                continue
            if class_name not in groups:
                groups[class_name] = []
            groups[class_name].append(file_path)
        
        logger.info(f"📁 Excluded {excluded_count} files from ignored folders")
        return groups

    def filter_classes(
        self, 
        dxf_files: List[Path],
        min_samples: int = 5,
        max_samples_per_class: int = None,
        include_classes: List[str] = None,
        exclude_classes: List[str] = None
    ) -> Tuple[List[Path], Dict[str, int]]:
        """Фильтрует файлы по классам"""
        groups = self.group_files_by_class(dxf_files)
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 СТАТИСТИКА КЛАССОВ (ДО ФИЛЬТРАЦИИ)")
        logger.info(f"{'='*60}")
        logger.info(f"Всего файлов: {len(dxf_files)}")
        logger.info(f"Всего классов: {len(groups)}")
        
        filtered_files = []
        class_stats = {}
        
        for class_name, files in sorted(groups.items(), key=lambda x: -len(x[1])):
            if include_classes and class_name not in include_classes:
                continue
            if exclude_classes and class_name in exclude_classes:
                continue
            
            if len(files) < min_samples:
                logger.debug(f"⚠️  Класс '{class_name}': {len(files)} < {min_samples} (пропущен)")
                continue
            
            if max_samples_per_class and len(files) > max_samples_per_class:
                import random
                files = random.sample(files, max_samples_per_class)
                logger.info(f"📉 Класс '{class_name}': обрезано до {max_samples_per_class}")
            
            filtered_files.extend(files)
            class_stats[class_name] = len(files)
            logger.info(f"✅ Класс '{class_name}': {len(files)} образцов")
        
        logger.info(f"{'='*60}")
        logger.info(f"ПОСЛЕ ФИЛЬТРАЦИИ:")
        logger.info(f"  Файлов: {len(filtered_files)}")
        logger.info(f"  Классов: {len(class_stats)}")
        logger.info(f"{'='*60}\n")
        
        return filtered_files, class_stats

    def balance_classes(
        self, 
        dxf_files: List[Path],
        target_samples: int,
        augmentor=None,
        output_dir: Path = None
    ) -> Tuple[List[Path], Dict[str, int]]:
        """
        Балансирует классы до целевого количества.
        - Большие классы: случайная выборка (downsample)
        - Малые классы: аугментация (upsample)
        """
        import random
        groups = self.group_files_by_class(dxf_files)
        
        logger.info(f"\n{'='*60}")
        logger.info("⚖️ БАЛАНСИРОВКА КЛАССОВ")
        logger.info(f"{'='*60}")
        logger.info(f"Целевое количество: {target_samples} образцов на класс")
        
        balanced_files = []
        class_stats = {}
        
        for class_name, files in sorted(groups.items(), key=lambda x: -len(x[1])):
            current_count = len(files)
            
            if current_count >= target_samples:
                # 🔽 Слишком много — берём случайную выборку
                selected = random.sample(files, target_samples)
                balanced_files.extend(selected)
                class_stats[class_name] = target_samples
                logger.info(f"📉 Класс '{class_name}': {current_count} → {target_samples} (downsampled)")
            else:
                # 🔼 Слишком мало — аугментируем
                if augmentor and output_dir:
                    needed = target_samples - current_count
                    all_files = list(files)
                    
                    class_dir = output_dir / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    aug_idx = 0
                    for _ in range(needed):
                        source_file = random.choice(files)
                        new_name = f"{source_file.stem}_aug{aug_idx}{source_file.suffix}"
                        output_path = class_dir / new_name
                        
                        if augmentor.augment_file(source_file, output_path):
                            all_files.append(output_path)
                            aug_idx += 1
                    
                    balanced_files.extend(all_files)
                    class_stats[class_name] = len(all_files)
                    logger.info(f"📈 Класс '{class_name}': {current_count} → {len(all_files)} (augmented)")
                else:
                    # Нет аугментора — оставляем как есть
                    balanced_files.extend(files)
                    class_stats[class_name] = current_count
                    logger.info(f"⚠️  Класс '{class_name}': {current_count} (no augmentation)")
        
        logger.info(f"{'='*60}")
        logger.info("ПОСЛЕ БАЛАНСИРОВКИ:")
        logger.info(f"  Файлов: {len(balanced_files)}")
        logger.info(f"  Классов: {len(class_stats)}")
        if len(class_stats) > 0:
            logger.info(f"  Среднее на класс: {len(balanced_files)/len(class_stats):.1f}")
        logger.info(f"{'='*60}\n")
        
        return balanced_files, class_stats

    def get_top_classes(self, dxf_files: List[Path], top_n: int = 20) -> List[str]:
        """Возвращает топ-N классов по количеству файлов"""
        groups = self.group_files_by_class(dxf_files)
        sorted_classes = sorted(groups.items(), key=lambda x: -len(x[1]))
        return [c[0] for c in sorted_classes[:top_n]]