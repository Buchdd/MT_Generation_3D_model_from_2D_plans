"""
revit_generator.py — генерация JSON команд для Dynamo/Revit.

Берёт ParsedFloorPlan из layer_parser и создаёт JSON который
GenerateRevit.dyn читает внутри Revit и размещает элементы.

Единицы в DXF: миллиметры (стандарт для российских чертежей).
Единицы Revit API: футы (1 фут = 304.8 мм).
Конвертация: значение_мм / 304.8 = значение_футах.
"""

import json
import math
from pathlib import Path
from src.layer_parser import parse_floor_plan, parse_directory, ParsedFloorPlan
from src.config import PROCESSED_DIR, PRED_RAW_DIR, REVIT_GENERATOR_CONFIG
from src.utils import setup_logger

logger = setup_logger("RevitGenerator")

# 1 фут = 304.8 мм — стандартный коэффициент Revit API
MM_TO_FT = 1.0 / 304.8


def mm_to_ft(val: float) -> float:
    return val * MM_TO_FT


def pt_to_ft(pt: tuple, offset: tuple = (0.0, 0.0)) -> list:
    """
    Конвертирует точку (x, y) из мм в футы с вычитанием offset.
    offset — минимальные координаты чертежа (origin_offset из ParsedFloorPlan).
    Центрирует чертёж относительно нуля Revit.
    Без этого элементы оказываются в километрах от начала координат
    если DXF хранится в геодезических координатах.
    """
    return [(pt[0] - offset[0]) * MM_TO_FT,
            (pt[1] - offset[1]) * MM_TO_FT,
            0.0]


# ---------------------------------------------------------------------------
# Формирование команд из ParsedFloorPlan
# ---------------------------------------------------------------------------

def _wall_commands(plan: ParsedFloorPlan, cfg: dict, offset=(0.0,0.0)) -> list:
    """
    Генерирует команды для наружных стен.
    Каждая стена: метод wall, осевая линия start→end, толщина, высота.
    """
    commands = []
    height_ft = cfg['default_wall_height'] * MM_TO_FT * 1000  # метры → мм → футы

    for i, wall in enumerate(plan.walls):
        length_mm = math.hypot(
            wall.end[0] - wall.start[0],
            wall.end[1] - wall.start[1]
        )
        # Минимальная длина стены 100мм — отсекаем шум
        if length_mm < 100:
            continue

        thickness_ft = wall.thickness * MM_TO_FT
        # Ограничиваем разумными пределами: 50мм — 1000мм
        thickness_ft = max(mm_to_ft(50), min(thickness_ft, mm_to_ft(1000)))

        commands.append({
            'id':       f"wall_{i}",
            'method':   'wall',
            'category': 'Walls',
            'layer':    wall.layer,
            'geometry': {
                'start': pt_to_ft(wall.start, offset),
                'end':   pt_to_ft(wall.end, offset),
            },
            'params': {
                'height_ft':    round(height_ft, 6),
                'thickness_ft': round(thickness_ft, 6),
                'is_outer':     wall.is_outer,
            },
        })

    return commands


def _partition_commands(plan: ParsedFloorPlan, cfg: dict, offset=(0.0,0.0)) -> list:
    """Команды для перегородок — то же что стены но тоньше."""
    commands = []
    height_ft = cfg['default_wall_height'] * MM_TO_FT * 1000

    for i, wall in enumerate(plan.partitions):
        length_mm = math.hypot(
            wall.end[0] - wall.start[0],
            wall.end[1] - wall.start[1]
        )
        if length_mm < 50:
            continue

        thickness_ft = wall.thickness * MM_TO_FT
        thickness_ft = max(mm_to_ft(50), min(thickness_ft, mm_to_ft(500)))

        commands.append({
            'id':       f"partition_{i}",
            'method':   'wall',
            'category': 'Walls',
            'layer':    wall.layer,
            'geometry': {
                'start': pt_to_ft(wall.start),
                'end':   pt_to_ft(wall.end),
            },
            'params': {
                'height_ft':    round(height_ft, 6),
                'thickness_ft': round(thickness_ft, 6),
                'is_outer':     False,
            },
        })

    return commands


def _door_commands(plan: ParsedFloorPlan, cfg: dict, offset=(0.0,0.0)) -> list:
    """
    Команды для дверей.
    Метод: family_instance по точке вставки + угол поворота.
    Revit сам найдёт ближайшую стену и врежет дверь.
    """
    commands = []
    for i, door in enumerate(plan.doors):
        width_ft = door.width * MM_TO_FT
        # Разумная ширина двери: 600мм — 3000мм
        width_ft = max(mm_to_ft(600), min(width_ft, mm_to_ft(3000)))

        commands.append({
            'id':         f"door_{i}",
            'method':     'door',
            'category':   'Doors',
            'layer':      door.layer,
            'block_name': door.block_name,
            'geometry': {
                'position': pt_to_ft(door.position, offset),
                'rotation': door.rotation,
            },
            'params': {
                'width_ft': round(width_ft, 6),
            },
        })

    return commands


def _window_commands(plan: ParsedFloorPlan, cfg: dict, offset=(0.0,0.0)) -> list:
    """Команды для окон."""
    commands = []
    for i, win in enumerate(plan.windows):
        width_ft  = win.width * MM_TO_FT
        width_ft  = max(mm_to_ft(300), min(width_ft, mm_to_ft(5000)))
        sill_ft   = cfg.get('default_sill_height', 0.9) * MM_TO_FT * 1000

        commands.append({
            'id':         f"window_{i}",
            'method':     'window',
            'category':   'Windows',
            'layer':      win.layer,
            'block_name': win.block_name,
            'geometry': {
                'position': pt_to_ft(win.position, offset),
                'rotation': win.rotation,
            },
            'params': {
                'width_ft': round(width_ft, 6),
                'sill_height_ft': round(sill_ft, 6),
            },
        })

    return commands


def _stair_commands(plan: ParsedFloorPlan, cfg: dict, offset=(0.0,0.0)) -> list:
    """Команды для лестниц — bbox контур."""
    commands = []
    for i, stair in enumerate(plan.stairs):
        if len(stair.points) < 2:
            continue
        commands.append({
            'id':       f"stair_{i}",
            'method':   'stair',
            'category': 'Stairs',
            'layer':    stair.layer,
            'geometry': {
                'points': [pt_to_ft(p, offset) for p in stair.points],
                'bbox': {
                    'min_x': (stair.bbox[0]-offset[0]) * MM_TO_FT,
                    'min_y': (stair.bbox[1]-offset[1]) * MM_TO_FT,
                    'max_x': (stair.bbox[2]-offset[0]) * MM_TO_FT,
                    'max_y': (stair.bbox[3]-offset[1]) * MM_TO_FT,
                },
            },
            'params': {},
        })

    return commands


# ---------------------------------------------------------------------------
# Главные функции
# ---------------------------------------------------------------------------

def generate_from_plan(plan: ParsedFloorPlan,
                        output_json: Path = None) -> dict:
    """
    Генерирует JSON команды из ParsedFloorPlan.
    Возвращает словарь с командами (и сохраняет JSON если указан путь).
    """
    cfg    = REVIT_GENERATOR_CONFIG
    offset = getattr(plan, 'origin_offset', (0.0, 0.0))

    logger.info(f"Origin offset: ({offset[0]:.0f}, {offset[1]:.0f}) мм")

    commands = []
    commands += _wall_commands(plan, cfg, offset)
    commands += _partition_commands(plan, cfg, offset)
    commands += _door_commands(plan, cfg, offset)
    commands += _window_commands(plan, cfg, offset)
    commands += _stair_commands(plan, cfg, offset)

    # Статистика
    by_method = {}
    for cmd in commands:
        m = cmd['method']
        by_method[m] = by_method.get(m, 0) + 1

    source = Path(plan.source_file).name
    logger.info(f"\n{source} — сгенерировано команд: {len(commands)}")
    for method, cnt in sorted(by_method.items()):
        logger.info(f"  {method:<15} {cnt}")

    result = {
        'source_dxf':    plan.source_file,
        'total':         len(commands),
        'by_method':     by_method,
        'level_name':    cfg.get('default_level_name', 'Level 1'),
        'commands':      commands,
    }

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON сохранён: {output_json}")

    return result


def generate_for_directory(dxf_dir: Path = None,
                            output_dir: Path = None,
                            predictions_dir=None,  # оставлен для совместимости
                            ):
    """
    Парсит все DXF в папке и генерирует JSON команды для каждого.
    """
    if dxf_dir    is None: dxf_dir    = PRED_RAW_DIR
    if output_dir is None: output_dir = PROCESSED_DIR / "revit_commands"

    dxf_dir    = Path(dxf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dxf_files = list(dxf_dir.glob("*.dxf"))
    if not dxf_files:
        logger.error(f"DXF файлы не найдены в {dxf_dir}")
        return []

    logger.info(f"Найдено {len(dxf_files)} DXF файлов")
    logger.info("="*60)

    results = []
    for dxf_path in dxf_files:
        logger.info(f"\nОбрабатываем: {dxf_path.name}")
        logger.info("-"*40)

        # Парсинг по слоям
        plan = parse_floor_plan(dxf_path)

        # Генерация JSON
        json_path = output_dir / f"{dxf_path.stem}_revit_commands.json"
        data = generate_from_plan(plan, output_json=json_path)
        results.append(data)

    logger.info("\n" + "="*60)
    logger.info(f"Готово. JSON файлы в: {output_dir}")
    logger.info("Следующий шаг: Revit → Dynamo → GenerateRevit.dyn")
    logger.info("="*60)

    return results
