"""
ifc_generator.py — генерация IFC файла из ParsedFloorPlan.

IFC (Industry Foundation Classes) — стандартный формат BIM.
Revit открывает его через Файл → Открыть → IFC без скриптов и Dynamo.

Что генерируем:
    IfcWall       — наружные стены и перегородки
    IfcDoor       — двери
    IfcWindow     — окна
    IfcSlab       — перекрытия (если есть)
    IfcStairFlight — лестницы (упрощённо)

Единицы: всё в миллиметрах (стандарт IFC).
"""

import math
import uuid
import time
import ifcopenshell
import ifcopenshell.api
from pathlib import Path
from src.layer_parser import parse_floor_plan, ParsedFloorPlan
from src.config import PROCESSED_DIR, PRED_RAW_DIR, REVIT_GENERATOR_CONFIG
from src.utils import setup_logger

logger = setup_logger("IFCGenerator")


def _f(*args) -> list:
    """Конвертирует координаты в чистые Python float для ifcopenshell."""
    return [float(a) for a in args]


def new_guid() -> str:
    return ifcopenshell.guid.compress(uuid.uuid4().hex)


def create_ifc_project(filename: str) -> tuple:
    """Создаёт базовую IFC структуру: проект, сайт, здание, этаж."""
    ifc = ifcopenshell.file(schema="IFC4")

    # --- Контекст ---
    ctx = ifc.createIfcGeometricRepresentationContext(
        None, "Model", 3, 1e-5,
        ifc.createIfcAxis2Placement3D(
            ifc.createIfcCartesianPoint(_f(0.0, 0.0, 0.0)),
            ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
            ifc.createIfcDirection(_f(1.0, 0.0, 0.0)),
        ),
        None
    )
    ctx.ContextType = "Model"

    plan_ctx = ifc.createIfcGeometricRepresentationSubContext(
        "FootPrint", "Model", None, None, None, None, ctx, None,
        "PLAN_VIEW", None
    )
    body_ctx = ifc.createIfcGeometricRepresentationSubContext(
        "Body", "Model", None, None, None, None, ctx, None,
        "MODEL_VIEW", None
    )

    # --- Единицы (миллиметры) ---
    mm_unit = ifc.createIfcSIUnit(None, "LENGTHUNIT", "MILLI", "METRE")
    unit_assign = ifc.createIfcUnitAssignment([mm_unit])

    # --- Проект ---
    project = ifc.createIfcProject(
        new_guid(), None,
        filename, None, None, None, None,
        [ctx], unit_assign
    )

    # --- Расположение по умолчанию ---
    origin = ifc.createIfcCartesianPoint(_f(0.0, 0.0, 0.0))
    z_axis = ifc.createIfcDirection(_f(0.0, 0.0, 1.0))
    x_axis = ifc.createIfcDirection(_f(1.0, 0.0, 0.0))
    placement3d = ifc.createIfcAxis2Placement3D(origin, z_axis, x_axis)
    local_placement = ifc.createIfcLocalPlacement(None, placement3d)

    # --- Сайт ---
    site = ifc.createIfcSite(
        new_guid(), None, "Site", None, None,
        local_placement, None, None,
        "ELEMENT", None, None, None, None, None
    )
    ifc.createIfcRelAggregates(
        new_guid(), None, None, None, project, [site]
    )

    # --- Здание ---
    building = ifc.createIfcBuilding(
        new_guid(), None, "Building", None, None,
        local_placement, None, None,
        "ELEMENT", None, None, None
    )
    ifc.createIfcRelAggregates(
        new_guid(), None, None, None, site, [building]
    )

    # --- Этаж ---
    storey_placement = ifc.createIfcLocalPlacement(
        local_placement,
        ifc.createIfcAxis2Placement3D(
            ifc.createIfcCartesianPoint(_f(0.0, 0.0, 0.0)),
            ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
            ifc.createIfcDirection(_f(1.0, 0.0, 0.0)),
        )
    )
    storey = ifc.createIfcBuildingStorey(
        new_guid(), None, "Level 1", None, None,
        storey_placement, None, None,
        "ELEMENT", 0.0
    )
    ifc.createIfcRelAggregates(
        new_guid(), None, None, None, building, [storey]
    )

    return ifc, storey, body_ctx, plan_ctx, local_placement


def make_wall_shape(ifc, sx, sy, ex, ey, thickness, height, body_ctx):
    """Создаёт 3D геометрию стены как вытянутый прямоугольник."""
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length < 1:
        return None

    angle = math.atan2(dy, dx)

    # Профиль стены — прямоугольник в плане
    half_t = thickness / 2.0
    profile = ifc.createIfcRectangleProfileDef(
        "AREA", None,
        ifc.createIfcAxis2Placement2D(
            ifc.createIfcCartesianPoint(_f(length / 2.0, 0.0)),
            ifc.createIfcDirection(_f(1.0, 0.0))
        ),
        length, thickness
    )

    # Вытягиваем по Z
    solid = ifc.createIfcExtrudedAreaSolid(
        profile,
        ifc.createIfcAxis2Placement3D(
            ifc.createIfcCartesianPoint(_f(0.0, 0.0, 0.0)),
            ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
            ifc.createIfcDirection(_f(1.0, 0.0, 0.0)),
        ),
        ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
        height
    )

    shape = ifc.createIfcShapeRepresentation(
        body_ctx, "Body", "SweptSolid", [solid]
    )
    return shape


def make_opening_shape(ifc, width, height, body_ctx):
    """Создаёт простую 3D геометрию для двери/окна."""
    profile = ifc.createIfcRectangleProfileDef(
        "AREA", None,
        ifc.createIfcAxis2Placement2D(
            ifc.createIfcCartesianPoint(_f(width / 2.0, 0.0)),
            ifc.createIfcDirection(_f(1.0, 0.0))
        ),
        width, 100.0  # символическая толщина
    )
    solid = ifc.createIfcExtrudedAreaSolid(
        profile,
        ifc.createIfcAxis2Placement3D(
            ifc.createIfcCartesianPoint(_f(0.0, 0.0, 0.0)),
            ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
            ifc.createIfcDirection(_f(1.0, 0.0, 0.0)),
        ),
        ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
        height
    )
    shape = ifc.createIfcShapeRepresentation(
        body_ctx, "Body", "SweptSolid", [solid]
    )
    return shape


def add_wall(ifc, storey, wall, body_ctx, local_pl,
             height_mm, is_outer=True):
    """Добавляет IfcWall в модель."""
    sx, sy = wall.start
    ex, ey = wall.end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1:
        return None

    angle = math.atan2(dy, dx)
    thickness = max(50.0, min(wall.thickness, 1000.0))

    # Размещение стены
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    placement = ifc.createIfcLocalPlacement(
        local_pl,
        ifc.createIfcAxis2Placement3D(
            ifc.createIfcCartesianPoint((float(sx), float(sy), 0.0)),
            ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
            ifc.createIfcDirection(_f(cos_a, sin_a, 0.0)),
        )
    )

    shape = make_wall_shape(
        ifc, 0, 0, length, 0, thickness, height_mm, body_ctx
    )
    if shape is None:
        return None

    prod_def = ifc.createIfcProductDefinitionShape(None, None, [shape])

    wall_el = ifc.createIfcWall(
        new_guid(), None,
        "Outer Wall" if is_outer else "Partition",
        None, None,
        placement, prod_def, None
    )

    # Связываем с этажом
    ifc.createIfcRelContainedInSpatialStructure(
        new_guid(), None, None, None,
        [wall_el], storey
    )
    return wall_el


def add_opening(ifc, storey, opening, body_ctx, local_pl,
                kind='door', height_mm=2100.0, sill_mm=0.0):
    """Добавляет IfcDoor или IfcWindow в модель."""
    px, py = opening.position
    angle_rad = math.radians(opening.rotation)
    width = max(300.0, min(opening.width, 5000.0))

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    placement = ifc.createIfcLocalPlacement(
        local_pl,
        ifc.createIfcAxis2Placement3D(
            ifc.createIfcCartesianPoint((float(px), float(py), float(sill_mm))),
            ifc.createIfcDirection(_f(0.0, 0.0, 1.0)),
            ifc.createIfcDirection(_f(cos_a, sin_a, 0.0)),
        )
    )

    shape = make_opening_shape(ifc, width, height_mm, body_ctx)
    prod_def = ifc.createIfcProductDefinitionShape(None, None, [shape])

    if kind == 'door':
        el = ifc.createIfcDoor(
            new_guid(), None, "Door", None, None,
            placement, prod_def, None,
            height_mm, width
        )
    else:
        el = ifc.createIfcWindow(
            new_guid(), None, "Window", None, None,
            placement, prod_def, None,
            height_mm, width
        )

    ifc.createIfcRelContainedInSpatialStructure(
        new_guid(), None, None, None,
        [el], storey
    )
    return el


def generate_ifc(plan: ParsedFloorPlan,
                 output_path: Path = None) -> Path:
    """
    Генерирует IFC файл из ParsedFloorPlan.
    Возвращает путь к созданному файлу.
    """
    cfg = REVIT_GENERATOR_CONFIG

    source_name = Path(plan.source_file).stem
    if output_path is None:
        output_path = (PROCESSED_DIR / "ifc" / f"{source_name}.ifc")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Параметры из конфига (метры → мм)
    wall_height_mm = cfg['default_wall_height'] * 1000.0   # 3м → 3000мм
    sill_height_mm = cfg.get('default_sill_height', 0.9) * 1000.0

    logger.info(f"\nГенерация IFC: {source_name}")
    logger.info(f"  Стены:       {len(plan.walls)}")
    logger.info(f"  Перегородки: {len(plan.partitions)}")
    logger.info(f"  Двери:       {len(plan.doors)}")
    logger.info(f"  Окна:        {len(plan.windows)}")
    logger.info(f"  Лестницы:    {len(plan.stairs)}")

    # --- ML/GNN Fallback для нераспознанных слоёв ---
    # Если есть слои с непонятными именами (слой '0', технические коды)
    # — пробуем классифицировать их через ML и GNN
    fallback_report = {}
    if plan.unknown_layers:
        try:
            from src.ml_fallback import enhance_plan_with_ml_gnn
            fallback_report = enhance_plan_with_ml_gnn(
                plan, Path(plan.source_file),
                use_ml=True, use_gnn=True,
            )
        except Exception as e:
            logger.warning(f"ML/GNN fallback не удался: {e}")

    ifc, storey, body_ctx, plan_ctx, local_pl = create_ifc_project(source_name)

    placed = {'walls': 0, 'partitions': 0, 'doors': 0, 'windows': 0}

    # Наружные стены
    for wall in plan.walls:
        el = add_wall(ifc, storey, wall, body_ctx, local_pl,
                      wall_height_mm, is_outer=True)
        if el:
            placed['walls'] += 1

    # Перегородки
    for wall in plan.partitions:
        el = add_wall(ifc, storey, wall, body_ctx, local_pl,
                      wall_height_mm, is_outer=False)
        if el:
            placed['partitions'] += 1

    # Двери
    for door in plan.doors:
        el = add_opening(ifc, storey, door, body_ctx, local_pl,
                         kind='door', height_mm=2100.0, sill_mm=0.0)
        if el:
            placed['doors'] += 1

    # Окна
    for win in plan.windows:
        el = add_opening(ifc, storey, win, body_ctx, local_pl,
                         kind='window', height_mm=1200.0,
                         sill_mm=sill_height_mm)
        if el:
            placed['windows'] += 1

    ifc.write(str(output_path))

    total = sum(placed.values())
    logger.info(f"  Размещено: {total} элементов")
    logger.info(f"    Стены: {placed['walls']}, Перегородки: {placed['partitions']}")
    logger.info(f"    Двери: {placed['doors']}, Окна: {placed['windows']}")
    logger.info(f"  IFC сохранён: {output_path}")

    return output_path


def generate_ifc_directory(dir_path: Path = None,
                            output_dir: Path = None) -> list:
    """Генерирует IFC для всех DXF в папке pred_raw."""
    if dir_path   is None: dir_path   = PRED_RAW_DIR
    if output_dir is None: output_dir = PROCESSED_DIR / "ifc"

    dir_path   = Path(dir_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dxf_files = list(dir_path.glob("*.dxf"))
    logger.info(f"Найдено {len(dxf_files)} DXF файлов")

    results = []
    for dxf_path in dxf_files:
        logger.info(f"\n{'='*50}")
        plan = parse_floor_plan(dxf_path)
        out  = output_dir / f"{dxf_path.stem}.ifc"
        try:
            ifc_path = generate_ifc(plan, out)
            results.append(str(ifc_path))
        except Exception as e:
            logger.error(f"Ошибка {dxf_path.name}: {e}")
            import traceback; traceback.print_exc()

    logger.info(f"\n{'='*50}")
    logger.info(f"Готово. IFC файлы в: {output_dir}")
    logger.info("Открывай в Revit: Файл → Открыть → IFC")
    return results
