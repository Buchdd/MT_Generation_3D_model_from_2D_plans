import math
import numpy as np
from src.dxf_reader import load_dxf_document, get_entities
from src.config import TARGET_ENTITY_TYPES
from src.utils import setup_logger

logger = setup_logger("MLFeatures")

# ---------------------------------------------------------------------------
# Контекст файла — признаки всего DXF целиком
# ---------------------------------------------------------------------------

def extract_file_context(msp, entity_types=None) -> dict:
    """
    Вычисляет признаки файла целиком — контекст для каждой entity внутри.

    Зачем это нужно:
    Модель раньше видела одну линию в изоляции. Стена из библиотечного символа
    Revit (5-10 entity, маленький bbox) и стена из реального чертежа (сотни
    entity, большой bbox) — неотличимы по одной линии. Но по контексту файла
    они очень разные.

    Все признаки имеют префикс 'ctx_' — легко отличить в CSV.
    """
    if entity_types is None:
        entity_types = TARGET_ENTITY_TYPES

    entities = list(get_entities(msp, entity_types))

    if not entities:
        return _empty_file_context()

    # --- Количество entity ---
    ctx_entity_count = len(entities)

    # --- Распределение по типам ---
    type_counts = {t: 0 for t in entity_types}
    for e in entities:
        t = e.dxftype()
        if t in type_counts:
            type_counts[t] += 1

    ctx_ratio_line      = type_counts.get('LINE', 0) / ctx_entity_count
    ctx_ratio_polyline  = (type_counts.get('POLYLINE', 0) +
                           type_counts.get('LWPOLYLINE', 0)) / ctx_entity_count
    ctx_ratio_arc       = type_counts.get('ARC', 0) / ctx_entity_count
    ctx_ratio_circle    = type_counts.get('CIRCLE', 0) / ctx_entity_count

    # --- Статистика длин ---
    lengths = []
    for e in entities:
        try:
            if e.dxftype() == 'LINE':
                s, en = e.dxf.start, e.dxf.end
                lengths.append(math.hypot(en[0]-s[0], en[1]-s[1]))
            elif e.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
                pts = list(e.points())
                if len(pts) > 1:
                    l = sum(math.hypot(pts[i][0]-pts[i-1][0],
                                       pts[i][1]-pts[i-1][1])
                            for i in range(1, len(pts)))
                    lengths.append(l)
        except Exception:
            pass

    if lengths:
        ctx_mean_length   = float(np.mean(lengths))
        ctx_median_length = float(np.median(lengths))
        ctx_max_length    = float(np.max(lengths))
        ctx_std_length    = float(np.std(lengths))
        # Коэффициент вариации — насколько разнородны длины в файле
        ctx_cv_length     = ctx_std_length / ctx_mean_length if ctx_mean_length > 0 else 0.0
    else:
        ctx_mean_length = ctx_median_length = ctx_max_length = 0.0
        ctx_std_length  = ctx_cv_length = 0.0

    # --- Bbox всего файла ---
    try:
        all_x, all_y = [], []
        for e in entities:
            try:
                if e.dxftype() == 'LINE':
                    all_x += [e.dxf.start[0], e.dxf.end[0]]
                    all_y += [e.dxf.start[1], e.dxf.end[1]]
                elif e.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
                    pts = list(e.points())
                    all_x += [p[0] for p in pts]
                    all_y += [p[1] for p in pts]
                elif e.dxftype() in ('CIRCLE', 'ARC'):
                    cx, cy = e.dxf.center[0], e.dxf.center[1]
                    r = e.dxf.radius
                    all_x += [cx - r, cx + r]
                    all_y += [cy - r, cy + r]
            except Exception:
                pass

        if all_x and all_y:
            ctx_file_width  = max(all_x) - min(all_x)
            ctx_file_height = max(all_y) - min(all_y)
            ctx_file_aspect = (ctx_file_width / ctx_file_height
                               if ctx_file_height > 0 else 0.0)
            ctx_file_area   = ctx_file_width * ctx_file_height
        else:
            ctx_file_width = ctx_file_height = ctx_file_aspect = ctx_file_area = 0.0
    except Exception:
        ctx_file_width = ctx_file_height = ctx_file_aspect = ctx_file_area = 0.0

    # --- Плотность: сколько entity на единицу площади файла ---
    ctx_entity_density = (ctx_entity_count / ctx_file_area
                          if ctx_file_area > 0 else 0.0)

    # --- Количество уникальных слоёв ---
    try:
        ctx_layer_count = len(set(e.dxf.layer for e in entities
                                  if hasattr(e.dxf, 'layer')))
    except Exception:
        ctx_layer_count = 0

    return {
        'ctx_entity_count':   ctx_entity_count,
        'ctx_ratio_line':     ctx_ratio_line,
        'ctx_ratio_polyline': ctx_ratio_polyline,
        'ctx_ratio_arc':      ctx_ratio_arc,
        'ctx_ratio_circle':   ctx_ratio_circle,
        'ctx_mean_length':    ctx_mean_length,
        'ctx_median_length':  ctx_median_length,
        'ctx_max_length':     ctx_max_length,
        'ctx_std_length':     ctx_std_length,
        'ctx_cv_length':      ctx_cv_length,
        'ctx_file_width':     ctx_file_width,
        'ctx_file_height':    ctx_file_height,
        'ctx_file_aspect':    ctx_file_aspect,
        'ctx_file_area':      ctx_file_area,
        'ctx_entity_density': ctx_entity_density,
        'ctx_layer_count':    ctx_layer_count,
    }


def _empty_file_context() -> dict:
    """Возвращает нулевой контекст если файл пустой."""
    return {k: 0.0 for k in [
        'ctx_entity_count', 'ctx_ratio_line', 'ctx_ratio_polyline',
        'ctx_ratio_arc', 'ctx_ratio_circle', 'ctx_mean_length',
        'ctx_median_length', 'ctx_max_length', 'ctx_std_length',
        'ctx_cv_length', 'ctx_file_width', 'ctx_file_height',
        'ctx_file_aspect', 'ctx_file_area', 'ctx_entity_density',
        'ctx_layer_count',
    ]}


# ---------------------------------------------------------------------------
# Признаки одной entity
# ---------------------------------------------------------------------------

def extract_bounding_box(msp):
    try:
        bbox = msp.extents()
        if bbox is None or bbox.is_empty:
            return 0.0, 0.0, 0.0
        width = bbox.xmax - bbox.xmin
        height = bbox.ymax - bbox.ymin
        aspect_ratio = width / height if height > 0 else 0.0
        return width, height, aspect_ratio
    except Exception:
        return 0.0, 0.0, 0.0


def extract_features_from_dxf(file_path):
    """Признаки файла целиком (используется для file-level классификации)."""
    doc, msp = load_dxf_document(file_path)
    if doc is None:
        return None
    entities = get_entities(msp, TARGET_ENTITY_TYPES)
    entity_count = len(entities)
    type_counts = {t: 0 for t in TARGET_ENTITY_TYPES}
    for e in entities:
        t = e.dxftype()
        if t in type_counts:
            type_counts[t] += 1
    total_length = 0.0
    for e in entities:
        try:
            if hasattr(e, 'length'):
                total_length += e.length
        except Exception:
            pass
    width, height, aspect_ratio = extract_bounding_box(msp)
    layer_count = len(set(e.dxf.layer for e in entities))
    features = {
        'file_name': file_path.name,
        'entity_count': entity_count,
        'total_length': total_length,
        'width': width, 'height': height,
        'aspect_ratio': aspect_ratio,
        'layer_count': layer_count,
    }
    for t, count in type_counts.items():
        features[f'count_{t.lower()}'] = count
    return features


def extract_entity_features(entity, file_context: dict = None, mode="3d") -> dict:
    """
    Признаки одной entity + опциональный контекст файла.
    
    file_context — результат extract_file_context() для того же файла.
    mode: "3d" (по умолчанию) или "2d" (для планов/видов сверху).
    """
    features = {
        'entity_type': entity.dxftype(),
        'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0',
        'length': 0.0, 'area': 0.0,
        'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0,
        'num_points': 0,
        'bbox_width': 0.0, 'bbox_height': 0.0, 'bbox_depth': 0.0,
        'angle': 0.0,
        'is_tall': False, 'is_thick': False,
        'perimeter_area_ratio': 0.0,
        # 🟢 Базовые 2D-признаки (всегда в dict, чтобы столбцы CSV не "прыгали")
        'is_closed_polyline': False,
        'has_thickness_attr': False,
    }
    
    try:
        if entity.dxftype() == 'LINE':
            start, end = entity.dxf.start, entity.dxf.end
            features['length'] = math.hypot(end[0]-start[0], end[1]-start[1])
            features['angle']  = math.atan2(end[1]-start[1], end[0]-start[0])
            features['center_x'] = (start[0] + end[0]) / 2
            features['center_y'] = (start[1] + end[1]) / 2
            
            # 🔵 Только для 3D
            if mode == "3d":
                z_diff = abs((end[2] if len(end) > 2 else 0) -
                             (start[2] if len(start) > 2 else 0))
                features['bbox_depth'] = z_diff

        elif entity.dxftype() == 'CIRCLE':
            features['radius']   = entity.dxf.radius
            features['area']     = math.pi * entity.dxf.radius ** 2
            features['center_x'] = entity.dxf.center[0]
            features['center_y'] = entity.dxf.center[1]
            
            # 🔵 Только для 3D
            if mode == "3d":
                features['center_z'] = (entity.dxf.center[2]
                                        if len(entity.dxf.center) > 2 else 0.0)

        elif entity.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
            points = list(entity.points())
            if points:
                features['num_points'] = len(points)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                features['bbox_width']  = max(xs) - min(xs)
                features['bbox_height'] = max(ys) - min(ys)
                
                # 🔵 Только для 3D
                if mode == "3d":
                    zs = [p[2] for p in points if len(p) > 2]
                    features['bbox_depth']  = max(zs) - min(zs) if zs else 0.0
                    features['center_z']    = sum(zs) / len(zs) if zs else 0.0
                else:
                    features['center_z'] = 0.0
                    # 🟢 2D: флаг замкнутости контура (важно для стен/перегородок)
                    features['is_closed_polyline'] = bool(getattr(entity, 'is_closed', False))

                features['area']        = features['bbox_width'] * features['bbox_height']
                features['center_x']    = sum(xs) / len(xs)
                features['center_y']    = sum(ys) / len(ys)

                perimeter = sum(
                    math.hypot(points[i][0] - points[i-1][0],
                               points[i][1] - points[i-1][1])
                    for i in range(1, len(points))
                )
                features['perimeter_area_ratio'] = (perimeter / features['area']
                                                    if features['area'] > 0 else 0.0)

        elif entity.dxftype() == 'ARC':
            features['radius']   = entity.dxf.radius
            features['area']     = math.pi * entity.dxf.radius ** 2
            features['center_x'] = entity.dxf.center[0]
            features['center_y'] = entity.dxf.center[1]
            
            # 🔵 Только для 3D
            if mode == "3d":
                features['center_z'] = (entity.dxf.center[2]
                                        if len(entity.dxf.center) > 2 else 0.0)

        # 🟢 2D: атрибут толщины линии/объекта (иногда используется вместо bbox_depth)
        if mode == "2d" and hasattr(entity.dxf, 'thickness'):
            features['has_thickness_attr'] = abs(entity.dxf.thickness) > 1e-3

        # --- Флаги геометрии ---
        if mode == "3d":
            features['is_tall']  = (features['bbox_depth'] > 1.5 or
                                    features['bbox_height'] > 2.5)
            features['is_thick'] = features['bbox_depth'] > 0.3
        else:
            # В 2D Z-глубины нет, поэтому флаги всегда False
            features['is_tall'] = False
            features['is_thick'] = False

        # --- Производные признаки (общие для 2D/3D) ---
        length = features['length']
        area   = features['area']
        bw     = features['bbox_width']
        bh     = features['bbox_height']

        features['log_length']      = math.log1p(length)
        features['log_area']        = math.log1p(area)
        features['length_to_width'] = length / bw if bw > 0 else 0.0
        features['bbox_aspect']     = bw / bh if bh > 0 else 0.0

    except Exception as e:
        logger.debug(f"Feature extraction failed: {e}")

    # --- Контекст файла ---
    if file_context:
        features.update(file_context)

    return features
