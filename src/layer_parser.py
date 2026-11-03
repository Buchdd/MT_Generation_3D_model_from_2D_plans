"""
layer_parser.py — парсинг DXF плана этажа по слоям.

Логика:
  1. Маппинг имён слоёв → классы (словарь + fuzzy)
  2. Стены — группируем параллельные полилинии попарно,
     берём среднюю осевую линию и толщину между ними
  3. Двери/окна — разворачиваем INSERT блоки,
     определяем тип по имени блока
  4. Лестницы — контуры полилиний
  5. UNKNOWN слои — передаём в ML как fallback
"""

import math
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import ezdxf
from src.utils import setup_logger

logger = setup_logger("LayerParser")

# ---------------------------------------------------------------------------
# Словарь маппинга имён слоёв → классы
# ---------------------------------------------------------------------------
LAYER_KEYWORD_MAP = {
    # Стены наружные
    'стен':          'Стены',
    'wall':          'Стены',
    'наруж':         'Стены',
    'exterior':      'Стены',
    'внешн':         'Стены',

    # Перегородки
    'перегород':     'Перегородки',
    'partition':     'Перегородки',
    'partion':       'Перегородки',
    'внутр':         'Перегородки',
    'interior wall': 'Перегородки',

    # Окна
    'окн':           'Окна',
    'window':        'Окна',
    'win':           'Окна',

    # Двери
    'двер':          'Двери',
    'door':          'Двери',

    # Смешанный слой окна+двери — делим по имени блока
    'окна и двери':  'Окна_Двери',
    'doors and':     'Окна_Двери',

    # Лестницы
    'лестниц':       'Лестницы',
    'лестни':        'Лестницы',
    'лест':          'Лестницы',
    'stair':         'Лестницы',

    # Перекрытия
    'перекрыт':      'Перекрытия',
    'floor':         'Перекрытия',
    'slab':          'Перекрытия',
    'потол':         'Потолки',
    'ceiling':       'Потолки',

    # Колонны
    'колонн':        'Колонны',
    'column':        'Колонны',

    # Игнорируем
    'размер':    'IGNORE', 'dimension': 'IGNORE',
    'текст':     'IGNORE', 'text':      'IGNORE',
    'tekst':     'IGNORE', 'mtext':     'IGNORE',
    'ось':       'IGNORE', 'оси':       'IGNORE',
    'axis':      'IGNORE', 'осевые':    'IGNORE',
    'штамп':     'IGNORE', 'stamp':     'IGNORE',
    'мебел':     'IGNORE', 'furniture': 'IGNORE',
    'штриховк':  'IGNORE', 'hatch':     'IGNORE',
    'шт_':       'IGNORE', 'санте':     'IGNORE',
    'santeh':    'IGNORE', 'ashade':    'IGNORE',
    'площ':      'IGNORE', 'наименован':'IGNORE',
    'каркас':    'IGNORE', 'кирпич':    'IGNORE',
    'razmer':    'IGNORE', 'основная':  'IGNORE',
    'тонкая':    'IGNORE', 'граница':   'IGNORE',
    'markup':    'IGNORE', 'марк':      'IGNORE',
    'крепл':     'IGNORE', 'гран':      'IGNORE',
}

GEOMETRY_TYPES = {'LINE', 'LWPOLYLINE', 'POLYLINE', 'ARC', 'CIRCLE', 'INSERT'}
SKIP_TYPES     = {'HATCH', 'MTEXT', 'TEXT', 'DIMENSION', 'POINT',
                  'WIPEOUT', 'ACAD_PROXY_ENTITY', 'SPLINE', 'ELLIPSE'}

# Ключевые слова в именах блоков для определения типа
DOOR_BLOCK_KEYWORDS   = ['дв', 'door', 'dr', 'двер']
WINDOW_BLOCK_KEYWORDS = ['ок', 'окн', 'win', 'window', 'wn']


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class WallGeometry:
    start:     Tuple[float, float]
    end:       Tuple[float, float]
    thickness: float
    layer:     str
    is_outer:  bool = True   # наружная или перегородка


@dataclass
class OpeningGeometry:
    position:   Tuple[float, float]
    rotation:   float
    width:      float
    kind:       str     # 'door' | 'window'
    layer:      str
    block_name: str = ''


@dataclass
class StairGeometry:
    points: List[Tuple[float, float]]
    bbox:   Tuple[float, float, float, float]
    layer:  str


@dataclass
class ParsedFloorPlan:
    source_file:    str
    walls:          List[WallGeometry]    = field(default_factory=list)
    partitions:     List[WallGeometry]    = field(default_factory=list)
    doors:          List[OpeningGeometry] = field(default_factory=list)
    windows:        List[OpeningGeometry] = field(default_factory=list)
    stairs:         List[StairGeometry]   = field(default_factory=list)
    unknown_layers: dict                  = field(default_factory=dict)
    layer_classes:  dict                  = field(default_factory=dict)
    origin_offset:  tuple                 = field(default=(0.0, 0.0))


# ---------------------------------------------------------------------------
# Маппинг имени слоя
# ---------------------------------------------------------------------------

def classify_layer(layer_name: str) -> str:
    name = layer_name.lower().strip()
    # Сначала длинные ключи (более специфичные)
    for kw in sorted(LAYER_KEYWORD_MAP, key=len, reverse=True):
        if kw in name:
            return LAYER_KEYWORD_MAP[kw]
    if name == '0':
        return 'UNKNOWN'
    # Технические коды типа АР2-6_1-03
    if re.match(r'^[а-яa-z]{1,3}\d', name):
        return 'IGNORE'
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# Геометрия полилинии
# ---------------------------------------------------------------------------

def _pts(entity) -> List[Tuple[float, float]]:
    try:
        if entity.dxftype() == 'LWPOLYLINE':
            return [(p[0], p[1]) for p in entity.get_points()]
        elif entity.dxftype() == 'LINE':
            s, e = entity.dxf.start, entity.dxf.end
            return [(s[0], s[1]), (e[0], e[1])]
        elif entity.dxftype() == 'POLYLINE':
            return [(v.dxf.location[0], v.dxf.location[1])
                    for v in entity.vertices]
    except Exception:
        pass
    return []


def _length(pts: List) -> float:
    if len(pts) < 2:
        return 0.0
    return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
               for i in range(1, len(pts)))


def _bbox(pts: List) -> Tuple:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(pts: List) -> Tuple[float, float]:
    return (sum(p[0] for p in pts)/len(pts),
            sum(p[1] for p in pts)/len(pts))


# ---------------------------------------------------------------------------
# Извлечение стен — группируем пары параллельных полилиний
# ---------------------------------------------------------------------------

def _polyline_angle(pts: List) -> float:
    """Основной угол полилинии — по самому длинному сегменту."""
    if len(pts) < 2:
        return 0.0
    best_len, best_ang = 0.0, 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        l  = math.hypot(dx, dy)
        if l > best_len:
            best_len = l
            best_ang = math.atan2(dy, dx)
    return best_ang


def _are_parallel(pts_a: List, pts_b: List, angle_tol=0.15) -> bool:
    """Проверяет параллельность двух полилиний."""
    a1 = _polyline_angle(pts_a) % math.pi
    a2 = _polyline_angle(pts_b) % math.pi
    diff = abs(a1 - a2)
    return diff < angle_tol or abs(diff - math.pi) < angle_tol


def _midline_of_pair(pts_a: List, pts_b: List) -> Tuple:
    """
    Вычисляет осевую линию между двумя параллельными полилиниями.
    Берём центроид A и центроид B как концы осевой.
    Толщина = расстояние между центроидами (приблизительно).
    """
    ca = _centroid(pts_a)
    cb = _centroid(pts_b)

    # Толщина — мин расстояние между ближайшими точками
    min_dist = float('inf')
    for pa in pts_a:
        for pb in pts_b:
            d = math.hypot(pa[0]-pb[0], pa[1]-pb[1])
            if d < min_dist:
                min_dist = d

    return ca, cb, max(min_dist, 50.0)


def extract_walls(entities: List, layer: str,
                  is_outer: bool = True) -> List[WallGeometry]:
    """
    Извлекает стены из списка entity.

    Стратегия:
    1. Если есть пары параллельных полилиний — берём осевую между ними
    2. Одиночные длинные линии — берём как есть
    """
    walls   = []
    polys   = [e for e in entities
               if e.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'LINE')]
    pts_list = [_pts(e) for e in polys]
    pts_list = [p for p in pts_list if len(p) >= 2]

    used = set()

    # Шаг 1: ищем пары параллельных полилиний
    for i in range(len(pts_list)):
        if i in used:
            continue
        best_j, best_dist = None, float('inf')
        for j in range(i+1, len(pts_list)):
            if j in used:
                continue
            if not _are_parallel(pts_list[i], pts_list[j]):
                continue
            ci = _centroid(pts_list[i])
            cj = _centroid(pts_list[j])
            d  = math.hypot(ci[0]-cj[0], ci[1]-cj[1])
            # Пара должна быть близко (толщина стены < 2000мм)
            if d < 2000 and d < best_dist:
                best_dist = d
                best_j    = j

        if best_j is not None:
            used.add(i); used.add(best_j)
            start, end, thick = _midline_of_pair(pts_list[i], pts_list[best_j])
            length = math.hypot(end[0]-start[0], end[1]-start[1])
            if length > 50:
                walls.append(WallGeometry(
                    start=start, end=end,
                    thickness=thick, layer=layer, is_outer=is_outer
                ))
        else:
            # Одиночная полилиния — берём как стену
            pts = pts_list[i]
            if i not in used:
                used.add(i)
                # Для замкнутых полилиний — разбиваем на сегменты
                if len(pts) > 2:
                    is_closed = (abs(pts[0][0]-pts[-1][0]) < 1 and
                                 abs(pts[0][1]-pts[-1][1]) < 1)
                    n = len(pts) - (1 if is_closed else 0)
                    for k in range(n):
                        a = pts[k]
                        b = pts[(k+1) % len(pts)]
                        seg_len = math.hypot(b[0]-a[0], b[1]-a[1])
                        if seg_len > 100:
                            walls.append(WallGeometry(
                                start=a, end=b,
                                thickness=200.0,
                                layer=layer, is_outer=is_outer
                            ))
                else:
                    length = _length(pts)
                    if length > 100:
                        walls.append(WallGeometry(
                            start=pts[0], end=pts[-1],
                            thickness=200.0,
                            layer=layer, is_outer=is_outer
                        ))

    return walls


# ---------------------------------------------------------------------------
# Разворачивание INSERT блоков
# ---------------------------------------------------------------------------

def _block_kind(block_name: str) -> str:
    """Определяет тип блока по имени."""
    name = block_name.lower()
    for kw in DOOR_BLOCK_KEYWORDS:
        if kw in name:
            return 'door'
    for kw in WINDOW_BLOCK_KEYWORDS:
        if kw in name:
            return 'window'
    return 'unknown'


def _block_width(doc, block_name: str) -> float:
    """Оценивает ширину блока по его геометрии."""
    if block_name not in doc.blocks:
        return 900.0
    block = doc.blocks[block_name]
    all_pts = []
    for e in block:
        p = _pts(e)
        all_pts.extend(p)
    if not all_pts:
        return 900.0
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)
    return max(w, h, 100.0)


def extract_openings_from_inserts(entities: List, layer: str,
                                   doc, default_kind='door') -> List[OpeningGeometry]:
    """Разворачивает INSERT блоки в точки вставки дверей/окон."""
    openings = []
    for e in entities:
        if e.dxftype() != 'INSERT':
            continue
        try:
            block_name = e.dxf.name
            pos        = e.dxf.insert
            rotation   = getattr(e.dxf, 'rotation', 0.0)
            xscale     = getattr(e.dxf, 'xscale', 1.0)

            kind = _block_kind(block_name)
            if kind == 'unknown':
                kind = default_kind

            base_width = _block_width(doc, block_name)
            width      = base_width * abs(xscale)

            openings.append(OpeningGeometry(
                position=(pos[0], pos[1]),
                rotation=rotation,
                width=width,
                kind=kind,
                layer=layer,
                block_name=block_name,
            ))
        except Exception:
            continue
    return openings


def extract_openings_from_lines(entities: List, layer: str,
                                 default_kind='door') -> List[OpeningGeometry]:
    """Извлекает проёмы из обычных линий (когда нет блоков)."""
    openings = []
    for e in entities:
        if e.dxftype() not in ('LINE', 'LWPOLYLINE'):
            continue
        pts = _pts(e)
        if len(pts) < 2:
            continue
        length = _length(pts)
        if length < 100:
            continue
        mid   = _centroid(pts)
        dx    = pts[-1][0] - pts[0][0]
        dy    = pts[-1][1] - pts[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        openings.append(OpeningGeometry(
            position=mid, rotation=angle,
            width=length, kind=default_kind, layer=layer,
        ))
    return openings


# ---------------------------------------------------------------------------
# Лестницы
# ---------------------------------------------------------------------------

def extract_stairs(entities: List, layer: str) -> List[StairGeometry]:
    stairs = []
    for e in entities:
        pts = _pts(e)
        if len(pts) < 2:
            continue
        bb = _bbox(pts)
        stairs.append(StairGeometry(points=pts, bbox=bb, layer=layer))
    return stairs


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def _find_origin_offset(layer_entities: dict) -> tuple:
    """
    Находит минимальные X,Y координаты по всем LINE entity в чертеже.
    Используется для центрирования координат к нулю перед передачей в Revit.
    DXF часто хранится в геодезических координатах (миллионы единиц).
    """
    min_x = None
    min_y = None

    for entities in layer_entities.values():
        for e in entities:
            if e.dxftype() != 'LINE':
                continue
            try:
                s  = e.dxf.start
                en = e.dxf.end
                sx, sy   = float(s.x),  float(s.y)
                enx, eny = float(en.x), float(en.y)
                lx = min(sx, enx)
                ly = min(sy, eny)
                if min_x is None or lx < min_x:
                    min_x = lx
                if min_y is None or ly < min_y:
                    min_y = ly
            except Exception:
                continue

    if min_x is None:
        return 0.0, 0.0
    return min_x, min_y


def parse_floor_plan(dxf_path: Path) -> ParsedFloorPlan:
    dxf_path = Path(dxf_path)
    result   = ParsedFloorPlan(source_file=str(dxf_path))

    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        logger.error(f"Не удалось открыть {dxf_path.name}: {e}")
        return result

    msp = doc.modelspace()

    # Группируем entity по слоям
    layer_entities: dict = defaultdict(list)
    for entity in msp:
        layer_entities[entity.dxf.layer].append(entity)

    # Находим смещение для центрирования координат к нулю
    ox, oy = _find_origin_offset(layer_entities)
    logger.info(f"  Смещение origin: ({ox:.0f}, {oy:.0f})")
    result.origin_offset = (ox, oy)

    logger.info(f"\n{dxf_path.name}: {len(layer_entities)} слоёв")

    for layer_name, entities in layer_entities.items():
        cls = classify_layer(layer_name)
        result.layer_classes[layer_name] = cls

        if cls == 'IGNORE':
            continue

        # Геометрические entity (без мусора)
        geom = [e for e in entities
                if e.dxftype() in GEOMETRY_TYPES
                and e.dxftype() not in SKIP_TYPES]
        if not geom:
            continue

        n = len(geom)

        if cls == 'Стены':
            w = extract_walls(geom, layer_name, is_outer=True)
            result.walls.extend(w)
            logger.info(f"  СТЕНЫ      '{layer_name}': {n} entity → {len(w)} стен")

        elif cls == 'Перегородки':
            w = extract_walls(geom, layer_name, is_outer=False)
            result.partitions.extend(w)
            logger.info(f"  ПЕРЕГОРОД  '{layer_name}': {n} entity → {len(w)} перегородок")

        elif cls == 'Двери':
            inserts = [e for e in geom if e.dxftype() == 'INSERT']
            lines   = [e for e in geom if e.dxftype() != 'INSERT']
            ops = (extract_openings_from_inserts(inserts, layer_name, doc, 'door') +
                   extract_openings_from_lines(lines, layer_name, 'door'))
            result.doors.extend(ops)
            logger.info(f"  ДВЕРИ      '{layer_name}': {n} entity → {len(ops)} дверей")

        elif cls == 'Окна':
            inserts = [e for e in geom if e.dxftype() == 'INSERT']
            lines   = [e for e in geom if e.dxftype() != 'INSERT']
            ops = (extract_openings_from_inserts(inserts, layer_name, doc, 'window') +
                   extract_openings_from_lines(lines, layer_name, 'window'))
            result.windows.extend(ops)
            logger.info(f"  ОКНА       '{layer_name}': {n} entity → {len(ops)} окон")

        elif cls == 'Окна_Двери':
            inserts = [e for e in geom if e.dxftype() == 'INSERT']
            lines   = [e for e in geom if e.dxftype() != 'INSERT']
            all_ops = (extract_openings_from_inserts(inserts, layer_name, doc, 'door') +
                       extract_openings_from_lines(lines, layer_name, 'door'))
            doors   = [o for o in all_ops if o.kind == 'door']
            windows = [o for o in all_ops if o.kind == 'window']
            result.doors.extend(doors)
            result.windows.extend(windows)
            logger.info(f"  ОК+ДВ      '{layer_name}': {n} entity → "
                        f"{len(doors)} дв, {len(windows)} ок")

        elif cls == 'Лестницы':
            s = extract_stairs(geom, layer_name)
            result.stairs.extend(s)
            logger.info(f"  ЛЕСТНИЦЫ   '{layer_name}': {n} entity → {len(s)} лестниц")

        elif cls == 'UNKNOWN':
            result.unknown_layers[layer_name] = n
            logger.info(f"  UNKNOWN    '{layer_name}': {n} entity")

    logger.info(f"\nИтог {dxf_path.name}:")
    logger.info(f"  Наружные стены:  {len(result.walls)}")
    logger.info(f"  Перегородки:     {len(result.partitions)}")
    logger.info(f"  Двери:           {len(result.doors)}")
    logger.info(f"  Окна:            {len(result.windows)}")
    logger.info(f"  Лестницы:        {len(result.stairs)}")
    logger.info(f"  Нераспознанных:  {len(result.unknown_layers)}")

    return result


def parse_directory(dir_path: Path = None) -> List[ParsedFloorPlan]:
    from src.config import PRED_RAW_DIR
    if dir_path is None:
        dir_path = PRED_RAW_DIR
    dxf_files = list(Path(dir_path).glob('*.dxf'))
    logger.info(f"Найдено {len(dxf_files)} DXF файлов")
    return [parse_floor_plan(f) for f in dxf_files]
