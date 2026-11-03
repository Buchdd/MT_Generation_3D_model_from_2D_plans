"""
visualize_plan.py — визуализация результата парсинга плана этажа.

Рисует план как он должен выглядеть:
  Наружные стены  — толстые красные линии
  Перегородки     — тонкие оранжевые линии  
  Двери           — зелёные дуги (символ открывания)
  Окна            — голубые линии с засечками
  Лестницы        — фиолетовые ступени
  Нераспознанное  — серое
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
from pathlib import Path
from src.layer_parser import ParsedFloorPlan, parse_floor_plan
from src.config import PROCESSED_DIR
from src.utils import setup_logger

logger = setup_logger("VisualizePlan")

STYLE = {
    'Стены':        {'color': '#E74C3C', 'lw': 3.5, 'zorder': 5},
    'Перегородки':  {'color': '#E67E22', 'lw': 2.0, 'zorder': 4},
    'Двери':        {'color': '#2ECC71', 'lw': 2.0, 'zorder': 6},
    'Окна':         {'color': '#3498DB', 'lw': 2.5, 'zorder': 6},
    'Лестницы':     {'color': '#9B59B6', 'lw': 1.5, 'zorder': 3},
    'Нераспознано': {'color': '#555566', 'lw': 0.8, 'zorder': 1},
}


def _draw_wall(ax, wall, style):
    ax.plot(
        [wall.start[0], wall.end[0]],
        [wall.start[1], wall.end[1]],
        color=style['color'], lw=style['lw'],
        zorder=style['zorder'], solid_capstyle='round'
    )


def _draw_door(ax, door, style):
    """Рисует дверь как линия + дуга открывания."""
    x, y = door.position
    angle_rad = math.radians(door.rotation)
    w = door.width if door.width > 0 else 900

    # Линия полотна двери
    ex = x + w * math.cos(angle_rad)
    ey = y + w * math.sin(angle_rad)
    ax.plot([x, ex], [y, ey], color=style['color'],
            lw=style['lw'], zorder=style['zorder'])

    # Дуга открывания
    start_angle = math.degrees(angle_rad)
    arc = Arc((x, y), w * 2, w * 2,
              angle=0, theta1=start_angle, theta2=start_angle + 90,
              color=style['color'], lw=1.2, zorder=style['zorder'])
    ax.add_patch(arc)


def _draw_window(ax, window, style):
    """Рисует окно как линия с двумя засечками."""
    x, y = window.position
    angle_rad = math.radians(window.rotation)
    w = window.width if window.width > 0 else 1000
    perp = angle_rad + math.pi / 2

    ex = x + w * math.cos(angle_rad)
    ey = y + w * math.sin(angle_rad)

    # Основная линия
    ax.plot([x, ex], [y, ey], color=style['color'],
            lw=style['lw'], zorder=style['zorder'])

    # Две засечки (символ окна)
    for frac in (0.33, 0.67):
        mx = x + w * frac * math.cos(angle_rad)
        my = y + w * frac * math.sin(angle_rad)
        tick = w * 0.08
        ax.plot(
            [mx - tick * math.cos(perp), mx + tick * math.cos(perp)],
            [my - tick * math.sin(perp), my + tick * math.sin(perp)],
            color=style['color'], lw=1.5, zorder=style['zorder']
        )


def _draw_stair(ax, stair, style):
    """Рисует лестницу как контур."""
    pts = stair.points
    if len(pts) < 2:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=style['color'],
            lw=style['lw'], zorder=style['zorder'])


def visualize_parsed_plan(plan: ParsedFloorPlan,
                           output_path: Path = None,
                           show_unknown: bool = True):
    """
    Рисует распарсенный план этажа.
    Каждый тип объекта своим стилем.
    """
    source = Path(plan.source_file)
    if output_path is None:
        output_path = (PROCESSED_DIR / "evaluation" / "segmentation"
                       / f"{source.stem}_parsed.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # --- Наружные стены ---
    style = STYLE['Стены']
    for wall in plan.walls:
        _draw_wall(ax, wall, style)

    # --- Перегородки ---
    style = STYLE['Перегородки']
    for part in plan.partitions:
        _draw_wall(ax, part, style)

    # --- Двери ---
    style = STYLE['Двери']
    for door in plan.doors:
        _draw_door(ax, door, style)

    # --- Окна ---
    style = STYLE['Окна']
    for window in plan.windows:
        _draw_window(ax, window, style)

    # --- Лестницы ---
    style = STYLE['Лестницы']
    for stair in plan.stairs:
        _draw_stair(ax, stair, style)

    # --- Нераспознанные слои ---
    if show_unknown and plan.unknown_layers:
        import ezdxf
        try:
            doc = ezdxf.readfile(plan.source_file)
            msp = doc.modelspace()
            style = STYLE['Нераспознано']
            for entity in msp:
                if entity.dxf.layer not in plan.unknown_layers:
                    continue
                if entity.dxftype() == 'LINE':
                    s, e = entity.dxf.start, entity.dxf.end
                    ax.plot([s[0], e[0]], [s[1], e[1]],
                            color=style['color'], lw=style['lw'],
                            alpha=0.4, zorder=style['zorder'])
                elif entity.dxftype() == 'LWPOLYLINE':
                    pts = list(entity.get_points())
                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        ax.plot(xs, ys, color=style['color'],
                                lw=style['lw'], alpha=0.4,
                                zorder=style['zorder'])
        except Exception:
            pass

    # --- Автомасштаб ---
    ax.autoscale_view()

    # --- Легенда ---
    legend_items = []
    if plan.walls:
        legend_items.append(mpatches.Patch(
            color=STYLE['Стены']['color'], label=f'Наружные стены ({len(plan.walls)})'))
    if plan.partitions:
        legend_items.append(mpatches.Patch(
            color=STYLE['Перегородки']['color'], label=f'Перегородки ({len(plan.partitions)})'))
    if plan.doors:
        legend_items.append(mpatches.Patch(
            color=STYLE['Двери']['color'], label=f'Двери ({len(plan.doors)})'))
    if plan.windows:
        legend_items.append(mpatches.Patch(
            color=STYLE['Окна']['color'], label=f'Окна ({len(plan.windows)})'))
    if plan.stairs:
        legend_items.append(mpatches.Patch(
            color=STYLE['Лестницы']['color'], label=f'Лестницы ({len(plan.stairs)})'))
    if show_unknown and plan.unknown_layers:
        legend_items.append(mpatches.Patch(
            color=STYLE['Нераспознано']['color'],
            label=f'Нераспознано ({len(plan.unknown_layers)} слоёв)',
            alpha=0.5))

    if legend_items:
        ax.legend(handles=legend_items,
                  loc='upper left', bbox_to_anchor=(1.01, 1),
                  fontsize=11, framealpha=0.9,
                  facecolor='#2a2a3e', labelcolor='white',
                  edgecolor='#444466')

    ax.set_title(f"Парсинг плана: {source.name}",
                 fontsize=14, color='white', pad=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()
    logger.info(f"Сохранено: {output_path}")
    return output_path


def visualize_directory(dir_path=None, output_dir=None):
    """Парсит и визуализирует все DXF в папке."""
    from src.layer_parser import parse_directory
    from src.config import PRED_RAW_DIR

    if dir_path is None:
        dir_path = PRED_RAW_DIR
    if output_dir is None:
        output_dir = PROCESSED_DIR / "evaluation" / "segmentation"

    plans = parse_directory(dir_path)
    for plan in plans:
        out = Path(output_dir) / f"{Path(plan.source_file).stem}_parsed.png"
        visualize_parsed_plan(plan, output_path=out)

    return plans
