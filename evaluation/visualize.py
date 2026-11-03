import os
import matplotlib
matplotlib.use('Agg')  # Принудительно без GUI — иначе tkinter падает при batch обработке
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import ezdxf
from matplotlib.patches import Circle, Arc
from src.config import PROCESSED_DIR
from src.utils import setup_logger

logger = setup_logger("Visualize")

_CLASS_COLORS = {}

def get_class_color(class_name: str) -> str:
    if class_name in _CLASS_COLORS:
        return _CLASS_COLORS[class_name]
    h = hash(class_name) & 0xFFFFFF
    color = f"#{h:06X}"
    _CLASS_COLORS[class_name] = color
    return color

def visualize_segmentation(dxf_path, predictions, output_path=None):
    """Отрисовка DXF (вызывается из predict.py)"""
    dxf_path = Path(dxf_path)
    if output_path is None:
        output_path = dxf_path.parent / f"{dxf_path.stem}_segmentation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        logger.error(f"❌ Failed to read DXF: {e}")
        return False

    msp = doc.modelspace()
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')
    ax.axis('off')

    all_x, all_y = [], []
    drawn_classes = set()

    for entity in msp:
        handle = getattr(entity.dxf, 'handle', None)
        key = handle or str(id(entity))
        pred_class = predictions.get(key) or predictions.get(str(id(entity)))
        if not pred_class: pred_class = 'Unknown'
            
        color = get_class_color(pred_class)

        try:
            if entity.dxftype() == 'LINE':
                s, e = entity.dxf.start, entity.dxf.end
                ax.plot([s[0], e[0]], [s[1], e[1]], color=color, lw=1.5, alpha=0.9)
                all_x.extend([s[0], e[0]]); all_y.extend([s[1], e[1]])
            elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                pts = list(entity.points())
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=color, lw=1.5, alpha=0.9)
                if len(pts) > 2: ax.fill(xs, ys, color=color, alpha=0.15)
                all_x.extend(xs); all_y.extend(ys)
            elif entity.dxftype() == 'CIRCLE':
                c, r = entity.dxf.center, entity.dxf.radius
                ax.add_patch(Circle((c[0], c[1]), r, fill=False, color=color, lw=1.5))
                all_x.extend([c[0]-r, c[0]+r]); all_y.extend([c[1]-r, c[1]+r])
            elif entity.dxftype() == 'ARC':
                c, r = entity.dxf.center, entity.dxf.radius
                ax.add_patch(Arc((c[0], c[1]), 2*r, 2*r, angle=0, theta1=entity.dxf.start_angle, theta2=entity.dxf.end_angle, color=color, lw=1.5))
                all_x.extend([c[0]-r, c[0]+r]); all_y.extend([c[1]-r, c[1]+r])
        except Exception:
            continue

        if pred_class not in drawn_classes:
            ax.plot([], [], color=color, label=pred_class)
            drawn_classes.add(pred_class)

    if all_x and all_y:
        margin = 0.05
        dx = (max(all_x) - min(all_x)) * margin
        dy = (max(all_y) - min(all_y)) * margin
        ax.set_xlim(min(all_x) - dx, max(all_x) + dx)
        ax.set_ylim(min(all_y) - dy, max(all_y) + dy)

    if drawn_classes:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    ax.set_title(f"Сегментация: {dxf_path.name}", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return True

def plot_training_history(history=None, output_path=None):
    """Графики обучения (если есть)"""
    if output_path is None:
        output_path = PROCESSED_DIR / "evaluation" / "training_history.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not history or len(history) == 0:
        # Ищем файл истории
        history_path = PROCESSED_DIR / "models" / "gnn_training_history.json"
        if history_path.exists():
            import json
            with open(history_path) as f: history = json.load(f)
        else:
            logger.info("📊 История обучения пуста. Пропуск графика.")
            return

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if 'train_loss' in df.columns:
        axes[0].plot(df['epoch'], df['train_loss'], marker='o'); axes[0].set_title('Loss')
    for col in [c for c in df.columns if 'acc' in c.lower()]:
        axes[1].plot(df['epoch'], df[col], marker='o'); axes[1].set_title('Accuracy')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"График сохранен: {output_path}")

def show_segmentation_results():
    """🆕 НОВАЯ ФУНКЦИЯ: Показывает сохраненные картинки сегментации"""
    segmentation_dir = PROCESSED_DIR / "evaluation" / "segmentation"
    
    if not segmentation_dir.exists():
        logger.info("❌ Папка с сегментацией не найдена. Запустите 'python main.py --stage predict'")
        return

    images = list(segmentation_dir.glob("*_segmentation.png"))
    if not images:
        logger.info("❌ Картинки не найдены. Возможно, predict не отработал.")
        return

    logger.info(f"🖼️ Найдено {len(images)} изображений сегментации:")
    
    # Открываем первые 3 изображения в стандартном просмотрщике Windows
    for i, img_path in enumerate(images):
        logger.info(f"   -> {img_path}")
        if i < 3:  # Чтобы не открыть 100 окон, открываем только первые 3
            try:
                os.startfile(str(img_path))
                logger.info(f"   📂 Открыто в просмотрщике.")
            except Exception as e:
                logger.warning(f"   Не удалось открыть: {e}")

if __name__ == "__main__":
    # Для теста можно запустить просмотр картинок
    show_segmentation_results()