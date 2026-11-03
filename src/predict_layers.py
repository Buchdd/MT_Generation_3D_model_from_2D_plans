"""
predict_layers.py — сегментация целых планов этажей через GNN по слоям.

Идея:
    Реальный план этажа содержит слои (layers) — каждый слой как правило
    соответствует одному типу объектов (стены, окна, двери и т.д.).
    Вместо того чтобы классифицировать каждую линию отдельно (как ML),
    мы берём все entity одного слоя, строим из них граф и спрашиваем GNN:
    "какой класс у этого слоя?"

    Это намного более естественно — GNN видит весь контекст слоя сразу,
    а не одну линию в изоляции.

Результат:
    - CSV с маппингом layer_name -> predicted_class
    - Раскрашенная картинка где каждый слой своим цветом
"""

import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

from src.config import PROCESSED_DIR, ML_OUTPUT_DIR
from src.dxf_reader import load_dxf_document
from src.gnn_builder import build_graph_from_dxf, NODE_FEATURE_DIM
from src.models.gnn_model import GNNClassifier
from src.utils import setup_logger

logger = setup_logger("PredictLayers")


def build_layer_graph(entities: list, file_ctx: dict = None):
    """
    Строит граф из списка entity одного слоя.
    Использует ту же логику что gnn_builder но для подмножества entity.
    """
    import math, torch
    from torch_geometric.data import Data
    from src.gnn_builder import ENTITY_TYPE_MAP, NODE_FEATURE_KEYS, _entity_center
    from src.ml_features import extract_entity_features, extract_file_context

    if not entities:
        return None

    node_features = []
    centers = []

    for e in entities:
        type_idx     = ENTITY_TYPE_MAP.get(e.dxftype(), -1)
        type_one_hot = [0] * len(ENTITY_TYPE_MAP)
        if type_idx >= 0:
            type_one_hot[type_idx] = 1

        feats   = extract_entity_features(e, file_context=file_ctx)
        numeric = [float(feats.get(k, 0.0) or 0.0) for k in NODE_FEATURE_KEYS]
        numeric = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in numeric]

        node_features.append(type_one_hot + numeric)
        centers.append(_entity_center(e))

    x = torch.tensor(node_features, dtype=torch.float)
    n = len(entities)

    # Рёбра: полный граф внутри слоя (все entity связаны друг с другом)
    # Оправдано — все entity одного слоя семантически связаны
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append([i, j])
            edges.append([j, i])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    data      = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(n, dtype=torch.long)
    return data


def predict_file_by_layers(dxf_path: Path,
                            gnn_model_path: Path = None,
                            output_dir: Path = None) -> pd.DataFrame | None:
    """
    Сегментирует DXF файл по слоям через GNN.

    Возвращает DataFrame с колонками:
        layer_name, predicted_class, confidence, entity_count
    """
    dxf_path = Path(dxf_path)
    if gnn_model_path is None:
        gnn_model_path = PROCESSED_DIR / "models" / "gnn_model.pt"
    if output_dir is None:
        output_dir = PROCESSED_DIR / "evaluation" / "segmentation"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем модель
    if not gnn_model_path.exists():
        logger.error(f"GNN модель не найдена: {gnn_model_path}")
        return None

    try:
        gnn = GNNClassifier.load(gnn_model_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки GNN: {e}")
        return None

    # Загружаем маппинг классов
    csv_path = ML_OUTPUT_DIR / "train_features.csv"
    if not csv_path.exists():
        logger.error(f"train_features.csv не найден: {csv_path}")
        return None

    df_train     = pd.read_csv(csv_path)
    all_classes  = sorted(df_train['label'].unique())
    idx_to_class = {i: c for i, c in enumerate(all_classes)}

    # Читаем DXF
    import ezdxf
    from src.ml_features import extract_file_context

    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        logger.error(f"Ошибка чтения DXF: {e}")
        return None

    msp = doc.modelspace()

    # Группируем entity по слоям
    target_types = {'LINE', 'LWPOLYLINE', 'POLYLINE', 'CIRCLE', 'ARC'}
    layer_entities = defaultdict(list)

    for entity in msp:
        if entity.dxftype() in target_types:
            layer = entity.dxf.layer
            layer_entities[layer].append(entity)

    logger.info(f"{dxf_path.name}: {len(layer_entities)} слоёв с геометрией")

    # Контекст всего файла — одинаковый для всех слоёв
    # Создаём фиктивный msp-подобный объект через итерацию
    class _FakeMsp:
        def __iter__(self):
            for ents in layer_entities.values():
                yield from ents
    file_ctx = extract_file_context(_FakeMsp())

    # Предсказываем класс для каждого слоя
    rows = []
    for layer_name, entities in layer_entities.items():
        if len(entities) < 2:
            # Слишком мало entity — пропускаем
            logger.debug(f"  Слой '{layer_name}': {len(entities)} entity — пропущен")
            continue

        graph = build_layer_graph(entities, file_ctx)
        if graph is None:
            continue

        try:
            graph = graph.to(gnn.device)
            class_idx, proba = gnn.predict_graph(graph)
            pred_class = idx_to_class.get(class_idx, f"unknown_{class_idx}")
            confidence = float(max(proba))
        except Exception as e:
            logger.warning(f"  Слой '{layer_name}': ошибка предсказания: {e}")
            continue

        rows.append({
            'layer_name':      layer_name,
            'predicted_class': pred_class,
            'confidence':      round(confidence, 4),
            'entity_count':    len(entities),
        })

        logger.info(
            f"  {layer_name:<40} → {pred_class:<20} "
            f"conf={confidence:.3f} ({len(entities)} entity)"
        )

    if not rows:
        logger.warning("Нет результатов для сохранения")
        return None

    df_result = pd.DataFrame(rows).sort_values('entity_count', ascending=False)

    # Сохраняем CSV
    csv_out = output_dir / f"{dxf_path.stem}_layers_gnn.csv"
    df_result.to_csv(csv_out, index=False, encoding='utf-8-sig')
    logger.info(f"CSV сохранён: {csv_out}")

    # Визуализация — раскрашиваем entity по предсказанному классу слоя
    _visualize_by_layers(dxf_path, df_result, output_dir)

    return df_result


def _visualize_by_layers(dxf_path: Path,
                          layer_predictions: pd.DataFrame,
                          output_dir: Path):
    """Рисует DXF раскрашивая каждый слой в цвет предсказанного класса."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Arc as MplArc

    # Маппинг layer -> predicted_class
    layer_map = dict(zip(
        layer_predictions['layer_name'],
        layer_predictions['predicted_class']
    ))

    # Цвета по классам
    CLASS_COLORS = {
        'Стены':          '#E74C3C',
        'Перекрытия':     '#3498DB',
        'Потолки':        '#9B59B6',
        'Базовая дверь':  '#2ECC71',
        'Базовое окно':   '#F39C12',
        'Лестницы':       '#1ABC9C',
    }

    def get_color(cls_name: str) -> str:
        if cls_name in CLASS_COLORS:
            return CLASS_COLORS[cls_name]
        h = hash(cls_name) & 0xFFFFFF
        return f"#{h:06X}"

    import ezdxf
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception:
        return

    msp = doc.modelspace()
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    all_x, all_y    = [], []
    drawn_classes   = {}   # class_name -> color (для легенды)
    unknown_color   = '#555566'

    for entity in msp:
        layer    = entity.dxf.layer
        cls_name = layer_map.get(layer)
        color    = get_color(cls_name) if cls_name else unknown_color
        alpha    = 0.9 if cls_name else 0.2
        lw       = 1.8 if cls_name else 0.6

        if cls_name and cls_name not in drawn_classes:
            drawn_classes[cls_name] = color

        try:
            t = entity.dxftype()
            if t == 'LINE':
                s, e = entity.dxf.start, entity.dxf.end
                ax.plot([s[0], e[0]], [s[1], e[1]],
                        color=color, lw=lw, alpha=alpha, solid_capstyle='round')
                all_x += [s[0], e[0]]; all_y += [s[1], e[1]]

            elif t in ('LWPOLYLINE', 'POLYLINE'):
                pts = list(entity.points())
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)
                if len(pts) > 2 and cls_name:
                    ax.fill(xs, ys, color=color, alpha=0.08)
                all_x += xs; all_y += ys

            elif t == 'CIRCLE':
                c, r = entity.dxf.center, entity.dxf.radius
                ax.add_patch(Circle((c[0], c[1]), r,
                                    fill=False, color=color, lw=lw, alpha=alpha))
                all_x += [c[0]-r, c[0]+r]; all_y += [c[1]-r, c[1]+r]

            elif t == 'ARC':
                c, r = entity.dxf.center, entity.dxf.radius
                ax.add_patch(MplArc(
                    (c[0], c[1]), 2*r, 2*r,
                    angle=0,
                    theta1=entity.dxf.start_angle,
                    theta2=entity.dxf.end_angle,
                    color=color, lw=lw, alpha=alpha
                ))
                all_x += [c[0]-r, c[0]+r]; all_y += [c[1]-r, c[1]+r]
        except Exception:
            continue

    # Границы
    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        margin     = max((xmax - xmin), (ymax - ymin)) * 0.03
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

    # Легенда
    if drawn_classes:
        handles = [
            plt.Line2D([0], [0], color=color, lw=3, label=cls)
            for cls, color in sorted(drawn_classes.items())
        ]
        # Добавляем "нераспознанные слои"
        handles.append(plt.Line2D([0], [0], color=unknown_color,
                                  lw=1, alpha=0.5, label='Нераспознанные слои'))
        ax.legend(handles=handles,
                  loc='upper left', bbox_to_anchor=(1.01, 1),
                  fontsize=11, framealpha=0.85,
                  facecolor='#2a2a3e', labelcolor='white',
                  edgecolor='#444466')

    ax.set_title(
        f"GNN сегментация по слоям: {dxf_path.name}",
        fontsize=14, color='white', pad=16
    )

    img_out = output_dir / f"{dxf_path.stem}_layers_gnn.png"
    plt.savefig(img_out, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()
    logger.info(f"Картинка сохранена: {img_out}")


def predict_directory_by_layers(dir_path: Path = None,
                                 gnn_model_path: Path = None,
                                 output_dir: Path = None):
    """Пакетная обработка всех DXF в папке."""
    from src.utils import get_dxf_files
    from src.config import PRED_RAW_DIR

    if dir_path   is None: dir_path = PRED_RAW_DIR
    if output_dir is None: output_dir = PROCESSED_DIR / "evaluation" / "segmentation"

    dir_path   = Path(dir_path)
    output_dir = Path(output_dir)

    dxf_files = get_dxf_files(dir_path)
    logger.info(f"Найдено {len(dxf_files)} DXF файлов для layer-сегментации")

    results = []
    for f in dxf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Обрабатываем: {f.name}")
        logger.info(f"{'='*60}")
        try:
            df = predict_file_by_layers(f, gnn_model_path, output_dir)
            if df is not None:
                results.append({
                    'file':        f.name,
                    'layers':      len(df),
                    'classes':     df['predicted_class'].nunique(),
                    'top_class':   df.loc[df['entity_count'].idxmax(), 'predicted_class'],
                })
        except Exception as e:
            logger.error(f"Ошибка {f.name}: {e}")
            import traceback; traceback.print_exc()

    if results:
        summary = pd.DataFrame(results)
        summary_path = output_dir / "summary_layers_gnn.csv"
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"\nСводка сохранена: {summary_path}")
        logger.info("\n" + summary.to_string(index=False))

    return results
