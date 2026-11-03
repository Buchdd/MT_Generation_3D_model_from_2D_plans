"""
train_gnn_from_plans.py — обучение GNN на реальных DXF планах из pred_raw.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.config import (
    PROCESSED_DIR, PRED_RAW_DIR, EVALUATION_DIR,
    GNN_TRAIN_CONFIG, CLASS_WEIGHT_MULTIPLIERS,
)
from src.layer_parser import classify_layer, GEOMETRY_TYPES, SKIP_TYPES
from src.ml_features import extract_entity_features, extract_file_context
from src.gnn_builder import (
    build_layer_graph, NODE_FEATURE_DIM, augment_dataset, augment_graph
)
from src.models.gnn_model import GNNClassifier
from src.utils import setup_logger

logger = setup_logger("TrainGNNPlans")

def build_dataset_from_plans(dxf_dir: Path = None,
                              min_entities: int = 3) -> tuple:
    import ezdxf

    if dxf_dir is None:
        dxf_dir = PRED_RAW_DIR

    dxf_files = list(Path(dxf_dir).glob("*.dxf"))
    logger.info(f"Найдено {len(dxf_files)} DXF файлов в {dxf_dir}")

    all_classes = set()
    layer_data  = []  # (file_path, layer_name, class_name, entities)

    for dxf_path in dxf_files:
        try:
            doc = ezdxf.readfile(str(dxf_path))
        except Exception as e:
            logger.warning(f"Не удалось прочитать {dxf_path.name}: {e}")
            continue

        msp = doc.modelspace()

        class _FakeMsp:
            def __iter__(self): return iter(list(msp))
        file_ctx = extract_file_context(_FakeMsp())

        layer_entities = defaultdict(list)
        for e in msp:
            layer_entities[e.dxf.layer].append(e)

        for layer_name, entities in layer_entities.items():
            cls = classify_layer(layer_name)

            if cls in ('UNKNOWN', 'IGNORE'):
                continue

            # Убираем мусорные типы
            geom = [e for e in entities
                    if e.dxftype() in GEOMETRY_TYPES
                    and e.dxftype() not in SKIP_TYPES]

            if len(geom) < min_entities:
                logger.debug(f"  Пропускаем '{layer_name}' ({len(geom)} entity < {min_entities})")
                continue

            all_classes.add(cls)
            layer_data.append((dxf_path, layer_name, cls, geom, file_ctx))
            logger.info(f"  {dxf_path.name}/{layer_name} → {cls} ({len(geom)} entity)")

    if not all_classes:
        logger.error("Нет подходящих слоёв! Проверьте DXF файлы.")
        return [], {}, {}

    # Маппинг классов
    sorted_classes = sorted(all_classes)
    class_to_idx   = {c: i for i, c in enumerate(sorted_classes)}
    idx_to_class   = {i: c for c, i in class_to_idx.items()}

    logger.info(f"\nКлассы в датасете: {sorted_classes}")

    # строим графы
    graphs = []
    skipped = 0

    for dxf_path, layer_name, cls, geom, file_ctx in layer_data:
        graph = build_layer_graph(geom, file_ctx)
        if graph is None:
            skipped += 1
            continue

        graph.y          = torch.tensor([class_to_idx[cls]], dtype=torch.long)
        graph.class_name = cls
        graph.layer_name = layer_name
        graph.file_name  = dxf_path.name
        graphs.append(graph)

    logger.info(f"\nПостроено {len(graphs)} графов, пропущено {skipped}")
    logger.info("Распределение по классам:")
    from collections import Counter
    counts = Counter(g.class_name for g in graphs)
    for cls, cnt in sorted(counts.items()):
        logger.info(f"  {cls:<30} {cnt} графов")

    return graphs, class_to_idx, idx_to_class


def train_gnn_on_plans(
    dxf_dir: Path = None,
    output_model_path: Path = None,
    output_eval_dir: Path = None,
    aug_factor: int = 8,
    epochs: int = 100,
    lr: float = 0.005,
    hidden_channels: int = 64,
    num_layers: int = 3,
    batch_size: int = 4,
    val_ratio: float = 0.25,
):
    from torch_geometric.loader import DataLoader
    import random

    if output_model_path is None:
        output_model_path = PROCESSED_DIR / "models" / "gnn_plans_model.pt"
    if output_eval_dir is None:
        output_eval_dir = PROCESSED_DIR / "evaluation" / "gnn_plans"

    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GNN ОБУЧЕНИЕ НА РЕАЛЬНЫХ ПЛАНАХ")
    logger.info("=" * 60)
    logger.info(f"Источник данных: {dxf_dir or PRED_RAW_DIR}")
    logger.info(f"NODE_FEATURE_DIM: {NODE_FEATURE_DIM}")
    logger.info(f"aug_factor: {aug_factor} | epochs: {epochs}")
    logger.info(f"lr: {lr} | hidden: {hidden_channels} | layers: {num_layers}")

    # Строим датасет
    graphs, class_to_idx, idx_to_class = build_dataset_from_plans(dxf_dir)

    if not graphs:
        logger.error("Пустой датасет — обучение невозможно")
        return None, None

    num_classes = len(class_to_idx)
    logger.info(f"\nИтого: {len(graphs)} графов, {num_classes} классов")

    # Стратифицированный split
    random.seed(42)
    per_class = defaultdict(list)
    for i, g in enumerate(graphs):
        per_class[g.class_name].append(i)

    train_idx, val_idx = [], []
    for cls, indices in per_class.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        # Гарантируем хотя бы 1 в трейне
        if len(indices) - n_val < 1:
            n_val = 0
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    logger.info(f"Train: {len(train_idx)} графов | Val: {len(val_idx)} графов")

    train_classes = set(graphs[i].class_name for i in train_idx)
    val_classes   = set(graphs[i].class_name for i in val_idx)
    logger.info(f"Классов в трейне: {len(train_classes)} | в вале: {len(val_classes)}")

    # Аугментация только трейна
    orig_train = [graphs[i] for i in train_idx]
    aug_train  = augment_dataset(orig_train, factor=aug_factor)
    logger.info(f"После аугментации x{aug_factor}: {len(orig_train)} → {len(aug_train)} графов")

    class _ListDS(torch.utils.data.Dataset):
        def __init__(self, g): self.g = g
        def __len__(self):     return len(self.g)
        def __getitem__(self, i): return self.g[i]

    val_graphs   = [graphs[i] for i in val_idx]
    train_loader = DataLoader(_ListDS(aug_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_ListDS(val_graphs), batch_size=batch_size, shuffle=False)

    # Веса классов
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    counts = defaultdict(int)
    for g in orig_train:
        counts[g.class_name] += 1
    total  = sum(counts.values())
    n      = num_classes
    weights = torch.ones(n, dtype=torch.float)
    for cls, idx in class_to_idx.items():
        cnt  = counts.get(cls, 1)
        base = total / (n * cnt)
        mult = CLASS_WEIGHT_MULTIPLIERS.get(cls, 1.0)
        weights[idx] = base * mult
    weights = weights.to(device)

    logger.info("\nВеса классов:")
    for cls, idx in sorted(class_to_idx.items()):
        logger.info(f"  {cls:<30} {weights[idx]:.3f}")

    # Модель
    model = GNNClassifier(
        in_channels=NODE_FEATURE_DIM,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        lr=lr,
        model_type='gcn',
    )
    model.criterion = torch.nn.CrossEntropyLoss(weight=weights)
    n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info(f"Параметров модели: {n_params:,}")

    # Обучение
    history = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_m = model.train_epoch(train_loader)
        val_m   = model.evaluate(val_loader)

        history.append({
            'epoch':      epoch,
            'train_loss': train_m['loss'],
            'train_acc':  train_m['acc'],
            'val_acc':    val_m['acc'],
        })

        if val_m['acc'] > best_val_acc:
            best_val_acc = val_m['acc']
            model.save(output_model_path)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_m['loss']:.4f} | "
                f"Train: {train_m['acc']:.4f} | "
                f"Val: {val_m['acc']:.4f}"
            )

    logger.info(f"\nОбучение завершено. Лучший val acc: {best_val_acc:.4f}")
    logger.info(f"Модель: {output_model_path}")

    # Финальная оценка лучшей модели
    best_model = GNNClassifier.load(output_model_path)
    best_model.criterion = torch.nn.CrossEntropyLoss(weight=weights)
    final_metrics = best_model.evaluate(val_loader)

    # Сохраняем idx_to_class в файл рядом с моделью
    import json
    meta_path = output_model_path.with_suffix('.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
            'num_classes':  num_classes,
            'model_type':   'gnn_plans',
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Метаданные: {meta_path}")

    # Графики
    _plot_training(history, best_val_acc, output_eval_dir)
    _plot_confusion(final_metrics, idx_to_class, output_eval_dir)
    _plot_f1(final_metrics, idx_to_class, output_eval_dir)
    _plot_class_distribution(graphs, class_to_idx, output_eval_dir)

    return best_model, {
        'history':      history,
        'best_val_acc': best_val_acc,
        'idx_to_class': idx_to_class,
        'class_to_idx': class_to_idx,
    }


def _plot_training(history: list, best_val_acc: float, output_dir: Path):
    """График потерь и точности по эпохам."""
    df = pd.DataFrame(history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'GNN на реальных планах — обучение\n(лучший val acc: {best_val_acc:.4f})',
        fontsize=13, fontweight='bold'
    )

    # Loss
    ax1.plot(df['epoch'], df['train_loss'], color='#E74C3C', lw=2, label='Train Loss')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title('Функция потерь')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(df['epoch'], df['train_acc'], color='#3498DB', lw=2, label='Train Acc')
    ax2.plot(df['epoch'], df['val_acc'],   color='#2ECC71', lw=2,
             label='Val Acc', linestyle='--')
    ax2.axhline(y=best_val_acc, color='#E67E22', linestyle=':',
                lw=1.5, label=f'Best val={best_val_acc:.3f}')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Точность')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"График обучения: {path}")


def _plot_confusion(metrics: dict, idx_to_class: dict, output_dir: Path):
    from sklearn.metrics import confusion_matrix

    preds  = metrics.get('preds', [])
    labels = metrics.get('labels', [])
    if not preds:
        return

    y_pred_names  = [idx_to_class.get(p, str(p)) for p in preds]
    y_true_names  = [idx_to_class.get(l, str(l)) for l in labels]
    classes       = sorted(set(y_true_names))

    if not classes:
        return

    cm = confusion_matrix(y_true_names, y_pred_names, labels=classes)

    acc = sum(p == l for p, l in zip(preds, labels)) / len(preds) if preds else 0

    fig, ax = plt.subplots(figsize=(max(6, len(classes)*1.5), max(5, len(classes)*1.4)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, label='Количество образцов')

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=11,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_title(f'Confusion Matrix — GNN на планах (Acc={acc:.2%})',
                 fontweight='bold')

    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix: {path}")


def _plot_f1(metrics: dict, idx_to_class: dict, output_dir: Path):
    from sklearn.metrics import classification_report

    preds  = metrics.get('preds', [])
    labels = metrics.get('labels', [])
    if not preds:
        return

    y_pred = [idx_to_class.get(p, str(p)) for p in preds]
    y_true = [idx_to_class.get(l, str(l)) for l in labels]
    classes = sorted(set(y_true))

    report = classification_report(
        y_true, y_pred, labels=classes,
        output_dict=True, zero_division=0
    )

    f1_scores = [report[c]['f1-score'] for c in classes if c in report]

    fig, ax = plt.subplots(figsize=(max(8, len(classes)*1.5), 5))
    colors = ['#2ECC71' if f >= 0.7 else '#E67E22' if f >= 0.4 else '#E74C3C'
              for f in f1_scores]
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor='white', lw=0.5)

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score по классам — GNN на реальных планах', fontweight='bold')
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Порог 0.7')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.tight_layout()
    path = output_dir / "f1_by_class.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"F1 chart: {path}")

    # Сохраняем CSV
    rows = []
    for c in classes:
        if c in report:
            rows.append({
                'class':     c,
                'precision': round(report[c]['precision'], 4),
                'recall':    round(report[c]['recall'],    4),
                'f1':        round(report[c]['f1-score'],  4),
                'support':   report[c]['support'],
            })
    pd.DataFrame(rows).to_csv(output_dir / "classification_report.csv",
                               index=False, encoding='utf-8-sig')
    logger.info(f"CSV: {output_dir / 'classification_report.csv'}")


def _plot_class_distribution(graphs: list, class_to_idx: dict,
                              output_dir: Path):
    from collections import Counter

    counts  = Counter(g.class_name for g in graphs)
    classes = sorted(counts.keys())
    values  = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes)*1.5), 5))
    colors  = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars    = ax.bar(classes, values, color=colors, edgecolor='white')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(val), ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Количество графов')
    ax.set_title('Распределение классов в датасете\n(реальные архитектурные планы)',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.tight_layout()

    path = output_dir / "class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Распределение классов: {path}")

def load_plans_model(model_path: Path = None) -> tuple:
    import json

    if model_path is None:
        model_path = PROCESSED_DIR / "models" / "gnn_plans_model.pt"

    if not Path(model_path).exists():
        logger.warning(f"Модель gnn_plans не найдена: {model_path}")
        return None, None

    meta_path = Path(model_path).with_suffix('.json')
    if not meta_path.exists():
        logger.warning(f"Метаданные не найдены: {meta_path}")
        return None, None

    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    idx_to_class = {int(k): v for k, v in meta['idx_to_class'].items()}

    model = GNNClassifier.load(model_path)
    logger.info(f"GNN Plans модель загружена: {model_path}")
    logger.info(f"Классы: {idx_to_class}")

    return model, idx_to_class


def classify_layer_with_plans_gnn(entities: list, layer_name: str,
                                   file_ctx: dict = None,
                                   model=None,
                                   idx_to_class: dict = None) -> tuple:
    if model is None or idx_to_class is None:
        return None, 0.0

    geom = [e for e in entities
            if e.dxftype() in GEOMETRY_TYPES
            and e.dxftype() not in SKIP_TYPES]

    if len(geom) < 2:
        return None, 0.0

    graph = build_layer_graph(geom, file_ctx)
    if graph is None:
        return None, 0.0

    try:
        graph = graph.to(model.device)
        class_idx, proba = model.predict_graph(graph)
        pred_class = idx_to_class.get(class_idx, f"unknown_{class_idx}")
        confidence = float(max(proba))
        return pred_class, confidence
    except Exception as e:
        logger.error(f"GNN Plans инференс ошибка: {e}")
        return None, 0.0