"""
evaluate.py — честная оценка модели на тестовой выборке.

Изменения:
- test_csv по умолчанию теперь указывает на test_features.csv (отдельный от трейна).
  Раньше evaluate запускался на том же CSV что и обучение — accuracy была ~99% (ложь).
- Добавлен per-class отчёт в консоль: сразу видно какой класс проседает.
- Добавлен класс-доминант детектор: если один класс занимает > 60% предсказаний —
  выводится предупреждение (именно эта проблема была у друга).
- ROC/PR кривые теперь опциональны и не падают если классов больше 2.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import EVALUATION_DIR, ML_OUTPUT_DIR, PROCESSED_DIR
from src.models.ml_model import MLClassifier
from src.utils import setup_logger

logger = setup_logger("Evaluate")

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 12,
})

# Колонки которые не являются признаками
NON_FEATURE_COLUMNS = [
    'label', 'file_path', 'file_name', 'entity_type',
    'layer', 'entity_id', 'class', 'predicted_class', 'entity_handle',
]


# ---------------------------------------------------------------------------
# Вспомогательные функции визуализации
# ---------------------------------------------------------------------------

def plot_confusion_matrix_enhanced(y_true, y_pred, labels, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 2)))

    sns.heatmap(
        cm,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Количество образцов'},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
    )

    ax.set_xlabel('Предсказанный класс', fontsize=10, fontweight='bold')
    ax.set_ylabel('Истинный класс', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {output_path}")
    return cm


def plot_f1_bar_chart(report: dict, output_path: Path):
    classes = [k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')]
    f1_scores = [report[k]['f1-score'] for k in classes]
    idx = np.argsort(f1_scores)[::-1]
    classes_s = [classes[i] for i in idx]
    f1_s = [f1_scores[i] for i in idx]
    n = min(20, len(classes_s))

    plt.figure(figsize=(10, max(6, n * 0.5)))
    colors = plt.cm.RdYlGn(np.array(f1_s[:n]))
    bars = plt.barh(range(n), f1_s[:n], color=colors)
    plt.yticks(range(n), [c[:30] + '…' if len(c) > 30 else c for c in classes_s[:n]], fontsize=8)
    plt.xlabel('F1-Score', fontsize=9)
    plt.title('F1-Score по классам', fontsize=11, fontweight='bold')
    plt.xlim([0, 1.05])
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (bar, score) in enumerate(zip(bars, f1_s[:n])):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"F1-Score chart saved: {output_path}")


def _check_dominant_class(y_pred, threshold: float = 0.60):
    """
    Проверяет не доминирует ли один класс в предсказаниях.
    Если да — выводит предупреждение. Это главный симптом проблемы друга.
    """
    from collections import Counter
    counts = Counter(y_pred)
    total = len(y_pred)
    for cls, cnt in counts.most_common(3):
        ratio = cnt / total
        if ratio > threshold:
            logger.warning(
                f"\n{'!'*60}\n"
                f"ВНИМАНИЕ: Класс '{cls}' занимает {ratio:.1%} всех предсказаний!\n"
                f"Признак доминирования одного класса — модель не сбалансирована.\n"
                f"Проверьте балансировку данных и CLASS_WEIGHT_MULTIPLIERS в config.py\n"
                f"{'!'*60}"
            )
        else:
            break  # Остальные классы ещё меньше — не выводим


def _print_per_class_report(report: dict, y_true, y_pred):
    """Печатает подробный отчёт в консоль — сразу видно где проседает."""
    classes = [k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')]

    logger.info("\n" + "="*70)
    logger.info(f"{'Класс':<35} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    logger.info("="*70)

    for cls in sorted(classes, key=lambda c: report[c]['f1-score']):
        r = report[cls]
        warn = " ← НИЗКИЙ" if r['f1-score'] < 0.5 else ""
        logger.info(
            f"{cls:<35} {r['precision']:>10.3f} {r['recall']:>10.3f} "
            f"{r['f1-score']:>8.3f} {int(r['support']):>9}{warn}"
        )

    logger.info("="*70)
    logger.info(f"{'macro avg':<35} {report['macro avg']['precision']:>10.3f} "
                f"{report['macro avg']['recall']:>10.3f} "
                f"{report['macro avg']['f1-score']:>8.3f}")
    logger.info(f"Accuracy: {report['accuracy']:.4f}")
    logger.info("="*70 + "\n")


# ---------------------------------------------------------------------------
# Основная функция оценки ML модели
# ---------------------------------------------------------------------------

def evaluate_ml_model(
    model_path=None,
    test_csv=None,
    output_dir=None,
    eval_subfolder="ml",
):
    """
    Оценивает ML-модель на тестовой выборке.

    По умолчанию использует test_features.csv — отдельный файл
    который train_ml.py сохраняет из невиданных моделью файлов.
    """
    if model_path is None:
        model_path = PROCESSED_DIR / "models" / "ml_model.pkl"
    if test_csv is None:
        # ВАЖНО: test_features.csv, не train_features.csv
        test_csv = ML_OUTPUT_DIR / "test_features.csv"
    if output_dir is None:
        output_dir = EVALUATION_DIR

    output_dir = Path(output_dir) / eval_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем что тестовый CSV существует
    test_csv = Path(test_csv)
    if not test_csv.exists():
        logger.error(
            f"Тестовый CSV не найден: {test_csv}\n"
            f"Запустите сначала: python main.py --stage train_ml\n"
            f"Он автоматически создаст test_features.csv из невиданных файлов."
        )
        return None

    logger.info(f"Загружаем модель: {model_path}")
    model = MLClassifier.load(model_path)

    logger.info(f"Загружаем тестовые данные: {test_csv}")
    df = pd.read_csv(test_csv)

    if df.empty:
        logger.error("Тестовый CSV пустой!")
        return None

    logger.info(f"Тестовая выборка: {len(df)} строк, {df['label'].nunique()} классов")

    # Подготавливаем признаки — берём только числовые, выравниваем с трейном
    X_df = df.drop(columns=['label'] + [c for c in NON_FEATURE_COLUMNS if c in df.columns],
                   errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])

    for col in model.feature_columns:
        if col not in X_df.columns:
            X_df[col] = 0.0
    X_df = X_df[model.feature_columns]

    X = model.scaler.transform(X_df.values)

    y_true_labels = df['label'].values
    y_pred_encoded = model.model.predict(X)
    y_pred_labels = model.label_encoder.inverse_transform(y_pred_encoded)

    # Детектор доминирующего класса
    _check_dominant_class(y_pred_labels)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    logger.info(f"\nAccuracy на тестовой выборке: {acc:.4f}")

    # Только те классы что есть в тестовой выборке
    labels_in_test = sorted(df['label'].unique())
    report = classification_report(
        y_true_labels, y_pred_labels,
        labels=labels_in_test,
        output_dict=True,
        zero_division=0,
    )

    _print_per_class_report(report, y_true_labels, y_pred_labels)

    # Визуализации
    plot_confusion_matrix_enhanced(
        y_true_labels, y_pred_labels,
        labels=labels_in_test,
        output_path=output_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — тест (Acc={acc:.2%})",
    )
    plot_f1_bar_chart(report, output_dir / "f1_score_by_class.png")

    # Сохраняем метрики
    metrics_summary = {
        'accuracy': acc,
        'n_test_samples': len(df),
        'n_classes': len(labels_in_test),
        'classes': labels_in_test,
        'test_csv': str(test_csv),
        'model_path': str(model_path),
        'report': report,
        'model_type': 'ml',
    }
    with open(output_dir / "metrics_summary.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv", encoding='utf-8-sig')

    logger.info(f"Все результаты сохранены в: {output_dir}")
    return {'accuracy': acc, 'report': report, 'model_type': 'ml'}


# ---------------------------------------------------------------------------
# GNN оценка — без изменений в логике, только подключено к новому логгеру
# ---------------------------------------------------------------------------

def evaluate_gnn_model(
    model_path=None,
    test_dataset=None,
    output_dir=None,
    eval_subfolder="gnn",
    batch_size=16,
    in_channels=7,
    num_classes=54,
    idx_to_class=None,   # {int -> class_name} для читаемых меток
):
    import torch
    from torch_geometric.loader import DataLoader
    from src.models.gnn_model import GNNClassifier

    if model_path is None:
        model_path = PROCESSED_DIR / "models" / "gnn_model.pt"
    if output_dir is None:
        output_dir = EVALUATION_DIR

    output_dir = Path(output_dir) / eval_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading GNN model from {model_path}")
    try:
        model = GNNClassifier.load(model_path, in_channels=in_channels, num_classes=num_classes)
    except Exception as e:
        logger.error(f"Failed to load GNN model: {e}")
        return None

    if test_dataset is None:
        logger.warning("No test dataset provided, skipping GNN evaluation")
        return None

    device = model.device
    model.model.to(device)
    model.model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model.model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    logger.info(f"GNN Accuracy: {acc:.4f}")

    # Если передан idx_to_class — используем имена классов вместо цифр
    if idx_to_class:
        y_true_labels = [idx_to_class.get(i, str(i)) for i in y_true]
        y_pred_labels = [idx_to_class.get(i, str(i)) for i in y_pred]
        labels = sorted(set(y_true_labels))
    else:
        y_true_labels = y_true
        y_pred_labels = y_pred
        labels = sorted(set(y_true))

    plot_confusion_matrix_enhanced(
        y_true_labels, y_pred_labels, labels,
        output_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — GNN (Acc={acc:.2%})",
    )

    report = classification_report(
        y_true_labels, y_pred_labels,
        labels=labels, output_dict=True, zero_division=0
    )
    plot_f1_bar_chart(report, output_dir / "f1_score_by_class.png")
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")

    with open(output_dir / "metrics_summary.json", 'w', encoding='utf-8') as f:
        json.dump({'accuracy': acc, 'n_samples': len(y_true), 'report': report}, f, indent=2)

    logger.info(f"GNN evaluation saved to {output_dir}")
    return {'accuracy': acc, 'report': report}
