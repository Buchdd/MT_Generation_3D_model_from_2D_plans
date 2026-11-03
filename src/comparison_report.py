"""
comparison_report.py — сравнительный отчёт Rule-based vs ML vs GNN.

Для каждого DXF файла показывает:
  - Что нашёл rule-based (по именам слоёв)
  - Что предсказал ML для нераспознанных слоёв
  - Что предсказал GNN для нераспознанных слоёв
  - Совпадают ли ML и GNN

Генерирует:
  - CSV таблицу с метриками
  - PNG с confusion matrix ML vs GNN (если есть ground truth)
  - Текстовый отчёт
"""

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from src.layer_parser import parse_floor_plan, classify_layer
from src.ml_fallback import enhance_plan_with_ml_gnn
from src.config import PROCESSED_DIR, PRED_RAW_DIR
from src.utils import setup_logger

logger = setup_logger("ComparisonReport")


def run_comparison(dir_path: Path = None,
                   output_dir: Path = None) -> pd.DataFrame:
    """
    Запускает сравнение трёх методов на всех DXF файлах.
    Возвращает DataFrame с результатами.
    """
    if dir_path   is None: dir_path   = PRED_RAW_DIR
    if output_dir is None: output_dir = PROCESSED_DIR / "evaluation" / "comparison"

    dir_path   = Path(dir_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dxf_files = list(dir_path.glob("*.dxf"))
    logger.info(f"Сравнение методов на {len(dxf_files)} файлах")

    all_rows = []

    for dxf_path in dxf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Файл: {dxf_path.name}")

        # --- Rule-based ---
        plan = parse_floor_plan(dxf_path)

        rule_based = {
            'walls':      len(plan.walls),
            'partitions': len(plan.partitions),
            'doors':      len(plan.doors),
            'windows':    len(plan.windows),
            'stairs':     len(plan.stairs),
            'unknown':    len(plan.unknown_layers),
        }
        rule_total = sum(v for k, v in rule_based.items() if k != 'unknown')

        logger.info(f"\nRule-based результат:")
        for k, v in rule_based.items():
            logger.info(f"  {k}: {v}")

        # --- ML + GNN fallback ---
        fallback = {}
        if plan.unknown_layers:
            fallback = enhance_plan_with_ml_gnn(
                plan, dxf_path, use_ml=True, use_gnn=True
            )

        ml_results  = fallback.get('ml_results', [])
        gnn_results = fallback.get('gnn_results', [])

        # Строим строки для каждого нераспознанного слоя
        for layer_name, n_entity in plan.unknown_layers.items():
            ml_pred  = next((r for r in ml_results  if r.layer_name == layer_name), None)
            gnn_pred = next((r for r in gnn_results if r.layer_name == layer_name), None)

            row = {
                'file':          dxf_path.name,
                'layer':         layer_name,
                'entity_count':  n_entity,
                'rule_based':    'UNKNOWN',
                'ml_class':      ml_pred.predicted_class  if ml_pred  else 'N/A',
                'ml_confidence': round(ml_pred.confidence, 3) if ml_pred else 0.0,
                'gnn_class':     gnn_pred.predicted_class if gnn_pred else 'N/A',
                'gnn_confidence':round(gnn_pred.confidence, 3) if gnn_pred else 0.0,
                'ml_gnn_agree':  (ml_pred.predicted_class == gnn_pred.predicted_class)
                                  if (ml_pred and gnn_pred) else None,
            }
            all_rows.append(row)

        # Общая строка по файлу
        all_rows.append({
            'file':          dxf_path.name,
            'layer':         '=== ИТОГО ===',
            'entity_count':  rule_total + fallback.get('ml_added', 0),
            'rule_based':    f"{rule_total} элементов",
            'ml_class':      f"+{fallback.get('ml_added', 0)} из UNKNOWN",
            'ml_confidence': '',
            'gnn_class':     f"{len(gnn_results)} слоёв распознано",
            'gnn_confidence': '',
            'ml_gnn_agree':  None,
        })

    df = pd.DataFrame(all_rows)

    # Сохраняем CSV
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"\nCSV сохранён: {csv_path}")

    # Генерируем визуализацию
    _plot_comparison(df, output_dir)
    _plot_confidence(df, output_dir)
    _save_text_report(df, output_dir, dxf_files)

    return df


def _plot_comparison(df: pd.DataFrame, output_dir: Path):
    """График: распределение предсказаний ML vs GNN по классам."""
    detail = df[df['layer'] != '=== ИТОГО ==='].copy()
    if detail.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Распределение предсказаний для нераспознанных слоёв',
                 fontsize=14, fontweight='bold')

    for ax, col, title in [
        (axes[0], 'ml_class',  'ML (Random Forest)'),
        (axes[1], 'gnn_class', 'GNN'),
    ]:
        counts = detail[col].value_counts()
        counts = counts[counts.index != 'N/A']
        if counts.empty:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        colors = plt.cm.Set3(range(len(counts)))
        bars = ax.bar(range(len(counts)), counts.values, color=colors)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Количество слоёв')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "ml_gnn_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"График предсказаний: {output_dir / 'ml_gnn_predictions.png'}")


def _plot_confidence(df: pd.DataFrame, output_dir: Path):
    """График: уверенность ML vs GNN."""
    detail = df[df['layer'] != '=== ИТОГО ==='].copy()
    detail = detail[(detail['ml_confidence'] > 0) | (detail['gnn_confidence'] > 0)]

    if detail.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(detail))
    width = 0.35

    ml_conf  = pd.to_numeric(detail['ml_confidence'],  errors='coerce').fillna(0)
    gnn_conf = pd.to_numeric(detail['gnn_confidence'], errors='coerce').fillna(0)

    bars1 = ax.bar([i - width/2 for i in x], ml_conf,  width, label='ML',  color='#3498DB', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], gnn_conf, width, label='GNN', color='#E74C3C', alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{r['file'][:10]}\n{r['layer'][:15]}" for _, r in detail.iterrows()],
        rotation=30, ha='right', fontsize=7
    )
    ax.set_ylabel('Уверенность (confidence)')
    ax.set_ylim(0, 1.1)
    ax.set_title('Уверенность ML vs GNN для нераспознанных слоёв',
                 fontweight='bold')
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Порог 0.5')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ml_gnn_confidence.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"График уверенности: {output_dir / 'ml_gnn_confidence.png'}")


def _save_text_report(df: pd.DataFrame, output_dir: Path, dxf_files: list):
    """Текстовый отчёт с выводами."""
    detail = df[df['layer'] != '=== ИТОГО ==='].copy()

    lines = [
        "=" * 70,
        "СРАВНИТЕЛЬНЫЙ ОТЧЁТ: Rule-based vs ML vs GNN",
        "=" * 70,
        "",
        f"Проанализировано файлов: {len(dxf_files)}",
        f"Нераспознанных слоёв:    {len(detail)}",
        "",
    ]

    # По каждому файлу
    for fname in detail['file'].unique():
        fdata = detail[detail['file'] == fname]
        lines.append(f"--- {fname} ---")
        for _, row in fdata.iterrows():
            agree = ""
            if row['ml_gnn_agree'] is True:
                agree = "✓ ML=GNN"
            elif row['ml_gnn_agree'] is False:
                agree = "✗ РАСХОДЯТСЯ"
            lines.append(
                f"  Слой '{row['layer']}' ({row['entity_count']} entity):"
            )
            lines.append(
                f"    ML:  {row['ml_class']:<25} conf={row['ml_confidence']}"
            )
            lines.append(
                f"    GNN: {row['gnn_class']:<25} conf={row['gnn_confidence']} {agree}"
            )
        lines.append("")

    # Статистика совпадений
    agree_data = detail[detail['ml_gnn_agree'].notna()]
    if not agree_data.empty:
        n_agree    = agree_data['ml_gnn_agree'].sum()
        n_total    = len(agree_data)
        agree_pct  = 100 * n_agree / n_total if n_total > 0 else 0
        lines += [
            "=" * 70,
            "СТАТИСТИКА СОВПАДЕНИЙ ML и GNN:",
            f"  Совпали: {n_agree}/{n_total} ({agree_pct:.1f}%)",
            "",
            "ВЫВОД:",
            "  Rule-based: точный результат для размеченных слоёв.",
            "  ML/GNN: fallback для неразмеченных слоёв (слой '0' и т.п.).",
            f"  Совпадение ML и GNN ({agree_pct:.1f}%) показывает надёжность",
            "  геометрических признаков для данной задачи.",
            "=" * 70,
        ]

    report_path = output_dir / "comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Текстовый отчёт: {report_path}")

    # Выводим в лог
    for line in lines:
        logger.info(line)
