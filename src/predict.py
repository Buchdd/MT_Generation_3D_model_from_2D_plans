"""
predict.py — предсказание классов для DXF файлов.

Изменения:
- Добавлен GNN инференс: predict_dxf теперь принимает параметр use_gnn.
  При use_gnn=True строит граф из файла и прогоняет через GNNClassifier.
- predict_directory принимает mode='ml'|'gnn'|'both' — в режиме 'both'
  результаты обоих методов сохраняются рядом для сравнения.
- Убран HARDCODE для Стены — он работал с метрами, а данные в мм.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import PRED_RAW_DIR, PROCESSED_DIR
from src.models.ml_model import MLClassifier
from src.dxf_reader import load_dxf_document, get_entities
from src.utils import setup_logger, get_dxf_files
from src.ml_features import extract_entity_features, extract_file_context
from evaluation.visualize import visualize_segmentation

logger = setup_logger("Predict")


# ---------------------------------------------------------------------------
# ML предсказание
# ---------------------------------------------------------------------------

def predict_dxf_ml(dxf_path, model_path=None) -> pd.DataFrame | None:
    """Предсказание класса каждой entity через Random Forest."""
    if model_path is None:
        model_path = PROCESSED_DIR / "models" / "ml_model.pkl"

    try:
        model = MLClassifier.load(model_path)
    except Exception as e:
        logger.error(f"ML модель не загружена: {e}")
        return None

    doc, msp = load_dxf_document(dxf_path)
    if msp is None:
        logger.error(f"Не удалось прочитать DXF: {dxf_path}")
        return None

    entities = get_entities(msp, ['LINE', 'CIRCLE', 'ARC', 'POLYLINE', 'LWPOLYLINE'])
    logger.info(f"ML: {len(entities)} entities в {Path(dxf_path).name}")

    file_ctx = extract_file_context(msp)
    rows = []

    for entity in entities:
        try:
            feats = extract_entity_features(entity, file_context=file_ctx)
            if not feats:
                continue

            X = pd.DataFrame([feats])
            for col in model.feature_columns:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[model.feature_columns]

            X_scaled    = model.scaler.transform(X.values)
            pred_class  = model.predict(X_scaled)[0]
            proba       = model.predict_proba(X_scaled)[0]
            confidence  = float(max(proba))

            rows.append({
                'entity_handle':   getattr(entity.dxf, 'handle', None) or str(id(entity)),
                'predicted_class': pred_class,
                'confidence':      confidence,
                'entity_type':     entity.dxftype(),
                'method':          'ML',
            })
        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    _log_class_counts(df, "ML")
    return df


# ---------------------------------------------------------------------------
# GNN предсказание
# ---------------------------------------------------------------------------

def predict_dxf_gnn(dxf_path, model_path=None) -> pd.DataFrame | None:
    """
    Предсказание класса файла через GNN.

    GNN работает на уровне ФАЙЛА (один граф = один класс),
    а не на уровне entity. Поэтому все entity файла получают
    одинаковый предсказанный класс — класс всего объекта.
    """
    if model_path is None:
        model_path = PROCESSED_DIR / "models" / "gnn_model.pt"

    if not Path(model_path).exists():
        logger.error(f"GNN модель не найдена: {model_path}")
        logger.error("Запустите: python main.py --stage train_gnn")
        return None

    # Загружаем маппинг классов — нужен для декодирования предсказания
    # Берём из train_features.csv (те же классы что при обучении)
    csv_path = PROCESSED_DIR / "ml" / "train_features.csv"
    if not csv_path.exists():
        logger.error(f"train_features.csv не найден: {csv_path}")
        return None

    try:
        df_train = pd.read_csv(csv_path)
        all_classes  = sorted(df_train['label'].unique())
        idx_to_class = {i: c for i, c in enumerate(all_classes)}
    except Exception as e:
        logger.error(f"Не удалось загрузить маппинг классов: {e}")
        return None

    # Загружаем GNN (архитектура берётся из checkpoint)
    try:
        from src.models.gnn_model import GNNClassifier
        gnn = GNNClassifier.load(model_path)
    except Exception as e:
        logger.error(f"GNN не загружена: {e}")
        return None

    # Строим граф из DXF
    try:
        from src.gnn_builder import build_graph_from_dxf
        graph = build_graph_from_dxf(Path(dxf_path))
    except Exception as e:
        logger.error(f"Построение графа не удалось: {e}")
        return None

    if graph is None:
        logger.warning(f"Пустой граф для {dxf_path}")
        return None

    # Инференс — GNN даёт один класс на весь файл
    try:
        class_idx, proba = gnn.predict_graph(graph)
        pred_class  = idx_to_class.get(class_idx, f"unknown_{class_idx}")
        confidence  = float(max(proba))
    except Exception as e:
        logger.error(f"GNN инференс не удался: {e}")
        return None

    # Читаем entity для сохранения handles (визуализация работает по handles)
    doc, msp = load_dxf_document(dxf_path)
    entities = get_entities(msp, ['LINE', 'CIRCLE', 'ARC', 'POLYLINE', 'LWPOLYLINE']) if msp else []

    logger.info(
        f"GNN: {Path(dxf_path).name} → '{pred_class}' "
        f"(conf={confidence:.3f}, {len(entities)} entities)"
    )

    rows = [
        {
            'entity_handle':   getattr(e.dxf, 'handle', None) or str(id(e)),
            'predicted_class': pred_class,
            'confidence':      confidence,
            'entity_type':     e.dxftype(),
            'method':          'GNN',
        }
        for e in entities
    ]

    # Выводим вероятности по всем классам
    logger.info("  Вероятности GNN:")
    for idx, prob in sorted(enumerate(proba), key=lambda x: -x[1])[:4]:
        cls = idx_to_class.get(idx, str(idx))
        logger.info(f"    {cls:<25} {prob:.3f}")

    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _log_class_counts(df: pd.DataFrame, method: str):
    counts = df['predicted_class'].value_counts()
    logger.info(f"{method}: предсказано {len(df)} entities")
    for cls, cnt in counts.items():
        logger.info(f"   {cls}: {cnt}")


def predict_and_save(dxf_path, mode='ml', ml_model_path=None,
                     gnn_model_path=None, output_dir=None):
    """
    Предсказание + сохранение CSV + визуализация для одного файла.

    mode: 'ml'   — только Random Forest
          'gnn'  — только GNN
          'both' — оба метода, результаты сохраняются раздельно
    """
    dxf_path = Path(dxf_path)
    if output_dir is None:
        output_dir = PROCESSED_DIR / "evaluation" / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if mode in ('ml', 'both'):
        df_ml = predict_dxf_ml(dxf_path, ml_model_path)
        if df_ml is not None:
            results['ML'] = df_ml
            csv_path = output_dir / f"{dxf_path.stem}_predictions_ml.csv"
            df_ml.to_csv(csv_path, index=False, encoding='utf-8-sig')
            pred_dict = dict(zip(df_ml['entity_handle'], df_ml['predicted_class']))
            img_path  = output_dir / f"{dxf_path.stem}_segmentation_ml.png"
            visualize_segmentation(dxf_path, pred_dict, img_path)
            logger.info(f"ML: CSV → {csv_path.name}, IMG → {img_path.name}")

    if mode in ('gnn', 'both'):
        df_gnn = predict_dxf_gnn(dxf_path, gnn_model_path)
        if df_gnn is not None:
            results['GNN'] = df_gnn
            csv_path = output_dir / f"{dxf_path.stem}_predictions_gnn.csv"
            df_gnn.to_csv(csv_path, index=False, encoding='utf-8-sig')
            pred_dict = dict(zip(df_gnn['entity_handle'], df_gnn['predicted_class']))
            img_path  = output_dir / f"{dxf_path.stem}_segmentation_gnn.png"
            visualize_segmentation(dxf_path, pred_dict, img_path)
            logger.info(f"GNN: CSV → {csv_path.name}, IMG → {img_path.name}")

    # Обратная совместимость: predict_directory ожидает DataFrame
    if 'ML' in results:
        return results['ML']
    if 'GNN' in results:
        return results['GNN']
    return None


def predict_dxf(dxf_path, model_path=None):
    """Обратная совместимость — ML предсказание."""
    return predict_dxf_ml(dxf_path, model_path)


def predict_and_visualize_file(dxf_path, model_path=None, output_dir=None):
    """Обратная совместимость."""
    return predict_and_save(dxf_path, mode='ml',
                            ml_model_path=model_path, output_dir=output_dir)


def predict_directory(dir_path=None, model_path=None, output_dir=None,
                      mode='both'):
    """
    Пакетное предсказание для папки.

    mode='both' — запускает и ML и GNN для каждого файла,
    сохраняет раздельные CSV и картинки для сравнения.
    """
    if dir_path is None:
        dir_path = PRED_RAW_DIR
    dir_path = Path(dir_path)

    if output_dir is None:
        output_dir = PROCESSED_DIR / "evaluation" / "segmentation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dxf_files = get_dxf_files(dir_path)
    logger.info(f"Найдено {len(dxf_files)} файлов | режим: {mode}")

    results = []
    for f in dxf_files:
        try:
            df = predict_and_save(
                f, mode=mode,
                ml_model_path=model_path,
                output_dir=output_dir,
            )
            if df is not None:
                results.append({
                    'file':     f.name,
                    'entities': len(df),
                    'classes':  df['predicted_class'].nunique(),
                    'mode':     mode,
                })
        except Exception as e:
            logger.error(f"Ошибка {f.name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(output_dir / "summary.csv", index=False)
        logger.info(f"Готово. Обработано: {len(results)} файлов")

    return results


if __name__ == "__main__":
    predict_directory()
