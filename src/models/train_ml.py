import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import ML_OUTPUT_DIR, PROCESSED_DIR, CLASS_WEIGHT_MULTIPLIERS, CLASS_FILTER_CONFIG
from src.models.ml_model import MLClassifier
from src.balancer import balance_dataframe
from src.utils import setup_logger

logger = setup_logger("TrainML")

NON_FEATURE_COLUMNS = [
    'label', 'file_path', 'file_name', 'entity_type',
    'layer', 'entity_id', 'class', 'predicted_class', 'entity_handle',
]


def build_class_weights(label_series: pd.Series) -> dict:
    classes   = label_series.unique()
    counts    = label_series.value_counts()
    total     = len(label_series)
    n_classes = len(classes)

    base_weights  = {cls: total / (n_classes * counts[cls]) for cls in classes}
    final_weights = {
        cls: base_weights[cls] * CLASS_WEIGHT_MULTIPLIERS.get(cls, 1.0)
        for cls in classes
    }

    logger.info("\nВеса классов:")
    for cls in sorted(final_weights, key=lambda c: -final_weights[c]):
        logger.info(
            f"  {cls:<30} base={base_weights[cls]:.3f} "
            f"x{CLASS_WEIGHT_MULTIPLIERS.get(cls, 1.0):.1f} "
            f"=> {final_weights[cls]:.3f}"
        )
    return final_weights


def split_by_files(df: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42):
    """
    Разделяет данные на train/test на уровне файлов (не строк).
    Для классов с 1 файлом — 30% строк копируется в тест.
    """
    if 'file_path' not in df.columns:
        logger.warning("file_path не найдена — обычный stratified split")
        return train_test_split(df, test_size=test_size,
                                random_state=random_state, stratify=df['label'])

    train_parts, test_parts = [], []

    for label, group in df.groupby('label'):
        files = group['file_path'].unique()

        if len(files) < 2:
            train_parts.append(group)
            test_sample = group.sample(frac=0.3, random_state=random_state)
            test_parts.append(test_sample)
            logger.debug(f"  '{label}': 1 файл → train={len(group)}, test={len(test_sample)} (копия)")
            continue

        n_test      = max(1, int(len(files) * test_size))
        rng         = np.random.default_rng(random_state)
        test_files  = set(rng.choice(files, size=n_test, replace=False))
        train_files = set(files) - test_files

        train_parts.append(group[group['file_path'].isin(train_files)])
        test_parts.append(group[group['file_path'].isin(test_files)])

    df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=random_state)
    df_test  = pd.concat(test_parts,  ignore_index=True).sample(frac=1, random_state=random_state)

    logger.info(f"Split: train={len(df_train)} строк | test={len(df_test)} строк")
    return df_train, df_test


def train_ml_model(csv_path=None, model_type=None, val_size=0.2, output_path=None):
    from src.config import MODEL_TYPE

    if csv_path    is None: csv_path    = PROCESSED_DIR / "ml" / "train_features.csv"
    if output_path is None: output_path = PROCESSED_DIR / "models" / "ml_model.pkl"
    if model_type  is None: model_type  = MODEL_TYPE

    logger.info(f"Загрузка данных: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.error("CSV пустой!")
        return None

    # фильтр по белому списку классов 
    include = CLASS_FILTER_CONFIG.get('include_classes')
    if include:
        before = len(df)
        df     = df[df['label'].isin(include)].copy()
        logger.info(f"Белый список: {before} → {len(df)} строк, классы: {sorted(df['label'].unique())}")

    logger.info(f"Загружено {len(df)} строк, {df['label'].nunique()} классов")

    # строгая балансировка — одинаковое кол-во строк на класс
    df = balance_dataframe(df, label_col='label')

    # split train/test
    df_train, df_test = split_by_files(df, test_size=val_size)

    # веса классов
    class_weights = build_class_weights(df_train['label'])

    # обучение
    model     = MLClassifier(model_type=model_type, class_weight=None)
    X_train, y_train = model.prepare_data(df_train)

    int_weight_dict = {
        idx: class_weights.get(name, 1.0)
        for idx, name in enumerate(model.label_encoder.classes_)
    }
    model.model.set_params(class_weight=int_weight_dict)
    metrics = model.train(X_train, y_train, val_size=None)

    # оценка на тестовых данных 
    if not df_test.empty:
        logger.info("\n--- Оценка на тестовой выборке ---")
        try:
            X_test_df = df_test.drop(
                columns=['label'] + [c for c in NON_FEATURE_COLUMNS if c in df_test.columns],
                errors='ignore'
            ).select_dtypes(include=[np.number])

            for col in model.feature_columns:
                if col not in X_test_df.columns:
                    X_test_df[col] = 0.0
            X_test_df = X_test_df[model.feature_columns]

            X_test_scaled  = model.scaler.transform(X_test_df.values)
            y_test_encoded = model.label_encoder.transform(df_test['label'].values)
            test_score     = model.model.score(X_test_scaled, y_test_encoded)

            logger.info(f"Test accuracy: {test_score:.4f}")
            metrics['test_acc'] = test_score

            test_csv = PROCESSED_DIR / "ml" / "test_features.csv"
            df_test.to_csv(test_csv, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"Ошибка оценки: {e}")

    model.save(output_path)
    logger.info(f"Модель сохранена: {output_path}")

    try:
        test_csv = PROCESSED_DIR / "ml" / "test_features.csv"
        if test_csv.exists():
            from evaluation.evaluate import evaluate_ml_model
            evaluate_ml_model(
                model_path=output_path,
                test_csv=test_csv,
                output_dir=PROCESSED_DIR / "evaluation"
            )
    except Exception as e:
        logger.error(f"Evaluate failed: {e}")

    return model, metrics


if __name__ == "__main__":
    train_ml_model()
