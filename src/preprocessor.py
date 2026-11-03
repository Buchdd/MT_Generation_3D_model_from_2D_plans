import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from src.config import TRAIN_RAW_DIR, ML_OUTPUT_DIR, GNN_OUTPUT_DIR, ensure_dirs, CLASS_FILTER_CONFIG
from src.utils import setup_logger, get_dxf_files, extract_class_from_filename, get_class_statistics
from src.gnn_builder import build_graph_from_dxf
from src.class_filter import RevitClassFilter
from src.data_augmentation import DXFAugmentor
from src.dxf_reader import load_dxf_document, get_entities
from src.ml_features import extract_entity_features

logger = setup_logger("Preprocessor")
MODE = "2d"

def _balance_dataframe_rows(df: pd.DataFrame, balancing_config: dict) -> pd.DataFrame:
    """
    Балансировка на уровне entity-строк CSV.

    Стратегия:
    - Крупные классы: берём случайную выборку целых файлов (file_path),
      пока не достигнем target. Файл берётся ЦЕЛИКОМ — ни одна линия не вырезается.
    - Малые классы: оставляем как есть. Не дублируем строки —
      модель получит class_weight='balanced' для компенсации.

    Такой подход сохраняет геометрическую целостность каждого объекта.
    """
    import random

    target = balancing_config.get('target_rows_per_class')
    if target is None:
        counts = df['label'].value_counts()
        target = int(counts.median())
        logger.info(f"⚖️  Auto target rows per class: {target} (median)")

    logger.info(f"\n{'='*60}")
    logger.info("⚖️  БАЛАНСИРОВКА СТРОК CSV (по файлам целиком)")
    logger.info(f"{'='*60}")
    logger.info(f"Целевое кол-во строк на класс: {target}")

    result_parts = []

    for label, group in df.groupby('label'):
        row_count = len(group)

        if row_count <= target:
            # Малый класс — берём всё что есть
            result_parts.append(group)
            logger.info(f"  ✅ '{label}': {row_count} строк (оставляем все)")
            continue

        # Крупный класс — downsample по файлам целиком
        if 'file_path' in group.columns:
            files = group['file_path'].unique().tolist()
            random.shuffle(files)

            selected_rows = []
            for fp in files:
                file_rows = group[group['file_path'] == fp]
                selected_rows.append(file_rows)
                if sum(len(r) for r in selected_rows) >= target:
                    break

            selected = pd.concat(selected_rows, ignore_index=True)
            # Если последний файл перебрал — обрезаем до target
            if len(selected) > target:
                selected = selected.iloc[:target]
        else:
            # Нет колонки file_path — простая случайная выборка строк
            selected = group.sample(n=target, random_state=42)

        result_parts.append(selected)
        logger.info(f"  📉 '{label}': {row_count} → {len(selected)} строк (downsampled по файлам)")

    balanced = pd.concat(result_parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"{'='*60}")
    logger.info(f"ИТОГО после балансировки: {len(balanced)} строк")
    logger.info(f"{'='*60}\n")
    return balanced

def process_ml_data(
    train_dir,
    output_csv,
    min_samples_per_class=5,
    target_samples_per_class=None,
    include_classes=None,
    use_revit_structure=True,
    augment_data=False,
    augmentation_factor=3,
    exclude_folders=None,
    ignored_folder_names=None,
    balancing_config=None,
):
    from src.config import BALANCING_CONFIG
    if balancing_config is None:
        balancing_config = BALANCING_CONFIG

    logger.info(f"Starting ML data processing from {train_dir}")

    path_class_mapping = CLASS_FILTER_CONFIG.get('path_class_mapping', {})
    class_filter = RevitClassFilter(
        base_library_path=None,
        exclude_folders=exclude_folders,
        ignored_folder_names=ignored_folder_names,
        path_class_mapping=path_class_mapping,
    ) if use_revit_structure else None

    dxf_files = get_dxf_files(train_dir, recursive=True)
    logger.info(f"Found {len(dxf_files)} DXF files")

    if class_filter:
        dxf_files, class_stats = class_filter.filter_classes(
            dxf_files,
            min_samples=min_samples_per_class,
            include_classes=include_classes
        )

    if not dxf_files:
        logger.error("No files after filtering!")
        return

    # Шаг 1: балансировка на уровне ФАЙЛОВ (downsample крупных классов)
    # Цель — не тянуть тысячи DXF Стен, но и не резать геометрию
    if target_samples_per_class and target_samples_per_class > 0:
        logger.info(f"⚖️ File-level balancing to {target_samples_per_class} files per class...")
        augmentor = DXFAugmentor() if augment_data else None
        dxf_files, _ = class_filter.balance_classes(
            dxf_files, target_samples_per_class, augmentor=augmentor,
            output_dir=train_dir.parent / "balanced"
        )
    elif augment_data:
        augmentor = DXFAugmentor()
        dxf_files = augmentor.augment_dataset(
            dxf_files, train_dir.parent / "augmented", augmentation_factor, class_filter
        )

    get_class_statistics(
        dxf_files,
        class_extractor=lambda f: class_filter.extract_class_from_path(f) if class_filter else None
    )

    # Шаг 2: извлечение признаков по сущностям
    # file_path сохраняем в CSV — нужен для row-level балансировки
    all_features = []
    for file_path in tqdm(dxf_files, desc="Extracting entity features"):
        class_name = (
            class_filter.extract_class_from_path(file_path)
            if class_filter else extract_class_from_filename(file_path.name)
        )
        if not class_name:
            continue

        doc, msp = load_dxf_document(file_path)
        if not msp:
            continue

        # Вычисляем контекст файла один раз — добавляем ко всем entity внутри.
        # Это даёт модели понять в каком окружении живёт каждая линия.
        from src.ml_features import extract_file_context, extract_entity_features
        file_ctx = extract_file_context(msp)

        for entity in get_entities(msp, entity_types=['LINE', 'CIRCLE', 'ARC', 'POLYLINE', 'LWPOLYLINE']):
            feats = extract_entity_features(entity, file_context=file_ctx, mode=MODE)
            if feats:
                feats['label'] = class_name
                feats['file_path'] = str(file_path)
                all_features.append(feats)

    if not all_features:
        logger.warning("No features extracted.")
        return

    df = pd.DataFrame(all_features)

    logger.info(f"\n📊 Строк до балансировки: {len(df)}")
    for lbl, cnt in df['label'].value_counts().items():
        logger.info(f"   {lbl}: {cnt}")

    # Шаг 3: балансировка на уровне СТРОК CSV (по файлам целиком)
    if balancing_config.get('strategy') == 'row_level':
        df = _balance_dataframe_rows(df, balancing_config)

    logger.info(f"\n📊 Строк после балансировки: {len(df)}")
    for lbl, cnt in df['label'].value_counts().items():
        logger.info(f"   {lbl}: {cnt}")

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"✅ Saved {len(df)} entity samples to {output_csv}")

def process_gnn_data(
    train_dir,
    output_dir,
    min_samples_per_class=5,
    target_samples_per_class=None,
    include_classes=None,  # 🔥 Добавлено
    use_revit_structure=True,
    augment_data=False,
    augmentation_factor=3,
    exclude_folders=None,
    ignored_folder_names=None
):
    logger.info(f"Starting GNN data processing from {train_dir}")
    class_filter = RevitClassFilter(
        base_library_path=None,
        exclude_folders=exclude_folders,
        ignored_folder_names=ignored_folder_names
    ) if use_revit_structure else None

    dxf_files = get_dxf_files(train_dir, recursive=True)
    logger.info(f"Found {len(dxf_files)} DXF files for GNN")

    if class_filter:
        # 🔥 ИСПРАВЛЕНИЕ: Передаём white-list в фильтр!
        dxf_files, class_stats = class_filter.filter_classes(
            dxf_files, 
            min_samples=min_samples_per_class,
            include_classes=include_classes
        )

    if target_samples_per_class and target_samples_per_class > 0:
        logger.info("⚖️ Starting GNN class balancing...")
        augmentor = DXFAugmentor(mode=MODE) if augment_data else None
        dxf_files, _ = class_filter.balance_classes(
            dxf_files, target_samples_per_class, augmentor=augmentor, output_dir=train_dir.parent / "balanced_gnn"
        )
    elif augment_data:
        augmentor = DXFAugmentor(mode=MODE)
        dxf_files = augmentor.augment_dataset(
            dxf_files, train_dir.parent / "augmented_gnn", augmentation_factor, class_filter
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0
    for file_path in tqdm(dxf_files, desc="Processing GNN data"):
        class_name = class_filter.extract_class_from_path(file_path) if class_filter else extract_class_from_filename(file_path.name)
        if class_name is None: continue
        
        graph_data = build_graph_from_dxf(file_path)
        if graph_data is not None:
            graph_data.class_name = class_name
            torch.save(graph_data, output_dir / f"{file_path.stem}.pt")
            processed_count += 1
            
    logger.info(f"✅ GNN Data saved to {output_dir} ({processed_count} graphs)")

def run_preprocessing(
    min_samples_per_class=None,
    target_samples_per_class=None,
    include_classes=None,
    use_revit_structure=None,
    augment_data=None,
    augmentation_factor=None,
    exclude_folders=None,
    ignored_folder_names=None,
    balancing_config=None,
):
    from src.config import BALANCING_CONFIG
    ensure_dirs()

    if min_samples_per_class is None: min_samples_per_class = CLASS_FILTER_CONFIG['min_samples_per_class']
    if target_samples_per_class is None: target_samples_per_class = CLASS_FILTER_CONFIG.get('target_samples_per_class')
    if include_classes is None: include_classes = CLASS_FILTER_CONFIG.get('include_classes')
    if use_revit_structure is None: use_revit_structure = CLASS_FILTER_CONFIG['use_revit_structure']
    if augment_data is None: augment_data = CLASS_FILTER_CONFIG['augment_data']
    if augmentation_factor is None: augmentation_factor = CLASS_FILTER_CONFIG['augmentation_factor']
    if exclude_folders is None: exclude_folders = CLASS_FILTER_CONFIG['exclude_folders']
    if ignored_folder_names is None: ignored_folder_names = CLASS_FILTER_CONFIG['ignored_folder_names']
    if balancing_config is None: balancing_config = BALANCING_CONFIG

    logger.info("="*50)
    logger.info("DXF Object Recognition - Data Preparation")
    logger.info("="*50)
    logger.info(f"Min samples per class (files): {min_samples_per_class}")
    logger.info(f"Target files per class:        {target_samples_per_class}")
    logger.info(f"Row balancing strategy:        {balancing_config['strategy']}")
    logger.info(f"Target rows per class:         {balancing_config.get('target_rows_per_class', 'auto (median)')}")

    if include_classes:
        logger.info(f"WHITE-LIST ACTIVE: {len(include_classes)} classes")
        for c in include_classes: logger.info(f"   {c}")
    else:
        logger.info("WHITE-LIST: DISABLED (all classes allowed)")

    logger.info(f"Augmentation: {augment_data} (x{augmentation_factor})")
    logger.info("="*50)

    process_ml_data(
        TRAIN_RAW_DIR, ML_OUTPUT_DIR / "train_features.csv",
        min_samples_per_class=min_samples_per_class,
        target_samples_per_class=target_samples_per_class,
        include_classes=include_classes,
        use_revit_structure=use_revit_structure,
        augment_data=augment_data, augmentation_factor=augmentation_factor,
        exclude_folders=exclude_folders, ignored_folder_names=ignored_folder_names,
        balancing_config=balancing_config,
    )

    process_gnn_data(
        TRAIN_RAW_DIR, GNN_OUTPUT_DIR,
        min_samples_per_class=min_samples_per_class,
        target_samples_per_class=target_samples_per_class,
        include_classes=include_classes,
        use_revit_structure=use_revit_structure,
        augment_data=augment_data, augmentation_factor=augmentation_factor,
        exclude_folders=exclude_folders, ignored_folder_names=ignored_folder_names,
    )

    logger.info("="*50)
    logger.info("Preprocessing completed successfully!")
    logger.info("="*50)