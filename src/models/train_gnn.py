"""
train_gnn.py — обучение GNN на той же выборке что и ML.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from src.config import (
    GNN_OUTPUT_DIR, PROCESSED_DIR, EVALUATION_DIR, ML_OUTPUT_DIR,
    USE_GRID_SEARCH, GNN_GRID_SEARCH_CONFIG, GNN_TRAIN_CONFIG,
    CLASS_WEIGHT_MULTIPLIERS,
)
from src.models.gnn_model import GNNClassifier
from src.gnn_builder import build_graph_from_dxf, NODE_FEATURE_DIM, augment_dataset
from src.utils import setup_logger

logger = setup_logger("TrainGNN")


class CSVGraphDataset(Dataset):
    """
    Dataset который строит графы из DXF-файлов на основе train_features.csv.
    Использует те же файлы что и ML — для честного сравнения двух методов.
    """

    def __init__(self, csv_path: Path, transform=None):
        super().__init__(transform=transform)

        df = pd.read_csv(csv_path)
        if 'file_path' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV должен содержать колонки file_path и label")

        # Фильтрация по белому списку классов
        from src.config import CLASS_FILTER_CONFIG
        include = CLASS_FILTER_CONFIG.get('include_classes')
        if include:
            df = df[df['label'].isin(include)].copy()

        # Один граф на файл (не на entity)
        file_label_pairs = (
            df.groupby('file_path')['label']
            .first()
            .reset_index()
        )

        # Строим маппинг классов
        all_classes = sorted(file_label_pairs['label'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Строим и кешируем графы
        self._graphs = []
        skipped = 0
        for _, row in file_label_pairs.iterrows():
            fp = Path(row['file_path'])
            label = row['label']

            if not fp.exists():
                skipped += 1
                continue

            graph = build_graph_from_dxf(fp)
            if graph is None:
                skipped += 1
                continue

            graph.y = torch.tensor(
                [self.class_to_idx[label]], dtype=torch.long
            )
            graph.class_name = label
            self._graphs.append(graph)

        logger.info(
            f"CSVGraphDataset: {len(self._graphs)} графов загружено, "
            f"{skipped} пропущено"
        )

        # Балансировка графов — одинаковое кол-во на класс (как в ML)
        # Используем тот же балансировщик что и train_ml для согласованности
        counts = {}
        for g in self._graphs:
            c = g.class_name
            counts[c] = counts.get(c, 0) + 1
        min_count = min(counts.values())
        logger.info(f"Балансировка графов: target={min_count} на класс")
        logger.info(f"  До: {dict(sorted(counts.items()))}")

        import random as _rnd
        _rnd.seed(42)
        balanced = []
        per_class = {}
        for g in self._graphs:
            c = g.class_name
            per_class.setdefault(c, []).append(g)
        for c, graphs in per_class.items():
            _rnd.shuffle(graphs)
            balanced.extend(graphs[:min_count])
            logger.info(f"  {c:<30} {len(graphs):>3} → {min_count:>3} графов")
        _rnd.shuffle(balanced)
        self._graphs = balanced
        logger.info(f"  После: {len(self._graphs)} графов ({len(per_class)} классов)")
        logger.info(f"Классы: {self.class_to_idx}")

    def len(self):
        return len(self._graphs)

    def get(self, idx):
        return self._graphs[idx]


def build_class_weights_tensor(dataset: CSVGraphDataset, device) -> torch.Tensor:
    """
    Веса классов для CrossEntropyLoss — те же мультипликаторы что в ML.
    Базовая стратегия: обратная частота, поверх — CLASS_WEIGHT_MULTIPLIERS.
    """
    counts = {}
    for g in dataset._graphs:
        cls = g.class_name
        counts[cls] = counts.get(cls, 0) + 1

    total = sum(counts.values())
    n = len(counts)
    weights = torch.ones(n, dtype=torch.float)

    for cls, idx in dataset.class_to_idx.items():
        cnt = counts.get(cls, 1)
        base = total / (n * cnt)
        mult = CLASS_WEIGHT_MULTIPLIERS.get(cls, 1.0)
        weights[idx] = base * mult

    logger.info("\n📊 Веса классов для GNN:")
    for cls, idx in dataset.class_to_idx.items():
        logger.info(f"   {cls:<30} => {weights[idx]:.3f}")

    return weights.to(device)


def train_gnn_model(
    data_dir=None,
    model_type=None,
    use_grid_search=None,
    hidden_channels=None,
    num_layers=None,
    lr=None,
    epochs=None,
    batch_size=None,
    val_ratio=None,
    output_path=None,
    run_evaluation=True,
):
    if output_path is None:
        output_path = PROCESSED_DIR / "models" / "gnn_model.pt"
    if use_grid_search is None:
        use_grid_search = USE_GRID_SEARCH

    # Параметры из конфига (с возможностью переопределить через аргументы)
    cfg = GNN_TRAIN_CONFIG
    final_hidden   = hidden_channels or cfg['hidden_channels']
    final_layers   = num_layers      or cfg['num_layers']
    final_lr       = lr              or cfg['lr']
    final_epochs   = epochs          or cfg['epochs']
    final_batch    = batch_size      or cfg['batch_size']
    final_val      = val_ratio       or cfg['val_ratio']
    final_mtype    = model_type      or cfg.get('model_type', 'gcn')

    logger.info("=" * 60)
    logger.info("GNN TRAINING")
    logger.info("=" * 60)
    logger.info(f"  model_type:      {final_mtype}")
    logger.info(f"  hidden_channels: {final_hidden}")
    logger.info(f"  num_layers:      {final_layers}")
    logger.info(f"  lr:              {final_lr}")
    logger.info(f"  epochs:          {final_epochs}")
    logger.info(f"  batch_size:      {final_batch}")
    logger.info(f"  NODE_FEATURE_DIM:{NODE_FEATURE_DIM}")
    logger.info("=" * 60)

    # --- Загрузка датасета из того же CSV что и ML ---
    csv_path = ML_OUTPUT_DIR / "train_features.csv"
    if not csv_path.exists():
        logger.error(f"train_features.csv не найден: {csv_path}")
        logger.error("Запустите сначала: python main.py --stage prepare")
        return None, None

    logger.info(f"Строим графы из: {csv_path}")
    dataset = CSVGraphDataset(csv_path)

    if len(dataset) == 0:
        logger.error("Датасет пустой — нет доступных DXF-файлов")
        return None, None

    num_classes = len(dataset.class_to_idx)
    logger.info(f"Графов: {len(dataset)}, классов: {num_classes}")

    # --- Стратифицированный split по классам ---
    # random_split не учитывает классы — малые классы (1 файл) могут
    # целиком уйти в val и не попасть в трейн. Делаем split вручную.
    from collections import defaultdict
    import random as _random

    _random.seed(42)
    class_indices = defaultdict(list)
    for i, g in enumerate(dataset._graphs):
        cls_idx = int(g.y.item())
        class_indices[cls_idx].append(i)

    train_idx, val_idx = [], []
    for cls_idx, indices in class_indices.items():
        _random.shuffle(indices)
        n_v = max(1, int(len(indices) * final_val))
        # Гарантируем хотя бы 1 в трейне
        if len(indices) - n_v < 1:
            n_v = max(0, len(indices) - 1)
        val_idx.extend(indices[:n_v])
        train_idx.extend(indices[n_v:])

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    logger.info(f"Train: {len(train_ds)} графов | Val: {len(val_ds)} графов")
    # Проверяем что все классы есть в трейне
    train_classes = set(int(dataset._graphs[i].y.item()) for i in train_idx)
    val_classes   = set(int(dataset._graphs[i].y.item()) for i in val_idx)
    logger.info(f"Классов в трейне: {len(train_classes)}, в вале: {len(val_classes)}")

    # Аугментируем только трейн — вал остаётся чистым для честной оценки
    AUG_FACTOR = 5
    orig_train_graphs = [dataset._graphs[i] for i in train_idx]
    augmented_train   = augment_dataset(orig_train_graphs, factor=AUG_FACTOR)
    logger.info(f"Аугментация трейна: {len(orig_train_graphs)} → {len(augmented_train)} графов")

    # Создаём простой список-датасет для DataLoader
    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, graphs): self.graphs = graphs
        def __len__(self):          return len(self.graphs)
        def __getitem__(self, i):   return self.graphs[i]

    train_loader = DataLoader(_ListDataset(augmented_train), batch_size=final_batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=final_batch, shuffle=False)

    # --- Модель ---
    model = GNNClassifier(
        in_channels=NODE_FEATURE_DIM,
        num_classes=num_classes,
        hidden_channels=final_hidden,
        num_layers=final_layers,
        lr=final_lr,
        model_type=final_mtype,
    )

    # Взвешенная loss для компенсации дисбаланса
    class_weights = build_class_weights_tensor(dataset, model.device)
    model.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info(f"Параметров модели: {n_params:,}")

    # --- Обучение ---
    best_val_acc = 0.0
    history = []

    for epoch in range(1, final_epochs + 1):
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
            model.save(output_path)  # сохраняем лучшую модель

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_m['loss']:.4f} | "
                f"Train: {train_m['acc']:.4f} | "
                f"Val: {val_m['acc']:.4f}"
            )

    logger.info(f"\nGNN обучение завершено. Лучший val acc: {best_val_acc:.4f}")
    logger.info(f"Модель сохранена: {output_path}")

    # --- Оценка ---
    if run_evaluation:
        try:
            from evaluation.evaluate import evaluate_gnn_model
            evaluate_gnn_model(
                model_path=output_path,
                test_dataset=val_ds,
                output_dir=EVALUATION_DIR,
                eval_subfolder="gnn",
                batch_size=final_batch,
                in_channels=NODE_FEATURE_DIM,
                num_classes=num_classes,
                idx_to_class=dataset.idx_to_class,
            )
        except Exception as e:
            logger.error(f"Оценка GNN не удалась: {e}")
            import traceback
            traceback.print_exc()

    return model, history


if __name__ == "__main__":
    train_gnn_model()
