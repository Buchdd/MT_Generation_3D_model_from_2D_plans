import torch
from pathlib import Path
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from itertools import product
import pandas as pd
import json
from src.config import GNN_OUTPUT_DIR, PROCESSED_DIR, EVALUATION_DIR, GNN_GRID_SEARCH_CONFIG
from src.models.gnn_model import GNNClassifier
from src.utils import setup_logger, extract_class_from_filename
from src.gnn_builder import ENTITY_TYPE_MAP

logger = setup_logger("GridSearchGNN")


class DXFGraphDataset(Dataset):
    """Dataset для загрузки графов из .pt файлов"""
    
    def __init__(self, root_dir, transform=None):
        super().__init__(root=str(root_dir), transform=transform)
        self.graph_files = list(Path(root_dir).glob("*.pt"))
        
        self.class_to_idx = {}
        for f in self.graph_files:
            data = torch.load(f, weights_only=False)
            class_name = getattr(data, 'class_name', None)
            
            if class_name is None:
                class_name = extract_class_from_filename(f.stem)
            
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
                
    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        file_path = self.graph_files[idx]
        data = torch.load(file_path, weights_only=False)
        
        class_name = getattr(data, 'class_name', None)
        
        if class_name is None:
            class_name = extract_class_from_filename(file_path.stem)
        
        data.y = torch.tensor([self.class_to_idx[class_name]], dtype=torch.long)
        
        return data


def train_single_config(
    train_loader,
    val_loader,
    in_channels,
    num_classes,
    hidden_channels,
    num_layers,
    lr,
    epochs=15,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Обучение одной конфигурации гиперпараметров"""
    model = GNNClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        lr=lr
    )
    
    model.model.to(device)
    
    best_val_acc = 0
    
    for epoch in range(1, epochs + 1):
        train_metrics = model.train_epoch(train_loader)
        val_metrics = model.evaluate(val_loader)
        
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
    
    n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    return {
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'lr': lr,
        'train_acc': train_metrics['acc'],
        'val_acc': best_val_acc,
        'n_parameters': n_params
    }


def grid_search_gnn(
    data_dir=None,
    param_grid=None,
    epochs=None,
    batch_size=None,
    val_ratio=None,
    n_folds=None,
    output_path=None,
    save_results=True
):
    """
    Grid Search для подбора оптимальных гиперпараметров GNN
    Параметры берутся из config, если не переданы
    """
    if data_dir is None:
        data_dir = GNN_OUTPUT_DIR
    if output_path is None:
        output_path = EVALUATION_DIR / "gnn" / "grid_search_results"
    
    # ✅ Загружаем параметры из config
    config = GNN_GRID_SEARCH_CONFIG
    if param_grid is None:
        param_grid = config['param_grid']
    if epochs is None:
        epochs = config['epochs']
    if batch_size is None:
        batch_size = config['batch_size']
    if val_ratio is None:
        val_ratio = config['val_ratio']
    if n_folds is None:
        n_folds = config['n_folds']
    
    logger.info("="*70)
    logger.info("🔬 GNN GRID SEARCH")
    logger.info("="*70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Epochs per config: {epochs}")
    logger.info(f"Number of folds: {n_folds}")
    logger.info("="*70)
    
    dataset = DXFGraphDataset(data_dir)
    
    if len(dataset) == 0:
        logger.error("No graph files found!")
        return None
    
    in_channels = len(ENTITY_TYPE_MAP) + 2
    num_classes = len(dataset.class_to_idx)
    
    logger.info(f"Loaded {len(dataset)} graphs, {num_classes} classes")
    logger.info(f"in_channels: {in_channels}")
    
    all_combinations = list(product(
        param_grid['hidden_channels'],
        param_grid['num_layers'],
        param_grid['lr']
    ))
    
    total_configs = len(all_combinations)
    logger.info(f"\n📋 Total configurations to test: {total_configs}")
    
    results = []
    
    for idx, (hidden_channels, num_layers, lr) in enumerate(all_combinations, 1):
        logger.info(f"\n[{idx}/{total_configs}] Testing: hidden={hidden_channels}, layers={num_layers}, lr={lr}")
        
        fold_results = []
        
        for fold in range(n_folds):
            train_size = int((1 - val_ratio) * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42 + fold)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            result = train_single_config(
                train_loader=train_loader,
                val_loader=val_loader,
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                lr=lr,
                epochs=epochs
            )
            
            fold_results.append(result['val_acc'])
            
            if n_folds > 1:
                logger.info(f"   Fold {fold + 1}/{n_folds}: Val Acc = {result['val_acc']:.4f}")
        
        avg_val_acc = sum(fold_results) / len(fold_results)
        std_val_acc = torch.std(torch.tensor(fold_results)).item() if len(fold_results) > 1 else 0
        
        result_entry = {
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'lr': lr,
            'val_acc_mean': avg_val_acc,
            'val_acc_std': std_val_acc,
            'train_acc': result['train_acc'],
            'n_parameters': result['n_parameters']
        }
        
        results.append(result_entry)
        logger.info(f"   ✅ Mean Val Acc: {avg_val_acc:.4f} (±{std_val_acc:.4f})")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_acc_mean', ascending=False)
    
    best_config_row = results_df.iloc[0]
    best_config = {
        'hidden_channels': int(best_config_row['hidden_channels']),
        'num_layers': int(best_config_row['num_layers']),
        'lr': float(best_config_row['lr']),
        'val_acc_mean': float(best_config_row['val_acc_mean']),
        'val_acc_std': float(best_config_row['val_acc_std']),
        'n_parameters': int(best_config_row['n_parameters'])
    }
    
    logger.info("\n" + "="*70)
    logger.info("🏆 GRID SEARCH RESULTS")
    logger.info("="*70)
    logger.info(f"\n🥇 BEST CONFIGURATION:")
    logger.info(f"   hidden_channels: {best_config['hidden_channels']}")
    logger.info(f"   num_layers: {best_config['num_layers']}")
    logger.info(f"   lr: {best_config['lr']}")
    logger.info(f"   Val Accuracy: {best_config['val_acc_mean']:.4f} (±{best_config['val_acc_std']:.4f})")
    
    logger.info(f"\n📈 TOP 5 CONFIGURATIONS:")
    for i, row in results_df.head(5).iterrows():
        logger.info(f"   {i+1}. hidden={int(row['hidden_channels'])}, layers={int(row['num_layers'])}, "
                   f"lr={row['lr']} → Acc={row['val_acc_mean']:.4f}")
    
    if save_results:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path / "grid_search_results.csv", index=False, encoding='utf-8-sig')
        
        with open(output_path / "best_config.json", 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Results saved to: {output_path}")
    
    logger.info("="*70)
    
    return best_config