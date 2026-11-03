"""
gnn_model.py

Фикс проблемы 1: При загрузке модели не совпадала архитектура — в checkpoint
был num_layers=3 (есть convs.0), а при загрузке создавался num_layers=2 (convs пустой).
Решение: сохраняем num_layers в checkpoint и восстанавливаем при загрузке.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool
)
from src.utils import setup_logger

logger = setup_logger("GNNModel")


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels)
             for _ in range(num_layers - 2)]
        )
        self.conv_last = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=0.3, training=self.training)
        x = self.conv_last(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.convs = torch.nn.ModuleList(
            [GATConv(hidden_channels, hidden_channels // heads, heads=heads)
             for _ in range(num_layers - 2)]
        )
        self.conv_last = GATConv(hidden_channels, hidden_channels, heads=1)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(F.elu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        for conv in self.convs:
            x = F.dropout(F.elu(conv(x, edge_index)), p=0.3, training=self.training)
        x = self.conv_last(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels)
             for _ in range(num_layers - 2)]
        )
        self.conv_last = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=0.3, training=self.training)
        x = self.conv_last(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        def mlp(in_c, out_c):
            return torch.nn.Sequential(
                torch.nn.Linear(in_c, out_c),
                torch.nn.ReLU(),
                torch.nn.Linear(out_c, out_c),
            )

        self.conv1 = GINConv(mlp(in_channels, hidden_channels))
        self.convs = torch.nn.ModuleList(
            [GINConv(mlp(hidden_channels, hidden_channels))
             for _ in range(num_layers - 2)]
        )
        self.conv_last = GINConv(mlp(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=0.3, training=self.training)
        x = self.conv_last(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


def get_gnn_model(model_type, in_channels, hidden_channels, out_channels, num_layers=2):
    models = {
        'gcn':       SimpleGCN,
        'gat':       GAT,
        'graphsage': GraphSAGE,
        'gin':       GIN,
    }
    cls = models.get(model_type)
    if cls is None:
        logger.warning(f"Unknown model type '{model_type}', using GCN")
        cls = SimpleGCN
    return cls(in_channels, hidden_channels, out_channels, num_layers)


class GNNClassifier:
    def __init__(self, in_channels, num_classes, hidden_channels=64,
                 num_layers=2, lr=0.01, model_type='gcn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type     = model_type
        self.in_channels    = in_channels
        self.num_classes    = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers     = num_layers

        logger.info(f"Device: {self.device} | Model: {model_type.upper()} | "
                    f"layers={num_layers} hidden={hidden_channels}")

        self.model = get_gnn_model(
            model_type, in_channels, hidden_channels, num_classes, num_layers
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_history = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out  = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pred    = out.argmax(dim=1)
            correct += pred.eq(batch.y).sum().item()
            total   += batch.y.size(0)
        return {'loss': total_loss / len(loader), 'acc': correct / total}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            out  = self.model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y).sum().item()
            total   += batch.y.size(0)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
        return {'acc': correct / total, 'preds': all_preds, 'labels': all_labels}

    @torch.no_grad()
    def predict_graph(self, graph_data):
        """Предсказание для одного графа — возвращает (class_idx, proba_array)."""
        self.model.eval()
        graph_data = graph_data.to(self.device)
        if graph_data.batch is None:
            graph_data.batch = torch.zeros(
                graph_data.x.size(0), dtype=torch.long, device=self.device
            )
        out   = self.model(graph_data.x, graph_data.edge_index, graph_data.batch)
        proba = torch.softmax(out, dim=1).cpu().numpy()[0]
        return int(proba.argmax()), proba

    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state':    self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history':         self.training_history,
            'model_type':      self.model_type,
            # Сохраняем параметры архитектуры — без них load воссоздаёт неверную модель
            'in_channels':     self.in_channels,
            'num_classes':     self.num_classes,
            'hidden_channels': self.hidden_channels,
            'num_layers':      self.num_layers,
        }, path)
        logger.info(f"GNN saved: {path}")

    @classmethod
    def load(cls, path, in_channels=None, num_classes=None):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Берём параметры архитектуры из checkpoint — не угадываем снаружи
        model_type      = checkpoint.get('model_type',      'gcn')
        in_ch           = in_channels  or checkpoint.get('in_channels',     7)
        n_cls           = num_classes  or checkpoint.get('num_classes',     6)
        hidden_channels = checkpoint.get('hidden_channels', 64)
        num_layers      = checkpoint.get('num_layers',      2)

        instance = cls(
            in_channels=in_ch,
            num_classes=n_cls,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            model_type=model_type,
        )
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state'])
        instance.training_history = checkpoint.get('history', [])
        instance.model.eval()
        logger.info(f"GNN loaded: {path} | {model_type} layers={num_layers} hidden={hidden_channels}")
        return instance
