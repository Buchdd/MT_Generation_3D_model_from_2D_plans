import math
import torch
import numpy as np
from torch_geometric.data import Data
from src.dxf_reader import load_dxf_document, get_entities
from src.config import TARGET_ENTITY_TYPES
from src.ml_features import extract_entity_features, extract_file_context
from src.utils import setup_logger

logger = setup_logger("GNNBuilder")

ENTITY_TYPE_MAP = {t: i for i, t in enumerate(TARGET_ENTITY_TYPES)}

# Признаки которые используются как фичи узлов графа.
# Должны совпадать с числовыми признаками в ML CSV — для честного сравнения.
NODE_FEATURE_KEYS = [
    'length', 'area', 'center_x', 'center_y', 'center_z',
    'num_points', 'bbox_width', 'bbox_height', 'bbox_depth',
    'angle', 'perimeter_area_ratio',
    'log_length', 'log_area', 'length_to_width', 'bbox_aspect',
    # Контекстные признаки файла
    'ctx_entity_count', 'ctx_ratio_line', 'ctx_ratio_polyline',
    'ctx_ratio_arc', 'ctx_ratio_circle',
    'ctx_mean_length', 'ctx_max_length', 'ctx_cv_length',
    'ctx_file_width', 'ctx_file_height', 'ctx_file_aspect',
    'ctx_entity_density', 'ctx_layer_count',
]

# Размерность признаков узла = one-hot типа (5) + числовые фичи
NODE_FEATURE_DIM = len(TARGET_ENTITY_TYPES) + len(NODE_FEATURE_KEYS)

# Порог пространственной близости для построения рёбер (в единицах чертежа).
# Два объекта связываются если расстояние между их центрами <= порога.
# None = только рёбра по слою (старое поведение).
SPATIAL_EDGE_THRESHOLD = None  # задаётся в config, пока авто

def _entity_center(entity) -> tuple:
    """Возвращает (cx, cy) центра entity."""
    try:
        t = entity.dxftype()
        if t == 'LINE':
            s, e = entity.dxf.start, entity.dxf.end
            return ((s[0] + e[0]) / 2, (s[1] + e[1]) / 2)
        elif t in ('CIRCLE', 'ARC'):
            c = entity.dxf.center
            return (c[0], c[1])
        elif t in ('POLYLINE', 'LWPOLYLINE'):
            pts = list(entity.points())
            if pts:
                return (sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts))
    except Exception:
        pass
    return (0.0, 0.0)

def build_graph_from_dxf(file_path, spatial_threshold=None):
    """
    Преобразует DXF-файл в объект графа PyTorch Geometric (Data).

    Каждая entity — узел графа.
    Рёбра строятся по двум критериям:
      1. Общий слой (entity на одном слое связаны).
      2. Пространственная близость (если spatial_threshold задан).

    Признаки узлов — те же геометрические фичи что в ML + one-hot типа.
    """
    doc, msp = load_dxf_document(file_path)
    if doc is None:
        return None

    entities = get_entities(msp, TARGET_ENTITY_TYPES)
    if not entities:
        return None

    num_nodes = len(entities)

    # --- Контекст файла (одинаков для всех узлов) ---
    file_ctx = extract_file_context(msp)

    # --- Признаки узлов ---
    node_features = []
    centers = []

    for e in entities:
        # One-hot тип entity
        type_idx    = ENTITY_TYPE_MAP.get(e.dxftype(), -1)
        type_one_hot = [0] * len(TARGET_ENTITY_TYPES)
        if type_idx >= 0:
            type_one_hot[type_idx] = 1

        # Геометрические признаки (те же что в ML)
        feats = extract_entity_features(e, file_context=file_ctx)

        numeric = [float(feats.get(k, 0.0) or 0.0) for k in NODE_FEATURE_KEYS]

        # Заменяем NaN/Inf на 0
        numeric = [0.0 if (math.isnan(v) or


math.isinf(v)) else v for v in numeric]

        node_features.append(type_one_hot + numeric)
        centers.append(_entity_center(e))

    x = torch.tensor(node_features, dtype=torch.float)

    # --- Рёбра ---
    edge_set = set()

    # 1. По слою
    layer_groups: dict = {}
    for i, e in enumerate(entities):
        layer = e.dxf.layer
        layer_groups.setdefault(layer, []).append(i)

    for nodes in layer_groups.values():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[b]
                edge_set.add((i, j))
                edge_set.add((j, i))

    # 2. По пространственной близости
    threshold = spatial_threshold
    if threshold is None and num_nodes > 1:
        # Авто-порог: медиана попарных расстояний (ограничено 200 парами)
        sample = min(num_nodes, 20)
        dists = []
        for i in range(sample):
            for j in range(i + 1, sample):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dists.append(math.hypot(dx, dy))
        if dists:
            threshold = float(np.median(dists)) * 1.5

    if threshold and threshold > 0:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if math.hypot(dx, dy) <= threshold:
                    edge_set.add((i, j))
                    edge_set.add((j, i))

    if edge_set:
        edges = list(edge_set)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Граф без рёбер — создаём самопетлю чтобы GCN не падал
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.file_name = file_path.name

    return data

# ---------------------------------------------------------------------------
# Графовая аугментация — увеличение выборки для GNN
# ---------------------------------------------------------------------------

def augment_graph(data: Data, strategy: str = 'random') -> Data:
    """
    Аугментация одного графа для увеличения обучающей выборки GNN.

    Стратегии:
      'drop_edges'   — случайно удаляем 20% рёбер (имитирует пропуски в чертеже)
      'add_noise'    — добавляем гауссов шум к признакам узлов
      'drop_nodes'   — случайно удаляем 10% узлов (имитирует неполный чертёж)
      'random'       — случайно выбираем одну из стратегий

    Возвращает новый объект Data с теми же метками y и class_name.
    """
    import random as _random
    import copy

    strategies = ['drop_edges', 'add_noise', 'drop_nodes']
    if strategy == 'random':
        strategy = _random.choice(strategies)

    data = copy.deepcopy(data)

    if strategy == 'drop_edges':
        # Удаляем случайные 20% рёбер
        n_edges = data.edge_index.size(1)
        if n_edges > 2:
            keep_mask = torch.rand(n_edges) > 0.20
            # Гарантируем хотя бы одно ребро
            if keep_mask.sum() == 0:
                keep_mask[0] = True
            data.edge_index = data.edge_index[:, keep_mask]

    elif strategy == 'add_noise':
        # Гауссов шум к числовым признакам (std=0.05 — мягкий шум)
        noise = torch.randn_like(data.x) * 0.05
        data.x = data.x + noise

    elif strategy == 'drop_nodes':
        n_nodes = data.x.size(0)
        if n_nodes > 3:
            n_drop  = max(1, int(n_nodes * 0.10))
            drop_idx = torch.randperm(n_nodes)[:n_drop]
            keep_mask = torch.ones(n_nodes, dtype=torch.bool)
            keep_mask[drop_idx] = False
            keep_idx = keep_mask.nonzero(as_tuple=True)[0]

            # Переиндексируем узлы
            new_idx = torch.full((n_nodes,), -1, dtype=torch.long)
            new_idx[keep_idx] = torch.arange(len(keep_idx))

            data.x = data.x[keep_mask]
            ei = data.edge_index
            valid = (new_idx[ei[0]] >= 0) & (new_idx[ei[1]] >= 0)
            if valid.sum() > 0:
                data.edge_index = torch.stack([
                    new_idx[ei[0][valid]],
                    new_idx[ei[1][valid]],
                ])
            else:
                data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    return data

def augment_dataset(graphs: list, factor: int = 3) -> list:
    """
    Аугментирует список графов в factor раз.
    Оригиналы сохраняются, добавляются аугментированные копии.

    factor=3 означает: итоговый размер = оригинал * (1 + factor)
    """
    augmented = list(graphs)
    strategies = ['drop_edges', 'add_noise', 'drop_nodes']

    for g in graphs:
        for i in range(factor):
            strategy = strategies[i % len(strategies)]
            aug = augment_graph(g, strategy=strategy)
            augmented.append(aug)

    return augmented

def build_layer_graph(entities: list, file_ctx: dict = None):
    """
    Строит граф PyTorch Geometric из списка entity одного слоя DXF.
    Используется в ml_fallback.py для GNN классификации нераспознанных слоёв.

    Отличие от build_graph_from_dxf:
      - принимает готовый список entity (не читает файл)
      - file_ctx передаётся снаружи (уже вычислен)
      - рёбра строятся как полный граф внутри слоя
    """
    import math as _math
    import torch as _torch
    from torch_geometric.data import Data

    if not entities:
        return None

    node_features = []

    for e in entities:
        type_idx     = ENTITY_TYPE_MAP.get(e.dxftype(), -1)
        type_one_hot = [0] * len(TARGET_ENTITY_TYPES)
        if type_idx >= 0:
            type_one_hot[type_idx] = 1

        feats   = extract_entity_features(e, file_context=file_ctx)
        numeric = [float(feats.get(k, 0.0) or 0.0) for k in NODE_FEATURE_KEYS]
        numeric = [0.0 if (_math.isnan(v) or _math.isinf(v)) else v for v in numeric]

        node_features.append(type_one_hot + numeric)

    n = len(node_features)
    x = _torch.tensor(node_features, dtype=_torch.float)

    # Полный граф — все entity слоя связаны между собой
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append([i, j])
            edges.append([j, i])

    if edges:
        edge_index = _torch.tensor(edges, dtype=_torch.long).t().contiguous()
    else:
        edge_index = _torch.tensor([[0], [0]], dtype=_torch.long)

    data       = Data(x=x, edge_index=edge_index)
    data.batch = _torch.zeros(n, dtype=_torch.long)
    return data