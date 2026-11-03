"""
ml_fallback.py — ML/GNN классификация для нераспознанных слоёв.

Когда rule-based парсер не может определить класс слоя
(имя '0', технические коды типа 'АР2-6_1-03') —
этот модуль пробует классифицировать каждую entity
через обученные ML и GNN модели.

Результат добавляется в ParsedFloorPlan наравне с rule-based данными.
Это позволяет сравнить три подхода:
  1. Rule-based  — по именам слоёв (точный, но требует разметки)
  2. ML          — по геометрическим признакам каждой entity
  3. GNN         — по графу связей между entity слоя

Для диплома это именно то что нужно:
  - Rule-based даёт эталон
  - ML и GNN пытаются восстановить то же самое без имён слоёв
  - Сравниваем результаты и делаем вывод о применимости
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.layer_parser import (
    ParsedFloorPlan, WallGeometry, OpeningGeometry, StairGeometry,
    extract_walls, extract_openings_from_lines, extract_openings_from_inserts,
    extract_stairs, GEOMETRY_TYPES, SKIP_TYPES
)
from src.ml_features import extract_entity_features, extract_file_context
from src.config import PROCESSED_DIR
from src.utils import setup_logger

logger = setup_logger("MLFallback")

# Маппинг классов ML/GNN → методы обработки геометрии
CLASS_TO_GEOMETRY = {
    'Стены':          'wall_outer',
    'Перегородки':    'wall_inner',
    'Базовая дверь':  'door',
    'Двери':          'door',
    'Базовое окно':   'window',
    'Окна':           'window',
    'Лестницы':       'stair',
    'Перекрытия':     'slab',
    'Потолки':        'ceiling',
}


@dataclass
class FallbackResult:
    """Результат ML/GNN классификации одного слоя."""
    layer_name:    str
    method:        str          # 'ML' или 'GNN'
    predicted_class: str
    confidence:    float
    entity_count:  int
    class_votes:   Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ML классификация
# ---------------------------------------------------------------------------

def classify_layer_ml(entities: list, layer_name: str,
                       msp, model_path: Path = None) -> Optional[FallbackResult]:
    """
    Классифицирует слой через ML (Random Forest).

    Для каждой entity извлекаем геометрические признаки,
    получаем предсказание класса, берём большинство голосов по слою.
    """
    if model_path is None:
        model_path = PROCESSED_DIR / "models" / "ml_model.pkl"

    if not model_path.exists():
        logger.warning(f"ML модель не найдена: {model_path}")
        return None

    try:
        from src.models.ml_model import MLClassifier
        model = MLClassifier.load(model_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки ML модели: {e}")
        return None

    # Контекст файла
    class _FakeMsp:
        def __iter__(self): return iter(entities)
    file_ctx = extract_file_context(_FakeMsp())

    votes = []
    confidences = []

    for e in entities:
        if e.dxftype() in SKIP_TYPES:
            continue
        try:
            feats = extract_entity_features(e, file_context=file_ctx)
            if not feats:
                continue

            X = pd.DataFrame([feats])
            for col in model.feature_columns:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[model.feature_columns]
            X_scaled = model.scaler.transform(X.values)

            pred   = model.predict(X_scaled)[0]
            proba  = model.predict_proba(X_scaled)[0]
            conf   = float(max(proba))

            votes.append(pred)
            confidences.append(conf)
        except Exception:
            continue

    if not votes:
        return None

    # Большинство голосов
    counter      = Counter(votes)
    top_class    = counter.most_common(1)[0][0]
    avg_conf     = float(np.mean(confidences))

    logger.info(
        f"  ML '{layer_name}': {len(votes)} votes → "
        f"'{top_class}' (conf={avg_conf:.3f}) | {dict(counter.most_common(3))}"
    )

    return FallbackResult(
        layer_name=layer_name,
        method='ML',
        predicted_class=top_class,
        confidence=avg_conf,
        entity_count=len(votes),
        class_votes=dict(counter),
    )


# ---------------------------------------------------------------------------
# GNN классификация
# ---------------------------------------------------------------------------

def classify_layer_gnn(entities: list, layer_name: str,
                        model_path: Path = None,
                        idx_to_class: dict = None) -> Optional[FallbackResult]:
    """
    Классифицирует слой через GNN.

    Строит граф из всех entity слоя и получает один класс на весь слой.
    """
    if model_path is None:
        model_path = PROCESSED_DIR / "models" / "gnn_model.pt"

    if not model_path.exists():
        logger.warning(f"GNN модель не найдена: {model_path}")
        return None

    # Загружаем маппинг классов если не передан
    if idx_to_class is None:
        csv_path = PROCESSED_DIR / "ml" / "train_features.csv"
        if not csv_path.exists():
            return None
        df_train     = pd.read_csv(csv_path)
        all_classes  = sorted(df_train['label'].unique())
        idx_to_class = {i: c for i, c in enumerate(all_classes)}

    try:
        from src.models.gnn_model import GNNClassifier
        # build_layer_graph импортируется локально — функция определена
        # либо в src.gnn_builder либо в src.predict_layers
        try:
            from src.gnn_builder import build_layer_graph
        except ImportError:
            from src.predict_layers import build_layer_graph
        gnn = GNNClassifier.load(model_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки GNN: {e}")
        return None

    geom_entities = [e for e in entities
                     if e.dxftype() in GEOMETRY_TYPES
                     and e.dxftype() not in SKIP_TYPES]
    if len(geom_entities) < 2:
        return None

    try:
        # Контекст файла для признаков узлов
        class _FakeMsp:
            def __iter__(self): return iter(geom_entities)
        file_ctx = extract_file_context(_FakeMsp())

        graph = build_layer_graph(geom_entities, file_ctx)
        if graph is None:
            return None

        graph = graph.to(gnn.device)
        class_idx, proba = gnn.predict_graph(graph)
        pred_class  = idx_to_class.get(class_idx, f"unknown_{class_idx}")
        confidence  = float(max(proba))

        # Топ-3 вероятности для лога
        top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
        top3_str = {idx_to_class.get(i, str(i)): round(float(p), 3) for i, p in top3}

        logger.info(
            f"  GNN '{layer_name}': {len(geom_entities)} entity → "
            f"'{pred_class}' (conf={confidence:.3f}) | {top3_str}"
        )

        return FallbackResult(
            layer_name=layer_name,
            method='GNN',
            predicted_class=pred_class,
            confidence=confidence,
            entity_count=len(geom_entities),
            class_votes={pred_class: len(geom_entities)},
        )
    except Exception as e:
        logger.error(f"GNN ошибка для '{layer_name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Применение результата к ParsedFloorPlan
# ---------------------------------------------------------------------------

def apply_fallback_to_plan(plan: ParsedFloorPlan,
                            result: FallbackResult,
                            layer_entities: dict,
                            doc) -> int:
    """
    Добавляет геометрию из нераспознанного слоя в plan
    на основании предсказанного класса.

    Возвращает количество добавленных элементов.
    """
    layer_name = result.layer_name
    pred_cls   = result.predicted_class
    entities   = layer_entities.get(layer_name, [])
    geom       = [e for e in entities
                  if e.dxftype() in GEOMETRY_TYPES
                  and e.dxftype() not in SKIP_TYPES]

    if not geom:
        return 0

    geo_type = CLASS_TO_GEOMETRY.get(pred_cls)
    added    = 0

    if geo_type == 'wall_outer':
        walls = extract_walls(geom, layer_name, is_outer=True)
        plan.walls.extend(walls)
        added = len(walls)

    elif geo_type == 'wall_inner':
        walls = extract_walls(geom, layer_name, is_outer=False)
        plan.partitions.extend(walls)
        added = len(walls)

    elif geo_type == 'door':
        inserts = [e for e in geom if e.dxftype() == 'INSERT']
        lines   = [e for e in geom if e.dxftype() != 'INSERT']
        ops = (extract_openings_from_inserts(inserts, layer_name, doc, 'door') +
               extract_openings_from_lines(lines, layer_name, 'door'))
        plan.doors.extend(ops)
        added = len(ops)

    elif geo_type == 'window':
        inserts = [e for e in geom if e.dxftype() == 'INSERT']
        lines   = [e for e in geom if e.dxftype() != 'INSERT']
        ops = (extract_openings_from_inserts(inserts, layer_name, doc, 'window') +
               extract_openings_from_lines(lines, layer_name, 'window'))
        plan.windows.extend(ops)
        added = len(ops)

    elif geo_type == 'stair':
        stairs = extract_stairs(geom, layer_name)
        plan.stairs.extend(stairs)
        added = len(stairs)

    if added > 0:
        logger.info(
            f"  → Добавлено {added} элементов класса '{pred_cls}' "
            f"из слоя '{layer_name}' [{result.method}]"
        )

    return added


# ---------------------------------------------------------------------------
# Главная функция — интеграция в parse_floor_plan
# ---------------------------------------------------------------------------

def enhance_plan_with_ml_gnn(plan: ParsedFloorPlan,
                              dxf_path: Path,
                              use_ml: bool = True,
                              use_gnn: bool = True,
                              ml_model_path: Path = None,
                              gnn_model_path: Path = None) -> dict:
    """
    Дополняет ParsedFloorPlan классификацией нераспознанных слоёв
    через ML и/или GNN.

    Возвращает словарь с результатами для отчёта:
    {
        'unknown_layers':  список нераспознанных слоёв,
        'ml_results':      [FallbackResult, ...],
        'gnn_results':     [FallbackResult, ...],
        'ml_added':        количество добавленных элементов через ML,
        'gnn_added':       количество добавленных элементов через GNN,
    }
    """
    import ezdxf
    from collections import defaultdict

    if not plan.unknown_layers:
        logger.info("Нет нераспознанных слоёв — ML/GNN fallback не нужен")
        return {'unknown_layers': [], 'ml_results': [], 'gnn_results': [],
                'ml_added': 0, 'gnn_added': 0}

    logger.info(f"\n{'='*60}")
    logger.info(f"ML/GNN Fallback для {Path(dxf_path).name}")
    logger.info(f"Нераспознанных слоёв: {len(plan.unknown_layers)}")
    logger.info(f"{'='*60}")

    # Читаем DXF заново чтобы получить entity по слоям
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        logger.error(f"Ошибка чтения DXF: {e}")
        return {}

    msp = doc.modelspace()
    layer_entities = defaultdict(list)
    for e in msp:
        layer_entities[e.dxf.layer].append(e)

    # Загружаем idx_to_class один раз для GNN
    idx_to_class = None
    if use_gnn:
        csv_path = PROCESSED_DIR / "ml" / "train_features.csv"
        if csv_path.exists():
            df_train     = pd.read_csv(csv_path)
            all_classes  = sorted(df_train['label'].unique())
            idx_to_class = {i: c for i, c in enumerate(all_classes)}

    # Загружаем GNN Plans модель — обучена на реальных планах
    # Более релевантна чем стандартная GNN для нераспознанных слоёв
    plans_model, plans_idx_to_class = None, None
    if use_gnn:
        try:
            from src.models.train_gnn_from_plans import load_plans_model
            plans_model, plans_idx_to_class = load_plans_model()
            if plans_model:
                logger.info("GNN Plans модель загружена — используем для fallback")
        except Exception as e:
            logger.debug(f"GNN Plans недоступна: {e}")

    ml_results  = []
    gnn_results = []
    ml_added    = 0
    gnn_added   = 0

    for layer_name in plan.unknown_layers:
        entities = layer_entities.get(layer_name, [])
        if not entities:
            continue

        logger.info(f"\nСлой '{layer_name}' ({len(entities)} entity):")

        # --- ML ---
        if use_ml:
            ml_res = classify_layer_ml(
                entities, layer_name, msp, ml_model_path
            )
            if ml_res:
                ml_results.append(ml_res)
                # Применяем к плану (создаём копию чтобы не портить original)
                added = apply_fallback_to_plan(plan, ml_res, layer_entities, doc)
                ml_added += added

        # --- GNN (Plans модель приоритетнее стандартной) ---
        if use_gnn:
            # Пробуем Plans GNN (обучена на реальных планах)
            if plans_model and plans_idx_to_class:
                try:
                    from src.models.train_gnn_from_plans import classify_layer_with_plans_gnn
                    class _FakeMsp:
                        def __iter__(self): return iter(entities)
                    from src.ml_features import extract_file_context
                    fctx = extract_file_context(_FakeMsp())
                    pred_cls, conf = classify_layer_with_plans_gnn(
                        entities, layer_name, fctx, plans_model, plans_idx_to_class
                    )
                    if pred_cls:
                        gnn_res = FallbackResult(
                            layer_name=layer_name,
                            method='GNN_Plans',
                            predicted_class=pred_cls,
                            confidence=conf,
                            entity_count=len(entities),
                            class_votes={pred_cls: len(entities)},
                        )
                        gnn_results.append(gnn_res)
                        logger.info(
                            f"  GNN_Plans '{layer_name}': "
                            f"'{pred_cls}' conf={conf:.3f}"
                        )
                except Exception as e:
                    logger.debug(f"GNN_Plans инференс: {e}")
                    # Fallback на стандартную GNN
                    gnn_res = classify_layer_gnn(
                        entities, layer_name, gnn_model_path, idx_to_class
                    )
                    if gnn_res:
                        gnn_results.append(gnn_res)
            else:
                # Стандартная GNN
                gnn_res = classify_layer_gnn(
                    entities, layer_name, gnn_model_path, idx_to_class
                )
                if gnn_res:
                    gnn_results.append(gnn_res)
                # GNN не добавляем в plan напрямую — только для сравнения
                # (ML уже добавил, GNN показываем как альтернативу)

    logger.info(f"\n{'='*60}")
    logger.info(f"Итог ML/GNN Fallback:")
    logger.info(f"  ML  предсказал {len(ml_results)} слоёв, добавил {ml_added} элементов")
    logger.info(f"  GNN предсказал {len(gnn_results)} слоёв")

    # Сравниваем ML vs GNN
    if ml_results and gnn_results:
        logger.info(f"\nСравнение ML vs GNN:")
        for ml_r in ml_results:
            gnn_r = next((g for g in gnn_results
                          if g.layer_name == ml_r.layer_name), None)
            if gnn_r:
                match = "✓ СОВПАЛО" if ml_r.predicted_class == gnn_r.predicted_class else "✗ РАСХОДЯТСЯ"
                logger.info(
                    f"  '{ml_r.layer_name}': "
                    f"ML='{ml_r.predicted_class}' "
                    f"GNN='{gnn_r.predicted_class}' {match}"
                )

    logger.info(f"{'='*60}")

    return {
        'unknown_layers': list(plan.unknown_layers.keys()),
        'ml_results':     ml_results,
        'gnn_results':    gnn_results,
        'ml_added':       ml_added,
        'gnn_added':      0,
    }
