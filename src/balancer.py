"""
balancer.py — единая функция балансировки для ML и GNN.

Используется в train_ml.py и train_gnn.py чтобы гарантировать
одинаковое количество экземпляров в каждом классе.

Что значит "сбалансировано":
    Класс А: 100 строк
    Класс Б: 100 строк  
    Класс В: 100 строк
    → каждый класс ровно N строк, N одинаковый для всех.

Стратегия:
    - Находим TARGET = количество строк в наименьшем классе
    - Классы с бОльшим числом строк — обрезаем до TARGET
    - Классы с меньшим числом строк — не бывает (TARGET = минимум)
    - Результат: все классы ровно TARGET строк
"""

import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger("Balancer")


def balance_dataframe(df: pd.DataFrame,
                      label_col: str = 'label',
                      target: int = None,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Балансирует DataFrame так чтобы в каждом классе было ровно target строк.

    Параметры:
        df           — исходный DataFrame с колонкой label
        label_col    — название колонки с метками классов
        target       — сколько строк оставить на класс.
                       None = автоматически (минимум среди всех классов).
        random_state — seed для воспроизводимости

    Возвращает сбалансированный DataFrame.
    """
    counts = df[label_col].value_counts()

    if target is None:
        target = int(counts.min())

    logger.info(f"Балансировка: target = {target} строк на класс")
    logger.info(f"Классов: {len(counts)}")

    parts = []
    for label, group in df.groupby(label_col):
        n = len(group)
        if n >= target:
            sampled = group.sample(n=target, random_state=random_state)
        else:
            # Класс меньше target — берём всё что есть
            # (не должно случаться если target=min, но на всякий случай)
            sampled = group
        parts.append(sampled)
        logger.info(f"  {label:<30} {n:>4} → {len(sampled):>4} строк")

    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Итого: {len(result)} строк, {result[label_col].nunique()} классов")
    # Финальная проверка
    final_counts = result[label_col].value_counts()
    assert final_counts.max() == final_counts.min(), \
        f"Балансировка не сработала: {dict(final_counts)}"
    logger.info(f"✓ Проверка: все классы по {final_counts.iloc[0]} строк")

    return result
