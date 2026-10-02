"""
  * группа строк с одинаковым значением колонки не может быть разорвана
    между двумя чанками (даты между чанками не пересекаются);
  * размер чанка должен быть >= chunk_size, ЗА ИСКЛЮЧЕНИЕМ последнего
    чанка, если данных для его "добора" больше не осталось (это прямое
    следствие первого условия и явно проверяется в тестах — см. пример
    в задании: диапазон 3-5 даёт хвостовой чанк размером 1);
  * память расходуется экономно: функция — генератор, чанки не
    накапливаются в списке и не удерживаются в памяти одновременно;
    группировка выполняется одним линейным проходом по уже
    отсортированной колонке (без построения полной карты групп, как
    это делает pandas.DataFrame.groupby).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def iter_chunks_by_column(
    df: pd.DataFrame,
    column: str,
    chunk_size: int,
    *,
    assume_sorted: bool = False,
) -> Iterator[pd.DataFrame]:
    """
    Лениво возвращает срезы df, разбитые по значениям df[column].

    :param df: исходный датафрейм
    :param column: колонка, по которой группируем (например, "dt")
    :param chunk_size: желаемый минимальный размер чанка (int >= 1)
    :param assume_sorted: если False (по умолчанию), функция проверит,
        что df[column] отсортирован по неубыванию, и бросит ValueError,
        если это не так — иначе строки с одинаковым значением, но
        расположенные не подряд, будут ошибочно разбиты на разные группы.
        Проверка дешёвая по памяти (один проход, без копий).
    :yields: последовательные срезы df (pd.DataFrame), не пересекающиеся
        по значениям column, суммарно покрывающие весь df.
    :raises ValueError: если chunk_size < 1, df[column] не отсортирован
        (при assume_sorted=False), или колонка отсутствует.
    """
    if column not in df.columns:
        raise ValueError(f"Колонка {column!r} отсутствует в датафрейме")
    if chunk_size < 1:
        raise ValueError("chunk_size должен быть >= 1")

    n = len(df)
    if n == 0:
        return

    series = df[column]
    if not assume_sorted and not series.is_monotonic_increasing:
        raise ValueError(
            f"Колонка {column!r} должна быть отсортирована по неубыванию"
        )

    values = series.to_numpy()

    chunk_start = 0   # начало текущего накапливаемого чанка
    group_start = 0    # начало текущей "неразрывной" группы одинаковых значений
    for i in range(1, n + 1):
        if i == n or values[i] != values[group_start]:
            group_end = i
            if group_end - chunk_start >= chunk_size:
                yield df.iloc[chunk_start:group_end]
                chunk_start = group_end
            group_start = i

    if chunk_start < n:
        yield df.iloc[chunk_start:n]


def sort_by_column_for_chunking(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Подготовка df к iter_chunks_by_column, когда порядок не гарантирован,
    но датафрейм целиком помещается в память.

    Использование:
        df_sorted = sort_by_column_for_chunking(df, "dt")
        for chunk in iter_chunks_by_column(df_sorted, "dt", size, assume_sorted=True):
            ...
    """
    order = np.argsort(df[column].to_numpy(), kind="mergesort")  # стабильная
    return df.iloc[order].reset_index(drop=True)
