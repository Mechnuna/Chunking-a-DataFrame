"""
Идея:
  Проход 1 — считаем, сколько строк
      приходится на каждое уникальное значение column. Память здесь
      ~ O(число уникальных значений column), а НЕ O(число строк) — для
      временных рядов с повторами это обычно на порядки меньше исходных
      данных.
  Между проходами — на этих маленьких агрегированных counts тем же жадным алгоритмом,
  что и в solution.py,
      определяем, какому результирующему чанку принадлежит каждое
      уникальное значение.
  Проход 2 — раскладываем строки по
      файлам на диске согласно определённому chunk_id. Память здесь
      ~ O(число одновременно открытых файлов), а НЕ O(число строк).

После второго прохода чанки лежат на диске готовыми файлами и их можно
читать по одному, уже не заботясь о памяти.
"""
from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import pandas as pd


def iter_chunks_external(
    batches_factory: Callable[[], Iterable[pd.DataFrame]],
    column: str,
    chunk_size: int,
    tmp_dir: str | None = None,
) -> Iterator[Path]:
    """
    :param column: колонка для группировки/разбиения
    :param chunk_size: желаемый минимальный размер чанка (как и раньше,
        последний чанк может оказаться меньше — это неизбежное следствие
        того, что группу с одинаковым значением нельзя разрывать)
    :param tmp_dir: каталог для временных файлов чанков
    :yields: пути к файлам на диске — по одному на выходной чанк, уже
        не пересекающимся по значениям column
    """
    if chunk_size < 1:
        raise ValueError("chunk_size должен быть >= 1")

    tmp_path = Path(tmp_dir or tempfile.mkdtemp())

    # --- Проход 1: считаем размер группы для каждого уникального значения ---
    counts: Counter = Counter()
    for batch in batches_factory():
        counts.update(batch[column].value_counts().to_dict())

    if not counts:
        return

    # --- Жадно определяем границы чанков на counts (их << чем строк) ---
    value_to_chunk: dict = {}
    chunk_id = 0
    acc = 0
    for value in sorted(counts):
        value_to_chunk[value] = chunk_id
        acc += counts[value]
        if acc >= chunk_size:
            chunk_id += 1
            acc = 0
    # если последняя группа не добрала chunk_size, она остаётся приклеенной
    # к последнему chunk_id — отсюда возможный меньший хвостовой чанк

    # --- Проход 2: раскладываем строки по файлам согласно value_to_chunk ---
    chunk_paths: dict[int, Path] = {}
    for batch in batches_factory():
        batch = batch.copy()
        batch["_chunk_id"] = batch[column].map(value_to_chunk)
        for cid, part in batch.groupby("_chunk_id"):
            part = part.drop(columns="_chunk_id")
            path = chunk_paths.get(cid)
            if path is None:
                path = tmp_path / f"chunk_{cid:06d}.csv"
                chunk_paths[cid] = path
                part.to_csv(path, index=False, mode="w")
            else:
                part.to_csv(path, index=False, mode="a", header=False)

    for cid in sorted(chunk_paths):
        yield chunk_paths[cid]
