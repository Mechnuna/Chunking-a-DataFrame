import random

import pandas as pd
import pytest

from solution import iter_chunks_by_column, sort_by_column_for_chunking
from solution_external import iter_chunks_external


def _make_df(seconds):
    return pd.DataFrame(
        {
            "dt": pd.to_datetime(seconds, unit="s", origin="2023-01-01"),
            "payload": range(len(seconds)),
        }
    )


# ---------- Случай А: не отсортировано, но помещается в память ----------


def test_sort_helper_then_chunk_matches_sorted_baseline():
    seconds = [
        1,
        3,
        2,
        1,
        2,
        2,
    ]  # тот же мультисет значений, что в примере, но вразнобой
    df = _make_df(seconds)
    df_sorted = sort_by_column_for_chunking(df, "dt")

    assert df_sorted["dt"].is_monotonic_increasing
    assert len(df_sorted) == len(df)
    assert sorted(df_sorted["payload"].tolist()) == sorted(df["payload"].tolist())

    sizes = [
        len(c) for c in iter_chunks_by_column(df_sorted, "dt", 2, assume_sorted=True)
    ]
    assert sizes == [2, 3, 1]


def test_sort_helper_is_stable_within_equal_values():
    # при равных значениях порядок исходных строк должен сохраняться
    df = pd.DataFrame({"dt": [2, 1, 1, 2], "payload": ["a", "b", "c", "d"]})
    out = sort_by_column_for_chunking(df, "dt")
    assert out["payload"].tolist() == ["b", "c", "a", "d"]


@pytest.mark.parametrize("seed", range(5))
def test_sort_helper_random_data_roundtrip(seed):
    rng = random.Random(seed)
    seconds = [rng.randint(0, 5) for _ in range(50)]
    df = _make_df(seconds)
    chunks = list(
        iter_chunks_by_column(
            sort_by_column_for_chunking(df, "dt"),
            "dt",
            4,
            assume_sorted=True,
        )
    )
    assert sum(len(c) for c in chunks) == len(df)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["dt"].max() < nxt["dt"].min()


# ---------- Случай Б: не отсортировано и не помещается в память (эмуляция) ----------


def _batches_factory(seconds, batch_size=2):
    def factory():
        df = _make_df(seconds)
        for start in range(0, len(df), batch_size):
            yield df.iloc[start : start + batch_size]

    return factory


def test_external_partitioning_matches_in_memory_baseline(tmp_path):
    seconds = [
        3,
        1,
        1,
        2,
        2,
        2,
    ]  # та же мультисет-группировка, что и в примере, вразнобой
    factory = _batches_factory(seconds, batch_size=2)

    paths = list(iter_chunks_external(factory, "dt", 2, tmp_dir=str(tmp_path)))
    sizes = [len(pd.read_csv(p)) for p in paths]
    assert sizes == [2, 3, 1]

    # чанки не пересекаются по значению dt
    frames = [pd.read_csv(p, parse_dates=["dt"]) for p in paths]
    for prev, nxt in zip(frames, frames[1:]):
        assert prev["dt"].max() < nxt["dt"].min()

    # вместе чанки покрывают все строки без потерь
    total_rows = sum(len(f) for f in frames)
    assert total_rows == len(seconds)


def test_external_partitioning_empty_source(tmp_path):
    factory = _batches_factory([], batch_size=2)
    assert list(iter_chunks_external(factory, "dt", 3, tmp_dir=str(tmp_path))) == []


def test_external_partitioning_large_chunk_returns_single_file(tmp_path):
    seconds = [3, 1, 1, 2, 2, 2]
    factory = _batches_factory(seconds, batch_size=2)
    paths = list(iter_chunks_external(factory, "dt", 100, tmp_dir=str(tmp_path)))
    assert len(paths) == 1
    assert len(pd.read_csv(paths[0])) == len(seconds)
