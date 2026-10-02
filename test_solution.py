import inspect

import pandas as pd
import pytest

from solution import iter_chunks_by_column


def _make_df(seconds):
    return pd.DataFrame({
        "dt": pd.to_datetime(seconds, unit="s", origin="2023-01-01")
    })


EXAMPLE = [1, 1, 2, 2, 2, 3]


def _sizes(df, column, chunk_size, **kwargs):
    return [len(c) for c in iter_chunks_by_column(df, column, chunk_size, **kwargs)]


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_example_small_chunk(chunk_size):
    df = _make_df(EXAMPLE)
    assert _sizes(df, "dt", chunk_size) == [2, 3, 1]


@pytest.mark.parametrize("chunk_size", [3, 4, 5])
def test_example_medium_chunk(chunk_size):
    df = _make_df(EXAMPLE)
    # хвостовой чанк МЕНЬШЕ желаемого размера — так задано в самом
    # примере из условия, это ключевой нюанс задачи, а не баг
    assert _sizes(df, "dt", chunk_size) == [5, 1]


@pytest.mark.parametrize("chunk_size", [6, 100])
def test_example_large_chunk_returns_whole_frame(chunk_size):
    df = _make_df(EXAMPLE)
    chunks = list(iter_chunks_by_column(df, "dt", chunk_size))
    assert len(chunks) == 1
    assert len(chunks[0]) == len(df)


def test_chunks_do_not_overlap_and_cover_whole_frame():
    df = _make_df(EXAMPLE)
    chunks = list(iter_chunks_by_column(df, "dt", 2))
    pd.testing.assert_frame_equal(pd.concat(chunks), df)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["dt"].max() < nxt["dt"].min()


def test_group_with_duplicates_never_split_even_with_chunk_size_1():
    df = _make_df([1, 1, 1, 1, 2])
    assert _sizes(df, "dt", 1) == [4, 1]


def test_empty_dataframe_yields_nothing():
    df = _make_df([])
    assert list(iter_chunks_by_column(df, "dt", 3)) == []


def test_invalid_chunk_size_raises():
    df = _make_df(EXAMPLE)
    with pytest.raises(ValueError):
        list(iter_chunks_by_column(df, "dt", 0))


def test_missing_column_raises():
    df = _make_df(EXAMPLE)
    with pytest.raises(ValueError):
        list(iter_chunks_by_column(df, "not_a_column", 1))


def test_unsorted_column_raises_by_default():
    df = _make_df([2, 1, 1, 3])
    with pytest.raises(ValueError):
        list(iter_chunks_by_column(df, "dt", 1))


def test_unsorted_column_allowed_with_assume_sorted_true():
    df = _make_df([2, 1, 1, 3])
    result = list(iter_chunks_by_column(df, "dt", 1, assume_sorted=True))
    assert sum(len(c) for c in result) == len(df)


def test_is_generator_not_eager_list():
    assert inspect.isgeneratorfunction(iter_chunks_by_column)


def test_task_dataframe_from_prompt():
    dfs = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:00:05", freq="s")
    df = pd.DataFrame({"dt": dfs.repeat(3)})
    chunks = list(iter_chunks_by_column(df, "dt", 4))
    assert sum(len(c) for c in chunks) == len(df)
    for c in chunks[:-1]:
        assert len(c) >= 4
    pd.testing.assert_frame_equal(pd.concat(chunks), df)
