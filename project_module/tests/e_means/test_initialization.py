"""Shared initialization invariants."""

import numpy as np
import pytest
from project_module.e_means.initialization import make_initial_centers


def test_initial_centers_are_distinct_reproducible_train_rows() -> None:
    x = np.array([[0, 1], [2, 3], [0, 1], [4, 5], [6, 7]], dtype=np.float32)
    centers = make_initial_centers(x, 3, 4)
    np.testing.assert_array_equal(centers, make_initial_centers(x, 3, 4))
    assert centers.dtype == np.float32
    assert len({tuple(row) for row in centers}) == 3
    assert all(any(np.array_equal(row, candidate) for candidate in x) for row in centers)


def test_too_few_unique_rows_rejected() -> None:
    x = np.array([[0, 1], [0, 1], [2, 3]], dtype=np.float32)
    with pytest.raises(ValueError, match="unique"):
        make_initial_centers(x, 3, 0)


def test_duplicate_multiplicity_does_not_bias_distinct_row_sampling() -> None:
    x = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    hits = sum(make_initial_centers(x, 1, seed)[0, 0] == 1 for seed in range(2000))
    assert 900 < hits < 1100


def test_signed_zero_rows_are_duplicates() -> None:
    x = np.array([[0.0, 1.0], [-0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="unique"):
        make_initial_centers(x, 3, 0)


def test_existing_seed_selects_same_centers() -> None:
    x = np.array([[1, 0], [0, 1], [1, 0], [4, 5], [6, 7], [4, 5]], dtype=np.float32)
    expected = np.array([[0, 1], [4, 5], [1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(make_initial_centers(x, 3, 4), expected)


def test_initialization_does_not_copy_all_rows_into_python_objects() -> None:
    import tracemalloc

    x = np.arange(4096 * 512, dtype=np.float32).reshape(4096, 512)
    tracemalloc.start()
    try:
        make_initial_centers(x, 3, 0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


def test_finite_validation_does_not_allocate_entire_input_mask() -> None:
    import tracemalloc

    x = np.arange(16384 * 256, dtype=np.float32).reshape(16384, 256)
    tracemalloc.start()
    try:
        make_initial_centers(x, 3, 0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


def test_hash_collision_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from project_module.e_means import initialization

    fake_hash = SimpleNamespace(
        blake2b=lambda *args, **kwargs: SimpleNamespace(digest=lambda: b"x" * 16)
    )
    monkeypatch.setattr(initialization, "hashlib", fake_hash)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="collision"):
        make_initial_centers(x, 1, 0)


def test_nonfinite_input_in_later_chunk_is_rejected() -> None:
    x = np.zeros((4097, 256), dtype=np.float32)
    x[-1, -1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_initial_centers(x, 1, 0)
