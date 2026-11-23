"""Exact squared-L2 evaluation against independent small numerical oracles."""

import numpy as np
import pytest
from project_module.e_means.evaluation import evaluate_centers, evaluate_groups


def test_exact_mse_matches_hand_computed_oracle() -> None:
    x = np.array([[0, 0], [1, 0], [3, 0], [5, 0]], dtype=np.float32)
    centers = np.array([[0, 0], [4, 0], [100, 0]], dtype=np.float32)

    result = evaluate_centers(x, centers, chunk_rows=2)

    assert result.mse == 0.75
    assert result.n_points == 4
    np.testing.assert_array_equal(result.cluster_sizes, [2, 2, 0])
    assert result.empty_clusters == 1


@pytest.mark.parametrize("chunk_rows", [1, 3, 17, 100])
def test_chunking_preserves_mse_and_empty_clusters(chunk_rows: int) -> None:
    x = np.arange(80, dtype=np.float32).reshape(40, 2)
    centers = np.array([[0, 1], [20, 21], [20, 21], [60, 61]], dtype=np.float32)
    oracle = np.square(x.astype(np.float64)[:, None] - centers.astype(np.float64)).sum(2)
    labels = np.argmin(oracle, axis=1)

    result = evaluate_centers(x, centers, chunk_rows=chunk_rows)

    assert result.mse == float(oracle[np.arange(len(x)), labels].mean())
    np.testing.assert_array_equal(result.cluster_sizes, np.bincount(labels, minlength=4))


def test_nonfinite_centroids_are_rejected() -> None:
    x = np.array([[0, 1]], dtype=np.float32)
    centers = np.array([[np.inf, 0]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        evaluate_centers(x, centers, chunk_rows=2)


def test_origin_is_valid_l2_input() -> None:
    x = np.zeros((1, 2), dtype=np.float32)
    centers = np.ones((1, 2), dtype=np.float32)

    result = evaluate_centers(x, centers, chunk_rows=1)

    assert result.mse == 2.0


def test_empty_group_is_null_not_failed() -> None:
    x = np.array([[0, 0], [2, 0]], dtype=np.float32)
    centers = np.zeros((1, 2), dtype=np.float32)
    groups = np.array([0, 256], dtype=np.int64)

    result = evaluate_groups(x, centers, groups, chunk_rows=1)

    assert result["0"] == {"count": 1, "mse": 0.0, "reason": None}
    assert result["1"] == {"count": 0, "mse": None, "reason": "empty_group"}
    assert result["256"] == {"count": 1, "mse": 4.0, "reason": None}
    assert len(result) == 257


def test_large_common_offset_does_not_cancel_distances() -> None:
    x = np.full((35, 3), 1e6, dtype=np.float32)
    x[:, 0] += 3
    centers = np.full((2, 3), 1e6, dtype=np.float32)
    centers[1, 0] += 4

    result = evaluate_centers(x, centers, chunk_rows=35)

    assert result.mse == 1.0
    np.testing.assert_array_equal(result.cluster_sizes, [0, 35])


@pytest.mark.parametrize("bad_group", [-1, 257])
def test_unknown_group_is_rejected(bad_group: int) -> None:
    x = np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="group"):
        evaluate_groups(x, x.copy(), np.array([bad_group], dtype=np.int64), chunk_rows=1)


@pytest.mark.parametrize("chunk_rows", [0, -1, True])
def test_invalid_chunk_rows_is_rejected(chunk_rows: int) -> None:
    x = np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="chunk_rows"):
        evaluate_centers(x, x.copy(), chunk_rows=chunk_rows)
