"""Independent exact squared-L2 evaluation with bounded-memory numeric checks."""

import logging
from collections.abc import Iterator, Mapping

import faiss
import numpy as np
import numpy.typing as npt

from project_module.e_means.contracts import Evaluation, JSONValue

_GROUP_COUNT = 257  # Synthetic component IDs 0..255 and outlier ID 256.
_LOG = logging.getLogger(__name__)


def _validate(
    x: npt.NDArray[np.float32], centroids: npt.NDArray[np.float32], chunk_rows: int
) -> None:
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    for name, value in (("x", x), ("centroids", centroids)):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.ndim != 2
            or not all(value.shape)
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be a nonempty C-contiguous float32 matrix")
    if x.shape[1] != centroids.shape[1]:
        raise ValueError("x and centroids must have the same dimension")
    if not np.isfinite(centroids).all():
        raise ValueError("centroids must be finite")


def _direct(point: npt.NDArray[np.float32], centers: npt.NDArray[np.float32]) -> tuple[float, int]:
    residual = centers - point
    distances = np.sum(residual * residual, axis=1, dtype=np.float32)
    label = int(np.argmin(distances))
    return float(distances[label]), label


def _oracle_matches(
    point: npt.NDArray[np.float32], centers: npt.NDArray[np.float32], distance: float
) -> bool:
    residual = centers.astype(np.float64) - point.astype(np.float64)
    minimum = float(np.min(np.sum(residual * residual, axis=1)))
    return bool(np.isclose(distance, minimum, rtol=2e-5, atol=1e-6))


def _nearest(
    x: npt.NDArray[np.float32], centroids: npt.NDArray[np.float32], chunk_rows: int
) -> Iterator[tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int64]]]:
    _validate(x, centroids, chunk_rows)
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    k_neighbors = min(2, len(centroids))
    max_center_norm = float(np.max(np.sum(centroids.astype(np.float64) ** 2, axis=1)))
    # Tiny problems are checked in full; large problems use fixed global rows,
    # independent of the caller's chunking and without an (n,k) distance matrix.
    spot_rows = (
        np.arange(len(x))
        if len(x) <= 128 and len(centroids) <= 256
        else np.unique(np.linspace(0, len(x) - 1, min(8, len(x)), dtype=np.int64))
    )
    fallback_rows = 0
    spot_checks = 0
    for start in range(0, len(x), chunk_rows):
        block = x[start : start + chunk_rows]
        if not np.isfinite(block).all():
            raise ValueError("x must be finite")
        faiss_distances, candidates = index.search(block, k_neighbors)
        if np.any(candidates < 0):
            raise ValueError("exact search could not represent finite FP32 distances")
        labels = np.ascontiguousarray(candidates[:, 0], dtype=np.int64)
        residual = block - centroids[labels]
        distances = np.sum(residual * residual, axis=1, dtype=np.float32)
        norms = np.sum(block.astype(np.float64) ** 2, axis=1) + max_center_norm
        # BLAS evaluates norm sums minus dot products. Near zero, that can
        # destroy all significant digits, so recompute every center directly.
        cancellation_bound = 64 * np.finfo(np.float32).eps * norms
        unsafe = (faiss_distances[:, 0] <= cancellation_bound) | (
            np.abs(faiss_distances[:, 0] - distances) > cancellation_bound
        )
        if k_neighbors == 2:
            second = block - centroids[candidates[:, 1]]
            second_distances = np.sum(second * second, axis=1, dtype=np.float32)
            unsafe |= second_distances <= distances
            unsafe |= faiss_distances[:, 1] - faiss_distances[:, 0] <= cancellation_bound
        for row in np.flatnonzero(unsafe):
            distances[row], labels[row] = _direct(block[row], centroids)
        fallback_rows += int(np.count_nonzero(unsafe))
        checks = spot_rows[(spot_rows >= start) & (spot_rows < start + len(block))] - start
        spot_checks += len(checks)
        if any(not _oracle_matches(block[row], centroids, float(distances[row])) for row in checks):
            _LOG.warning(
                "FP64 spot discrepancy; recomputing evaluation chunk at row %d directly", start
            )
            fallback_rows += len(block) - int(np.count_nonzero(unsafe))
            for direct_row in range(len(block)):
                distances[direct_row], labels[direct_row] = _direct(block[direct_row], centroids)
            if any(
                not _oracle_matches(block[row], centroids, float(distances[row])) for row in checks
            ):
                raise ValueError("FP32 distances disagree with the independent FP64 oracle")
        if not np.isfinite(distances).all() or np.any(distances < 0):
            raise ValueError("squared L2 distance is nonfinite or negative")
        yield start, distances, labels
    _LOG.info(
        "Exact Faiss CPU evaluation: rows=%d centers=%d chunk_rows=%d direct_fallback_rows=%d fp64_spot_rows=%d",
        len(x),
        len(centroids),
        chunk_rows,
        fallback_rows,
        spot_checks,
    )


def evaluate_centers(
    x: npt.NDArray[np.float32], centroids: npt.NDArray[np.float32], *, chunk_rows: int
) -> Evaluation:
    """Evaluate final centers with independent exact squared-L2 assignments.

    Parameters
    ----------
    x : numpy.ndarray
        Finite C-contiguous float32 matrix, shape (n_points, dimension).
        The origin is a valid L2 input.
    centroids : numpy.ndarray
        Finite C-contiguous float32 final centers, shape (k, dimension).
        Equal distances choose the smallest center index.
    chunk_rows : int
        Positive evaluation query chunk size, unrelated to training batches.

    Returns
    -------
    Evaluation
        Point-mean squared L2 (not divided by dimension), FP64 aggregation
        of FP32 distances, and int64 cluster sizes of shape (k,).

    Raises
    ------
    ValueError
        Invalid shape/dtype, nonfinite inputs/distances, or failed FP64 check.
    """
    _validate(x, centroids, chunk_rows)
    sizes = np.zeros(len(centroids), dtype=np.int64)
    # Per-point accumulation in a fixed order makes aggregation chunk independent.
    total = np.float64(0)
    for _, distances, labels in _nearest(x, centroids, chunk_rows):
        sizes += np.bincount(labels, minlength=len(centroids))
        for distance in distances:
            total += np.float64(distance)
    return Evaluation(
        mse=float(total / len(x)),
        n_points=len(x),
        empty_clusters=int(np.count_nonzero(sizes == 0)),
        cluster_sizes=sizes,
    )


def evaluate_groups(
    x: npt.NDArray[np.float32],
    centroids: npt.NDArray[np.float32],
    group_ids: npt.NDArray[np.int64],
    *,
    chunk_rows: int,
) -> Mapping[str, JSONValue]:
    """Evaluate the 256 known components and the optional outlier group.

    Parameters
    ----------
    x, centroids : numpy.ndarray
        Finite C-contiguous float32 matrices shaped (n_points, dimension)
        and (k, dimension), respectively.
    group_ids : numpy.ndarray
        C-contiguous int64 shape (n_points,), known IDs in 0..256.
    chunk_rows : int
        Positive query chunk size.

    Returns
    -------
    Mapping[str, JSONValue]
        All IDs 0..256 mapped to count, mse and reason. Empty groups have
        count=0, mse=null, reason=empty_group; populated groups reason=null.

    Raises
    ------
    ValueError
        Invalid arrays, unknown group IDs, or failed numeric validation.
    """
    if (
        not isinstance(group_ids, np.ndarray)
        or group_ids.dtype != np.int64
        or group_ids.ndim != 1
        or not group_ids.flags.c_contiguous
        or len(group_ids) != len(x)
        or np.any(group_ids < 0)
        or np.any(group_ids >= _GROUP_COUNT)
    ):
        raise ValueError("group_ids must be aligned C-contiguous int64 IDs in 0..256")
    counts = np.bincount(group_ids, minlength=_GROUP_COUNT)
    totals = np.zeros(_GROUP_COUNT, dtype=np.float64)
    for start, distances, _ in _nearest(x, centroids, chunk_rows):
        np.add.at(totals, group_ids[start : start + len(distances)], distances)
    return {
        str(group): {
            "count": int(count),
            "mse": float(totals[group] / count) if count else None,
            "reason": None if count else "empty_group",
        }
        for group, count in enumerate(counts)
    }
