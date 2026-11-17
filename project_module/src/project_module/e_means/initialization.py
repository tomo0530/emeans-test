"""Reproducible shared initialization from distinct training vectors."""

import hashlib

import numpy as np
import numpy.typing as npt

from project_module.e_means.contracts import derive_seed


def make_initial_centers(x: npt.NDArray[np.float32], k: int, seed: int) -> npt.NDArray[np.float32]:
    """Sample uniformly from distinct rows of finite float32 ``(n, d)`` training data.

    Returns a C-contiguous float32 matrix shaped ``(k, d)``.
    """
    if (
        not isinstance(x, np.ndarray)
        or x.dtype != np.float32
        or x.ndim != 2
        or not x.flags.c_contiguous
        or not all(x.shape)
    ):
        raise ValueError("x must be a nonempty finite C-contiguous float32 matrix")
    flattened = x.reshape(-1)
    for start in range(0, flattened.size, 1024 * 1024):
        if not np.isfinite(flattened[start : start + 1024 * 1024]).all():
            raise ValueError("x must be a nonempty finite C-contiguous float32 matrix")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be positive")
    # Fixed-size row digests avoid retaining an n-by-d collection of Python
    # bytes. First occurrences are sorted by row index to preserve the original
    # candidate order and hence the initial centers for existing seeds.
    digests = np.empty(len(x), dtype="V16")
    for i, row in enumerate(x):
        canonical = row.copy()
        canonical[canonical == 0] = 0
        digests[i] = hashlib.blake2b(canonical, digest_size=16).digest()
    _, first, inverse, counts = np.unique(
        digests, return_index=True, return_inverse=True, return_counts=True
    )
    repeated_rows = np.flatnonzero(counts[inverse] > 1)
    for start in range(0, len(repeated_rows), 4096):
        rows = repeated_rows[start : start + 4096]
        if not np.array_equal(x[rows], x[first[inverse[rows]]]):
            raise ValueError("row hash collision")
    if len(first) < k:
        raise ValueError("fewer unique rows than k")
    indices = np.sort(first)
    rng = np.random.default_rng(derive_seed(seed, "init"))
    selected = rng.choice(indices, size=k, replace=False)
    return np.ascontiguousarray(x[selected], dtype=np.float32)
