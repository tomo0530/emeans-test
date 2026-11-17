"""Upstream NumPy e-means adapter."""

import math

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from project_module.e_means.contracts import RunSpec, TrainingResult, derive_seed
from project_module.e_means.methods import (
    BeforeTrain,
    Matrix,
    TrainingClock,
    emeans_params,
    validate,
)


def train(
    x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
) -> TrainingResult:
    """Train the upstream NumPy implementation on the full eligible dataset."""
    validate(
        x,
        init,
        spec,
        "numpy_e_means",
        "cpu",
        {"batch_size", "assignment_chunk_size", "alpha0", "batch_law", "unused_threshold"},
    )
    params = emeans_params(spec)
    faiss.omp_set_num_threads(spec.threads)
    clock = TrainingClock(before_train)
    model = NumpyEmeans(
        x.shape[1],
        spec.k,
        nepochs=int(spec.budget),
        seed=derive_seed(spec.seed, "method/numpy_e_means"),
        max_points_per_centroid=math.ceil(len(x) / spec.k),
        capture_assignments=False,
        spherical=False,
        **params,
    )
    centroids, _ = model.train(x, init_centroids=init.copy())
    result_timing = clock.finish()
    recipe = model.recipe
    return TrainingResult(
        np.ascontiguousarray(centroids, dtype=np.float32),
        result_timing,
        {
            "initial_centers": "supplied",
            "backend": recipe["backend"],
            "implementation": "numpy",
            "dtype": "float32",
            "actual_epochs": len(recipe["epoch_rows"]),
            "processed_rows": sum(recipe["epoch_rows"]),
            "revivals": model.n_splits,
            "recipe": recipe,
            "resolved_params": {
                **params,
                "capture_assignments": False,
                "spherical": False,
                "max_points_per_centroid": math.ceil(len(x) / spec.k),
            },
        },
    )
