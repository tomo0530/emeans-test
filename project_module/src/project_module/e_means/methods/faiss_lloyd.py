"""Faiss CPU full-data Lloyd adapter."""

import math

import faiss
import numpy as np

from project_module.e_means.contracts import RunSpec, TrainingResult, derive_seed
from project_module.e_means.methods import BeforeTrain, Matrix, TrainingClock, validate


def train(
    x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
) -> TrainingResult:
    """Train Faiss Lloyd with all rows and externally supplied centers."""
    validate(x, init, spec, "faiss_lloyd", "cpu", set())
    clock = TrainingClock(before_train)
    faiss.omp_set_num_threads(spec.threads)
    model = faiss.Kmeans(
        x.shape[1],
        spec.k,
        niter=int(spec.budget),
        nredo=1,
        seed=derive_seed(spec.seed, "method/faiss_lloyd"),
        max_points_per_centroid=math.ceil(len(x) / spec.k),
        min_points_per_centroid=1,
        spherical=False,
    )
    model.train(x, init_centroids=init.copy())
    result_timing = clock.finish()
    iteration_stats = model.iteration_stats
    if iteration_stats is None:
        raise RuntimeError("Faiss did not return iteration statistics")
    iterations = len(iteration_stats)
    revivals = sum((int(stat["nsplit"]) for stat in iteration_stats), start=0)
    return TrainingResult(
        np.ascontiguousarray(model.centroids, dtype=np.float32),
        result_timing,
        {
            "initial_centers": "supplied",
            "backend": "faiss_cpu",
            "dtype": "float32",
            "actual_iterations": iterations,
            "processed_rows": len(x) * iterations,
            "revivals": revivals,
            "internal_objective": "faiss evaluates assignments in each iteration",
            "resolved_params": {
                key: getattr(model.cp, key)
                for key in (
                    "afkmc2_chain_length",
                    "check_input_data_for_NaNs",
                    "decode_block_size",
                    "early_stop_threshold",
                    "frozen_centroids",
                    "init_method",
                    "int_centroids",
                    "max_points_per_centroid",
                    "min_points_per_centroid",
                    "niter",
                    "nredo",
                    "seed",
                    "spherical",
                    "update_index",
                    "use_faster_subsampling",
                    "use_super_kmeans",
                    "verbose",
                )
            },
        },
    )
