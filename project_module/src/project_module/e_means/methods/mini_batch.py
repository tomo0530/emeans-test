"""Scikit-learn MiniBatchKMeans with explicit sampling and work budget."""

import math

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from project_module.e_means.contracts import RunSpec, TrainingResult, derive_seed
from project_module.e_means.methods import BeforeTrain, Matrix, TrainingClock, validate


def train(
    x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
) -> TrainingResult:
    """Run a fixed number of replacement-sampled partial fits."""
    validate(x, init, spec, "mini_batch", "cpu", set())
    batch_size = min(len(x), max(4096, spec.k))
    steps = math.ceil(spec.budget * len(x) / batch_size)
    seed = derive_seed(spec.seed, "method/mini_batch")
    with threadpool_limits(limits=spec.threads):
        clock = TrainingClock(before_train)
        rng = np.random.default_rng(seed)
        model = MiniBatchKMeans(
            n_clusters=spec.k,
            init=init.copy(),
            n_init=1,
            batch_size=batch_size,
            compute_labels=False,
            random_state=seed,
            tol=0.0,
            max_no_improvement=None,
        )
        for _ in range(steps):
            model.partial_fit(x[rng.integers(0, len(x), size=batch_size)])
        result_timing = clock.finish()
    resolved_params = model.get_params()
    resolved_params["init"] = "supplied"
    return TrainingResult(
        np.ascontiguousarray(model.cluster_centers_, dtype=np.float32),
        result_timing,
        {
            "initial_centers": "supplied",
            "backend": "sklearn",
            "dtype": "float32",
            "batch_size": batch_size,
            "actual_batches": steps,
            "processed_rows": steps * batch_size,
            "sample_with_replacement": True,
            "reassignment_ratio": float(model.reassignment_ratio),
            "revivals": None,
            "revivals_reason": "sklearn does not expose count",
            "resolved_params": resolved_params,
        },
    )
