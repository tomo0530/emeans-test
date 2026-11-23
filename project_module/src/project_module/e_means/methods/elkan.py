"""Scikit-learn full-data Elkan adapter."""

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from project_module.e_means.contracts import RunSpec, TrainingResult, derive_seed
from project_module.e_means.methods import BeforeTrain, Matrix, TrainingClock, validate


def train(
    x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
) -> TrainingResult:
    """Run Elkan with a maximum iteration budget and supplied centers."""
    validate(x, init, spec, "elkan", "cpu", set())
    with threadpool_limits(limits=spec.threads):
        clock = TrainingClock(before_train)
        model = KMeans(
            n_clusters=spec.k,
            init=init.copy(),
            n_init=1,
            max_iter=int(spec.budget),
            tol=0.0,
            algorithm="elkan",
            random_state=derive_seed(spec.seed, "method/elkan"),
        )
        model.fit(x)
        result_timing = clock.finish()
    resolved_params = model.get_params()
    resolved_params["init"] = "supplied"
    return TrainingResult(
        np.ascontiguousarray(model.cluster_centers_, dtype=np.float32),
        result_timing,
        {
            "initial_centers": "supplied",
            "backend": "sklearn_elkan",
            "dtype": "float32",
            "actual_iterations": int(model.n_iter_),
            "processed_rows": len(x) * int(model.n_iter_),
            "revivals": None,
            "revivals_reason": "sklearn does not expose count",
            "stop_condition": "strict assignment convergence or iteration budget",
            "internal_objective": "sklearn performs final label/inertia calculation",
            "resolved_params": resolved_params,
        },
    )
