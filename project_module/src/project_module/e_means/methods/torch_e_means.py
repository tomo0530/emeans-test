"""Upstream native FP32 Torch e-means adapter."""

import math
from time import perf_counter

import numpy as np

from project_module.e_means.contracts import JSONValue, RunSpec, TrainingResult, derive_seed
from project_module.e_means.methods import (
    BeforeTrain,
    Matrix,
    TrainingClock,
    emeans_params,
    positive_int_param,
    validate,
)


def train(
    x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
) -> TrainingResult:
    """Run upstream QuantizedEmeans without quantization on CUDA."""
    validate(
        x,
        init,
        spec,
        "torch_e_means",
        "cuda",
        {
            "batch_size",
            "assignment_chunk_size",
            "centroid_chunk_size",
            "alpha0",
            "batch_law",
            "unused_threshold",
        },
    )
    params = emeans_params(spec)
    width = positive_int_param(spec, "centroid_chunk_size", 4096)
    try:
        import torch
        from faiss.contrib.e_means import QuantizedEmeans
        from faiss.contrib.e_means.torch_backend import TorchBackend
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("native torch assignment backend is unavailable") from exc
    if not torch.cuda.is_available() or TorchBackend is None:
        raise RuntimeError("native CUDA backend is unavailable")
    torch.cuda.synchronize()
    transfer_start = perf_counter()
    data = torch.from_numpy(x).to("cuda")
    initial = torch.from_numpy(init).to("cuda")
    torch.cuda.synchronize()
    transfer = perf_counter() - transfer_start
    clock = TrainingClock(before_train)
    model = QuantizedEmeans(
        x.shape[1],
        spec.k,
        nepochs=int(spec.budget),
        seed=derive_seed(spec.seed, "method/torch_e_means"),
        quantizer=None,
        assignment_backend="native",
        centroid_chunk_size=width,
        max_points_per_centroid=math.ceil(len(x) / spec.k),
        capture_assignments=False,
        spherical=False,
        **params,
    )
    graph_eligibility: list[JSONValue] = []
    upstream_eligible = model._replay_eligible

    def observe_eligibility(rows: object) -> bool:
        eligible: bool = upstream_eligible(rows)
        graph_eligibility.append({"upstream_eligible": eligible, "effective_eligible": eligible})
        return eligible

    # Observe the upstream decision without changing capture operation identities.
    model._replay_eligible = observe_eligibility
    try:
        centers, _ = model.train(data, init_centroids=initial)
    finally:
        # Restore class lookup and release the bound-method reference cycle.
        del model._replay_eligible
    torch.cuda.synchronize()
    timing = clock.finish(transfer=transfer, synchronized=True)
    transfer_start = perf_counter()
    centroids = centers.cpu().numpy().copy()
    torch.cuda.synchronize()
    transfer += perf_counter() - transfer_start
    timing = timing.model_copy(update={"transfer_seconds": transfer})
    recipe = model.recipe
    if (
        recipe["backend"] != "torch"
        or recipe["assignment_backend"] != "native"
        or recipe["representation"] != "float32"
    ):
        raise RuntimeError("native FP32 torch backend is unavailable")
    return TrainingResult(
        np.ascontiguousarray(centroids, dtype=np.float32),
        timing,
        {
            "initial_centers": "supplied",
            "backend": "torch",
            "dtype": "float32",
            "assignment_backend": "native",
            "graph_policy": "upstream_auto",
            "graph_eligibility": graph_eligibility,
            "graph_actual_replays": None,
            "actual_epochs": len(recipe["epoch_rows"]),
            "processed_rows": sum(recipe["epoch_rows"]),
            "revivals": model.n_splits,
            "recipe": recipe,
            "resolved_params": {
                **params,
                "capture_assignments": False,
                "spherical": False,
                "quantizer": None,
                "assignment_backend": "native",
                "centroid_chunk_size": width,
                "max_points_per_centroid": math.ceil(len(x) / spec.k),
            },
        },
    )
