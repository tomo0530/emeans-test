"""Full-data Lloyd updates using upstream native Torch assignment."""

from time import perf_counter

import numpy as np

from project_module.e_means.contracts import RunSpec, TrainingResult
from project_module.e_means.methods import (
    BeforeTrain,
    Matrix,
    TrainingClock,
    positive_int_param,
    validate,
)


def train(
    x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
) -> TrainingResult:
    """Run full-data FP32 Lloyd iterations on CUDA and retain empty centers."""
    validate(x, init, spec, "torch_lloyd", "cuda", {"assignment_chunk_size", "centroid_chunk_size"})
    rows = positive_int_param(spec, "assignment_chunk_size", 4096)
    width = positive_int_param(spec, "centroid_chunk_size", 4096)
    assert rows is not None and width is not None
    try:
        import torch
        from faiss.contrib.e_means.torch_backend import TorchBackend
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("native torch assignment backend is unavailable") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA backend is unavailable")
    torch.cuda.synchronize()
    transfer_start = perf_counter()
    data = torch.from_numpy(x).to("cuda")
    centers = torch.from_numpy(init).to("cuda")
    torch.cuda.synchronize()
    transfer = perf_counter() - transfer_start
    clock = TrainingClock(before_train)
    backend = TorchBackend(centroid_chunk_size=width)
    for _ in range(int(spec.budget)):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(spec.k, dtype=torch.float32, device="cuda")
        for begin in range(0, len(x), rows):
            block = data[begin : begin + rows]
            _, labels = backend.assign(block, centers)
            sums.index_add_(0, labels, block)
            counts.index_add_(0, labels, torch.ones(len(block), dtype=torch.float32, device="cuda"))
        centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
    torch.cuda.synchronize()
    timing = clock.finish(transfer=transfer, synchronized=True)
    transfer_start = perf_counter()
    centroids = centers.cpu().numpy().copy()
    torch.cuda.synchronize()
    transfer += perf_counter() - transfer_start
    timing = timing.model_copy(update={"transfer_seconds": transfer})
    return TrainingResult(
        np.ascontiguousarray(centroids),
        timing,
        {
            "initial_centers": "supplied",
            "backend": "torch",
            "dtype": "float32",
            "actual_iterations": int(spec.budget),
            "processed_rows": len(x) * int(spec.budget),
            "revivals": 0,
            "empty_center_policy": "retain",
            "assignment_backend": "native",
            "assignment_chunk_size": rows,
            "centroid_chunk_size": width,
        },
    )
