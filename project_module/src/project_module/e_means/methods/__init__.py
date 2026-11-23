"""Explicit dispatch and shared validation for training adapters."""

from collections.abc import Callable
from importlib import import_module
from time import monotonic_ns, perf_counter, time_ns
from typing import Protocol

import numpy as np
import numpy.typing as npt

from project_module.e_means.contracts import JSONValue, RunSpec, TimingBreakdown, TrainingResult

type Matrix = npt.NDArray[np.float32]
type BeforeTrain = Callable[[], None]


class Adapter(Protocol):
    """Callable interface shared by every adapter."""

    def __call__(
        self, x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
    ) -> TrainingResult: ...


def validate(
    x: Matrix, init: Matrix, spec: RunSpec, method: str, device: str, allowed: set[str]
) -> None:
    """Validate array and adapter boundaries before execution."""
    if spec.method != method or spec.device != device:
        raise ValueError(f"{method} requires method={method} and device={device}")
    for name, value in (("x", x), ("init", init)):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.ndim != 2
            or not value.flags.c_contiguous
            or not all(value.shape)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be a finite C-contiguous float32 matrix")
    if x.shape[0] != spec.n_train or init.shape != (spec.k, x.shape[1]):
        raise ValueError("input shapes do not match RunSpec")
    unknown = spec.params.keys() - allowed
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")


def positive_int_param(spec: RunSpec, key: str, default: int | None) -> int | None:
    """Read an optional positive integer without accepting booleans."""
    value = spec.params.get(key, default)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{key} must be a positive integer")
    return value


def emeans_params(spec: RunSpec) -> dict[str, JSONValue]:
    """Validate settings common to both upstream e-means paths."""
    params: dict[str, JSONValue] = {}
    for key in ("batch_size", "assignment_chunk_size"):
        params[key] = positive_int_param(spec, key, None)
    for key in ("alpha0", "unused_threshold"):
        value = spec.params.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
            or (key == "alpha0" and value == 0)
        ):
            raise ValueError(f"invalid {key}")
        params[key] = value
    law = spec.params.get("batch_law", "auto")
    if law not in ("auto", "quality", "throughput"):
        raise ValueError("invalid batch_law")
    params["batch_law"] = law
    return params


class TrainingClock:
    """Capture the actual training window, after the optional rendezvous."""

    def __init__(self, before_train: BeforeTrain | None = None) -> None:
        if before_train is not None:
            before_train()
        self.started_at_ns = time_ns()
        self.started_monotonic_ns = monotonic_ns()
        self.started = perf_counter()

    def finish(self, *, transfer: float = 0, synchronized: bool = False) -> TimingBreakdown:
        """Stop the clock and create a breakdown with actual window timestamps."""
        duration = perf_counter() - self.started
        finished_monotonic_ns = monotonic_ns()
        finished_at_ns = time_ns()
        return TimingBreakdown(
            transfer_seconds=transfer,
            train_seconds=duration,
            assign_seconds=0,
            evaluate_seconds=0,
            save_seconds=0,
            e2e_seconds=0,
            metadata={
                "clock": "synchronized perf_counter wall" if synchronized else "perf_counter wall",
                "scope": "training and finalize",
                "train_started_at_ns": self.started_at_ns,
                "train_finished_at_ns": finished_at_ns,
                "train_started_monotonic_ns": self.started_monotonic_ns,
                "train_finished_monotonic_ns": finished_monotonic_ns,
            },
        )


def train(
    x: Matrix, init: Matrix, spec: RunSpec, *, before_train: BeforeTrain | None = None
) -> TrainingResult:
    """Train the selected method with supplied float32 data and centers.

    Parameters
    ----------
    x : Matrix
        Finite C-contiguous float32 training rows of shape (n, d).
    init : Matrix
        Finite C-contiguous float32 initial centers of shape (k, d).
    spec : RunSpec
        Method, budget, seed, device, and parameters for the run.

    before_train : BeforeTrain or None
        Optional callback after preparation, immediately before training.

    Returns
    -------
    TrainingResult
        Final CPU centroids, elapsed phases, and actual work metadata.
    """
    modules = {
        name: f"{__name__}.{name}"
        for name in (
            "faiss_lloyd",
            "numpy_e_means",
            "mini_batch",
            "elkan",
            "torch_e_means",
            "torch_lloyd",
        )
    }
    if spec.method not in modules:
        raise ValueError(f"unknown method: {spec.method}")
    adapter: Adapter = import_module(modules[spec.method]).train
    return adapter(x, init, spec, before_train=before_train)
