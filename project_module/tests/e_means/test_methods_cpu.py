"""Behavioral checks for CPU training adapters."""

import numpy as np
import pytest
from project_module.e_means.contracts import RunSpec
from project_module.e_means.methods import train


def make_spec(method: str, **changes: object) -> RunSpec:
    values = dict(
        run_id="r",
        dataset_id="d",
        condition_id="c",
        method=method,
        device="cpu",
        measurement_id="m",
        n_train=4,
        k=2,
        seed=17,
        data_seed=1,
        threads=1,
        timing_repeat=0,
        budget=1,
        params={},
    )
    values.update(changes)
    return RunSpec.model_validate(values)


@pytest.mark.parametrize("method", ["faiss_lloyd", "numpy_e_means", "mini_batch", "elkan"])
def test_methods_use_supplied_centers_and_report_actual_work(method: str) -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    init = np.array([[0.0], [10.0]], dtype=np.float32)
    result = train(x, init, make_spec(method))
    assert result.centroids.shape == (2, 1)
    assert result.centroids.dtype == np.float32
    assert np.isfinite(result.centroids).all()
    assert result.stats["processed_rows"] >= 4
    assert result.stats["initial_centers"] == "supplied"


def test_faiss_one_iteration_uses_initial_centers() -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    result = train(x, np.array([[0.0], [10.0]], dtype=np.float32), make_spec("faiss_lloyd"))
    np.testing.assert_allclose(result.centroids, [[1.0], [11.0]], rtol=1e-5, atol=1e-6)


def test_minibatch_first_batch_accepts_k_over_4096() -> None:
    x = np.arange(4100, dtype=np.float32).reshape(-1, 1)
    result = train(x, x.copy(), make_spec("mini_batch", n_train=4100, k=4100, budget=0.25))
    assert result.stats["batch_size"] == 4100
    assert result.stats["processed_rows"] == 4100


def test_unknown_params_are_rejected() -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="unknown"):
        train(x, x[[0, 2]], make_spec("faiss_lloyd", params={"bad": 1}))


def test_e_means_accepts_upstream_quality_batch_law() -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    result = train(x, x[[0, 2]], make_spec("numpy_e_means", params={"batch_law": "quality"}))
    assert result.stats["recipe"]["batch_law"] == "quality"


@pytest.mark.parametrize("method", ["faiss_lloyd", "numpy_e_means", "mini_batch", "elkan"])
def test_before_train_runs_once_before_measured_window(method: str) -> None:
    import time

    x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    events: list[int] = []

    def before_train() -> None:
        events.append(time.time_ns())

    result = train(x, x[[0, 2]], make_spec(method), before_train=before_train)
    metadata = result.timing.metadata
    assert len(events) == 1
    assert events[0] <= metadata["train_started_at_ns"] < metadata["train_finished_at_ns"]
    assert metadata["train_finished_at_ns"] <= time.time_ns()
    elapsed_ns = metadata["train_finished_monotonic_ns"] - metadata["train_started_monotonic_ns"]
    assert elapsed_ns > 0
    assert abs(result.timing.train_seconds - elapsed_ns / 1e9) < 0.002


def test_before_train_is_not_called_for_invalid_input() -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    events: list[str] = []
    with pytest.raises(ValueError, match="unknown"):
        train(
            x,
            x[[0, 2]],
            make_spec("elkan", params={"bad": 1}),
            before_train=lambda: events.append("called"),
        )
    assert events == []


@pytest.mark.parametrize("method", ["faiss_lloyd", "numpy_e_means", "mini_batch", "elkan"])
def test_before_train_error_propagates(method: str) -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)

    def fail_gate() -> None:
        raise RuntimeError("barrier aborted")

    with pytest.raises(RuntimeError, match="barrier aborted"):
        train(x, x[[0, 2]], make_spec(method), before_train=fail_gate)
