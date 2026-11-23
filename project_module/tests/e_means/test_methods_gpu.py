"""Native GPU Lloyd and e-means checks on small inputs."""

import numpy as np
import pytest
from project_module.e_means.contracts import RunSpec
from project_module.e_means.methods import train

pytestmark = pytest.mark.gpu


def spec(method: str, budget: int = 1, **changes: object) -> RunSpec:
    values = dict(
        run_id="r",
        dataset_id="d",
        condition_id="c",
        method=method,
        device="cuda",
        measurement_id="m",
        n_train=4,
        k=2,
        seed=17,
        data_seed=1,
        threads=1,
        timing_repeat=0,
        budget=budget,
        params={},
    )
    values.update(changes)
    return RunSpec.model_validate(values)


def data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32),
        np.array([[0.0], [10.0]], dtype=np.float32),
    )


def test_torch_lloyd_one_iteration_matches_fp64_oracle() -> None:
    x, init = data()
    result = train(x, init, spec("torch_lloyd"))
    np.testing.assert_allclose(result.centroids, [[1.0], [11.0]], rtol=1e-4, atol=1e-6)
    assert result.stats["processed_rows"] == 4
    assert result.stats["backend"] == "torch"


def test_empty_center_is_retained() -> None:
    x, init = data()
    init[1, 0] = 100
    result = train(x, init, spec("torch_lloyd"))
    np.testing.assert_allclose(result.centroids, [[6.0], [100.0]], rtol=1e-4, atol=1e-6)


def test_lloyd_objective_does_not_increase() -> None:
    x, init = data()
    result = train(x, init, spec("torch_lloyd", budget=3))
    initial_mse = np.min((x[:, None, :] - init[None, :, :]) ** 2, axis=1).sum()
    final_mse = np.min((x[:, None, :] - result.centroids[None, :, :]) ** 2, axis=1).sum()
    assert final_mse <= initial_mse


def test_lloyd_iteration_prefix_agrees() -> None:
    x, init = data()
    once = train(x, init, spec("torch_lloyd"))
    twice = train(x, once.centroids, spec("torch_lloyd"))
    direct = train(x, init, spec("torch_lloyd", budget=2))
    np.testing.assert_allclose(twice.centroids, direct.centroids, rtol=1e-4, atol=1e-6)


def test_torch_e_means_runs_native_fp32() -> None:
    x, init = data()
    result = train(x, init, spec("torch_e_means"))
    assert result.centroids.dtype == np.float32
    assert result.stats["backend"] == "torch"
    assert result.stats["processed_rows"] > 0
    assert result.stats["graph_policy"] == "upstream_auto"
    observations = result.stats["graph_eligibility"]
    assert observations and all(
        isinstance(row["upstream_eligible"], bool)
        and row["upstream_eligible"] == row["effective_eligible"]
        for row in observations
    )
    assert result.stats["graph_actual_replays"] is None


def test_unavailable_native_backend_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    import faiss.contrib.e_means.torch_backend as backend

    x, init = data()
    monkeypatch.delattr(backend, "TorchBackend")
    with pytest.raises(RuntimeError, match="native"):
        train(x, init, spec("torch_lloyd"))


def test_duplicate_centers_leave_empty_center_unchanged() -> None:
    x = np.array([[0.0], [0.0], [2.0], [2.0]], dtype=np.float32)
    init = np.array([[0.0], [0.0]], dtype=np.float32)
    result = train(x, init, spec("torch_lloyd"))
    np.testing.assert_allclose(result.centroids, [[1.0], [0.0]], rtol=1e-4, atol=1e-6)


def test_e_means_cpu_and_gpu_share_algorithm_recipe() -> None:
    x, init = data()
    gpu = train(x, init, spec("torch_e_means", budget=2))
    cpu_spec = spec("numpy_e_means", budget=2, device="cpu")
    cpu = train(x, init, cpu_spec)
    for key in ("batch_law", "batch_size", "alpha0", "epoch_rows", "unused_threshold"):
        assert gpu.stats["recipe"][key] == cpu.stats["recipe"][key]
    assert gpu.stats["recipe"]["assignment_backend"] == "native"
    assert gpu.stats["recipe"]["representation"] == "float32"


@pytest.mark.parametrize("method", ["torch_lloyd", "torch_e_means"])
def test_before_train_follows_synchronized_gpu_transfer(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import torch

    x, init = data()
    torch.cuda.synchronize()
    initial_allocation = torch.cuda.memory_allocated()
    sync_count = 0
    real_synchronize = torch.cuda.synchronize
    events: list[tuple[int, int, int]] = []
    copies_to_cpu: list[tuple[int, int]] = []
    real_cpu = torch.Tensor.cpu

    def synchronize(*args: object, **kwargs: object) -> None:
        nonlocal sync_count
        real_synchronize(*args, **kwargs)
        sync_count += 1

    def before_train() -> None:
        events.append((time.time_ns(), sync_count, torch.cuda.memory_allocated()))

    def copy_to_cpu(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        copies_to_cpu.append((time.time_ns(), sync_count))
        return real_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.Tensor, "cpu", copy_to_cpu)
    result = train(x, init, spec(method), before_train=before_train)
    metadata = result.timing.metadata
    assert len(events) == 1
    callback_ns, syncs_at_callback, allocated = events[0]
    assert syncs_at_callback >= 2
    assert allocated > initial_allocation
    assert callback_ns <= metadata["train_started_at_ns"] < metadata["train_finished_at_ns"]
    assert metadata["train_finished_at_ns"] <= time.time_ns()
    assert copies_to_cpu[-1][0] >= metadata["train_finished_at_ns"]
    assert copies_to_cpu[-1][1] > syncs_at_callback
    elapsed_ns = metadata["train_finished_monotonic_ns"] - metadata["train_started_monotonic_ns"]
    assert elapsed_ns > 0
    assert abs(result.timing.train_seconds - elapsed_ns / 1e9) < 0.002
