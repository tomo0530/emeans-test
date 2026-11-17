"""Single-host rendezvous: complete groups synchronize or fail closed."""

import multiprocessing as mp
import time
from pathlib import Path

import pytest


def _participate(root: str, member: str, identity: str, output: object, delay: float = 0.0) -> None:
    from project_module.e_means._sentinel import SentinelGroup

    group = SentinelGroup(Path(root), "group", ["gpu-a", "gpu-b"], member, 2.0, {"plan": "same"})
    try:
        group.join()
        if delay:
            time.sleep(delay)
        waited = group.wait({"condition": "S0", "repeat": 0}, {"init_hash": identity})
        group.finish()
        output.put((member, "success", waited))
    except Exception as exc:
        output.put((member, type(exc).__name__, str(exc)))
    finally:
        group.close()


def test_delayed_participants_wait_for_one_release(tmp_path: Path) -> None:
    from project_module.e_means._sentinel import (
        SentinelCoordinationError,
        SentinelGroup,
        inspect_group,
    )

    ctx = mp.get_context("fork")
    output = ctx.Queue()
    first = ctx.Process(target=_participate, args=(str(tmp_path), "gpu-a", "same", output))
    second = ctx.Process(target=_participate, args=(str(tmp_path), "gpu-b", "same", output, 0.2))
    first.start()
    second.start()
    results = [output.get(timeout=6), output.get(timeout=6)]
    first.join(3)
    second.join(3)
    assert {result[1] for result in results} == {"success"}
    assert max(result[2] for result in results) >= 0.2
    state = inspect_group(tmp_path, "group")
    assert state["status"] == "complete"
    assert state["completed_members"] == ["gpu-a", "gpu-b"]
    again = SentinelGroup(tmp_path, "group", ["gpu-a", "gpu-b"], "gpu-a", 0.1, {"plan": "same"})
    with pytest.raises(SentinelCoordinationError, match="already complete"):
        again.join()
    again.abort("unrelated_rejoin")
    assert inspect_group(tmp_path, "group") == state


def test_mismatched_inputs_abort_both_members(tmp_path: Path) -> None:
    ctx = mp.get_context("fork")
    output = ctx.Queue()
    first = ctx.Process(target=_participate, args=(str(tmp_path), "gpu-a", "first", output))
    second = ctx.Process(target=_participate, args=(str(tmp_path), "gpu-b", "second", output))
    first.start()
    second.start()
    results = [output.get(timeout=6), output.get(timeout=6)]
    first.join(3)
    second.join(3)
    assert {result[1] for result in results} == {"SentinelCoordinationError"}


def test_timeout_poisoned_group_cannot_be_reused(tmp_path: Path) -> None:
    from project_module.e_means._sentinel import SentinelCoordinationError, SentinelGroup

    group = SentinelGroup(tmp_path, "group", ["gpu-a", "gpu-b"], "gpu-a", 0.1, {"plan": "same"})
    group.join()
    with pytest.raises(SentinelCoordinationError, match="timeout"):
        group.wait({"condition": "S0"}, {"init_hash": "same"})
    group.close()
    again = SentinelGroup(tmp_path, "group", ["gpu-a", "gpu-b"], "gpu-a", 0.1, {"plan": "same"})
    with pytest.raises(SentinelCoordinationError):
        again.join()


def _claim_and_wait(root: str, ready: object) -> None:
    from project_module.e_means._sentinel import SentinelGroup

    group = SentinelGroup(Path(root), "crash", ["gpu-a", "gpu-b"], "gpu-a", 5.0, {"plan": "same"})
    group.join()
    ready.put("claimed")
    group.wait({"condition": "S0"}, {"init_hash": "same"})


def test_crashed_participant_aborts_group(tmp_path: Path) -> None:
    from project_module.e_means._sentinel import (
        SentinelCoordinationError,
        SentinelGroup,
        inspect_group,
    )

    ctx = mp.get_context("fork")
    ready = ctx.Queue()
    process = ctx.Process(target=_claim_and_wait, args=(str(tmp_path), ready))
    process.start()
    assert ready.get(timeout=3) == "claimed"
    deadline = time.monotonic() + 3
    while not list(tmp_path.glob(".emeans-sentinel/*/trials/*/arrival-*.json")):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    process.terminate()
    process.join(3)
    other = SentinelGroup(tmp_path, "crash", ["gpu-a", "gpu-b"], "gpu-b", 1, {"plan": "same"})
    with pytest.raises(SentinelCoordinationError, match="no longer alive"):
        other.join()
    assert inspect_group(tmp_path, "crash")["status"] == "aborted"


def test_duplicate_claim_aborts_incomplete_generation(tmp_path: Path) -> None:
    from project_module.e_means._sentinel import (
        SentinelCoordinationError,
        SentinelGroup,
        inspect_group,
    )

    first = SentinelGroup(tmp_path, "duplicate", ["gpu-a", "gpu-b"], "gpu-a", 1, {"plan": "same"})
    duplicate = SentinelGroup(
        tmp_path, "duplicate", ["gpu-a", "gpu-b"], "gpu-a", 1, {"plan": "same"}
    )
    first.join()
    try:
        with pytest.raises(SentinelCoordinationError, match="already claimed"):
            duplicate.join()
        assert inspect_group(tmp_path, "duplicate")["status"] == "aborted"
    finally:
        first.close()


def test_different_group_descriptor_aborts_generation(tmp_path: Path) -> None:
    from project_module.e_means._sentinel import SentinelCoordinationError, SentinelGroup

    first = SentinelGroup(tmp_path, "same-id", ["gpu-a", "gpu-b"], "gpu-a", 1, {"plan": "first"})
    other = SentinelGroup(tmp_path, "same-id", ["gpu-a", "gpu-b"], "gpu-b", 1, {"plan": "second"})
    first.join()
    try:
        with pytest.raises(SentinelCoordinationError, match="descriptor mismatch"):
            other.join()
    finally:
        first.close()


def _eight_member_trial(root: str, member: str, members: list[str], output: object) -> None:
    from project_module.e_means._sentinel import SentinelGroup

    group = SentinelGroup(Path(root), "eight", members, member, 5, {"plan": "same"})
    try:
        group.join()
        group.wait({"condition": "S0", "repeat": 0}, {"init_hash": "same"})
        begin = time.monotonic_ns()
        time.sleep(0.1)
        end = time.monotonic_ns()
        group.finish()
        output.put((member, begin, end, None))
    except Exception as exc:
        output.put((member, 0, 0, type(exc).__name__))
    finally:
        group.close()


def test_eight_processes_release_into_overlapping_work_windows(tmp_path: Path) -> None:
    from project_module.e_means._sentinel import inspect_group

    ctx = mp.get_context("fork")
    output = ctx.Queue()
    members = [f"gpu-{index}" for index in range(8)]
    processes = [
        ctx.Process(target=_eight_member_trial, args=(str(tmp_path), member, members, output))
        for member in members
    ]
    for process in processes:
        process.start()
    results = [output.get(timeout=8) for _ in members]
    for process in processes:
        process.join(3)
    assert all(result[3] is None for result in results)
    assert min(result[2] for result in results) > max(result[1] for result in results)
    state = inspect_group(tmp_path, "eight")
    assert state["status"] == "complete"
    assert state["completed_members"] == members


def _runner_member(
    root: str, output_dir: str, config: object, spec: object, member: str, results: object
) -> None:
    import json

    from project_module.e_means import runner
    from project_module.e_means._sentinel import SentinelGroup
    from project_module.e_means.data import load_dataset

    data = Path(root)
    output = Path(output_dir)
    descriptor = runner._sentinel_descriptor(config, [spec], output)
    group = SentinelGroup(data, "pipeline", ["gpu-a", "gpu-b"], member, 5, descriptor)
    try:
        group.join()
        dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
        attempt = runner._trial(
            spec,
            dataset,
            output,
            config,
            resume=True,
            retry_failed=False,
            retry_reason=None,
            gpu=None,
            warm=set(),
            minimum_budget=1,
            sentinel=group,
        )
        group.finish()
        results.put(
            (
                member,
                json.loads((attempt / "status.json").read_text())["status"],
                json.loads((attempt / "metrics.json").read_text())["timing"]["metadata"],
            )
        )
    except Exception as exc:
        group.abort("test_failure")
        results.put((member, type(exc).__name__, {}))
    finally:
        group.close()


def test_runner_trials_coordinate_matching_inputs_through_adapter_hook(tmp_path: Path) -> None:
    from project_module.e_means import runner
    from project_module.e_means._sentinel import inspect_group
    from project_module.e_means.contracts import ExperimentConfig
    from project_module.e_means.data import prepare_condition

    base = {
        "schema_version": 1,
        "plan_revision": "test",
        "phase": "pilot",
        "conditions": [
            {
                "condition_id": "tiny",
                "dataset_id": "synthetic",
                "n_train": 32,
                "k": 2,
                "dimension": 3,
                "data_seeds": [7],
                "generator_params": {
                    "n_components": 2,
                    "validation_size": 8,
                    "test_size": 8,
                    "variant": "base",
                },
            }
        ],
        "methods": [{"method": "faiss_lloyd", "budgets": [1], "params": {}}],
        "seeds": [7],
        "measurement": {
            "cohort": "sentinel-parallel-gpu-a",
            "repeats": 1,
            "evaluate_test": False,
            "repeat_rules": [],
            "sentinel": {
                "group_id": "pipeline",
                "participants": 2,
                "members": ["gpu-a", "gpu-b"],
                "timeout_seconds": 5,
            },
        },
        "resources": {
            "cpu_threads": 1,
            "cpu_affinity": [],
            "gpu_pool": [0],
            "gpu_memory_limit_bytes": 25769803776,
            "host_memory_limit_bytes": 1000000000,
        },
    }
    first = ExperimentConfig.model_validate(base)
    second = first.model_copy(
        update={
            "measurement": first.measurement.model_copy(
                update={"cohort": "sentinel-parallel-gpu-b"}
            )
        }
    )
    data = tmp_path / "data"
    prepare_condition(first.conditions[0], data, 7)
    configs = [first, second]
    outputs = [tmp_path / "a", tmp_path / "b"]
    jobs = [
        runner.prepare_jobs(config, data, out)[0]
        for config, out in zip(configs, outputs, strict=True)
    ]
    ctx = mp.get_context("fork")
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_runner_member, args=(str(data), str(out), config, spec, member, results)
        )
        for out, config, spec, member in zip(
            outputs, configs, jobs, ["gpu-a", "gpu-b"], strict=True
        )
    ]
    for process in processes:
        process.start()
    values = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(3)
    assert [value[1] for value in values] == ["success", "success"]
    assert all(value[2]["coordination_wait_seconds"] > 0 for value in values)
    assert inspect_group(data, "pipeline")["status"] == "complete"
