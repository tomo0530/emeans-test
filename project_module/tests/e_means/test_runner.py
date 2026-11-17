"""Runner ledger, resume, and artifact invariants."""

import json
from pathlib import Path

import numpy as np
import pytest
from project_module.e_means.contracts import ExperimentConfig, SentinelConfig
from project_module.e_means.data import prepare_condition


def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
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
                "cohort": "smoke",
                "repeats": 2,
                "evaluate_test": False,
                "repeat_rules": [],
            },
            "resources": {
                "cpu_threads": 1,
                "cpu_affinity": [],
                "gpu_pool": [0],
                "gpu_memory_limit_bytes": 25769803776,
                "host_memory_limit_bytes": 412316860416,
            },
        }
    )


def _prepare(tmp_path: Path) -> tuple[ExperimentConfig, Path, Path]:
    config = _config()
    data = tmp_path / "data"
    prepare_condition(config.conditions[0], data, 7)
    return config, data, tmp_path / "runs"


def test_timing_repeat_and_measurement_id_never_reuse_success(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    jobs = runner.prepare_jobs(config, data, output)
    assert len(jobs) == 2
    assert len({job.run_id for job in jobs}) == 2
    assert [job.timing_repeat for job in jobs] == [0, 1]
    assert len(json.loads((output / "jobs.json").read_text())) == 2


def test_pilot_never_evaluates_test(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    runner.run_phase(config, data, output, phase="cpu", resume=True)
    jobs = json.loads((output / "jobs.json").read_text())
    metrics = json.loads((output / jobs[0]["run_id"] / "attempt-0001/metrics.json").read_text())
    assert metrics["test"] is None
    assert metrics["test_reason"] == "not_evaluated"
    centroids = np.load(output / jobs[0]["run_id"] / "attempt-0001/centroids.npy")
    assert np.isfinite(centroids).all()


def test_changed_hash_never_reuses_or_overwrites_success(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    runner.prepare_jobs(config, data, output)
    changed = config.model_copy(update={"plan_revision": "changed"})
    with pytest.raises(runner.FrozenIdentityError, match="snapshot"):
        runner.prepare_jobs(changed, data, output)


def test_resume_retries_interrupted_attempt_without_deleting_history(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    runner.run_phase(config, data, output, phase="cpu", resume=True)
    job = json.loads((output / "jobs.json").read_text())[0]
    first = output / job["run_id"] / "attempt-0001"
    status = json.loads((first / "status.json").read_text())
    status["status"] = "interrupted"
    (first / "status.json").write_text(json.dumps(status))
    runner.run_phase(config, data, output, phase="cpu", resume=True)
    assert json.loads((first / "status.json").read_text())["status"] == "interrupted"
    assert (
        json.loads((first.parent / "attempt-0002/status.json").read_text())["status"] == "success"
    )


def test_resume_preserves_failed_terminal_status(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    jobs = runner.prepare_jobs(config, data, output)
    runner._crash_record(jobs[0], output, "worker_crash")
    runner.run_phase(config, data, output, phase="cpu", resume=True)
    job = json.loads((output / "jobs.json").read_text())[0]
    first = output / job["run_id"] / "attempt-0001"
    assert json.loads((first / "status.json").read_text())["status"] == "failed"
    assert not (first.parent / "attempt-0002").exists()
    runner.run_phase(
        config,
        data,
        output,
        phase="cpu",
        resume=True,
        retry_failed=True,
        retry_reason="investigated",
    )
    assert (
        json.loads((first.parent / "attempt-0002/status.json").read_text())["status"] == "success"
    )


def test_missing_artifact_never_marks_success(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    runner.run_phase(config, data, output, phase="cpu", resume=True)
    job = json.loads((output / "jobs.json").read_text())[0]
    (output / job["run_id"] / "attempt-0001/centroids.npy").unlink()
    with pytest.raises(RuntimeError, match="frozen identity"):
        runner.run_phase(config, data, output, phase="cpu", resume=True)


def test_logs_do_not_contain_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from project_module.e_means import runner

    monkeypatch.setenv("HF_TOKEN", "fakecanary_secret_8675309")
    config, data, output = _prepare(tmp_path)
    runner.run_phase(config, data, output, phase="cpu", resume=True)
    for path in output.rglob("*.json"):
        assert "fakecanary_secret_8675309" not in path.read_text()


def test_synthetic_seeds_are_paired_without_cartesian_product(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    condition = config.conditions[0].model_copy(update={"data_seeds": [7, 8, 9]})
    config = config.model_copy(update={"conditions": [condition], "seeds": [7, 8, 9]})
    for seed in (8, 9):
        prepare_condition(condition, data, seed)
    jobs = runner.prepare_jobs(config, data, output)
    assert {(s.seed, s.data_seed) for s in jobs} == {(7, 7), (8, 8), (9, 9)}
    assert len(jobs) == 6


def test_ten_synthetic_main_seeds_create_ten_jobs(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    seeds = list(range(7, 17))
    condition = config.conditions[0].model_copy(update={"data_seeds": seeds})
    measurement = config.measurement.model_copy(update={"repeats": 1})
    config = config.model_copy(
        update={
            "conditions": [condition],
            "seeds": seeds,
            "measurement": measurement,
            "phase": "main",
        }
    )
    for seed in seeds[1:]:
        prepare_condition(condition, data, seed)
    jobs = runner.prepare_jobs(config, data, output)
    assert len(jobs) == 10
    assert {(s.seed, s.data_seed) for s in jobs} == {(seed, seed) for seed in seeds}


def test_worker_failure_before_attempt_preserves_other_jobs(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    jobs = runner.prepare_jobs(config, data, output)
    runner._crash_record(jobs[0], output, "worker_crash")
    runner.run_phase(config, data, output, phase="cpu", resume=True)
    failed = output / jobs[0].run_id / "attempt-0001/status.json"
    completed = output / jobs[1].run_id / "attempt-0001/status.json"
    assert json.loads(failed.read_text())["status"] == "failed"
    assert json.loads(completed.read_text())["status"] == "success"


def test_cpu_memory_limit_records_infeasible(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    resources = config.resources.model_copy(update={"host_memory_limit_bytes": 1})
    config = config.model_copy(update={"resources": resources})
    jobs = runner.run_phase(config, data, output, phase="cpu", resume=True)
    status = json.loads((output / jobs[0].run_id / "attempt-0001/status.json").read_text())
    assert status["status"] == "infeasible"
    assert status["reason_code"] == "memory_limit"


def test_input_cache_rejects_modified_data(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, _ = _prepare(tmp_path)
    cache = runner._InputCache(data)
    dataset, _, _, _ = cache.get(config.conditions[0], 7, 7)
    train_path = dataset.manifest_path.parent / "train.npy"
    modified = np.load(train_path, mmap_mode="r+")
    modified[0, 0] += 1
    modified.flush()
    with pytest.raises(runner.FrozenIdentityError, match="frozen identity"):
        cache.get(config.conditions[0], 7, 7)


def test_parallel_sentinel_shares_phase_lock_and_excludes_same_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    members = [f"GPU-{index}" for index in range(8)]
    sentinel = SentinelConfig(group_id="test-group", participants=8, members=members)
    measurement = config.measurement.model_copy(
        update={"cohort": "sentinel-parallel-GPU-0", "sentinel": sentinel}
    )
    resources = config.resources.model_copy(update={"host_memory_limit_bytes": 48 * 1024**3})
    method = config.methods[0].model_copy(update={"method": "torch_lloyd"})
    config = config.model_copy(
        update={"measurement": measurement, "resources": resources, "methods": [method]}
    )
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: f"GPU-{handle}",
        nvmlDeviceGetCount=lambda: 8,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(runner, "_hardware_fingerprint", lambda config: {"gpu": "test"})

    def inspect_locks(*args: object, **kwargs: object) -> None:
        with (data / ".emeans-measurement.lock").open("a+b") as phase_lock:
            fcntl.flock(phase_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                fcntl.flock(phase_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (data / ".emeans-gpu-0.lock").open("a+b") as gpu_lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    monkeypatch.setattr(runner, "_gpu_phase", inspect_locks)
    monkeypatch.setattr(
        runner,
        "SentinelGroup",
        lambda *args: SimpleNamespace(
            join=lambda: None, finish=lambda: None, close=lambda: None, abort=lambda reason: None
        ),
    )
    runner.run_phase(config, data, output, phase="gpu", devices=[0], resume=True)


def test_parallel_sentinel_rejects_wrong_gpu_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    members = [f"GPU-{index}" for index in range(8)]
    sentinel = SentinelConfig(group_id="test-group", participants=8, members=members)
    measurement = config.measurement.model_copy(
        update={"cohort": "sentinel-parallel-GPU-wrong", "sentinel": sentinel}
    )
    resources = config.resources.model_copy(update={"host_memory_limit_bytes": 48 * 1024**3})
    config = config.model_copy(update={"measurement": measurement, "resources": resources})
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda index: index,
            nvmlDeviceGetUUID=lambda handle: f"GPU-{handle}",
            nvmlDeviceGetCount=lambda: 8,
        ),
    )
    with pytest.raises(ValueError, match="exact GPU UUID"):
        runner.run_phase(config, data, output, phase="gpu", devices=[0])


def test_parallel_sentinel_rejects_excessive_host_limit(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    sentinel = SentinelConfig(
        group_id="test-group", participants=8, members=[f"GPU-{index}" for index in range(8)]
    )
    measurement = config.measurement.model_copy(
        update={"cohort": "sentinel-parallel-GPU-0", "sentinel": sentinel}
    )
    config = config.model_copy(update={"measurement": measurement})
    with pytest.raises(ValueError, match="48 GiB"):
        runner.run_phase(config, data, output, phase="gpu", devices=[0])


def test_error_log_redacts_secret_in_exception_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from project_module.e_means import methods, runner
    from project_module.e_means.data import load_dataset

    token = "fakecanary_secret_8675309"
    monkeypatch.setenv("HF_TOKEN", token)
    config, data, output = _prepare(tmp_path)
    job = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError(f"backend failed token={token}")

    monkeypatch.setattr(methods, "train", fail)
    attempt = runner._trial(
        job,
        dataset,
        output,
        config,
        resume=True,
        retry_failed=False,
        retry_reason=None,
        gpu=None,
        warm=set(),
        minimum_budget=1,
    )
    log = (attempt / "error.log").read_text()
    assert "ValueError" in log and "backend failed" in log and "[REDACTED]" in log
    assert token not in log
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".log"}:
            assert token not in path.read_text()


def test_spawn_sets_blas_threads_before_numpy_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import multiprocessing as mp
    import os
    import subprocess
    import sys

    from project_module.e_means import runner

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "8")
    ctx = mp.get_context("spawn")
    target = tmp_path / "threads.json"
    script = "import json,os,sys,numpy; from threadpoolctl import threadpool_info; json.dump([[os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')],[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']],open(sys.argv[1],'w'))"
    worker = runner._spawn_with_threads(
        ctx, subprocess.run, ([sys.executable, "-c", script, str(target)],), 1
    )
    worker.join(timeout=20)
    assert worker.exitcode == 0
    actual, pools = json.loads(target.read_text())
    assert set(actual) == {"1"}
    assert pools and set(pools) == {1}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"


@pytest.mark.parametrize(
    "changed", [{"gpu_model": "A100", "driver": "535"}, {"gpu_model": "V100", "driver": "550"}]
)
def test_hardware_change_rejects_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: dict[str, str]
) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    monkeypatch.setattr(
        runner,
        "_hardware_fingerprint",
        lambda config: {"gpu_model": "V100", "driver": "535"},
        raising=False,
    )
    runner.prepare_jobs(config, data, output)
    monkeypatch.setattr(
        runner,
        "_hardware_fingerprint",
        lambda config: changed,
        raising=False,
    )
    with pytest.raises(runner.FrozenIdentityError, match="hardware snapshot"):
        runner.prepare_jobs(config, data, output)


def test_changed_frozen_run_inputs_are_rejected(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    jobs = runner.prepare_jobs(config, data, output)
    path = output / "run-inputs.json"
    inputs = json.loads(path.read_text())
    inputs[jobs[0].run_id]["init_hash"] = "0" * 64
    path.write_text(json.dumps(inputs))
    with pytest.raises(runner.FrozenIdentityError, match="run inputs"):
        runner.prepare_jobs(config, data, output)


def test_success_copies_and_validates_frozen_manifest(tmp_path: Path) -> None:
    import hashlib

    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    jobs = runner.run_phase(config, data, output, phase="cpu", resume=True)
    attempt = output / jobs[0].run_id / "attempt-0001"
    inputs = json.loads((output / "run-inputs.json").read_text())[jobs[0].run_id]
    manifest = attempt / "dataset-manifest.json"
    metrics = json.loads((attempt / "metrics.json").read_text())
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        == inputs["manifest_hash"]
        == metrics["manifest_hash"]
    )
    manifest.write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="frozen identity"):
        runner.run_phase(config, data, output, phase="cpu", resume=True)


def _main_parallel_config() -> ExperimentConfig:
    config = _config()
    return config.model_copy(
        update={
            "phase": "main",
            "measurement": config.measurement.model_copy(update={"cohort": "main-parallel"}),
            "resources": config.resources.model_copy(
                update={"cpu_threads": 16, "cpu_affinity": list(range(16)), "gpu_pool": [0, 1]}
            ),
        }
    )


@pytest.mark.parametrize("devices", [[0], [1]])
def test_main_parallel_rejects_partial_gpu_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, devices: list[int]
) -> None:
    from project_module.e_means import runner

    monkeypatch.setattr(runner, "prepare_jobs", lambda *args: [])
    with pytest.raises(ValueError, match="main-parallel requires the complete GPU pool"):
        runner.run_phase(
            _main_parallel_config(),
            tmp_path / "data",
            tmp_path / "runs",
            phase="gpu",
            devices=devices,
            dry_run=True,
        )


def test_main_parallel_accepts_reordered_complete_gpu_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from project_module.e_means import runner

    monkeypatch.setattr(runner, "prepare_jobs", lambda *args: [])
    assert (
        runner.run_phase(
            _main_parallel_config(),
            tmp_path / "data",
            tmp_path / "runs",
            phase="gpu",
            devices=[1, 0],
            dry_run=True,
        )
        == []
    )


def test_changed_code_aborts_phase_without_terminal_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import multiprocessing as mp

    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    jobs = runner.prepare_jobs(config, data, output)
    original = runner._code_identity
    monkeypatch.setattr(
        runner, "_code_identity", lambda: {**original(), "source_sha256": "changed"}
    )
    original_context = mp.get_context
    monkeypatch.setattr(runner.mp, "get_context", lambda method: original_context("fork"))
    with pytest.raises(RuntimeError, match="frozen identity"):
        runner._cpu_phase(jobs, config, data, output, True, False, None)
    assert not any(output.glob("*/attempt-*"))


def test_modified_dataset_aborts_phase_without_terminal_attempts(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    jobs = runner.prepare_jobs(config, data, output)
    train_path = data / "processed/synthetic/tiny/7/train.npy"
    array = np.load(train_path, mmap_mode="r+")
    array[0, 0] += 1
    array.flush()
    with pytest.raises(RuntimeError, match="frozen identity"):
        runner._cpu_phase(jobs, config, data, output, True, False, None)
    assert not any(output.glob("*/attempt-*"))


def test_hardware_change_during_phase_aborts_without_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    monkeypatch.setattr(runner, "_hardware_fingerprint", lambda config: {"driver": "535"})
    jobs = runner.prepare_jobs(config, data, output)
    monkeypatch.setattr(runner, "_hardware_fingerprint", lambda config: {"driver": "550"})
    with pytest.raises(runner.FrozenIdentityError, match="frozen identity"):
        runner._cpu_phase(jobs, config, data, output, True, False, None)
    assert not any(output.glob("*/attempt-*"))


def test_code_change_after_training_marks_attempt_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from project_module.e_means import methods, runner
    from project_module.e_means.data import load_dataset

    config, data, output = _prepare(tmp_path)
    job = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
    original_code = runner._code_identity
    original_train = methods.train
    state = {"calls": 0, "changed": False}

    def code() -> dict[str, object]:
        current = original_code()
        if state["changed"]:
            current["source_sha256"] = "changed"
        return current

    def train(*args: object, **kwargs: object) -> object:
        state["calls"] += 1
        result = original_train(*args, **kwargs)
        if state["calls"] == 2:
            state["changed"] = True
        return result

    monkeypatch.setattr(runner, "_code_identity", code)
    monkeypatch.setattr(methods, "train", train)
    with pytest.raises(runner.FrozenIdentityError, match="source changed"):
        runner._trial(
            job,
            dataset,
            output,
            config,
            resume=True,
            retry_failed=False,
            retry_reason=None,
            gpu=None,
            warm=set(),
            minimum_budget=1,
        )
    status = json.loads((output / job.run_id / "attempt-0001/status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["reason_code"] == "frozen_identity_changed"


def test_resident_pipeline_excludes_slow_evaluation_and_times_train_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from project_module.e_means import runner
    from project_module.e_means.data import load_dataset

    config, data, output = _prepare(tmp_path)
    job = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
    original = runner._evaluation

    def slow(*args: object, **kwargs: object) -> object:
        time.sleep(0.15)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_evaluation", slow)
    attempt = runner._trial(
        job,
        dataset,
        output,
        config,
        resume=True,
        retry_failed=False,
        retry_reason=None,
        gpu=None,
        warm=set(),
        minimum_budget=1,
    )
    metrics = json.loads((attempt / "metrics.json").read_text())
    timing = metrics["timing"]
    assert timing["evaluate_seconds"] >= 0.15
    assert timing["e2e_seconds"] < timing["evaluate_seconds"]
    assert timing["assign_seconds"] > 0
    assert timing["metadata"]["scope_version"] == "resident-input-pipeline-v1"


def test_gpu_host_memory_sum_aborts_and_interrupts_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import queue
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner
    from project_module.e_means.contracts import RunSpec

    config, data, output = _prepare(tmp_path)
    config = config.model_copy(
        update={"resources": config.resources.model_copy(update={"host_memory_limit_bytes": 1})}
    )
    jobs = runner.prepare_jobs(config, data, output)
    keys = json.loads((output / "run-keys.json").read_text())

    class FakeQueue:
        def put(self, item: object) -> None:
            if isinstance(item, RunSpec):
                path = output / item.run_id / "attempt-0001"
                path.mkdir(parents=True)
                runner._write_json(path / "spec.json", item.model_dump(mode="json"))
                runner._record(path, item, keys[item.run_id], 1, "running")

        def get(self, timeout: float) -> object:
            raise queue.Empty

    class FakeProcess:
        pid = 123
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            pass

    monkeypatch.setattr(runner.mp, "get_context", lambda method: SimpleNamespace(Queue=FakeQueue))
    monkeypatch.setattr(runner, "_spawn_with_threads", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        runner.psutil,
        "Process",
        lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100)),
    )
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda index: index,
            nvmlDeviceGetUUID=lambda handle: f"GPU-{handle}",
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=0),
        ),
    )
    with pytest.raises(RuntimeError, match="host memory sum exceeded"):
        runner._gpu_phase(jobs, [0], config, data, output, True, False, None)
    status = json.loads((output / jobs[0].run_id / "attempt-0001/status.json").read_text())
    assert status["status"] == "interrupted"
    assert not (output / jobs[1].run_id).exists()


def test_sentinel_hook_runs_once_before_measured_training(tmp_path: Path) -> None:
    import time

    from project_module.e_means import runner
    from project_module.e_means.data import load_dataset

    class FakeSentinel:
        calls: list[tuple[dict[str, object], dict[str, object]]] = []

        def wait(self, trial: dict[str, object], inputs: dict[str, object]) -> float:
            self.calls.append((trial, inputs))
            begin = time.perf_counter()
            time.sleep(0.05)
            return time.perf_counter() - begin

        def abort(self, reason: str) -> None:
            raise AssertionError(reason)

    config, data, output = _prepare(tmp_path)
    job = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
    sentinel = FakeSentinel()
    attempt = runner._trial(
        job,
        dataset,
        output,
        config,
        resume=True,
        retry_failed=False,
        retry_reason=None,
        gpu=None,
        warm=set(),
        minimum_budget=1,
        sentinel=sentinel,
    )
    metrics = json.loads((attempt / "metrics.json").read_text())
    timing = metrics["timing"]
    assert len(sentinel.calls) == 1
    assert sentinel.calls[0][0]["condition_id"] == "tiny"
    assert sentinel.calls[0][1]["init_hash"] == metrics["init_hash"]
    assert timing["metadata"]["coordination_wait_seconds"] >= 0.05
    assert timing["e2e_seconds"] >= timing["metadata"]["coordination_wait_seconds"]
    assert timing["train_seconds"] < timing["metadata"]["coordination_wait_seconds"]
    assert timing["metadata"]["train_started_at_ns"] < timing["metadata"]["train_finished_at_ns"]


def test_missing_sentinel_hook_interrupts_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from project_module.e_means import methods, runner
    from project_module.e_means.data import load_dataset

    class FakeSentinel:
        reason: str | None = None

        def abort(self, reason: str) -> None:
            self.reason = reason

    config, data, output = _prepare(tmp_path)
    job = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
    original = methods.train

    def without_hook(x: object, init: object, spec: object, **kwargs: object) -> object:
        return original(x, init, spec)

    monkeypatch.setattr(methods, "train", without_hook)
    sentinel = FakeSentinel()
    with pytest.raises(runner.SentinelCoordinationError, match="hook was not called"):
        runner._trial(
            job,
            dataset,
            output,
            config,
            resume=True,
            retry_failed=False,
            retry_reason=None,
            gpu=None,
            warm=set(),
            minimum_budget=1,
            sentinel=sentinel,
        )
    status = json.loads((output / job.run_id / "attempt-0001/status.json").read_text())
    assert status["status"] == "interrupted"
    assert sentinel.reason == "sentinel_missing_hook"


def test_parallel_sentinel_requires_frozen_eight_members(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    measurement = config.measurement.model_copy(update={"cohort": "sentinel-parallel-GPU-0"})
    config = config.model_copy(update={"measurement": measurement})
    with pytest.raises(ValueError, match="eight sorted unique"):
        runner.run_phase(config, data, output, phase="gpu", devices=[0])


def test_direct_cuda_trial_is_rejected(tmp_path: Path) -> None:
    from project_module.e_means import runner
    from project_module.e_means.data import load_dataset

    config, data, output = _prepare(tmp_path)
    job = runner.prepare_jobs(config, data, output)[0].model_copy(update={"device": "cuda"})
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
    with pytest.raises(ValueError, match="monitored GPU phase"):
        runner.run_trial(job, dataset, output)


def test_pilot_real_dataset_requires_pilot_condition_id(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    condition = config.conditions[0].model_copy(update={"dataset_id": "arxiv_nomic"})
    config = config.model_copy(update={"conditions": [condition]})
    with pytest.raises(ValueError, match="condition ID must match pilot"):
        runner.prepare_jobs(config, data, output)


def test_train_group_diagnostics_are_excluded_from_assignment_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from project_module.e_means import evaluation, runner
    from project_module.e_means.data import load_dataset

    config, data, output = _prepare(tmp_path)
    job = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
    original = evaluation.evaluate_groups

    def slow_groups(*args: object, **kwargs: object) -> object:
        time.sleep(0.15)
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_groups", slow_groups)
    attempt = runner._trial(
        job,
        dataset,
        output,
        config,
        resume=True,
        retry_failed=False,
        retry_reason=None,
        gpu=None,
        warm=set(),
        minimum_budget=1,
    )
    timing = json.loads((attempt / "metrics.json").read_text())["timing"]
    assert timing["assign_seconds"] > 0
    assert timing["evaluate_seconds"] >= 0.15
    assert timing["assign_seconds"] < timing["evaluate_seconds"]


@pytest.mark.parametrize(("enabled", "expected"), [(True, ":4096:8"), (False, None)])
def test_gpu_worker_spawn_sets_workspace_before_import_and_restores_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, expected: str | None
) -> None:
    import multiprocessing as mp
    import os
    import subprocess
    import sys

    from project_module.e_means import runner

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    target = tmp_path / "workspace.json"
    script = "import json,os,sys; json.dump(os.environ.get('CUBLAS_WORKSPACE_CONFIG'),open(sys.argv[1],'w'))"
    process = runner._spawn_with_threads(
        mp.get_context("spawn"),
        subprocess.run,
        ([sys.executable, "-c", script, str(target)],),
        1,
        gpu_deterministic_algorithms=enabled,
    )
    process.join(20)
    assert process.exitcode == 0
    assert json.loads(target.read_text()) == expected
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_gpu_determinism_changes_frozen_config_identity(tmp_path: Path) -> None:
    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    stock_jobs = runner.prepare_jobs(config, data, output)
    resources = config.resources.model_copy(update={"gpu_deterministic_algorithms": True})
    changed = config.model_copy(update={"resources": resources})
    with pytest.raises(runner.FrozenIdentityError, match="snapshot"):
        runner.prepare_jobs(changed, data, output)
    deterministic_root = tmp_path / "deterministic"
    deterministic_jobs = runner.prepare_jobs(changed, data, deterministic_root)
    assert stock_jobs[0].run_id != deterministic_jobs[0].run_id
    stock_hardware = json.loads((output / "hardware.snapshot.json").read_text())
    det_hardware = json.loads((deterministic_root / "hardware.snapshot.json").read_text())
    assert stock_hardware["gpu_runtime_profile"]["cublas_workspace_requested"] is None
    assert det_hardware["gpu_runtime_profile"]["cublas_workspace_requested"] == ":4096:8"


def test_legacy_gpu_profile_records_disabled_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            are_deterministic_algorithms_enabled=lambda: False,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
        ),
    )
    state = runner._gpu_runtime_state(False)
    assert state["deterministic_algorithms_requested"] is False
    assert state["deterministic_algorithms_enabled"] is False
    assert state["cublas_workspace_config"] is None


@pytest.mark.parametrize(
    ("actual", "warn_only", "workspace"),
    [
        (False, False, ":4096:8"),
        (True, True, ":4096:8"),
        (True, False, ":16:8"),
    ],
)
def test_gpu_profile_drift_aborts(
    actual: bool, warn_only: bool, workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            are_deterministic_algorithms_enabled=lambda: actual,
            is_deterministic_algorithms_warn_only_enabled=lambda: warn_only,
        ),
    )
    with pytest.raises(runner.FrozenIdentityError, match="GPU runtime profile"):
        runner._gpu_runtime_state(True)


@pytest.mark.parametrize("requested", [False, True])
def test_gpu_profile_configures_strict_determinism(
    monkeypatch: pytest.MonkeyPatch, requested: bool
) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    state = {"enabled": not requested, "warn_only": True}

    def configure(enabled: bool, *, warn_only: bool) -> None:
        state.update(enabled=enabled, warn_only=warn_only)

    if requested:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            use_deterministic_algorithms=configure,
            are_deterministic_algorithms_enabled=lambda: state["enabled"],
            is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
        ),
    )
    result = runner._configure_gpu_runtime(requested)
    assert result["deterministic_algorithms_requested"] is requested
    assert result["deterministic_algorithms_enabled"] is requested
    assert result["deterministic_warn_only"] is False
    assert result["cublas_workspace_config"] == (":4096:8" if requested else None)


def test_gpu_profile_drift_during_training_interrupts_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import methods, runner
    from project_module.e_means.data import load_dataset

    config, data, output = _prepare(tmp_path)
    resources = config.resources.model_copy(update={"gpu_deterministic_algorithms": True})
    config = config.model_copy(update={"resources": resources})
    spec = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)
    state = {"enabled": False, "calls": 0}
    original = methods.train

    def configure(enabled: bool, *, warn_only: bool) -> None:
        assert warn_only is False
        state["enabled"] = enabled

    def train(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        state["calls"] += 1
        if state["calls"] == 2:
            state["enabled"] = False
        return result

    class Monitor:
        rss = 0
        gpu_memory = 0

        def __init__(self, gpu: object) -> None:
            pass

        def __enter__(self) -> "Monitor":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    cuda = SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None, reset_peak_memory_stats=lambda: None
    )
    torch = SimpleNamespace(
        cuda=cuda,
        use_deterministic_algorithms=configure,
        are_deterministic_algorithms_enabled=lambda: state["enabled"],
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        set_num_threads=lambda value: None,
        get_num_interop_threads=lambda: 1,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(methods, "train", train)
    monkeypatch.setattr(runner, "_Monitor", Monitor)
    with pytest.raises(runner.FrozenIdentityError, match="GPU runtime profile"):
        runner._trial(
            spec,
            dataset,
            output,
            config,
            resume=True,
            retry_failed=False,
            retry_reason=None,
            gpu=0,
            warm=set(),
            minimum_budget=1,
            gpu_runtime_initialized={"initialized": False},
        )
    status = json.loads((output / spec.run_id / "attempt-0001/status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["reason_code"] == "frozen_identity_changed"


def test_cpu_only_run_does_not_import_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    from project_module.e_means import runner
    from project_module.e_means.data import load_dataset

    original_import = builtins.__import__

    def reject_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("CPU-only execution imported Torch")
        return original_import(name, *args, **kwargs)

    config, data, output = _prepare(tmp_path)
    monkeypatch.setattr(builtins, "__import__", reject_torch)
    spec = runner.prepare_jobs(config, data, output)[0]
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
    )
    assert json.loads((attempt / "status.json").read_text())["status"] == "success"


def test_gpu_profile_records_matmul_and_uninitialized_memory_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            are_deterministic_algorithms_enabled=lambda: True,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
            get_float32_matmul_precision=lambda: "highest",
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))
            ),
            utils=SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=True)),
        ),
    )
    profile = runner._gpu_runtime_state(True)
    assert profile["cuda_matmul_allow_tf32"] is False
    assert profile["float32_matmul_precision"] == "highest"
    assert profile["fill_uninitialized_memory"] is True


def test_unsupported_deterministic_operation_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from project_module.e_means import methods, runner
    from project_module.e_means.data import load_dataset

    config, data, output = _prepare(tmp_path)
    config = config.model_copy(
        update={
            "resources": config.resources.model_copy(update={"gpu_deterministic_algorithms": True})
        }
    )
    spec = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)

    def unsupported(*args: object, **kwargs: object) -> object:
        raise RuntimeError("operation does not have a deterministic implementation")

    monkeypatch.setattr(methods, "train", unsupported)
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
    )
    status = json.loads((attempt / "status.json").read_text())
    assert status["status"] == "unsupported"
    assert status["reason_code"] == "deterministic_unsupported"


def test_deterministic_unsupported_worker_reports_phase_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import queue

    from project_module.e_means import runner

    config, data, output = _prepare(tmp_path)
    config = config.model_copy(
        update={
            "resources": config.resources.model_copy(update={"gpu_deterministic_algorithms": True})
        }
    )
    spec = runner.prepare_jobs(config, data, output)[0]
    path = output / spec.run_id / "attempt-0001"
    path.mkdir(parents=True)
    key = json.loads((output / "run-keys.json").read_text())[spec.run_id]
    runner._record(path, spec, key, 1, "unsupported", "deterministic_unsupported")
    monkeypatch.setattr(runner, "_trial", lambda *args, **kwargs: path)
    monkeypatch.setattr(runner.os, "sched_setaffinity", lambda *args: None)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "test")
    requests: queue.Queue = queue.Queue()
    responses: queue.Queue = queue.Queue()
    requests.put(spec)
    requests.put(None)
    runner._worker_loop(
        0, "GPU-test", config, data, output, requests, responses, True, False, None, 0, {}, None
    )
    _, run_id, error, _ = responses.get_nowait()
    assert run_id == spec.run_id
    assert error == "DeterministicProfileUnsupported"
    assert json.loads((path / "status.json").read_text())["status"] == "unsupported"


def test_missing_determinism_api_aborts_as_frozen_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(runner.FrozenIdentityError, match="GPU runtime profile"):
        runner._gpu_runtime_state(True)


def test_sentinel_deterministic_unsupported_interrupts_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from project_module.e_means import methods, runner
    from project_module.e_means.data import load_dataset

    class FakeSentinel:
        reason: str | None = None

        def abort(self, reason: str) -> None:
            self.reason = reason

    config, data, output = _prepare(tmp_path)
    config = config.model_copy(
        update={
            "resources": config.resources.model_copy(update={"gpu_deterministic_algorithms": True})
        }
    )
    spec = runner.prepare_jobs(config, data, output)[0]
    dataset = load_dataset("synthetic", data, condition_id="tiny", data_seed=7)

    def unsupported(*args: object, **kwargs: object) -> object:
        raise RuntimeError("operation does not have a deterministic implementation")

    monkeypatch.setattr(methods, "train", unsupported)
    sentinel = FakeSentinel()
    with pytest.raises(runner.SentinelCoordinationError, match="deterministic"):
        runner._trial(
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
            sentinel=sentinel,
        )
    status = json.loads((output / spec.run_id / "attempt-0001/status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["reason_code"] == "sentinel_deterministic_unsupported"
    assert sentinel.reason == "sentinel_deterministic_unsupported"


def test_missing_determinism_setter_aborts_as_frozen_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    from project_module.e_means import runner

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(runner.FrozenIdentityError, match="GPU runtime profile"):
        runner._configure_gpu_runtime(True)
