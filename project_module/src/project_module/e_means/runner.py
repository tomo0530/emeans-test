"""Immutable experiment ledger, attempt publication and phase execution."""

import hashlib
import importlib.metadata
import io
import json
import logging
import multiprocessing as mp
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil

from project_module.e_means._sentinel import SentinelCoordinationError, SentinelGroup
from project_module.e_means.contracts import (
    DatasetManifest,
    Evaluation,
    EvaluationMetrics,
    ExperimentConfig,
    PreparedDataset,
    RunMetrics,
    RunRecord,
    RunSpec,
    TimingBreakdown,
)
from project_module.e_means.data import load_dataset
from project_module.e_means.initialization import make_initial_centers

_GPU_METHODS = {"torch_e_means", "torch_lloyd"}
_SAMPLE_SECONDS = 0.1
_DETERMINISTIC_WORKSPACE = ":4096:8"


class FrozenIdentityError(RuntimeError):
    """A frozen input, environment, or existing artifact no longer agrees."""


class DeterministicProfileUnsupported(RuntimeError):
    """The required deterministic profile cannot execute a planned operation."""


def _frozen_files(root: Path) -> dict[str, bytes]:
    try:
        return {
            name: (root / name).read_bytes()
            for name in (
                "config.snapshot.json",
                "hardware.snapshot.json",
                "jobs.json",
                "run-keys.json",
                "run-inputs.json",
            )
        }
    except OSError as exc:
        raise FrozenIdentityError("frozen identity files are unavailable") from exc


def _check_frozen_hardware(
    config: ExperimentConfig, root: Path, frozen_files: dict[str, bytes]
) -> None:
    try:
        if (
            _frozen_files(root) != frozen_files
            or _canonical(_hardware_fingerprint(config)) != frozen_files["hardware.snapshot.json"]
        ):
            raise FrozenIdentityError("frozen identity hardware or ledger changed")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise FrozenIdentityError("frozen identity hardware or ledger changed") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic(path, _canonical(value))


def _sanitize_error(exc: BaseException) -> str:
    text = "".join(traceback.format_exception(exc))
    sensitive = (
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "API_KEY",
        "ACCESS_KEY",
        "PRIVATE_KEY",
        "CREDENTIAL",
    )
    values = [
        value
        for name, value in os.environ.items()
        if value and any(part in name.upper() for part in sensitive)
    ]
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return re.sub(
        r"(?i)(token|secret|password|api[_-]?key|credential)([\"'\s:=]+)([^\s,;\"']+)",
        r"\1\2[REDACTED]",
        text,
    )


def _code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    source = root / "project_module/src/project_module/e_means"
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        if path.name == "analysis.py":
            continue
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    cli = root / "scripts/e_means/run.py"
    digest.update(cli.read_bytes())
    lock = root / "uv.lock"
    versions: dict[str, str | None] = {}
    for name in ("faiss-cpu", "torch", "numpy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, check=True
        ).stdout
    except OSError, subprocess.CalledProcessError:
        head, dirty = None, b""
    return {
        "source_sha256": digest.hexdigest(),
        "git_head": head,
        "dirty_diff_sha256": _sha(dirty),
        "lock_sha256": _sha(lock.read_bytes()) if lock.exists() else None,
        "versions": versions,
    }


def _hardware_fingerprint(config: ExperimentConfig) -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo").read_text()
    models = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        }
    )
    cpus = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 0))
    )
    topology = []
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        topology.append(
            {
                "cpu": cpu,
                "package": (path / "topology/physical_package_id").read_text().strip(),
                "core": (path / "topology/core_id").read_text().strip(),
                "nodes": sorted(p.name for p in path.glob("node[0-9]*")),
            }
        )
    result: dict[str, Any] = {
        "cpu_models": models,
        "cpu_topology": topology,
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": sys.version,
        "gpu_count": 0,
        "gpu_pool": [],
        "driver": None,
        "cuda_runtime_distribution": None,
        "torch_distribution": None,
        "gpu_runtime_profile": _gpu_runtime_requested(
            config.resources.gpu_deterministic_algorithms
        ),
    }
    for name, key in (
        ("nvidia-cuda-runtime-cu12", "cuda_runtime_distribution"),
        ("torch", "torch_distribution"),
    ):
        try:
            result[key] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if any(method.method in _GPU_METHODS for method in config.methods):
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                result["driver"] = pynvml.nvmlSystemGetDriverVersion()
                result["gpu_count"] = pynvml.nvmlDeviceGetCount()
                devices = []
                for index in config.resources.gpu_pool:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    devices.append(
                        {
                            "model": pynvml.nvmlDeviceGetName(handle),
                            "compute_capability": list(
                                pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                            ),
                            "memory_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                        }
                    )
                result["gpu_pool"] = sorted(devices, key=lambda d: _canonical(d))
            finally:
                pynvml.nvmlShutdown()
        except Exception as exc:
            raise RuntimeError("cannot determine GPU hardware fingerprint") from exc
    return result


def _gpu_runtime_requested(enabled: bool) -> dict[str, Any]:
    return {
        "deterministic_algorithms_requested": enabled,
        "deterministic_warn_only_requested": False,
        "cublas_workspace_requested": _DETERMINISTIC_WORKSPACE if enabled else None,
        "graph_policy": "upstream_auto",
    }


def _gpu_runtime_state(enabled: bool) -> dict[str, Any]:
    import torch

    try:
        actual = {
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise FrozenIdentityError("GPU runtime profile cannot be observed") from exc
    if (
        actual["deterministic_algorithms_enabled"],
        actual["deterministic_warn_only"],
        actual["cublas_workspace_config"],
    ) != (enabled, False, _DETERMINISTIC_WORKSPACE if enabled else None):
        raise FrozenIdentityError("GPU runtime profile differs from the frozen request")
    try:
        actual["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    except AttributeError, RuntimeError:
        actual["cuda_matmul_allow_tf32"] = None
    try:
        actual["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except AttributeError, RuntimeError:
        actual["float32_matmul_precision"] = None
    try:
        actual["fill_uninitialized_memory"] = torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
    except AttributeError, RuntimeError:
        actual["fill_uninitialized_memory"] = None
    return {**_gpu_runtime_requested(enabled), **actual}


def _configure_gpu_runtime(enabled: bool) -> dict[str, Any]:
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (_DETERMINISTIC_WORKSPACE if enabled else None):
        raise FrozenIdentityError("GPU runtime profile workspace differs before initialization")
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=False)
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise FrozenIdentityError("GPU runtime profile could not be configured") from exc
    return _gpu_runtime_state(enabled)


def _spawn_with_threads(
    context: Any,
    target: Any,
    args: tuple[Any, ...],
    threads: int,
    *,
    gpu_deterministic_algorithms: bool | None = None,
) -> Any:
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    process = context.Process(target=target, args=args)
    try:
        for name in names:
            os.environ[name] = str(threads)
        if gpu_deterministic_algorithms is True:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _DETERMINISTIC_WORKSPACE
        elif gpu_deterministic_algorithms is False:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        process.start()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if gpu_deterministic_algorithms is not None:
            if previous_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
    return process


def _manifest(
    condition: Any, data_seed: int, data_root: Path
) -> tuple[PreparedDataset, DatasetManifest]:
    for value in (condition.dataset_id, condition.condition_id):
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError("unsafe dataset or condition identifier")
    dataset = load_dataset(
        condition.dataset_id, data_root, condition_id=condition.condition_id, data_seed=data_seed
    )
    manifest = DatasetManifest.model_validate_json(dataset.manifest_path.read_bytes())
    if (manifest.n_train, manifest.dimension, manifest.generator_params) != (
        condition.n_train,
        condition.dimension,
        condition.generator_params,
    ):
        raise ValueError("manifest condition mismatch")
    return dataset, manifest


def _dataset_signature(dataset: PreparedDataset) -> tuple[tuple[str, int, int, int], ...]:
    directory = dataset.manifest_path.parent
    paths = sorted(directory.glob("*.npy")) + [dataset.manifest_path]
    return tuple(
        (path.name, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    )


class _InputCache:
    """Keep at most two verified read-only datasets and two initial matrices."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.datasets: OrderedDict[
            tuple[str, int], tuple[PreparedDataset, tuple[tuple[str, int, int, int], ...]]
        ] = OrderedDict()
        self.initial: OrderedDict[tuple[str, int, int, int], np.ndarray] = OrderedDict()

    def get(
        self, condition: Any, data_seed: int, seed: int
    ) -> tuple[PreparedDataset, np.ndarray, bool, bool]:
        dataset_key = (condition.condition_id, data_seed)
        entry = self.datasets.get(dataset_key)
        try:
            hit = entry is not None and _dataset_signature(entry[0]) == entry[1]
            if not hit:
                dataset, _ = _manifest(condition, data_seed, self.data_root)
                self.datasets[dataset_key] = (dataset, _dataset_signature(dataset))
        except (ValueError, OSError) as exc:
            raise FrozenIdentityError("frozen identity dataset no longer matches") from exc
        if not hit:
            for key in list(self.initial):
                if key[:2] == dataset_key:
                    del self.initial[key]
        else:
            assert entry is not None
            dataset = entry[0]
        self.datasets.move_to_end(dataset_key)
        if len(self.datasets) > 2:
            evicted, _ = self.datasets.popitem(last=False)
            for key in list(self.initial):
                if key[:2] == evicted:
                    del self.initial[key]
        init_key = (*dataset_key, condition.k, seed)
        init = self.initial.get(init_key)
        init_hit = init is not None
        if init is None:
            init = make_initial_centers(dataset.train, condition.k, seed)
            self.initial[init_key] = init
        self.initial.move_to_end(init_key)
        if len(self.initial) > 2:
            self.initial.popitem(last=False)
        return dataset, init, hit, init_hit


def _identity(
    spec: RunSpec,
    config_hash: str,
    dataset: PreparedDataset,
    init_hash: str,
    code: dict[str, Any],
    hardware_hash: str,
) -> str:
    value = spec.model_dump(mode="json")
    value.pop("run_id")
    return _sha(
        _canonical(
            {
                "spec": value,
                "config_hash": config_hash,
                "manifest_hash": _sha(dataset.manifest_path.read_bytes()),
                "data_hash": dataset.data_hash,
                "split_hash": dataset.split_hash,
                "init_hash": init_hash,
                "hardware_hash": hardware_hash,
                "code": {key: value for key, value in code.items() if key != "dirty_diff_sha256"},
            }
        )
    )


def prepare_jobs(config: ExperimentConfig, data_root: Path, output_root: Path) -> list[RunSpec]:
    """Validate datasets and freeze the complete CPU/GPU job ledger.

    Parameters
    ----------
    config : ExperimentConfig
        Frozen experiment and measurement policy.
    data_root : Path
        Prepared input dataset root.
    output_root : Path
        Immutable snapshot and ledger destination.

    Returns
    -------
    list[RunSpec]
        All jobs, regardless of execution phase.
    """
    snapshot = _canonical(config.model_dump(mode="json"))
    snapshot_path = output_root / "config.snapshot.json"
    if snapshot_path.exists() and snapshot_path.read_bytes() != snapshot:
        raise FrozenIdentityError("config snapshot differs from existing snapshot")
    hardware = _canonical(_hardware_fingerprint(config))
    hardware_path = output_root / "hardware.snapshot.json"
    if hardware_path.exists() and hardware_path.read_bytes() != hardware:
        raise FrozenIdentityError("hardware snapshot differs from current environment")
    if len({c.condition_id for c in config.conditions}) != len(config.conditions):
        raise ValueError("duplicate condition")
    if len({m.method for m in config.methods}) != len(config.methods):
        raise ValueError("duplicate method")
    for condition in config.conditions:
        if condition.dataset_id != "synthetic" and condition.condition_id.startswith("pilot") != (
            config.phase == "pilot"
        ):
            raise ValueError("real dataset condition ID must match pilot phase")
    config_hash = _sha(snapshot)
    code = _code_identity()
    jobs: list[RunSpec] = []
    keys: dict[str, str] = {}
    inputs: dict[str, dict[str, str]] = {}
    ids: set[str] = set()
    for condition in config.conditions:
        paired = condition.dataset_id == "synthetic" and not (
            config.phase == "pilot" and len(condition.data_seeds) == 1
        )
        if paired and set(condition.data_seeds) != set(config.seeds):
            raise ValueError("synthetic data seeds must match experiment seeds")
        for data_seed in condition.data_seeds:
            try:
                dataset, _ = _manifest(condition, data_seed, data_root)
            except (OSError, ValueError) as exc:
                if snapshot_path.exists():
                    raise FrozenIdentityError("frozen identity dataset no longer matches") from exc
                raise
            for seed in config.seeds:
                if paired and seed != data_seed:
                    continue
                init = make_initial_centers(dataset.train, condition.k, seed)
                init_hash = _sha(init.tobytes())
                for method in config.methods:
                    device: Literal["cuda", "cpu"] = (
                        "cuda" if method.method in _GPU_METHODS else "cpu"
                    )
                    for budget in method.budgets:
                        rules = [
                            r
                            for r in config.measurement.repeat_rules
                            if (r.condition_id, r.method, r.budget)
                            == (condition.condition_id, method.method, budget)
                        ]
                        if len(rules) > 1:
                            raise ValueError("duplicate repeat rule")
                        repeats = rules[0].repeats if rules else config.measurement.repeats
                        for repeat in range(repeats):
                            spec = RunSpec(
                                run_id="",
                                dataset_id=condition.dataset_id,
                                condition_id=condition.condition_id,
                                method=method.method,
                                device=device,
                                measurement_id=config.measurement.cohort,
                                n_train=condition.n_train,
                                k=condition.k,
                                seed=seed,
                                data_seed=data_seed,
                                threads=1 if device == "cuda" else config.resources.cpu_threads,
                                timing_repeat=repeat,
                                budget=budget,
                                params=method.params,
                            )
                            key = _identity(
                                spec, config_hash, dataset, init_hash, code, _sha(hardware)
                            )
                            spec = spec.model_copy(update={"run_id": key[:20]})
                            if spec.run_id in ids:
                                raise ValueError("duplicate or colliding run id")
                            ids.add(spec.run_id)
                            keys[spec.run_id] = key
                            inputs[spec.run_id] = {
                                "config_hash": config_hash,
                                "manifest_hash": _sha(dataset.manifest_path.read_bytes()),
                                "init_hash": init_hash,
                            }
                            jobs.append(spec)
    encoded = _canonical([s.model_dump(mode="json") for s in jobs])
    ledger = output_root / "jobs.json"
    if ledger.exists() and ledger.read_bytes() != encoded:
        raise FrozenIdentityError("existing job ledger differs from current inputs")
    key_path = output_root / "run-keys.json"
    key_bytes = _canonical(keys)
    if key_path.exists() and key_path.read_bytes() != key_bytes:
        raise FrozenIdentityError("existing run keys differ from current inputs")
    input_path = output_root / "run-inputs.json"
    input_bytes = _canonical(inputs)
    if input_path.exists() and input_path.read_bytes() != input_bytes:
        raise FrozenIdentityError("existing run inputs differ from current inputs")
    output_root.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.exists():
        _atomic(snapshot_path, snapshot)
    if not hardware_path.exists():
        _atomic(hardware_path, hardware)
    if not ledger.exists():
        _atomic(ledger, encoded)
    if not key_path.exists():
        _atomic(key_path, key_bytes)
    if not input_path.exists():
        _atomic(input_path, input_bytes)
    return jobs


def _record(
    path: Path,
    spec: RunSpec,
    key: str,
    attempt: int,
    status: Literal[
        "pending", "running", "success", "failed", "infeasible", "unsupported", "interrupted"
    ],
    reason: str | None = None,
    started: datetime | None = None,
) -> None:
    record = RunRecord(
        run_key=key,
        status=status,
        attempt=attempt,
        reason_code=reason,
        started_at=started,
        finished_at=datetime.now(UTC) if status not in {"pending", "running"} else None,
        artifact_paths={
            name: name
            for name in (
                "spec.json",
                "environment.json",
                "metrics.json",
                "centroids.npy",
                "dataset-manifest.json",
            )
            if (path / name).is_file()
        },
    )
    _write_json(path / "status.json", record.model_dump(mode="json"))


def _evaluation(
    x: Any, centers: Any, groups: Any, *, result: Evaluation | None = None
) -> EvaluationMetrics:
    from project_module.e_means.evaluation import evaluate_centers, evaluate_groups

    if result is None:
        result = evaluate_centers(x, centers, chunk_rows=4096)
    group_result = (
        dict(evaluate_groups(x, centers, groups, chunk_rows=4096)) if groups is not None else None
    )
    return EvaluationMetrics(
        mse=result.mse,
        n_points=result.n_points,
        empty_clusters=result.empty_clusters,
        cluster_sizes=result.cluster_sizes.tolist(),
        group_mse=group_result,
    )


class _Monitor:
    def __init__(self, gpu: int | None) -> None:
        self.gpu = gpu
        self.rss = 0
        self.gpu_memory: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        handle = None
        nvml = None
        if self.gpu is not None:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu)
                nvml = pynvml
            except Exception:
                pass
        while not self._stop.is_set():
            self.rss = max(self.rss, process.memory_info().rss)
            if handle is not None and nvml is not None:
                try:
                    used = int(nvml.nvmlDeviceGetMemoryInfo(handle).used)
                    self.gpu_memory = max(self.gpu_memory or 0, used)
                except Exception:
                    pass
            self._stop.wait(_SAMPLE_SECONDS)
        if nvml is not None:
            nvml.nvmlShutdown()

    def __enter__(self) -> "_Monitor":
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        self._thread.join()


def _environment(
    spec: RunSpec,
    gpu: int | None,
    monitor: _Monitor,
    code: dict[str, Any],
    *,
    gpu_deterministic_algorithms: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": spec.threads,
        "device": spec.device,
        "gpu_index": gpu,
        "peak_rss_bytes": monitor.rss,
        "peak_gpu_memory_bytes": monitor.gpu_memory,
        "memory_sampling_seconds": _SAMPLE_SECONDS,
        "gpu_uuid": None,
        "gpu_name": None,
        "driver": None,
        "cuda_runtime": None,
    }
    if gpu is not None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            result.update(
                gpu_uuid=pynvml.nvmlDeviceGetUUID(handle),
                gpu_name=pynvml.nvmlDeviceGetName(handle),
                driver=pynvml.nvmlSystemGetDriverVersion(),
            )
            pynvml.nvmlShutdown()
            import torch
            from threadpoolctl import threadpool_info

            result["cuda_runtime"] = torch.version.cuda
            result["torch_num_threads"] = torch.get_num_threads()
            result["torch_num_interop_threads"] = torch.get_num_interop_threads()
            pools = [
                {
                    key: pool.get(key)
                    for key in ("user_api", "internal_api", "prefix", "version", "num_threads")
                }
                for pool in threadpool_info()
            ]
            result["threadpools"] = pools
        except Exception:
            pass
        if (
            result.get("torch_num_threads") != 1
            or result.get("torch_num_interop_threads") != 1
            or any(pool["num_threads"] != 1 for pool in result.get("threadpools", []))
        ):
            raise RuntimeError("GPU host thread limit was not enforced")
        result["gpu_runtime_profile"] = _gpu_runtime_state(gpu_deterministic_algorithms)
    return result


def _verify_success(
    path: Path,
    spec: RunSpec,
    key: str,
    config: ExperimentConfig,
    config_hash: str,
    manifest_hash: str,
    init_hash: str,
) -> None:
    if path.is_symlink() or any(
        (path / name).is_symlink()
        for name in (
            "spec.json",
            "metrics.json",
            "environment.json",
            "centroids.npy",
            "status.json",
            "dataset-manifest.json",
        )
    ):
        raise ValueError("artifact symlinks are forbidden")
    stored = RunSpec.model_validate_json((path / "spec.json").read_bytes())
    metrics = RunMetrics.model_validate_json((path / "metrics.json").read_bytes())
    manifest_bytes = (path / "dataset-manifest.json").read_bytes()
    manifest = DatasetManifest.model_validate_json(manifest_bytes)
    environment = json.loads((path / "environment.json").read_bytes())
    centers = np.load(path / "centroids.npy", allow_pickle=False)
    if (
        stored != spec
        or not isinstance(environment, dict)
        or (
            metrics.run_key,
            metrics.config_hash,
            metrics.manifest_hash,
            metrics.init_hash,
            metrics.phase,
        )
        != (key, config_hash, manifest_hash, init_hash, config.phase)
    ):
        raise ValueError("artifact identity mismatch")
    if spec.device == "cuda":
        requested = config.resources.gpu_deterministic_algorithms
        profile = environment.get("gpu_runtime_profile")
        expected = {
            **_gpu_runtime_requested(requested),
            "deterministic_algorithms_enabled": requested,
            "deterministic_warn_only": False,
            "cublas_workspace_config": _DETERMINISTIC_WORKSPACE if requested else None,
        }
        if not isinstance(profile, dict) or any(
            profile.get(name) != value for name, value in expected.items()
        ):
            raise ValueError("GPU runtime profile artifact mismatch")
    if (
        centers.shape
        != (
            spec.k,
            next(c.dimension for c in config.conditions if c.condition_id == spec.condition_id),
        )
        or centers.dtype != np.float32
        or not np.isfinite(centers).all()
    ):
        raise ValueError("invalid centroids")
    if (metrics.test is not None) != config.measurement.evaluate_test:
        raise ValueError("test evaluation policy mismatch")
    condition = next(c for c in config.conditions if c.condition_id == spec.condition_id)
    if (
        _sha(manifest_bytes) != manifest_hash
        or (
            manifest.dataset_id,
            manifest.condition_id,
            manifest.data_seed,
            manifest.n_train,
            manifest.dimension,
            manifest.generator_params,
        )
        != (
            spec.dataset_id,
            spec.condition_id,
            spec.data_seed,
            spec.n_train,
            condition.dimension,
            condition.generator_params,
        )
        or metrics.train.n_points != manifest.n_train
        or metrics.validation.n_points != manifest.n_validation
        or (metrics.test is not None and metrics.test.n_points != manifest.n_test)
        or any(
            len(item.cluster_sizes) != spec.k
            for item in (metrics.train, metrics.validation, metrics.test)
            if item is not None
        )
    ):
        raise ValueError("dataset manifest or evaluation dimensions mismatch")


def _trial(
    spec: RunSpec,
    dataset: PreparedDataset,
    root: Path,
    config: ExperimentConfig,
    *,
    resume: bool,
    retry_failed: bool,
    retry_reason: str | None,
    gpu: int | None,
    warm: set[tuple[str, str]],
    minimum_budget: float,
    data_load_seconds: float | None = None,
    initial_centers: np.ndarray | None = None,
    data_cache_hit: bool | None = None,
    init_cache_hit: bool | None = None,
    frozen_files: dict[str, bytes] | None = None,
    sentinel: SentinelGroup | None = None,
    gpu_runtime_initialized: dict[str, bool] | None = None,
) -> Path:
    from project_module.e_means.methods import train

    trial_start = time.perf_counter()
    if (
        Path(spec.run_id).name != spec.run_id
        or len(spec.run_id) != 20
        or any(c not in "0123456789abcdef" for c in spec.run_id)
    ):
        raise FrozenIdentityError("frozen identity contains unsafe run id")
    run_dir = root / spec.run_id
    if run_dir.is_symlink():
        raise FrozenIdentityError("frozen identity run directory cannot be a symlink")
    init = (
        initial_centers
        if initial_centers is not None
        else make_initial_centers(dataset.train, spec.k, spec.seed)
    )
    try:
        current_files = _frozen_files(root)
        if frozen_files is not None and current_files != frozen_files:
            raise FrozenIdentityError("frozen identity files changed during phase")
        config_hash = _sha(current_files["config.snapshot.json"])
        manifest_hash = _sha(dataset.manifest_path.read_bytes())
        signature = _dataset_signature(dataset)
        init_hash = _sha(init.tobytes())
        code = _code_identity()
        key = _identity(
            spec,
            config_hash,
            dataset,
            init_hash,
            code,
            _sha(current_files["hardware.snapshot.json"]),
        )
        expected_inputs = json.loads(current_files["run-inputs.json"]).get(spec.run_id)
        expected_key = json.loads(current_files["run-keys.json"]).get(spec.run_id)
        if spec.model_dump(mode="json") not in json.loads(current_files["jobs.json"]):
            raise FrozenIdentityError("frozen identity job is not in the ledger")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FrozenIdentityError("frozen identity files are invalid") from exc
    if expected_inputs != {
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "init_hash": init_hash,
    }:
        raise FrozenIdentityError("frozen identity run inputs mismatch")
    if spec.run_id != key[:20] or expected_key != key:
        raise FrozenIdentityError("frozen identity run key differs from ledger or collides")
    if (
        gpu is not None
        and gpu_runtime_initialized is not None
        and gpu_runtime_initialized.get("initialized", False)
    ):
        _gpu_runtime_state(config.resources.gpu_deterministic_algorithms)
    paths = sorted(run_dir.glob("attempt-*")) if run_dir.exists() else []
    if paths:
        last = paths[-1]
        if last.is_symlink():
            raise FrozenIdentityError("frozen identity attempt cannot be a symlink")
        try:
            record = RunRecord.model_validate_json((last / "status.json").read_bytes())
            stored_spec = RunSpec.model_validate_json((last / "spec.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise FrozenIdentityError("frozen identity existing attempt is invalid") from exc
        if record.run_key != key or stored_spec != spec:
            raise FrozenIdentityError("frozen identity existing attempt mismatch")
        if record.status == "success":
            try:
                _verify_success(last, spec, key, config, config_hash, manifest_hash, init_hash)
            except (OSError, ValueError) as exc:
                raise FrozenIdentityError(
                    "frozen identity existing attempt validation failed"
                ) from exc
            return last
        if record.status == "running":
            _record(
                last,
                spec,
                key,
                record.attempt,
                "interrupted",
                "previous_worker_interrupted",
                record.started_at,
            )
            record = RunRecord.model_validate_json((last / "status.json").read_bytes())
        if record.status in {"failed", "unsupported", "infeasible"}:
            if not retry_failed or record.status != "failed":
                return last
            if not retry_reason or not retry_reason.strip():
                raise FrozenIdentityError("frozen identity retry reason is required")
        elif not resume:
            raise FrozenIdentityError("frozen identity resume is required for an interrupted run")
    attempt = len(paths) + 1
    path = run_dir / f"attempt-{attempt:04d}"
    path.mkdir(parents=True, exist_ok=False)
    _write_json(path / "spec.json", spec.model_dump(mode="json"))
    started = datetime.now(UTC)
    _record(
        path,
        spec,
        key,
        attempt,
        "running",
        retry_reason if attempt > 1 and retry_failed else None,
        started,
    )
    try:
        cold = None
        warm_key = (spec.method, spec.condition_id)
        with _Monitor(gpu) as monitor:
            if warm_key not in warm:
                cold_start = time.perf_counter()
                if gpu is not None:
                    try:
                        import torch
                    except ImportError as exc:
                        raise RuntimeError("native torch backend is unavailable") from exc
                    from threadpoolctl import threadpool_limits

                    if gpu_runtime_initialized is None or not gpu_runtime_initialized.get(
                        "initialized", False
                    ):
                        _configure_gpu_runtime(config.resources.gpu_deterministic_algorithms)
                        if gpu_runtime_initialized is not None:
                            gpu_runtime_initialized["initialized"] = True
                    else:
                        _gpu_runtime_state(config.resources.gpu_deterministic_algorithms)
                    torch.set_num_threads(1)
                    if torch.get_num_interop_threads() != 1:
                        torch.set_num_interop_threads(1)
                    threadpool_limits(limits=1)
                train(
                    dataset.train, init.copy(), spec.model_copy(update={"budget": minimum_budget})
                )
                cold = time.perf_counter() - cold_start
                warm.add(warm_key)
            if gpu is not None:
                import torch
                from threadpoolctl import threadpool_limits

                _gpu_runtime_state(config.resources.gpu_deterministic_algorithms)
                threadpool_limits(limits=1)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            init_start = time.perf_counter()
            prepared_init = init.copy()
            init_prepare_seconds = time.perf_counter() - init_start
            coordination_wait_seconds = 0.0
            barrier_calls = 0

            def before_train() -> None:
                nonlocal barrier_calls, coordination_wait_seconds
                assert sentinel is not None
                barrier_calls += 1
                if barrier_calls != 1:
                    sentinel.abort("sentinel_duplicate_hook")
                    raise SentinelCoordinationError("sentinel training hook called more than once")
                trial = {
                    field: getattr(spec, field)
                    for field in (
                        "dataset_id",
                        "condition_id",
                        "data_seed",
                        "method",
                        "budget",
                        "seed",
                        "timing_repeat",
                    )
                }
                inputs = {
                    "manifest_hash": manifest_hash,
                    "data_hash": dataset.data_hash,
                    "split_hash": dataset.split_hash,
                    "init_hash": init_hash,
                }
                coordination_wait_seconds = sentinel.wait(trial, inputs)

            pipeline_start = time.perf_counter()
            result = train(
                dataset.train,
                prepared_init,
                spec,
                before_train=before_train if sentinel is not None else None,
            )
            if sentinel is not None and barrier_calls != 1:
                sentinel.abort("sentinel_missing_hook")
                raise SentinelCoordinationError("sentinel training hook was not called")
            save_start = time.perf_counter()
            temp = path / ".centroids.npy.tmp"
            with temp.open("wb") as stream:
                np.save(stream, result.centroids, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path / "centroids.npy")
            save_seconds = time.perf_counter() - save_start
            pipeline_seconds = time.perf_counter() - pipeline_start
            if gpu is not None:
                _gpu_runtime_state(config.resources.gpu_deterministic_algorithms)
            gpu_allocated = int(torch.cuda.max_memory_allocated()) if gpu is not None else None
            gpu_reserved = int(torch.cuda.max_memory_reserved()) if gpu is not None else None

            import faiss
            from threadpoolctl import threadpool_info

            from project_module.e_means.evaluation import evaluate_centers

            evaluation_threads = {
                "faiss_threads": faiss.omp_get_max_threads(),
                "threadpools": [
                    {key: pool.get(key) for key in ("user_api", "internal_api", "num_threads")}
                    for pool in threadpool_info()
                ],
            }
            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            logger = logging.getLogger("project_module.e_means.evaluation")
            previous_level = logger.level
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            try:
                assignment_start = time.perf_counter()
                train_result = evaluate_centers(dataset.train, result.centroids, chunk_rows=4096)
                assign_seconds = time.perf_counter() - assignment_start
                evaluation_start = time.perf_counter()
                train_metrics = _evaluation(
                    dataset.train, result.centroids, dataset.train_groups, result=train_result
                )
                validation = _evaluation(
                    dataset.validation, result.centroids, dataset.validation_groups
                )
                test = (
                    _evaluation(dataset.test, result.centroids, dataset.test_groups)
                    if config.measurement.evaluate_test
                    else None
                )
                evaluate_seconds = time.perf_counter() - evaluation_start
            finally:
                logger.removeHandler(handler)
                logger.setLevel(previous_level)
                handler.close()
            _atomic(path / "evaluation.log", log_stream.getvalue().encode())
        if monitor.rss > config.resources.host_memory_limit_bytes or (
            monitor.gpu_memory is not None
            and monitor.gpu_memory > config.resources.gpu_memory_limit_bytes
        ):
            raise MemoryError("frozen memory boundary exceeded")
        environment = _environment(
            spec,
            gpu,
            monitor,
            code,
            gpu_deterministic_algorithms=config.resources.gpu_deterministic_algorithms,
        )
        environment.update(
            gpu_peak_allocated_bytes=gpu_allocated,
            gpu_peak_reserved_bytes=gpu_reserved,
            gpu_memory_scope="NVML total-device sampled including context and other processes",
            gpu_allocator_scope="per trial after warmup; includes input transfer",
        )
        _write_json(path / "environment.json", environment)
        try:
            manifest_bytes = dataset.manifest_path.read_bytes()
        except OSError as exc:
            raise FrozenIdentityError("frozen identity manifest is unavailable") from exc
        if _sha(manifest_bytes) != manifest_hash:
            raise FrozenIdentityError("frozen identity manifest changed during trial")
        _atomic(path / "dataset-manifest.json", manifest_bytes)
        timing_values = result.timing.model_dump(mode="json")
        timing_values.update(
            cold_seconds=cold,
            assign_seconds=assign_seconds,
            evaluate_seconds=evaluate_seconds,
            save_seconds=save_seconds,
            e2e_seconds=pipeline_seconds,
            metadata={
                **result.timing.metadata,
                "scope_version": "resident-input-pipeline-v1",
                "e2e_scope": "resident-input-pipeline-v1",
                "assignment_scope": "train exact labels and MSE fused; excludes group evaluation",
                "evaluator_threads": evaluation_threads,
                "e2e_includes_warmup": False,
                "data_load_seconds": data_load_seconds,
                "init_prepare_seconds": init_prepare_seconds,
                "data_cache_hit": data_cache_hit,
                "init_cache_hit": init_cache_hit,
                "e2e_includes_dataset_loading": False,
                "coordination_wait_seconds": coordination_wait_seconds,
                "whole_trial_wall_seconds": time.perf_counter()
                - trial_start
                + (data_load_seconds or 0.0),
            },
        )
        metrics = RunMetrics(
            run_key=key,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            init_hash=init_hash,
            phase=config.phase,
            train=train_metrics,
            validation=validation,
            test=test,
            test_reason=None if test is not None else "not_evaluated",
            timing=TimingBreakdown.model_validate(timing_values),
            stats=dict(result.stats),
        )
        _write_json(path / "metrics.json", metrics.model_dump(mode="json"))
        if _frozen_files(root) != current_files or _dataset_signature(dataset) != signature:
            raise FrozenIdentityError("frozen identity inputs changed during trial")
        current_code = _code_identity()
        if {k: v for k, v in current_code.items() if k != "dirty_diff_sha256"} != {
            k: v for k, v in code.items() if k != "dirty_diff_sha256"
        }:
            raise FrozenIdentityError("frozen identity source changed during trial")
        _verify_success(path, spec, key, config, config_hash, manifest_hash, init_hash)
        if gpu is not None:
            _gpu_runtime_state(config.resources.gpu_deterministic_algorithms)
        _record(
            path,
            spec,
            key,
            attempt,
            "success",
            retry_reason if attempt > 1 and retry_failed else None,
            started,
        )
    except (FrozenIdentityError, SentinelCoordinationError) as exc:
        _record(
            path,
            spec,
            key,
            attempt,
            "interrupted",
            "sentinel_coordination"
            if isinstance(exc, SentinelCoordinationError)
            else "frozen_identity_changed",
            started,
        )
        raise
    except Exception as exc:
        _atomic(path / "error.log", _sanitize_error(exc).encode())
        message = str(exc).lower()
        deterministic_unsupported = (
            isinstance(exc, RuntimeError)
            and "deterministic" in message
            and ("not" in message or "implementation" in message or "unavailable" in message)
        )
        if sentinel is not None:
            reason = (
                "sentinel_deterministic_unsupported"
                if deterministic_unsupported
                else "sentinel_trial_failed"
            )
            sentinel.abort(reason)
            _record(path, spec, key, attempt, "interrupted", reason, started)
            raise SentinelCoordinationError(reason) from exc
        status: Literal["failed", "infeasible", "unsupported"] = (
            "infeasible"
            if isinstance(exc, MemoryError) or "out of memory" in message
            else "unsupported"
            if isinstance(exc, RuntimeError)
            and (("native" in message and "unavailable" in message) or deterministic_unsupported)
            else "failed"
        )
        reason = "deterministic_unsupported" if deterministic_unsupported else type(exc).__name__
        _record(path, spec, key, attempt, status, reason, started)
    return path


def run_trial(spec: RunSpec, dataset: PreparedDataset, output_root: Path) -> Path:
    """Execute one ledgered trial and return its attempt directory.

    Parameters
    ----------
    spec : RunSpec
        Immutable job specification already present in the ledger.
    dataset : PreparedDataset
        Verified, read-only input splits.
    output_root : Path
        Root containing the frozen configuration and ledger.

    Returns
    -------
    Path
        Published attempt directory.
    """
    if spec.device == "cuda":
        raise ValueError("CUDA trials require the monitored GPU phase")
    config = ExperimentConfig.model_validate_json(
        (output_root / "config.snapshot.json").read_bytes()
    )
    if (output_root / "hardware.snapshot.json").read_bytes() != _canonical(
        _hardware_fingerprint(config)
    ):
        raise FrozenIdentityError("hardware snapshot differs from current environment")
    jobs = [
        RunSpec.model_validate(item)
        for item in json.loads((output_root / "jobs.json").read_bytes())
    ]
    if spec not in jobs:
        raise FrozenIdentityError("run spec is not in the frozen ledger")
    minimum = min(m.budgets for m in config.methods if m.method == spec.method)
    return _trial(
        spec,
        dataset,
        output_root,
        config,
        resume=True,
        retry_failed=False,
        retry_reason=None,
        gpu=None,
        warm=set(),
        minimum_budget=min(minimum),
    )


def _worker(*args: Any) -> None:
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        _worker_loop(*args)


def _worker_loop(
    device: int,
    device_uuid: str,
    config: ExperimentConfig,
    data_root: Path,
    root: Path,
    requests: Any,
    responses: Any,
    resume: bool,
    retry_failed: bool,
    retry_reason: str | None,
    core: int,
    frozen_files: dict[str, bytes],
    sentinel_params: tuple[Any, ...] | None,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})
    warm: set[tuple[str, str]] = set()
    cache = _InputCache(data_root)
    gpu_runtime_initialized = {"initialized": False}
    sentinel = SentinelGroup(*sentinel_params) if sentinel_params is not None else None
    while True:
        spec = requests.get()
        if spec is None:
            return
        try:
            condition = next(c for c in config.conditions if c.condition_id == spec.condition_id)
            load_start = time.perf_counter()
            dataset, init, data_hit, init_hit = cache.get(condition, spec.data_seed, spec.seed)
            data_load_seconds = time.perf_counter() - load_start
            minimum = min(next(m.budgets for m in config.methods if m.method == spec.method))
            attempt = _trial(
                spec,
                dataset,
                root,
                config,
                resume=resume,
                retry_failed=retry_failed,
                retry_reason=retry_reason,
                gpu=device,
                warm=warm,
                minimum_budget=minimum,
                data_load_seconds=data_load_seconds,
                initial_centers=init,
                data_cache_hit=data_hit,
                init_cache_hit=init_hit,
                frozen_files=frozen_files,
                sentinel=sentinel,
                gpu_runtime_initialized=gpu_runtime_initialized,
            )
            record = RunRecord.model_validate_json((attempt / "status.json").read_bytes())
            if (
                config.resources.gpu_deterministic_algorithms
                and record.reason_code == "deterministic_unsupported"
            ):
                raise DeterministicProfileUnsupported(
                    "required deterministic GPU operation is unsupported"
                )
            responses.put((device, spec.run_id, None, None))
        except Exception as exc:
            responses.put((device, spec.run_id, type(exc).__name__, _sanitize_error(exc)))


def _cpu_worker(
    config: ExperimentConfig,
    data_root: Path,
    root: Path,
    requests: Any,
    responses: Any,
    resume: bool,
    retry_failed: bool,
    retry_reason: str | None,
    frozen_files: dict[str, bytes],
) -> None:
    from threadpoolctl import threadpool_limits

    if hasattr(os, "sched_setaffinity") and config.resources.cpu_affinity:
        os.sched_setaffinity(0, set(config.resources.cpu_affinity))
    warm: set[tuple[str, str]] = set()
    cache = _InputCache(data_root)
    with threadpool_limits(limits=config.resources.cpu_threads):
        while True:
            spec = requests.get()
            if spec is None:
                return
            try:
                condition = next(
                    c for c in config.conditions if c.condition_id == spec.condition_id
                )
                load_start = time.perf_counter()
                dataset, init, data_hit, init_hit = cache.get(condition, spec.data_seed, spec.seed)
                data_load_seconds = time.perf_counter() - load_start
                minimum = min(next(m.budgets for m in config.methods if m.method == spec.method))
                _trial(
                    spec,
                    dataset,
                    root,
                    config,
                    resume=resume,
                    retry_failed=retry_failed,
                    retry_reason=retry_reason,
                    gpu=None,
                    warm=warm,
                    minimum_budget=minimum,
                    data_load_seconds=data_load_seconds,
                    initial_centers=init,
                    data_cache_hit=data_hit,
                    init_cache_hit=init_hit,
                    frozen_files=frozen_files,
                )
                responses.put((spec.run_id, None, None))
            except Exception as exc:
                responses.put((spec.run_id, type(exc).__name__, _sanitize_error(exc)))


def _crash_record(
    spec: RunSpec, root: Path, reason: str, *, infeasible: bool = False, detail: str | None = None
) -> bool:
    path = root / spec.run_id
    attempts = sorted(path.glob("attempt-*")) if path.exists() else []
    if attempts:
        last = attempts[-1]
        try:
            record = RunRecord.model_validate_json((last / "status.json").read_bytes())
            if record.status == "running":
                if detail is not None:
                    _atomic(last / "error.log", detail.encode())
                _record(
                    last,
                    spec,
                    record.run_key,
                    record.attempt,
                    "infeasible" if infeasible else "failed",
                    reason,
                    record.started_at,
                )
                return True
            if record.status not in {"pending", "interrupted"}:
                return False
        except OSError, ValueError:
            pass
    keys = json.loads((root / "run-keys.json").read_bytes())
    key = keys[spec.run_id]
    if len(key) != 64 or not key.startswith(spec.run_id):
        raise FrozenIdentityError("invalid frozen run key")
    attempt = len(attempts) + 1
    target = path / f"attempt-{attempt:04d}"
    target.mkdir(parents=True, exist_ok=False)
    _write_json(target / "spec.json", spec.model_dump(mode="json"))
    if detail is not None:
        _atomic(target / "error.log", detail.encode())
    _record(target, spec, key, attempt, "infeasible" if infeasible else "failed", reason)
    return True


def _interrupt_record(spec: RunSpec, root: Path, reason: str) -> None:
    attempts = sorted((root / spec.run_id).glob("attempt-*"))
    if not attempts:
        return
    path = attempts[-1]
    try:
        record = RunRecord.model_validate_json((path / "status.json").read_bytes())
    except OSError, ValueError:
        return
    if record.status == "running":
        _record(
            path, spec, record.run_key, record.attempt, "interrupted", reason, record.started_at
        )


def _cpu_phase(
    jobs: list[RunSpec],
    config: ExperimentConfig,
    data_root: Path,
    root: Path,
    resume: bool,
    retry_failed: bool,
    retry_reason: str | None,
) -> None:
    context = mp.get_context("spawn")
    responses = context.Queue()
    frozen_files = _frozen_files(root)

    def start() -> tuple[Any, Any]:
        requests = context.Queue()
        process = context.Process(
            target=_cpu_worker,
            args=(
                config,
                data_root,
                root,
                requests,
                responses,
                resume,
                retry_failed,
                retry_reason,
                frozen_files,
            ),
        )
        names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in names}
        affinity = (
            os.sched_getaffinity(0)
            if config.resources.cpu_affinity and hasattr(os, "sched_getaffinity")
            else None
        )
        try:
            for name in names:
                os.environ[name] = str(config.resources.cpu_threads)
            if affinity is not None:
                os.sched_setaffinity(0, set(config.resources.cpu_affinity))
            process.start()
        finally:
            try:
                if affinity is not None:
                    os.sched_setaffinity(0, affinity)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        return process, requests

    process, requests = start()
    try:
        for spec in jobs:
            _check_frozen_hardware(config, root, frozen_files)
            requests.put(spec)
            while True:
                try:
                    run_id, error, detail = responses.get(timeout=_SAMPLE_SECONDS)
                    if run_id != spec.run_id:
                        raise RuntimeError("CPU worker returned wrong job")
                    if error:
                        if error == "FrozenIdentityError":
                            raise FrozenIdentityError("frozen identity changed during CPU phase")
                        if not _crash_record(spec, root, f"worker_{error}", detail=detail):
                            raise RuntimeError(f"existing attempt validation failed: {error}")
                    break
                except queue.Empty:
                    pass
                if not process.is_alive():
                    process.join()
                    _crash_record(spec, root, "worker_crash")
                    process, requests = start()
                    break
                try:
                    rss = psutil.Process(process.pid).memory_info().rss
                except psutil.NoSuchProcess:
                    continue
                if rss > config.resources.host_memory_limit_bytes:
                    process.terminate()
                    process.join()
                    _crash_record(spec, root, "memory_limit", infeasible=True)
                    process, requests = start()
                    break
    finally:
        if process.is_alive():
            requests.put(None)
            process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()


def _gpu_phase(
    jobs: list[RunSpec],
    devices: list[int],
    config: ExperimentConfig,
    data_root: Path,
    root: Path,
    resume: bool,
    retry_failed: bool,
    retry_reason: str | None,
    sentinel: SentinelGroup | None = None,
) -> None:
    context = mp.get_context("spawn")
    responses = context.Queue()
    frozen_files = _frozen_files(root)
    available = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    physical: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in available:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        identity = (
            (topology / "physical_package_id").read_text().strip(),
            (topology / "core_id").read_text().strip(),
        )
        if identity not in seen:
            seen.add(identity)
            physical.append(cpu)
    if not devices or max(devices) >= len(physical):
        raise ValueError("not enough host CPU cores for GPU workers")
    processes: dict[int, Any] = {}
    queues: dict[int, Any] = {}
    active: dict[int, RunSpec] = {}

    def start(device: int) -> None:
        requests = context.Queue()
        process = _spawn_with_threads(
            context,
            _worker,
            (
                device,
                pynvml.nvmlDeviceGetUUID(handles[device]),
                config,
                data_root,
                root,
                requests,
                responses,
                resume,
                retry_failed,
                retry_reason,
                physical[device],
                frozen_files,
                (
                    data_root,
                    config.measurement.sentinel.group_id,
                    sentinel.members,
                    sentinel.member,
                    sentinel.timeout,
                    sentinel.descriptor["plan"],
                )
                if sentinel is not None and config.measurement.sentinel is not None
                else None,
            ),
            1,
            gpu_deterministic_algorithms=config.resources.gpu_deterministic_algorithms,
        )
        queues[device] = requests
        processes[device] = process

    pending: dict[int, deque[RunSpec]] = {device: deque() for device in devices}
    method_order = {method.method: i for i, method in enumerate(config.methods)}
    for spec in jobs:
        index = (config.seeds.index(spec.seed) + method_order[spec.method]) % len(devices)
        pending[devices[index]].append(spec)
    try:
        import pynvml

        pynvml.nvmlInit()
        handles = {device: pynvml.nvmlDeviceGetHandleByIndex(device) for device in devices}
    except Exception as exc:
        raise RuntimeError("NVML monitoring is required for GPU execution") from exc
    try:
        for device in devices:
            start(device)
        while any(pending.values()) or active:
            if sentinel is not None:
                sentinel.assert_active()
            for device in devices:
                if device not in active and pending[device]:
                    _check_frozen_hardware(config, root, frozen_files)
                    spec = pending[device].popleft()
                    queues[device].put(spec)
                    active[device] = spec
            try:
                device, run_id, error, detail = responses.get(timeout=0.25)
                spec = active.pop(device)
                if spec.run_id != run_id:
                    raise RuntimeError("GPU worker returned wrong job")
                if error:
                    if error == "FrozenIdentityError":
                        raise FrozenIdentityError("frozen identity changed during GPU phase")
                    if error == "SentinelCoordinationError":
                        raise SentinelCoordinationError("sentinel coordination failed")
                    if error == "DeterministicProfileUnsupported":
                        raise DeterministicProfileUnsupported(
                            "required deterministic GPU operation is unsupported"
                        )
                    if sentinel is not None:
                        sentinel.abort("sentinel_worker_failed")
                        _interrupt_record(spec, root, "sentinel_worker_failed")
                        raise SentinelCoordinationError("sentinel worker failed")
                    if not _crash_record(spec, root, f"worker_{error}", detail=detail):
                        raise RuntimeError(f"existing attempt validation failed: {error}")
            except queue.Empty:
                pass
            rss = 0
            for process in processes.values():
                if process.is_alive():
                    try:
                        rss += psutil.Process(process.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            host_exceeded = rss > config.resources.host_memory_limit_bytes
            for device, spec in list(active.items()):
                gpu_used = int(pynvml.nvmlDeviceGetMemoryInfo(handles[device]).used)
                if host_exceeded or gpu_used > config.resources.gpu_memory_limit_bytes:
                    if host_exceeded:
                        raise RuntimeError("host memory sum exceeded; GPU phase aborted")
                    if sentinel is not None:
                        sentinel.abort("sentinel_gpu_memory_limit")
                        raise SentinelCoordinationError("sentinel GPU memory limit exceeded")
                    processes[device].terminate()
                    processes[device].join()
                    _crash_record(spec, root, "memory_limit", infeasible=True)
                    del active[device]
                    start(device)
                    continue
                if not processes[device].is_alive():
                    if sentinel is not None:
                        sentinel.abort("sentinel_worker_crash")
                        raise SentinelCoordinationError("sentinel worker crashed")
                    processes[device].join()
                    _crash_record(spec, root, "worker_crash")
                    del active[device]
                    start(device)
    finally:
        if sys.exc_info()[0] is not None:
            for device, spec in active.items():
                if processes[device].is_alive():
                    processes[device].terminate()
                    processes[device].join()
                _interrupt_record(spec, root, "phase_aborted")
        for device, process in processes.items():
            if process.is_alive():
                queues[device].put(None)
                process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
        pynvml.nvmlShutdown()


def _validate_cpu_affinity(config: ExperimentConfig) -> None:
    if config.resources.cpu_threads != 16:
        return
    cpus = config.resources.cpu_affinity
    if len(cpus) != 16 or len(set(cpus)) != 16:
        raise ValueError("CPU measurement requires sixteen distinct cores")
    topology = Path("/sys/devices/system/cpu")
    cores: set[tuple[str, str]] = set()
    nodes: set[str] = set()
    for cpu in cpus:
        directory = topology / f"cpu{cpu}"
        core = (directory / "topology/core_id").read_text().strip()
        package = (directory / "topology/physical_package_id").read_text().strip()
        matching_nodes = [node.name for node in directory.glob("node[0-9]*")]
        if len(matching_nodes) != 1:
            raise ValueError("cannot verify CPU NUMA topology")
        cores.add((package, core))
        nodes.add(matching_nodes[0])
    if len(cores) != 16 or len(nodes) != 1:
        raise ValueError("CPU measurement requires one NUMA node and sixteen physical cores")


def _sentinel_descriptor(
    config: ExperimentConfig, jobs: list[RunSpec], root: Path
) -> dict[str, Any]:
    normalized = config.model_dump(mode="json")
    normalized["measurement"]["cohort"] = "sentinel-parallel"
    normalized["resources"]["gpu_pool"] = []
    schedule = []
    for spec in jobs:
        item = spec.model_dump(mode="json")
        item.pop("run_id")
        item.pop("measurement_id")
        schedule.append(item)
    code = _code_identity()
    code.pop("dirty_diff_sha256", None)
    return {
        "config": normalized,
        "schedule": schedule,
        "code": code,
        "hardware": json.loads((root / "hardware.snapshot.json").read_bytes()),
    }


def run_phase(
    config: ExperimentConfig,
    data_root: Path,
    output_root: Path,
    *,
    phase: Literal["cpu", "gpu"],
    devices: list[int] | None = None,
    resume: bool = False,
    dry_run: bool = False,
    retry_failed: bool = False,
    retry_reason: str | None = None,
) -> list[RunSpec]:
    """Freeze the complete ledger and execute its selected phase.

    Parameters
    ----------
    config : ExperimentConfig
        Frozen experiment configuration.
    data_root, output_root : Path
        Prepared dataset and output roots.
    phase : {'cpu', 'gpu'}
        CPU or GPU training phase.
    devices : list[int] or None
        GPU indices; defaults to the configured pool.
    resume, dry_run, retry_failed : bool
        Resume interrupted work, only freeze the ledger, or retry failures.
    retry_reason : str or None
        Required explanation for failed-run retries.

    Returns
    -------
    list[RunSpec]
        Complete frozen ledger.
    """
    import fcntl

    if phase not in {"cpu", "gpu"}:
        raise ValueError("invalid phase")
    if config.phase == "main" and config.resources.cpu_threads != 16:
        raise ValueError("main CPU measurement requires sixteen threads")
    if retry_failed and (not retry_reason or not retry_reason.strip()):
        raise ValueError("retry reason is required")
    selected_devices = devices if devices is not None else config.resources.gpu_pool
    if phase == "gpu" and (
        not selected_devices
        or len(set(selected_devices)) != len(selected_devices)
        or any(d not in config.resources.gpu_pool for d in selected_devices)
    ):
        raise ValueError("invalid GPU devices")
    if (
        phase == "gpu"
        and config.measurement.cohort == "main-isolated"
        and len(selected_devices) != 1
    ):
        raise ValueError("main-isolated requires exactly one GPU")
    if (
        phase == "gpu"
        and config.measurement.cohort == "main-parallel"
        and set(selected_devices) != set(config.resources.gpu_pool)
    ):
        raise ValueError("main-parallel requires the complete GPU pool")
    sentinel_parallel = config.measurement.cohort.startswith("sentinel-parallel-")
    if sentinel_parallel:
        if config.phase != "pilot" or phase != "gpu" or len(selected_devices) != 1:
            raise ValueError("parallel sentinel requires pilot and one GPU")
        sentinel_config = config.measurement.sentinel
        if (
            sentinel_config is None
            or sentinel_config.participants != 8
            or len(sentinel_config.members) != 8
            or sentinel_config.members != sorted(sentinel_config.members)
        ):
            raise ValueError("parallel sentinel requires eight sorted unique GPU UUID members")
        if retry_failed:
            raise ValueError("parallel sentinel requires a fresh group rather than retries")
        if config.resources.host_memory_limit_bytes > 48 * 1024**3:
            raise ValueError("parallel sentinel host limit must be at most 48 GiB per GPU")
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(selected_devices[0])
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            available_uuids = {
                pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index))
                for index in range(pynvml.nvmlDeviceGetCount())
            }
            pynvml.nvmlShutdown()
        except Exception as exc:
            raise RuntimeError("cannot verify sentinel GPU UUID") from exc
        if config.measurement.cohort != f"sentinel-parallel-{uuid}":
            raise ValueError("sentinel cohort must contain its exact GPU UUID")
        if (
            uuid not in sentinel_config.members
            or not set(sentinel_config.members) <= available_uuids
        ):
            raise ValueError("sentinel members must be available exact GPU UUIDs")
    output_root.mkdir(parents=True, exist_ok=True)
    # A common data-root lock serializes CPU and GPU phases, including across roots.
    lock_path = data_root / ".emeans-measurement.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context(lock_path.open("a+b"))
        fcntl.flock(lock, fcntl.LOCK_SH if sentinel_parallel else fcntl.LOCK_EX)
        if sentinel_parallel:
            gpu_lock = stack.enter_context(
                (data_root / f".emeans-gpu-{selected_devices[0]}.lock").open("a+b")
            )
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        jobs = prepare_jobs(config, data_root, output_root)
        if dry_run:
            return jobs
        selected = [s for s in jobs if (s.device == "cpu") == (phase == "cpu")]
        if sentinel_parallel and not selected:
            raise ValueError("parallel sentinel requires at least one GPU job")
        if phase == "gpu":
            group = None
            if sentinel_parallel:
                assert config.measurement.sentinel is not None
                group = SentinelGroup(
                    data_root,
                    config.measurement.sentinel.group_id,
                    config.measurement.sentinel.members,
                    uuid,
                    config.measurement.sentinel.timeout_seconds,
                    _sentinel_descriptor(config, jobs, output_root),
                )
                stack.callback(group.close)
                group.join()
            try:
                _gpu_phase(
                    selected,
                    selected_devices,
                    config,
                    data_root,
                    output_root,
                    resume,
                    retry_failed,
                    retry_reason,
                    group,
                )
                if group is not None:
                    group.finish()
            except BaseException:
                if group is not None:
                    group.abort("sentinel_phase_failed")
                raise
        else:
            _validate_cpu_affinity(config)
            _cpu_phase(selected, config, data_root, output_root, resume, retry_failed, retry_reason)
        return jobs
