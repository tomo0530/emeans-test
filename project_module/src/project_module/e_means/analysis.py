"""Frozen-cohort paired-seed summaries, observed attainment, tables and figures."""

import csv
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from project_module.e_means.contracts import (
    DatasetManifest,
    ExperimentConfig,
    JSONValue,
    RunMetrics,
    RunRecord,
    RunSpec,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

_LOG = logging.getLogger(__name__)
_TERMINAL = {"success", "failed", "infeasible", "unsupported"}
_LEVELS = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
_PIPELINE_SCOPE = "resident-input-pipeline-v1"
_TIMING_DIAGNOSTICS = (
    "e2e_scope",
    "assignment_scope",
    "evaluator_threads",
    "whole_trial_wall_seconds",
    "data_load_seconds",
    "init_prepare_seconds",
    "data_cache_hit",
    "init_cache_hit",
    "coordination_wait_seconds",
    "train_started_at_ns",
    "train_finished_at_ns",
    "train_started_monotonic_ns",
    "train_finished_monotonic_ns",
)
type _Identity = tuple[str, str, float, int, int]
type _Row = dict[str, JSONValue]


@dataclass(frozen=True)
class _Attempt:
    spec: RunSpec
    number: int
    status: str
    run_key: str | None
    metrics: RunMetrics | None
    path: Path
    reason: str | None = None


@dataclass(frozen=True)
class _RunBinding:
    run_key: str
    config_hash: str
    manifest_hash: str
    init_hash: str


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _read_json(path: Path) -> JSONValue:
    return cast(
        "JSONValue", json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    )


def _write_json(path: Path, value: JSONValue) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _root_for(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / "config.snapshot.json").exists() or (candidate / "jobs.json").exists():
            return candidate
    return None


def _identity(spec: RunSpec) -> _Identity:
    return spec.condition_id, spec.method, spec.budget, spec.seed, spec.timing_repeat


def _repeat_count(config: ExperimentConfig, condition: str, method: str, budget: float) -> int:
    rules = [
        rule
        for rule in config.measurement.repeat_rules
        if (rule.condition_id, rule.method, rule.budget) == (condition, method, budget)
    ]
    if len(rules) > 1:
        raise ValueError("duplicate timing repeat rule")
    return rules[0].repeats if rules else config.measurement.repeats


def _job_matches_scope(spec: RunSpec, config: ExperimentConfig) -> bool:
    condition = next((c for c in config.conditions if c.condition_id == spec.condition_id), None)
    method = next((m for m in config.methods if m.method == spec.method), None)
    if condition is None or method is None:
        return False
    if (
        spec.dataset_id != condition.dataset_id
        or spec.n_train != condition.n_train
        or spec.k != condition.k
        or spec.data_seed not in condition.data_seeds
        or spec.seed not in config.seeds
        or spec.budget not in method.budgets
        or spec.params != method.params
        or spec.timing_repeat >= _repeat_count(config, spec.condition_id, spec.method, spec.budget)
    ):
        return False
    if config.phase == "main" and spec.dataset_id == "synthetic" and spec.data_seed != spec.seed:
        return False
    if spec.dataset_id in {"arxiv_nomic", "landmark_nomic"} and spec.data_seed != 42:
        return False
    if (
        config.phase == "pilot"
        and spec.dataset_id == "synthetic"
        and len(condition.data_seeds) > 1
        and spec.data_seed != spec.seed
    ):
        return False
    return True


def _is_digest(value: JSONValue) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _read_bindings(
    root: Path, config: ExperimentConfig, jobs: list[RunSpec], gaps: list[JSONValue]
) -> dict[str, _RunBinding]:
    paths = (root / "run-keys.json", root / "run-inputs.json")
    if any(path.is_symlink() for path in paths):
        raise ValueError("frozen run identity ledgers must not be symlinks")
    keys, inputs = (_read_json(path) for path in paths)
    if not isinstance(keys, dict) or not isinstance(inputs, dict):
        raise ValueError("run identity ledgers must be JSON objects keyed by run_id")
    ids = {spec.run_id for spec in jobs}
    for name, ledger in (("run-keys.json", keys), ("run-inputs.json", inputs)):
        if set(ledger) != ids:
            gaps.append(
                {"code": "identity_ledger_scope", "detail": f"{name} run IDs differ from jobs.json"}
            )
    config_bytes = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    bindings: dict[str, _RunBinding] = {}
    for run_id in ids:
        key, hashes = keys.get(run_id), inputs.get(run_id)
        if (
            not _is_digest(key)
            or not isinstance(hashes, dict)
            or set(hashes) != {"config_hash", "manifest_hash", "init_hash"}
            or not all(_is_digest(value) for value in hashes.values())
            or hashes["config_hash"] != config_hash
        ):
            gaps.append(
                {
                    "code": "invalid_run_identity",
                    "run_id": run_id,
                    "detail": "frozen key/input hashes are missing, invalid, or differ from config",
                }
            )
            continue
        bindings[run_id] = _RunBinding(
            cast("str", key),
            hashes["config_hash"],
            cast("str", hashes["manifest_hash"]),
            cast("str", hashes["init_hash"]),
        )
    return bindings


def _validate_manifest(
    path: Path, spec: RunSpec, config: ExperimentConfig, metrics: RunMetrics, binding: _RunBinding
) -> None:
    raw = (path / "dataset-manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding.manifest_hash:
        raise ValueError("dataset manifest bytes differ from the frozen manifest hash")
    manifest = DatasetManifest.model_validate_json(raw)
    condition = next(c for c in config.conditions if c.condition_id == spec.condition_id)
    if (
        (manifest.dataset_id, manifest.condition_id, manifest.data_seed)
        != (spec.dataset_id, spec.condition_id, spec.data_seed)
        or manifest.dimension != condition.dimension
        or manifest.n_train != spec.n_train
        or manifest.n_train != condition.n_train
        or manifest.generator_params != condition.generator_params
    ):
        raise ValueError("dataset manifest identity/shape differs from spec or frozen config")
    if (
        metrics.train.n_points != manifest.n_train
        or metrics.validation.n_points != manifest.n_validation
        or (metrics.test is not None and metrics.test.n_points != manifest.n_test)
    ):
        raise ValueError("dataset manifest split counts differ from evaluation counts")


def _attempt(
    spec: RunSpec, path: Path, config: ExperimentConfig, binding: _RunBinding | None
) -> _Attempt:
    number = 1
    try:
        number = int(path.name.removeprefix("attempt-"))
        if path.is_symlink() or any(item.is_symlink() for item in path.iterdir()):
            raise ValueError("artifact symlinks are forbidden")
        stored = RunSpec.model_validate(_read_json(path / "spec.json"))
        record = RunRecord.model_validate(_read_json(path / "status.json"))
        if binding is None or record.run_key != binding.run_key:
            raise ValueError("status run key differs from the frozen run identity")
        if stored != spec or record.attempt != number:
            raise ValueError("spec or attempt differs from the immutable ledger")
        for relative in record.artifact_paths.values():
            artifact = path / relative
            if Path(relative).is_absolute() or not artifact.resolve().is_relative_to(
                path.resolve()
            ):
                raise ValueError("artifact path escapes the run attempt directory")
        metrics = None
        if record.status == "success":
            metrics = RunMetrics.model_validate(_read_json(path / "metrics.json"))
            if metrics.timing.metadata.get("e2e_scope") != _PIPELINE_SCOPE:
                raise ValueError(
                    "timing scope is not resident-input-pipeline-v1; legacy E2E cannot be pooled"
                )
            if metrics.run_key != record.run_key or metrics.phase != config.phase:
                raise ValueError("run key or phase mismatch")
            if (metrics.config_hash, metrics.manifest_hash, metrics.init_hash) != (
                binding.config_hash,
                binding.manifest_hash,
                binding.init_hash,
            ):
                raise ValueError("metrics input hashes differ from the frozen run inputs")
            _validate_manifest(path, spec, config, metrics, binding)
            if config.measurement.evaluate_test and metrics.test is None:
                raise ValueError("frozen main evaluation requires test metrics")
            if not config.measurement.evaluate_test and metrics.test is not None:
                raise ValueError("frozen cohort forbids test evaluation")
            environment = _read_json(path / "environment.json")
            if not isinstance(environment, dict):
                raise ValueError("environment must be a JSON object")
            centers = np.load(path / "centroids.npy", mmap_mode="r", allow_pickle=False)
            condition = next(c for c in config.conditions if c.condition_id == spec.condition_id)
            if (
                centers.dtype != np.float32
                or centers.shape != (spec.k, condition.dimension)
                or not centers.flags.c_contiguous
                or not np.isfinite(centers).all()
            ):
                raise ValueError("invalid final centers")
            evaluations = [metrics.train, metrics.validation]
            if metrics.test is not None:
                evaluations.append(metrics.test)
            if metrics.train.n_points != spec.n_train or any(
                len(m.cluster_sizes) != spec.k for m in evaluations
            ):
                raise ValueError("evaluation dimensions differ from run spec")
        return _Attempt(
            spec, number, record.status, record.run_key, metrics, path, record.reason_code
        )
    except (OSError, ValueError, ValidationError, StopIteration) as exc:
        return _Attempt(spec, number, "invalid", None, None, path, str(exc))


def _load_runs(
    root: Path,
    config: ExperimentConfig,
    jobs: list[RunSpec],
    gaps: list[JSONValue],
    bindings: Mapping[str, _RunBinding],
) -> tuple[dict[_Identity, _Attempt], list[_Attempt], list[_Row]]:
    selected: dict[_Identity, _Attempt] = {}
    all_attempts: list[_Attempt] = []
    raw: list[_Row] = []
    cohort = config.measurement.cohort
    identity_counts = Counter(_identity(spec) for spec in jobs if spec.measurement_id == cohort)
    for spec in jobs:
        if spec.measurement_id != cohort:
            continue
        identity = _identity(spec)
        duplicate = identity_counts[identity] > 1
        if duplicate:
            gaps.append(
                {
                    "code": "duplicate_job",
                    "run_id": spec.run_id,
                    "detail": "duplicate scientific identity",
                }
            )
        if Path(spec.run_id).name != spec.run_id or spec.run_id in {"", ".", ".."}:
            gaps.append(
                {
                    "code": "invalid_run_id",
                    "run_id": spec.run_id,
                    "detail": "run_id must be a safe basename",
                }
            )
            continue
        run_dir = root / spec.run_id
        if run_dir.is_symlink():
            attempts = [
                _Attempt(spec, 1, "invalid", None, None, run_dir, "run directory is a symlink")
            ]
        else:
            paths = sorted(p for p in run_dir.glob("attempt-*") if p.is_dir())
            attempts = [_attempt(spec, path, config, bindings.get(spec.run_id)) for path in paths]
        if not attempts:
            attempts = [_Attempt(spec, 1, "missing", None, None, run_dir, "run artifact is absent")]
        if duplicate or not _job_matches_scope(spec, config):
            attempts = [
                replace(
                    a,
                    status="invalid",
                    metrics=None,
                    reason="job identity is duplicated or outside frozen scope",
                )
                for a in attempts
            ]
        all_attempts.extend(attempts)
        # Resuming interrupted work is allowed; a scientific terminal outcome
        # (including failure) remains the primary result for all later retries.
        primary = next((a for a in attempts if a.status != "interrupted"), attempts[-1])
        selected[identity] = primary
        for attempt in attempts:
            raw.append(_raw_row(attempt, primary=attempt is primary))
            if attempt is not primary and attempt.status == "invalid":
                gaps.append(
                    {
                        "code": "invalid",
                        "run_id": spec.run_id,
                        "detail": f"attempt {attempt.number}: {attempt.reason}",
                    }
                )
        if primary.status not in _TERMINAL:
            gaps.append(
                {
                    "code": primary.status,
                    "run_id": spec.run_id,
                    "detail": primary.reason or "run has not reached a terminal status",
                }
            )
    return selected, all_attempts, raw


def _raw_row(attempt: _Attempt, *, primary: bool) -> _Row:
    spec, metrics = attempt.spec, attempt.metrics
    row: _Row = {
        "run_id": spec.run_id,
        "run_key": attempt.run_key,
        "config_hash": metrics.config_hash if metrics else None,
        "manifest_hash": metrics.manifest_hash if metrics else None,
        "init_hash": metrics.init_hash if metrics else None,
        "dataset_manifest": str(attempt.path / "dataset-manifest.json") if metrics else None,
        "dataset_id": spec.dataset_id,
        "condition_id": spec.condition_id,
        "method": spec.method,
        "budget": spec.budget,
        "seed": spec.seed,
        "data_seed": spec.data_seed,
        "timing_repeat": spec.timing_repeat,
        "measurement_id": spec.measurement_id,
        "attempt": attempt.number,
        "primary": primary,
        "status": attempt.status,
        "reason": attempt.reason,
        "train_seconds": metrics.timing.train_seconds if metrics else None,
        "e2e_seconds": metrics.timing.e2e_seconds if metrics else None,
        "transfer_seconds": metrics.timing.transfer_seconds if metrics else None,
        "assign_seconds": metrics.timing.assign_seconds if metrics else None,
        "evaluate_seconds": metrics.timing.evaluate_seconds if metrics else None,
        "save_seconds": metrics.timing.save_seconds if metrics else None,
        "cold_seconds": metrics.timing.cold_seconds if metrics else None,
        "train_mse": metrics.train.mse if metrics else None,
        "validation_mse": metrics.validation.mse if metrics else None,
        "test_mse": metrics.test.mse if metrics and metrics.test else None,
        "empty_clusters": metrics.train.empty_clusters if metrics else None,
        "cluster_sizes": list(metrics.train.cluster_sizes) if metrics else None,
        "group_mse": metrics.test.group_mse if metrics and metrics.test else None,
        "train_group_mse": metrics.train.group_mse if metrics else None,
        "validation_group_mse": metrics.validation.group_mse if metrics else None,
        "max_mean_cluster_size_ratio": max(metrics.train.cluster_sizes) / (spec.n_train / spec.k)
        if metrics
        else None,
        "stats": dict(metrics.stats) if metrics else None,
        "artifact_dir": str(attempt.path),
    }
    row.update(
        {key: metrics.timing.metadata.get(key) if metrics else None for key in _TIMING_DIAGNOSTICS}
    )
    return row


def _coverage(config: ExperimentConfig, jobs: list[RunSpec], gaps: list[JSONValue]) -> None:
    if len(set(config.seeds)) != len(config.seeds):
        gaps.append({"code": "duplicate_seed", "detail": "seed IDs must be independent and unique"})
    cohort_jobs = [spec for spec in jobs if spec.measurement_id == config.measurement.cohort]
    keys = {_identity(spec) for spec in cohort_jobs}
    ids = [spec.run_id for spec in cohort_jobs]
    if len(set(ids)) != len(ids):
        gaps.append({"code": "duplicate_run_id", "detail": "jobs.json repeats a run_id"})
    for condition in config.conditions:
        if not any(spec.condition_id == condition.condition_id for spec in cohort_jobs):
            gaps.append({"code": "condition_absent", "detail": condition.condition_id})
        for method in config.methods:
            for budget in method.budgets:
                repeats = _repeat_count(config, condition.condition_id, method.method, budget)
                for seed in config.seeds:
                    for repeat in range(repeats):
                        key = (condition.condition_id, method.method, budget, seed, repeat)
                        if key not in keys:
                            gaps.append({"code": "planned_job_absent", "detail": repr(key)})
    for dataset in {condition.dataset_id for condition in config.conditions}:
        if not any(spec.dataset_id == dataset for spec in cohort_jobs):
            gaps.append({"code": "dataset_absent", "detail": dataset})
    for method in config.methods:
        if not any(spec.method == method.method for spec in cohort_jobs):
            gaps.append({"code": "method_absent", "detail": method.method})
    for spec in cohort_jobs:
        if not _job_matches_scope(spec, config):
            gaps.append(
                {
                    "code": "job_outside_snapshot",
                    "run_id": spec.run_id,
                    "detail": "job differs from frozen scope",
                }
            )
    if config.phase == "main" and sorted(config.seeds) != list(range(10)):
        gaps.append(
            {"code": "main_seed_scope", "detail": "main curves require independent seeds 0..9"}
        )


def _seed_row(
    selected: Mapping[_Identity, _Attempt],
    config: ExperimentConfig,
    condition: str,
    method: str,
    budget: float,
    seed: int,
) -> _Row:
    repeats = _repeat_count(config, condition, method, budget)
    attempts = [
        selected.get((condition, method, budget, seed, repeat)) for repeat in range(repeats)
    ]
    successful = [
        a for a in attempts if a is not None and a.status == "success" and a.metrics is not None
    ]
    quality = (
        attempts[0].metrics if attempts[0] is not None and attempts[0].status == "success" else None
    )
    complete = len(successful) == repeats
    metrics = [a.metrics for a in successful if a.metrics is not None]
    return {
        "seed": seed,
        "status": "success" if complete else "incomplete",
        "failure_status": next(
            (a.status if a else "missing" for a in attempts if a is None or a.status != "success"),
            "success",
        ),
        "repeat_statuses": [a.status if a else "missing" for a in attempts],
        "expected_repeats": repeats,
        "successful_repeats": len(successful),
        "train_seconds": float(np.median([m.timing.train_seconds for m in metrics]))
        if complete
        else None,
        "e2e_seconds": float(np.median([m.timing.e2e_seconds for m in metrics]))
        if complete
        else None,
        "train_mse": quality.train.mse if quality else None,
        "validation_mse": quality.validation.mse if quality else None,
        "test_mse": quality.test.mse if quality and quality.test else None,
        "repeat_mse": [m.test.mse if m.test else m.validation.mse for m in metrics],
        "repeat_train_seconds": [m.timing.train_seconds for m in metrics],
    }


def _ci(
    values: npt.NDArray[np.float64], indices: npt.NDArray[np.int64], *, median: bool = False
) -> list[JSONValue]:
    draws = values[indices]
    estimates = np.median(draws, axis=1) if median else np.mean(draws, axis=1)
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def _points(config: ExperimentConfig, selected: Mapping[_Identity, _Attempt]) -> list[_Row]:
    seeds = sorted(config.seeds)
    indices = np.random.default_rng(1729).integers(
        0, len(seeds), size=(10000, len(seeds)), dtype=np.int64
    )
    points: list[_Row] = []
    for condition in config.conditions:
        references = {
            seed: _seed_row(selected, config, condition.condition_id, "faiss_lloyd", 100.0, seed)
            for seed in seeds
        }
        for method in config.methods:
            for budget in sorted(method.budgets):
                rows = [
                    _seed_row(selected, config, condition.condition_id, method.method, budget, seed)
                    for seed in seeds
                ]
                count = sum(row["status"] == "success" for row in rows)
                complete = (
                    count == len(seeds)
                    and len(set(seeds)) == len(seeds)
                    and (config.phase == "pilot" or seeds == list(range(10)))
                )
                point: _Row = {
                    "dataset_id": condition.dataset_id,
                    "condition_id": condition.condition_id,
                    "method": method.method,
                    "budget": budget,
                    "status": "complete" if complete else "incomplete",
                    "successful_seeds": count,
                    "expected_seeds": len(seeds),
                    "seed_values": cast("list[JSONValue]", rows),
                }
                for metric in (
                    "train_seconds",
                    "e2e_seconds",
                    "train_mse",
                    "validation_mse",
                    "test_mse",
                ):
                    available = complete and all(
                        isinstance(row[metric], (int, float)) for row in rows
                    )
                    values = (
                        np.array([row[metric] for row in rows], dtype=np.float64)
                        if available
                        else None
                    )
                    time = metric.endswith("seconds")
                    point[metric] = (
                        float(np.median(values) if time else np.mean(values))
                        if values is not None
                        else None
                    )
                    point[f"{metric}_ci95"] = (
                        _ci(values, indices, median=time) if values is not None else None
                    )
                deltas: list[float] = []
                reference_ok = True
                quality_key = "test_mse" if config.measurement.evaluate_test else "validation_mse"
                repeat_differences: list[float] = []
                for row in rows:
                    seed = cast("int", row["seed"])
                    ref = references[seed][quality_key]
                    quality = row[quality_key]
                    defined = (
                        isinstance(ref, (float, int))
                        and ref > 0
                        and references[seed]["status"] == "success"
                    )
                    reference_ok &= defined
                    delta = (
                        100 * (float(cast("float", quality)) / float(cast("float", ref)) - 1)
                        if defined and isinstance(quality, (float, int))
                        else None
                    )
                    row["delta_percent"] = delta
                    row["reference_status"] = "defined" if defined else "reference_undefined"
                    if delta is not None:
                        deltas.append(delta)
                    repeat_values = cast("list[float]", row["repeat_mse"])
                    if isinstance(quality, (float, int)):
                        repeat_differences.extend(abs(value - quality) for value in repeat_values)
                relative_complete = complete and reference_ok and len(deltas) == len(seeds)
                point["relative_status"] = (
                    "reference_undefined" if not reference_ok else point["status"]
                )
                point["delta_percent"] = float(np.mean(deltas)) if relative_complete else None
                point["delta_percent_ci95"] = (
                    _ci(np.array(deltas, dtype=np.float64), indices) if relative_complete else None
                )
                point["repeat_mse_max_abs_difference"] = max(repeat_differences, default=0.0)
                points.append(point)
    return points


def _segments(points: list[_Row]) -> list[_Row]:
    groups = sorted({(str(p["condition_id"]), str(p["method"])) for p in points})
    result: list[_Row] = []
    for condition, method in groups:
        segments: list[JSONValue] = []
        current: list[JSONValue] = []
        for point in sorted(
            (p for p in points if (p["condition_id"], p["method"]) == (condition, method)),
            key=lambda p: float(cast("float", p["budget"])),
        ):
            if point["status"] == "complete":
                current.append(point["budget"])
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        result.append({"condition_id": condition, "method": method, "segments": segments})
    return result


def _attainment(points: list[_Row], config: ExperimentConfig) -> list[_Row]:
    results: list[_Row] = []
    groups = sorted({(str(p["condition_id"]), str(p["method"])) for p in points})
    for condition, method in groups:
        group = [p for p in points if (p["condition_id"], p["method"]) == (condition, method)]
        for level in _LEVELS:
            reached = [
                p
                for p in group
                if p["status"] == "complete"
                and isinstance(p["delta_percent"], (int, float))
                and p["delta_percent"] <= level
            ]
            best = (
                min(reached, key=lambda p: float(cast("float", p["train_seconds"])))
                if reached
                else None
            )
            status = (
                "reached"
                if best
                else "reference_undefined"
                if any(p["relative_status"] == "reference_undefined" for p in group)
                else "incomplete"
                if any(p["status"] != "complete" for p in group)
                else "unreached"
            )
            seed_results: list[JSONValue] = []
            for seed in sorted(config.seeds):
                rows = [
                    next(row for row in cast("list[_Row]", p["seed_values"]) if row["seed"] == seed)
                    for p in group
                ]
                seed_reached = [
                    row
                    for row in rows
                    if row["status"] == "success"
                    and isinstance(row["delta_percent"], (int, float))
                    and row["delta_percent"] <= level
                ]
                seed_best = (
                    min(seed_reached, key=lambda r: float(cast("float", r["train_seconds"])))
                    if seed_reached
                    else None
                )
                seed_status = (
                    "reached"
                    if seed_best
                    else "reference_undefined"
                    if any(r["reference_status"] == "reference_undefined" for r in rows)
                    else "incomplete"
                    if any(r["status"] != "success" for r in rows)
                    else "unreached"
                )
                seed_results.append(
                    {
                        "seed": seed,
                        "status": seed_status,
                        "train_seconds": seed_best["train_seconds"] if seed_best else None,
                    }
                )
            ci = best["delta_percent_ci95"] if best else None
            results.append(
                {
                    "condition_id": condition,
                    "method": method,
                    "delta_percent": level,
                    "status": status,
                    "train_seconds": best["train_seconds"] if best else None,
                    "budget": best["budget"] if best else None,
                    "ci_upper_below_level": bool(cast("list[float]", ci)[1] <= level)
                    if ci
                    else None,
                    "reached_seeds": sum(
                        cast("dict[str, JSONValue]", r)["status"] == "reached" for r in seed_results
                    ),
                    "expected_seeds": len(config.seeds),
                    "seeds": seed_results,
                }
            )
    return results


def _write_table(path: Path, rows: Sequence[_Row]) -> None:
    rows = [
        {
            key.replace("e2e_seconds", "resident_input_pipeline_seconds"): value
            for key, value in row.items()
        }
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _validation_selection(points: list[_Row]) -> list[_Row]:
    selections: list[_Row] = []
    for condition, method in sorted({(str(p["condition_id"]), str(p["method"])) for p in points}):
        group = [p for p in points if (p["condition_id"], p["method"]) == (condition, method)]
        complete = all(p["status"] == "complete" for p in group)
        eligible = [
            p for p in group if p["status"] == "complete" and p["validation_mse"] is not None
        ]
        best = (
            min(
                eligible,
                key=lambda p: (
                    float(cast("float", p["validation_mse"])),
                    float(cast("float", p["train_seconds"])),
                    float(cast("float", p["budget"])),
                ),
            )
            if eligible and complete
            else None
        )
        selections.append(
            {
                "condition_id": condition,
                "method": method,
                "status": "complete" if best else "incomplete",
                "selection_criterion": "minimum_mean_validation_mse_then_median_train_seconds_then_budget",
                "budget": best["budget"] if best else None,
                "validation_mse": best["validation_mse"] if best else None,
                "test_mse": best["test_mse"] if best else None,
                "train_seconds": best["train_seconds"] if best else None,
                "e2e_seconds": best["e2e_seconds"] if best else None,
            }
        )
    return selections


def _plot(points: list[_Row], config: ExperimentConfig, output: Path) -> None:
    colors = plt.get_cmap("tab10")
    panels = [
        (
            "train_seconds",
            "test_mse" if config.measurement.evaluate_test else "validation_mse",
            "Warm training (s)",
            "Test MSE" if config.measurement.evaluate_test else "Validation MSE",
        ),
        ("train_seconds", "delta_percent", "Warm training (s)", "Paired excess loss (%)"),
        (
            "e2e_seconds",
            "test_mse" if config.measurement.evaluate_test else "validation_mse",
            "Resident-input pipeline (s)",
            "Test MSE" if config.measurement.evaluate_test else "Validation MSE",
        ),
        ("train_seconds", "train_mse", "Warm training (s)", "Train MSE"),
    ]
    for condition in config.conditions:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for method_index, method in enumerate(config.methods):
            ordered = sorted(
                (
                    p
                    for p in points
                    if p["condition_id"] == condition.condition_id and p["method"] == method.method
                ),
                key=lambda p: float(cast("float", p["budget"])),
            )
            for ax, (xkey, ykey, xlabel, ylabel) in zip(axes.flat, panels, strict=True):
                # NaNs exist only in the plotting array, never serialized JSON.
                xs = [
                    p[xkey] if p["status"] == "complete" and p[ykey] is not None else np.nan
                    for p in ordered
                ]
                ys = [
                    p[ykey] if p["status"] == "complete" and p[ykey] is not None else np.nan
                    for p in ordered
                ]
                ax.plot(xs, ys, "o-", label=method.method, color=colors(method_index))
                for point in ordered:
                    if point["status"] != "complete" or point[ykey] is None:
                        continue
                    x = float(cast("float", point[xkey]))
                    y = float(cast("float", point[ykey]))
                    xci = cast("list[float]", point[f"{xkey}_ci95"])
                    yci = cast("list[float]", point[f"{ykey}_ci95"])
                    ax.errorbar(
                        x,
                        y,
                        xerr=np.array([[max(0, x - xci[0])], [max(0, xci[1] - x)]]),
                        yerr=np.array([[max(0, y - yci[0])], [max(0, yci[1] - y)]]),
                        color=colors(method_index),
                        alpha=0.5,
                        capsize=2,
                    )
                    ax.annotate(
                        f"{point['budget']:g}",
                        (x, y),
                        xytext=(3, 4),
                        textcoords="offset points",
                        fontsize=7,
                    )
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.grid(alpha=0.2)
        incomplete = [
            f"{p['method']}@{p['budget']:g}: {p['successful_seeds']}/{p['expected_seeds']}"
            for p in points
            if p["condition_id"] == condition.condition_id and p["status"] != "complete"
        ]
        fig.suptitle(
            f"{condition.condition_id} | {config.measurement.cohort} | fixed-grid descriptive results\nSeed bootstrap 95% pointwise CI (10,000 draws); missing points break lines",
            fontsize=10,
        )
        axes.flat[0].legend(fontsize=7)
        if incomplete:
            remaining = (
                f"; +{len(incomplete) - 4} points (see failure map / points.csv)"
                if len(incomplete) > 4
                else ""
            )
            fig.supxlabel(
                "Incomplete: " + "; ".join(incomplete[:4]) + remaining, fontsize=7, wrap=True
            )
        filename = _safe_name(condition.condition_id)
        for extension in ("png", "svg"):
            fig.savefig(output / f"curves-{filename}.{extension}", dpi=170)
        plt.close(fig)
        _failure_plot(points, config, condition.condition_id, output, filename)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _failure_plot(
    points: list[_Row], config: ExperimentConfig, condition: str, output: Path, filename: str
) -> None:
    group = [p for p in points if p["condition_id"] == condition]
    categories = [
        "success",
        "failed",
        "infeasible",
        "unsupported",
        "missing",
        "invalid",
        "interrupted",
        "pending",
        "running",
    ]
    matrix = np.array(
        [
            [
                categories.index(str(row["failure_status"]))
                for row in cast("list[_Row]", p["seed_values"])
            ]
            for p in group
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(8, max(3, len(group) * 0.22)), layout="constrained")
    cmap = ListedColormap(
        [
            "#009E73",
            "#D55E00",
            "#E69F00",
            "#CC79A7",
            "#BBBBBB",
            "#000000",
            "#56B4E9",
            "#F0E442",
            "#0072B2",
        ]
    )
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories)),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(config.seeds)), [str(seed) for seed in sorted(config.seeds)])
    ax.set_yticks(
        range(len(group)),
        [
            f"{p['method']} @ {p['budget']:g} ({p['successful_seeds']}/{p['expected_seeds']})"
            for p in group
        ],
        fontsize=7,
    )
    ax.set(
        xlabel="Independent seed",
        title=f"{condition}: primary run outcomes (first unsuccessful timing repeat)",
    )
    bar = fig.colorbar(image, ax=ax, ticks=range(len(categories)))
    bar.ax.set_yticklabels(categories, fontsize=7)
    for extension in ("png", "svg"):
        fig.savefig(output / f"failures-{filename}.{extension}", dpi=170)
    plt.close(fig)


def _report(summary: _Row, output: Path) -> None:
    scope = cast("_Row", summary["scope"])
    text = (
        "# e-means 評価集計\n\n"
        f"状態: {summary['status']}。phase={scope.get('phase')}、採用cohort={scope.get('cohort')}。\n\n"
        "主軸はwarm train時間、品質はrepeat0の最終中心に対する独立exact squared L2点平均です。"
        "seed内の時間中央値を先に取り、seed間で時間中央値とMSE算術平均を集計します。"
        "同一セル・seedのFaiss Lloyd@100を基準に、seed別比の平均を相対損失とします。\n\n"
        "Resident-input pipelineはresident-input-pipeline-v1の連続計時窓で、常駐済み入力に対する"
        "train呼出し直前から最終中心のfsync直後までを測ります。評価、初回hash検証、入力/初期中心の"
        "準備、git/NVML呼出しを含めません。旧E2E artifactは同じ時間として集約しません。"
        "assign_secondsはtrain最終exact labelsとMSEの融合評価、evaluate_secondsは残りのgroup/holdout評価です。"
        "実評価thread数・入力準備・whole trial wallはraw_seeds.csvの診断列に分離しています。\n\n"
        "図のCIはseed indexを単位とし、乱数1729で同セルの全手法・予算・両軸に共用する"
        "10,000回の対応bootstrapのpercentile 95%点ごと区間です。曲線全体や最良点の同時保証ではありません。\n\n"
        "10/10成功点のみ主曲線に描き、欠落点は接続しません。失敗・不適合・未対応は終端台帳に残します。"
        "初回の失敗を後の成功attemptで置換せず、retry_attemptsへ分離しました。"
        "未到達、失敗による不完全、基準未定義を区別します。\n\n"
        "観測到達時間は事前固定水準以下となる測定済み予算点の最小時間です。未測定時刻や途中到達を補間しません。"
        "到達seedだけの条件付き時間で優劣を断定しません。test固定格子の結果は記述であり、実用設定の選択はvalidationのみで行います。\n\n"
        "全seed・attemptの生値はraw_seeds.csv、曲線の元表はpoints.csv、到達結果はattainment.csv、"
        "状態・不足一覧はsummary.jsonを参照してください。巨大配列をreportへコピーしません。\n\n"
        "validation_selected.csvは全予定点が成功した手法内で平均validation MSEが最小の予算を選び、"
        "対応するtest MSEと時間を別表に示します。同率時はwarm時間、次に予算で決め、testは選択に使いません。"
        "この基準は時間・品質の別の実用上の効用まで最適化するものではありません。\n\n"
        f"不足記録: {len(cast('list[JSONValue]', summary['gaps']))}件。\n"
    )
    (output / "report.md").write_text(text, encoding="utf-8")


def summarize_runs(run_dirs: Sequence[Path], out_dir: Path) -> Path:
    """Summarize an immutable job ledger, always retaining a partial report.

    Parameters
    ----------
    run_dirs : Sequence[pathlib.Path]
        Run directories (or their shared root). An ancestor must contain
        config.snapshot.json, jobs.json, run-keys.json and run-inputs.json.
        Only the frozen cohort is selected; successful attempts must retain
        the exact frozen dataset-manifest.json bytes.
    out_dir : pathlib.Path
        Destination for finite summary.json, CSV source tables, standalone
        PNG/SVG figures and a Japanese methodological report.

    Returns
    -------
    pathlib.Path
        summary.json. Its complete/status/gaps/counts fields describe ledger
        completeness; individual terminal failures leave curve points incomplete.

    Raises
    ------
    OSError
        The output directory cannot be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gaps: list[JSONValue] = []
    points: list[_Row] = []
    attainment: list[_Row] = []
    raw: list[_Row] = []
    attempts: list[_Attempt] = []
    selected: dict[_Identity, _Attempt] = {}
    scope: _Row = {"phase": None, "cohort": None, "expected_runs": 0, "expected_seeds": 0}
    jobs: list[RunSpec] = []
    config: ExperimentConfig | None = None
    try:
        roots = {_root_for(Path(path).resolve()) for path in run_dirs}
        if len(roots) != 1 or None in roots:
            raise ValueError("all run directories must share one immutable snapshot root")
        root = cast("Path", next(iter(roots)))
        config = ExperimentConfig.model_validate(_read_json(root / "config.snapshot.json"))
        ledger = _read_json(root / "jobs.json")
        if not isinstance(ledger, list):
            raise ValueError("jobs.json must be a list of RunSpec objects")
        jobs = [RunSpec.model_validate(value) for value in ledger]
        scope = {
            "phase": config.phase,
            "cohort": config.measurement.cohort,
            "expected_runs": sum(
                len(config.seeds) * _repeat_count(config, c.condition_id, m.method, budget)
                for c in config.conditions
                for m in config.methods
                for budget in m.budgets
            ),
            "ledger_runs": sum(spec.measurement_id == config.measurement.cohort for spec in jobs),
            "expected_seeds": len(config.seeds),
            "conditions": [c.condition_id for c in config.conditions],
            "methods": [m.method for m in config.methods],
            "plan_revision": config.plan_revision,
        }
        _coverage(config, jobs, gaps)
        bindings = _read_bindings(root, config, jobs, gaps)
        selected, attempts, raw = _load_runs(root, config, jobs, gaps, bindings)
        points = _points(config, selected)
        attainment = _attainment(points, config)
    except (OSError, ValueError, ValidationError) as exc:
        _LOG.warning("Incomplete run ledger: %s", exc)
        gaps.append({"code": "invalid_snapshot", "detail": str(exc)})
    counts: _Row = {
        "expected": scope["expected_runs"],
        "terminal": sum(a.status in _TERMINAL for a in selected.values()),
    }
    statuses = Counter(a.status for a in selected.values())
    counts.update(
        {
            status: statuses[status]
            for status in (
                "success",
                "failed",
                "infeasible",
                "unsupported",
                "interrupted",
                "missing",
                "invalid",
                "pending",
                "running",
            )
        }
    )
    counts["missing"] = statuses["missing"] + sum(
        isinstance(gap, dict) and gap.get("code") == "planned_job_absent" for gap in gaps
    )
    complete = not gaps
    summary: _Row = {
        "schema_version": 1,
        "complete": complete,
        "status": "complete" if complete else "incomplete",
        "scope": scope,
        "timing_scopes": {"e2e_seconds": _PIPELINE_SCOPE},
        "counts": counts,
        "gaps": gaps,
        "points": cast("list[JSONValue]", points),
        "attainment": cast("list[JSONValue]", attainment),
        "validation_selection": cast("list[JSONValue]", _validation_selection(points)),
        "curve_segments": cast("list[JSONValue]", _segments(points)),
        "retry_attempts": cast("list[JSONValue]", [row for row in raw if not row["primary"]]),
        "bootstrap": {
            "seed": 1729,
            "resamples": 10000,
            "unit": "independent_seed",
            "ci": "percentile_95_pointwise",
        },
    }
    _write_table(out_dir / "raw_seeds.csv", raw)
    _write_table(
        out_dir / "points.csv",
        [{key: value for key, value in point.items() if key != "seed_values"} for point in points],
    )
    _write_table(out_dir / "attainment.csv", attainment)
    _write_table(out_dir / "validation_selected.csv", _validation_selection(points))
    _write_table(out_dir / "failures.csv", [row for row in raw if row["status"] != "success"])
    if config is not None and points:
        _plot(points, config, out_dir)
    _report(summary, out_dir)
    target = out_dir / "summary.json"
    _write_json(target, summary)
    _LOG.info("Wrote %s (%s; %d points)", target, summary["status"], len(points))
    return target
