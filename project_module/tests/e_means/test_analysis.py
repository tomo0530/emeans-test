"""Paired-seed statistics and immutable run-ledger reporting."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from project_module.e_means.analysis import summarize_runs
from project_module.e_means.contracts import (
    DatasetManifest,
    EvaluationMetrics,
    ExperimentConfig,
    Method,
    RunMetrics,
    RunRecord,
    RunSpec,
    TimingBreakdown,
)


def _make_runs(root: Path, *, repeats: int = 1) -> tuple[Path, list[RunSpec]]:
    root.mkdir()
    config = ExperimentConfig.model_validate(
        dict(
            plan_revision="test",
            phase="main",
            conditions=[
                {
                    "condition_id": "tiny",
                    "dataset_id": "synthetic",
                    "n_train": 4,
                    "k": 2,
                    "dimension": 1,
                    "data_seeds": list(range(10)),
                    "generator_params": {},
                }
            ],
            methods=[
                {"method": "faiss_lloyd", "budgets": [100], "params": {}},
                {"method": "numpy_e_means", "budgets": [1, 2, 3], "params": {}},
            ],
            seeds=list(range(10)),
            measurement={"cohort": "main-isolated", "repeats": repeats, "evaluate_test": True},
            resources={
                "cpu_threads": 1,
                "cpu_affinity": [0],
                "gpu_pool": [0],
                "gpu_memory_limit_bytes": 1024,
                "host_memory_limit_bytes": 4096,
            },
        )
    )
    (root / "config.snapshot.json").write_text(config.model_dump_json(), encoding="utf-8")
    specs = [
        RunSpec(
            run_id=f"{method}-{budget}-{seed}-{repeat}",
            dataset_id="synthetic",
            condition_id="tiny",
            method=cast("Method", method),
            device="cpu",
            measurement_id="main-isolated",
            n_train=4,
            k=2,
            seed=seed,
            data_seed=seed,
            threads=1,
            timing_repeat=repeat,
            budget=budget,
            params={},
        )
        for method, budgets in [("faiss_lloyd", [100]), ("numpy_e_means", [1, 2, 3])]
        for budget in budgets
        for seed in range(10)
        for repeat in range(repeats)
    ]
    (root / "jobs.json").write_text(json.dumps([s.model_dump() for s in specs]), encoding="utf-8")
    config_hash = _config_hash(root)
    keys = {
        spec.run_id: hashlib.sha256(f"key:{spec.run_id}".encode()).hexdigest() for spec in specs
    }
    inputs = {
        spec.run_id: {
            "config_hash": config_hash,
            "manifest_hash": hashlib.sha256(_manifest_bytes(spec)).hexdigest(),
            "init_hash": hashlib.sha256(
                f"init:{spec.condition_id}:{spec.seed}".encode()
            ).hexdigest(),
        }
        for spec in specs
    }
    (root / "run-keys.json").write_text(json.dumps(keys), encoding="utf-8")
    (root / "run-inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    for spec in specs:
        _write_attempt(root, spec, mse=float(spec.seed + 1), seconds=float(spec.seed + 1))
    return root, specs


def _config_hash(root: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            json.loads((root / "config.snapshot.json").read_text()),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _manifest_bytes(spec: RunSpec) -> bytes:
    return (
        DatasetManifest(
            dataset_id=spec.dataset_id,
            condition_id=spec.condition_id,
            data_seed=spec.data_seed,
            source_revision="fixture-v1",
            source_hash="source",
            source_size_bytes=16,
            data_hash="data",
            group_hash=None,
            split_hash="split",
            split_seed=42,
            split_algorithm="fixture",
            dimension=1,
            n_train=spec.n_train,
            n_validation=4,
            n_test=4,
            normalized=False,
            generator_params={},
            license="CC0",
            attribution="synthetic fixture",
        )
        .model_dump_json()
        .encode()
    )


def _write_attempt(
    root: Path,
    spec: RunSpec,
    *,
    mse: float = 1,
    seconds: float = 1,
    status: str = "success",
    attempt: int = 1,
) -> None:
    target = root / spec.run_id / f"attempt-{attempt:04d}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "spec.json").write_text(spec.model_dump_json(), encoding="utf-8")
    run_key = json.loads((root / "run-keys.json").read_text())[spec.run_id]
    record = RunRecord.model_validate({"run_key": run_key, "status": status, "attempt": attempt})
    (target / "status.json").write_text(record.model_dump_json(), encoding="utf-8")
    if status != "success":
        return
    metric = EvaluationMetrics(mse=mse, n_points=4, empty_clusters=0, cluster_sizes=[2, 2])
    inputs = json.loads((root / "run-inputs.json").read_text())[spec.run_id]
    (target / "dataset-manifest.json").write_bytes(_manifest_bytes(spec))
    metrics = RunMetrics(
        run_key=run_key,
        config_hash=inputs["config_hash"],
        manifest_hash=inputs["manifest_hash"],
        init_hash=inputs["init_hash"],
        phase="main",
        train=metric,
        validation=metric,
        test=metric,
        test_reason=None,
        timing=TimingBreakdown(
            transfer_seconds=0,
            train_seconds=seconds,
            assign_seconds=0.25,
            evaluate_seconds=0.75,
            save_seconds=0,
            e2e_seconds=seconds + 1,
            metadata={
                "e2e_scope": "resident-input-pipeline-v1",
                "evaluator_threads": {"faiss_threads": 1, "threadpools": []},
                "data_load_seconds": 3.0,
                "init_prepare_seconds": 0.1,
                "whole_trial_wall_seconds": seconds + 11,
                "coordination_wait_seconds": 0.0,
            },
        ),
        stats={},
    )
    (target / "metrics.json").write_text(metrics.model_dump_json(), encoding="utf-8")
    np.save(target / "centroids.npy", np.zeros((2, 1), dtype=np.float32))
    (target / "environment.json").write_text("{}", encoding="utf-8")


def _summary(root: Path, output: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", json.loads(summarize_runs([root], output).read_text(encoding="utf-8"))
    )


def _point(summary: dict[str, Any], budget: int = 1) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        next(
            p for p in summary["points"] if p["method"] == "numpy_e_means" and p["budget"] == budget
        ),
    )


def test_summary_mean_quality_median_time_and_repeat_order(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs", repeats=3)
    for spec in specs:
        _write_attempt(
            root,
            spec,
            mse=spec.seed + 1 if spec.timing_repeat == 0 else 1000,
            seconds=[spec.seed + 1, 100, spec.seed + 3][spec.timing_repeat],
        )

    summary = _summary(root, tmp_path / "report")
    point = _point(summary)

    assert point["test_mse"] == 5.5
    assert point["train_seconds"] == 7.5
    assert point["successful_seeds"] == 10
    assert point["status"] == "complete"
    assert point["repeat_mse_max_abs_difference"] == 999


def test_reference_is_same_cell_and_seed(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    for spec in specs:
        _write_attempt(root, spec, mse=1 if spec.method == "numpy_e_means" else spec.seed + 1)

    point = _point(_summary(root, tmp_path / "report"))

    assert point["delta_percent"] == pytest.approx(
        100 * (sum(1 / n for n in range(1, 11)) / 10 - 1)
    )
    assert point["delta_percent"] != pytest.approx(100 * (1 / 5.5 - 1))


def test_paired_bootstrap_uses_seed_as_unit(tmp_path: Path) -> None:
    root, _ = _make_runs(tmp_path / "runs")
    indices = np.random.default_rng(1729).integers(0, 10, size=(10000, 10))
    draws = np.arange(1, 11)[indices]

    point = _point(_summary(root, tmp_path / "report"))

    np.testing.assert_allclose(point["test_mse_ci95"], np.percentile(draws.mean(1), [2.5, 97.5]))
    np.testing.assert_allclose(
        point["train_seconds_ci95"], np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    )
    assert point["delta_percent_ci95"] == [0.0, 0.0]


def test_incomplete_points_are_not_bridged(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = next(s for s in specs if s.method == "numpy_e_means" and s.budget == 2 and s.seed == 4)
    _write_attempt(root, spec, status="failed")

    summary = _summary(root, tmp_path / "report")

    assert _point(summary, 2)["status"] == "incomplete"
    assert _point(summary, 2)["test_mse"] is None
    segments = next(
        s["segments"] for s in summary["curve_segments"] if s["method"] == "numpy_e_means"
    )
    assert segments == [[1.0], [3.0]]
    assert summary["complete"] is True  # failed is terminal, while its curve point is incomplete


def test_failed_first_attempt_is_not_replaced_by_retry_success(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = next(s for s in specs if s.method == "numpy_e_means" and s.budget == 1 and s.seed == 0)
    _write_attempt(root, spec, status="failed")
    _write_attempt(root, spec, attempt=2, mse=0.01)

    summary = _summary(root, tmp_path / "report")

    assert _point(summary)["successful_seeds"] == 9
    assert summary["counts"]["failed"] == 1
    assert any(
        a["run_id"] == spec.run_id and a["status"] == "success" for a in summary["retry_attempts"]
    )


def test_interrupted_attempt_can_resume_without_becoming_failed(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = specs[0]
    _write_attempt(root, spec, status="interrupted")
    _write_attempt(root, spec, attempt=2)

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is True
    assert summary["counts"]["success"] == 40


def test_paired_summary_preserves_unreached_and_failures(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    for spec in specs:
        _write_attempt(root, spec, mse=1 if spec.method == "faiss_lloyd" else 2)

    summary = _summary(root, tmp_path / "report")
    attainment = next(
        a
        for a in summary["attainment"]
        if a["method"] == "numpy_e_means" and a["delta_percent"] == 0
    )

    assert attainment["status"] == "unreached"
    assert attainment["train_seconds"] is None
    assert attainment["reached_seeds"] == 0
    assert all(s["status"] == "unreached" for s in attainment["seeds"])


def test_attainment_uses_fastest_observed_point_without_interpolation(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    for spec in specs:
        _write_attempt(
            root,
            spec,
            mse=1 if spec.method == "faiss_lloyd" or spec.budget == 2 else 2,
            seconds=7 if spec.budget == 2 else 1,
        )

    summary = _summary(root, tmp_path / "report")
    attainment = next(
        a
        for a in summary["attainment"]
        if a["method"] == "numpy_e_means" and a["delta_percent"] == 0
    )

    assert attainment["status"] == "reached"
    assert attainment["train_seconds"] == 7
    assert attainment["budget"] == 2


def test_zero_reference_retains_absolute_quality_and_marks_undefined(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    _write_attempt(root, specs[0], mse=0)

    summary = _summary(root, tmp_path / "report")

    assert _point(summary)["test_mse"] == 5.5
    assert _point(summary)["relative_status"] == "reference_undefined"
    assert _point(summary)["delta_percent"] is None
    attainment = next(a for a in summary["attainment"] if a["method"] == "numpy_e_means")
    assert attainment["status"] == "reference_undefined"


def test_missing_run_still_produces_partial_report_and_source_tables(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    (root / specs[0].run_id / "attempt-0001" / "status.json").unlink()
    output = tmp_path / "report"

    summary = _summary(root, output)

    assert summary["complete"] is False
    assert summary["status"] == "incomplete"
    assert summary["gaps"]
    assert (output / "raw_seeds.csv").is_file()
    assert (output / "points.csv").is_file()
    assert (output / "curves-tiny.png").is_file()
    assert (output / "curves-tiny.svg").is_file()
    assert (output / "failures-tiny.png").is_file()
    assert (output / "report.md").is_file()


def test_failure_heatmap_data_preserves_terminal_categories(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = next(s for s in specs if s.method == "numpy_e_means" and s.budget == 1 and s.seed == 0)
    _write_attempt(root, spec, status="infeasible")

    summary = _summary(root, tmp_path / "report")

    assert _point(summary)["seed_values"][0]["failure_status"] == "infeasible"
    assert summary["counts"]["infeasible"] == 1


def test_reference_from_other_cell_does_not_define_ratio(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    reference = next(s for s in specs if s.method == "faiss_lloyd" and s.seed == 2)
    (root / reference.run_id / "attempt-0001" / "centroids.npy").unlink()

    summary = _summary(root, tmp_path / "report")

    assert _point(summary)["relative_status"] == "reference_undefined"
    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


def test_invalid_run_key_is_retained_as_partial_report(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    metric_path = root / specs[0].run_id / "attempt-0001" / "metrics.json"
    metric = json.loads(metric_path.read_text())
    metric["run_key"] = "other-run"
    metric_path.write_text(json.dumps(metric))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


def test_method_absent_from_expected_jobs_is_a_scope_gap(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    (root / "jobs.json").write_text(
        json.dumps([s.model_dump() for s in specs if s.method == "faiss_lloyd"])
    )

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert any(g["code"] == "method_absent" for g in summary["gaps"])
    assert summary["counts"]["expected"] == 40
    assert summary["counts"]["missing"] == 30


def test_duplicate_seed_in_snapshot_is_not_an_independent_replicate(tmp_path: Path) -> None:
    root, _ = _make_runs(tmp_path / "runs")
    path = root / "config.snapshot.json"
    config = json.loads(path.read_text())
    config["seeds"] = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    path.write_text(json.dumps(config))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert all(point["status"] != "complete" for point in summary["points"])


def test_cli_strict_mode_reports_missing_root_with_exit_two(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts/e_means/analyze.py"
    output = tmp_path / "report"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runs-root",
            str(tmp_path / "absent"),
            "--output-root",
            str(output),
            "--require-complete",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    assert json.loads((output / "summary.json").read_text())["complete"] is False


def test_cli_strict_mode_accepts_terminal_failure_but_preserves_incomplete_curve(
    tmp_path: Path,
) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    _write_attempt(root, specs[-1], status="failed")
    script = Path(__file__).resolve().parents[3] / "scripts/e_means/analyze.py"
    output = tmp_path / "report"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runs-root",
            str(root),
            "--output-root",
            str(output),
            "--require-complete",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] is True
    assert _point(summary, 3)["status"] == "incomplete"


def test_validation_selected_table_never_uses_test_to_choose_budget(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    for spec in specs:
        if spec.method == "numpy_e_means":
            path = root / spec.run_id / "attempt-0001" / "metrics.json"
            metric = json.loads(path.read_text())
            metric["validation"]["mse"] = float(spec.budget)
            metric["test"]["mse"] = 20.0 / spec.budget
            path.write_text(json.dumps(metric))

    summary = _summary(root, tmp_path / "report")
    selected = next(
        row for row in summary["validation_selection"] if row["method"] == "numpy_e_means"
    )

    assert selected["budget"] == 1
    assert selected["test_mse"] == 20
    assert (tmp_path / "report/validation_selected.csv").is_file()


def test_main_synthetic_data_seed_must_match_paired_learning_seed(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = next(s for s in specs if s.method == "numpy_e_means" and s.budget == 1 and s.seed == 2)
    wrong = spec.model_copy(update={"data_seed": 3})
    changed = [wrong if s.run_id == spec.run_id else s for s in specs]
    (root / "jobs.json").write_text(json.dumps([s.model_dump() for s in changed]))
    _write_attempt(root, wrong)

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert _point(summary)["status"] == "incomplete"
    assert any(g["code"] == "job_outside_snapshot" for g in summary["gaps"])


def test_artifact_paths_cannot_escape_attempt_directory(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    path = root / specs[0].run_id / "attempt-0001" / "status.json"
    status = json.loads(path.read_text())
    status["artifact_paths"] = {"metrics.json": "../../outside.json"}
    path.write_text(json.dumps(status))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


@pytest.mark.parametrize("field", ["manifest_hash", "init_hash"])
def test_metrics_hash_is_checked_against_frozen_run_inputs(tmp_path: Path, field: str) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    path = root / specs[0].run_id / "attempt-0001/metrics.json"
    metric = json.loads(path.read_text())
    metric[field] = "f" * 64
    path.write_text(json.dumps(metric))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


def test_matching_status_and_metric_keys_cannot_replace_frozen_run_key(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    target = root / specs[0].run_id / "attempt-0001"
    for name in ["status.json", "metrics.json"]:
        value = json.loads((target / name).read_text())
        value["run_key"] = "f" * 64
        (target / name).write_text(json.dumps(value))
    output = tmp_path / "report"
    script = Path(__file__).resolve().parents[3] / "scripts/e_means/analyze.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runs-root",
            str(root),
            "--output-root",
            str(output),
            "--require-complete",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["counts"]["invalid"] == 1


@pytest.mark.parametrize("filename", ["run-keys.json", "run-inputs.json"])
def test_frozen_identity_ledgers_are_required(tmp_path: Path, filename: str) -> None:
    root, _ = _make_runs(tmp_path / "runs")
    (root / filename).unlink()

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["gaps"]
    assert (tmp_path / "report/report.md").is_file()


@pytest.mark.parametrize("filename", ["run-keys.json", "run-inputs.json"])
def test_missing_run_identity_entry_invalidates_the_attempt(tmp_path: Path, filename: str) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    path = root / filename
    ledger = json.loads(path.read_text())
    del ledger[specs[0].run_id]
    path.write_text(json.dumps(ledger))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


@pytest.mark.parametrize("mode", ["missing", "raw_bytes_changed"])
def test_success_requires_the_exact_manifest_bytes(tmp_path: Path, mode: str) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    path = root / specs[0].run_id / "attempt-0001/dataset-manifest.json"
    if mode == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" ")

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "other-dataset"),
        ("condition_id", "other-condition"),
        ("data_seed", 99),
        ("dimension", 2),
        ("n_train", 5),
        ("n_validation", 5),
        ("n_test", 5),
    ],
)
def test_manifest_identity_shape_and_split_counts_match_spec_and_metrics(
    tmp_path: Path,
    field: str,
    value: str | int,
) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = specs[0]
    target = root / spec.run_id / "attempt-0001"
    manifest = json.loads((target / "dataset-manifest.json").read_text())
    manifest[field] = value
    manifest_bytes = json.dumps(manifest).encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    (target / "dataset-manifest.json").write_bytes(manifest_bytes)
    inputs = json.loads((root / "run-inputs.json").read_text())
    inputs[spec.run_id]["manifest_hash"] = digest
    (root / "run-inputs.json").write_text(json.dumps(inputs))
    metric = json.loads((target / "metrics.json").read_text())
    metric["manifest_hash"] = digest
    (target / "metrics.json").write_text(json.dumps(metric))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


def test_terminal_worker_failure_needs_key_but_no_manifest(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = specs[-1]
    _write_attempt(root, spec, status="failed")
    (root / spec.run_id / "attempt-0001/dataset-manifest.json").unlink()

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is True
    assert summary["counts"]["failed"] == 1


def test_terminal_failure_with_wrong_frozen_key_is_invalid(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = specs[-1]
    _write_attempt(root, spec, status="failed")
    path = root / spec.run_id / "attempt-0001/status.json"
    record = json.loads(path.read_text())
    record["run_key"] = "f" * 64
    path.write_text(json.dumps(record))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


def test_invalid_retry_identity_is_a_gap_without_replacing_primary_failure(tmp_path: Path) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    spec = specs[-1]
    _write_attempt(root, spec, status="failed")
    _write_attempt(root, spec, attempt=2)
    path = root / spec.run_id / "attempt-0002/status.json"
    record = json.loads(path.read_text())
    record["run_key"] = "f" * 64
    path.write_text(json.dumps(record))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["failed"] == 1
    assert any(gap["code"] == "invalid" and gap["run_id"] == spec.run_id for gap in summary["gaps"])


@pytest.mark.parametrize("scope", [None, "legacy-full-trial-e2e"])
def test_success_without_resident_input_pipeline_scope_is_invalid(
    tmp_path: Path, scope: str | None
) -> None:
    root, specs = _make_runs(tmp_path / "runs")
    path = root / specs[0].run_id / "attempt-0001/metrics.json"
    metric = json.loads(path.read_text())
    metric["timing"]["metadata"] = {"e2e_scope": scope} if scope else {}
    path.write_text(json.dumps(metric))

    summary = _summary(root, tmp_path / "report")

    assert summary["complete"] is False
    assert summary["counts"]["invalid"] == 1


def test_resident_input_pipeline_label_and_diagnostics_are_explicit(tmp_path: Path) -> None:
    root, _ = _make_runs(tmp_path / "runs")
    output = tmp_path / "report"

    summary = _summary(root, output)
    with (output / "raw_seeds.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))

    assert summary["timing_scopes"]["e2e_seconds"] == "resident-input-pipeline-v1"
    assert row["resident_input_pipeline_seconds"] == "2.0"
    assert "e2e_seconds" not in row
    assert row["e2e_scope"] == "resident-input-pipeline-v1"
    assert row["assign_seconds"] == "0.25"
    assert row["evaluate_seconds"] == "0.75"
    assert row["whole_trial_wall_seconds"] == "12.0"
    assert json.loads(row["evaluator_threads"])["faiss_threads"] == 1
    assert "Resident-input pipeline" in (output / "report.md").read_text()
