"""Behavioral tests for the shared experiment contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from project_module.e_means.contracts import (
    DatasetManifest,
    Evaluation,
    ExperimentConfig,
    PreparedDataset,
    RunMetrics,
    RunRecord,
    RunSpec,
    TimingBreakdown,
    TrainingResult,
    derive_seed,
)
from pydantic import ValidationError


def run_spec(**changes: object) -> RunSpec:
    values = dict(
        run_id="r",
        dataset_id="d",
        condition_id="c",
        method="faiss_lloyd",
        device="cpu",
        measurement_id="m",
        n_train=4,
        k=2,
        seed=1,
        data_seed=42,
        threads=1,
        timing_repeat=0,
        budget=2.0,
        params={},
    )
    values.update(changes)
    return RunSpec.model_validate(values)


def config_data() -> dict[str, object]:
    return dict(
        schema_version=1,
        plan_revision="p1",
        phase="pilot",
        conditions=[
            dict(
                condition_id="c",
                dataset_id="d",
                n_train=4,
                k=2,
                dimension=2,
                data_seeds=[42],
                generator_params={"scale": 1.0},
            )
        ],
        methods=[dict(method="faiss_lloyd", budgets=[1, 2], params={})],
        seeds=[1, 2],
        measurement=dict(cohort="main-parallel", repeats=1, evaluate_test=False),
        resources=dict(
            cpu_threads=1,
            cpu_affinity=[0],
            gpu_pool=[0],
            gpu_memory_limit_bytes=1024,
            host_memory_limit_bytes=4096,
        ),
    )


def timing_data() -> dict[str, object]:
    return dict(
        cold_seconds=None,
        transfer_seconds=0.1,
        train_seconds=0.2,
        assign_seconds=0.3,
        evaluate_seconds=0.4,
        save_seconds=0.5,
        e2e_seconds=1.5,
        metadata={"clock": "wall"},
    )


def test_run_spec_rejects_unknown_fields_and_invalid_budget() -> None:
    for changes in (
        {"unknown": 1},
        {"budget": 0},
        {"budget": float("inf")},
        {"budget": 1.5},
        {"method": "unknown"},
        {"k": 5},
    ):
        with pytest.raises(ValidationError):
            run_spec(**changes)
    assert run_spec(method="mini_batch", budget=0.25).budget == 0.25


def test_timing_rejects_negative_seconds() -> None:
    with pytest.raises(ValidationError):
        TimingBreakdown.model_validate({**timing_data(), "train_seconds": -1})


def test_run_spec_is_frozen() -> None:
    spec = run_spec()
    with pytest.raises(ValidationError):
        spec.budget = 5


def test_config_yaml_json_round_trip() -> None:
    config = ExperimentConfig.model_validate(yaml.safe_load(yaml.safe_dump(config_data())))
    assert ExperimentConfig.model_validate_json(config.model_dump_json()) == config
    assert config.schema_version == 1
    assert config.conditions[0].generator_params["scale"] == 1.0
    bad = config_data()
    bad["measurement"] = {"cohort": "x", "repeats": 1, "evaluate_test": False, "unexpected": 1}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_pilot_config_rejects_test_evaluation() -> None:
    data = config_data()
    data["measurement"] = {"cohort": "x", "repeats": 1, "evaluate_test": True}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_artifact_round_trip_rejects_nonfinite() -> None:
    metrics = dict(
        schema_version=1,
        run_key="key",
        config_hash="c",
        manifest_hash="m",
        init_hash="i",
        phase="pilot",
        train=dict(mse=1.0, n_points=2, empty_clusters=0, cluster_sizes=[1, 1], group_mse=None),
        validation=dict(
            mse=2.0, n_points=2, empty_clusters=0, cluster_sizes=[1, 1], group_mse=None
        ),
        test=None,
        test_reason="not_evaluated",
        timing=timing_data(),
        stats={"nested": [1.0]},
    )
    valid = RunMetrics.model_validate(metrics)
    assert RunMetrics.model_validate_json(valid.model_dump_json()) == valid
    assert json.loads(valid.model_dump_json())["test_reason"] == "not_evaluated"
    for changed in (
        {**metrics, "stats": {"nested": [float("nan")]}},
        {**metrics, "train": {**metrics["train"], "mse": float("inf")}},
        {**metrics, "test": metrics["train"], "test_reason": None},
    ):
        with pytest.raises(ValidationError):
            RunMetrics.model_validate(changed)


def test_named_random_streams_are_reproducible_signed_safe() -> None:
    assert derive_seed(42, "init") == 211240967
    assert derive_seed(42, "init") == derive_seed(42, "init")
    assert derive_seed(42, "init") != derive_seed(42, "data/noise")
    assert all(0 <= derive_seed(i, "method/faiss_lloyd") <= 2**31 - 1 for i in range(100))


def test_prepared_dataset_checks_shapes_groups_and_finiteness() -> None:
    train = np.ones((4, 2), dtype=np.float32)
    held = np.ones((2, 2), dtype=np.float32)
    args = dict(
        train=train,
        validation=held,
        test=held,
        train_groups=np.arange(4, dtype=np.int64),
        validation_groups=None,
        test_groups=None,
        n_train=4,
        split_hash="s",
        data_hash="d",
        manifest_path=Path("manifest.json"),
    )
    assert PreparedDataset(**args).n_train == 4
    for changed in (
        {"n_train": 3},
        {"test": np.ones((2, 3), dtype=np.float32)},
        {"train_groups": np.arange(3, dtype=np.int64)},
        {"train": train.astype(np.float64)},
        {"train": np.full((4, 2), np.nan, dtype=np.float32)},
        {"train": train[:, ::-1]},
    ):
        with pytest.raises(ValueError):
            PreparedDataset(**(args | changed))


def test_array_result_validation() -> None:
    timing = TimingBreakdown.model_validate(timing_data())
    result = TrainingResult(
        centroids=np.ones((2, 2), dtype=np.float32), timing=timing, stats={"iterations": 2}
    )
    assert result.centroids.shape == (2, 2)
    with pytest.raises(ValueError):
        TrainingResult(centroids=np.full((2, 2), np.inf, dtype=np.float32), timing=timing, stats={})
    evaluation = Evaluation(
        mse=1.0,
        n_points=2,
        empty_clusters=1,
        cluster_sizes=np.array([2, 0], dtype=np.int64),
        group_mse={"0": {"count": 0, "mse": None, "reason": "empty_group"}},
    )
    assert evaluation.empty_clusters == 1
    with pytest.raises(ValueError):
        Evaluation(
            mse=float("nan"),
            n_points=2,
            empty_clusters=0,
            cluster_sizes=np.array([1, 1], dtype=np.int64),
        )


def test_record_and_manifest_json_round_trip() -> None:
    record = RunRecord.model_validate(
        dict(
            schema_version=1,
            run_key="k",
            status="success",
            attempt=1,
            reason_code=None,
            started_at="2026-11-17T00:00:00Z",
            finished_at="2026-11-17T00:01:00Z",
            artifact_paths={"metrics": "metrics.json"},
        )
    )
    assert RunRecord.model_validate_json(record.model_dump_json()) == record
    manifest = DatasetManifest.model_validate(
        dict(
            schema_version=1,
            dataset_id="d",
            condition_id="c",
            data_seed=42,
            source_revision="v1",
            source_hash="s",
            source_size_bytes=10,
            data_hash="d",
            group_hash=None,
            split_hash="x",
            split_seed=42,
            split_algorithm="group",
            dimension=2,
            n_train=4,
            n_validation=2,
            n_test=2,
            normalized=False,
            generator_params={},
            license="CC0",
            attribution="example",
        )
    )
    assert DatasetManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_nested_json_values_reject_nonfinite_numbers() -> None:
    with pytest.raises(ValidationError):
        run_spec(params={"nested": {"values": [float("nan")]}})
    with pytest.raises(ValidationError):
        TimingBreakdown.model_validate({**timing_data(), "metadata": {"clock": float("inf")}})
    with pytest.raises(ValueError):
        TrainingResult(
            centroids=np.ones((1, 1), dtype=np.float32),
            timing=TimingBreakdown.model_validate(timing_data()),
            stats={"nested": [float("inf")]},
        )


def test_evaluation_rejects_inconsistent_sizes_and_group_mse() -> None:
    for values in (
        dict(mse=1.0, n_points=3, empty_clusters=0, cluster_sizes=np.array([1, 1], dtype=np.int64)),
        dict(mse=1.0, n_points=2, empty_clusters=1, cluster_sizes=np.array([1, 1], dtype=np.int64)),
        dict(
            mse=1.0,
            n_points=2,
            empty_clusters=0,
            cluster_sizes=np.array([1, 1], dtype=np.int64),
            group_mse={"0": {"count": 0, "mse": 0.0, "reason": None}},
        ),
    ):
        with pytest.raises(ValueError):
            Evaluation(**values)


def test_mutated_nested_json_cannot_be_serialized_as_nonfinite() -> None:
    spec = run_spec(params={"samples": [1.0]})
    samples = spec.params["samples"]
    assert isinstance(samples, list)
    samples.append(float("nan"))
    with pytest.raises(ValueError):
        spec.model_dump_json()


def test_tiny_config_fixture_matches_contract() -> None:
    fixture = Path(__file__).parent / "fixtures" / "tiny-config.json"
    config = ExperimentConfig.model_validate_json(fixture.read_text(encoding="utf-8"))
    assert config.conditions[0].n_train == 4
    assert config.phase == "pilot"


def test_measurement_supports_per_budget_repetition_rules() -> None:
    data = config_data()
    data["measurement"] = {
        "cohort": "main-parallel",
        "repeats": 1,
        "evaluate_test": False,
        "repeat_rules": [{"condition_id": "c", "method": "faiss_lloyd", "budget": 2, "repeats": 3}],
    }
    config = ExperimentConfig.model_validate(data)
    assert config.measurement.repeat_rules[0].repeats == 3
    assert ExperimentConfig.model_validate_json(config.model_dump_json()) == config


def test_training_stats_reject_non_json_values() -> None:
    with pytest.raises(ValueError):
        TrainingResult(
            centroids=np.ones((1, 1), dtype=np.float32),
            timing=TimingBreakdown.model_validate(timing_data()),
            stats={"invalid": object()},
        )


def sentinel_config_data() -> dict[str, object]:
    data = config_data()
    data["measurement"] = {
        "cohort": "sentinel-parallel-GPU-0",
        "repeats": 3,
        "evaluate_test": False,
        "sentinel": {
            "group_id": "pilot-generation-1",
            "participants": 8,
            "members": [f"GPU-{i}" for i in range(8)],
            "timeout_seconds": 300,
        },
    }
    return data


def test_sentinel_configuration_round_trips_frozen_group_members() -> None:
    config = ExperimentConfig.model_validate(sentinel_config_data())
    assert config.measurement.sentinel is not None
    assert config.measurement.sentinel.participants == 8
    assert config.measurement.sentinel.group_id == "pilot-generation-1"
    assert ExperimentConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("members", ["GPU-0"] * 8),
        ("participants", 7),
        ("timeout_seconds", 0),
        ("group_id", "../../other"),
    ],
)
def test_sentinel_rejects_invalid_group_membership(field: str, value: object) -> None:
    data = json.loads(json.dumps(sentinel_config_data()))
    data["measurement"]["sentinel"][field] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize(
    ("phase", "cohort"), [("main", "sentinel-parallel-GPU-0"), ("pilot", "main-parallel")]
)
def test_sentinel_configuration_is_only_valid_for_parallel_pilot(phase: str, cohort: str) -> None:
    data = json.loads(json.dumps(sentinel_config_data()))
    data["phase"] = phase
    data["measurement"]["cohort"] = cohort
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_gpu_determinism_profile_defaults_false_and_round_trips() -> None:
    legacy = ExperimentConfig.model_validate(config_data())
    assert legacy.resources.gpu_deterministic_algorithms is False
    data = config_data()
    resources = dict(data["resources"])
    resources["gpu_deterministic_algorithms"] = True
    data["resources"] = resources
    deterministic = ExperimentConfig.model_validate(data)
    assert deterministic.resources.gpu_deterministic_algorithms is True
    assert ExperimentConfig.model_validate_json(deterministic.model_dump_json()) == deterministic
