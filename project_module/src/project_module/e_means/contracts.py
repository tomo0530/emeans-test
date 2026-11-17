"""Shared experiment configuration, array, and artifact contracts."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
type Method = Literal[
    "faiss_lloyd", "numpy_e_means", "mini_batch", "elkan", "torch_e_means", "torch_lloyd"
]
type PositiveInt = Annotated[int, Field(gt=0)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
type NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _check_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_finite(item)


class ContractModel(BaseModel):
    """Strict shared model with finite numbers and immutable fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def check_finite_input(cls, value: object) -> object:
        """Reject nonfinite numbers, including within nested JSON values."""
        _check_finite(value)
        return value

    @model_serializer(mode="wrap")
    def serialize_finite(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Reject nonfinite nested values even after a mutable value changes."""
        result = cast("dict[str, object]", handler(self))
        _check_finite(result)
        return result


def _check_budget(method: Method, budget: float) -> None:
    if method != "mini_batch" and not budget.is_integer():
        raise ValueError("iteration and epoch budgets must be integers")


class RunSpec(ContractModel):
    """One independently completed run and its measurement identity."""

    run_id: str
    dataset_id: str
    condition_id: str
    method: Method
    device: Literal["cpu", "cuda"]
    measurement_id: str
    n_train: PositiveInt
    k: PositiveInt
    seed: int
    data_seed: int
    threads: PositiveInt
    timing_repeat: NonNegativeInt
    budget: PositiveFloat
    params: dict[str, JSONValue]

    @model_validator(mode="after")
    def check_run(self) -> Self:
        """Check method budget and train cardinality."""
        _check_budget(self.method, self.budget)
        if self.k > self.n_train:
            raise ValueError("k cannot exceed n_train")
        return self


class TimingBreakdown(ContractModel):
    """Wall-clock seconds by phase; metadata records clock and scope."""

    cold_seconds: NonNegativeFloat | None = None
    transfer_seconds: NonNegativeFloat
    train_seconds: NonNegativeFloat
    assign_seconds: NonNegativeFloat
    evaluate_seconds: NonNegativeFloat
    save_seconds: NonNegativeFloat
    e2e_seconds: NonNegativeFloat
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ConditionConfig(ContractModel):
    """Dataset condition and the data seeds to prepare."""

    condition_id: str
    dataset_id: str
    n_train: PositiveInt
    k: PositiveInt
    dimension: PositiveInt
    data_seeds: list[int] = Field(min_length=1)
    generator_params: dict[str, JSONValue]

    @model_validator(mode="after")
    def check_cardinality(self) -> Self:
        """Reject a condition with more centers than points."""
        if self.k > self.n_train:
            raise ValueError("k cannot exceed n_train")
        return self


class MethodConfig(ContractModel):
    """Frozen method budgets and parameters."""

    method: Method
    budgets: list[PositiveFloat] = Field(min_length=1)
    params: dict[str, JSONValue]

    @model_validator(mode="after")
    def check_budgets(self) -> Self:
        """Require integer iteration and epoch budgets."""
        for budget in self.budgets:
            _check_budget(self.method, budget)
        return self


class RepeatRule(ContractModel):
    """Override timing repetitions for one condition, method, and budget."""

    condition_id: str
    method: Method
    budget: PositiveFloat
    repeats: PositiveInt

    @model_validator(mode="after")
    def check_budget(self) -> Self:
        """Check method-specific budget units."""
        _check_budget(self.method, self.budget)
        return self


class SentinelConfig(ContractModel):
    """Single-use parallel sentinel generation and exact participant membership."""

    group_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    participants: PositiveInt
    members: list[str] = Field(min_length=1)
    timeout_seconds: PositiveFloat = 300

    @model_validator(mode="after")
    def check_members(self) -> Self:
        """Require a complete, unique set of nonempty participant identifiers."""
        if (
            len(self.members) != self.participants
            or len(set(self.members)) != len(self.members)
            or any(not member for member in self.members)
        ):
            raise ValueError("sentinel members must be unique and match participants")
        return self


class MeasurementConfig(ContractModel):
    """Measurement cohort, repetition count, and holdout policy."""

    cohort: str
    repeats: PositiveInt
    evaluate_test: bool
    repeat_rules: list[RepeatRule] = Field(default_factory=list)
    sentinel: SentinelConfig | None = None


class ResourceConfig(ContractModel):
    """CPU affinity, GPU pool, and memory boundaries in bytes."""

    cpu_threads: PositiveInt
    cpu_affinity: list[NonNegativeInt]
    gpu_pool: list[NonNegativeInt]
    gpu_memory_limit_bytes: PositiveInt
    host_memory_limit_bytes: PositiveInt


class ExperimentConfig(ContractModel):
    """Versioned experiment conditions and frozen measurement settings."""

    schema_version: Literal[1] = 1
    plan_revision: str
    phase: Literal["pilot", "main"]
    conditions: list[ConditionConfig] = Field(min_length=1)
    methods: list[MethodConfig] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    measurement: MeasurementConfig
    resources: ResourceConfig

    @model_validator(mode="after")
    def check_phase(self) -> Self:
        """Forbid looking at test data during pilot."""
        if self.phase == "pilot" and self.measurement.evaluate_test:
            raise ValueError("pilot must not evaluate test")
        if self.measurement.sentinel is not None and (
            self.phase != "pilot" or not self.measurement.cohort.startswith("sentinel-parallel-")
        ):
            raise ValueError("sentinel coordination is only valid for a parallel pilot")
        return self


class DatasetManifest(ContractModel):
    """Versioned source, processed data, split, and license provenance."""

    schema_version: Literal[1] = 1
    dataset_id: str
    condition_id: str
    data_seed: int
    source_revision: str
    source_hash: str
    source_size_bytes: NonNegativeInt
    data_hash: str
    group_hash: str | None
    split_hash: str
    split_seed: int
    split_algorithm: str
    dimension: PositiveInt
    n_train: PositiveInt
    n_validation: PositiveInt
    n_test: PositiveInt
    normalized: bool
    generator_params: dict[str, JSONValue]
    license: str
    attribution: str


class RunRecord(ContractModel):
    """Versioned run state and its artifact paths."""

    schema_version: Literal[1] = 1
    run_key: str
    status: Literal[
        "pending", "running", "success", "failed", "infeasible", "unsupported", "interrupted"
    ]
    attempt: PositiveInt
    reason_code: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)


def _check_groups(groups: Mapping[str, JSONValue] | None) -> None:
    if groups is None:
        return
    _check_json(groups)
    for group in groups.values():
        if not isinstance(group, dict):
            raise ValueError("group evaluation must be a mapping")
        count = group.get("count")
        mse = group.get("mse")
        reason = group.get("reason")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("group count must be a nonnegative integer")
        if count == 0:
            if mse is not None or reason != "empty_group":
                raise ValueError("empty group must have null mse and empty_group reason")
        elif (
            not isinstance(mse, (float, int))
            or isinstance(mse, bool)
            or mse < 0
            or reason is not None
        ):
            raise ValueError("nonempty group must have a nonnegative mse and no reason")


class EvaluationMetrics(ContractModel):
    """JSON representation of exact MSE and cluster/group counts."""

    mse: NonNegativeFloat
    n_points: PositiveInt
    empty_clusters: NonNegativeInt
    cluster_sizes: list[NonNegativeInt] = Field(min_length=1)
    group_mse: dict[str, JSONValue] | None = None

    @model_validator(mode="after")
    def check_counts(self) -> Self:
        """Check the cluster totals and group results."""
        if (
            sum(self.cluster_sizes) != self.n_points
            or self.cluster_sizes.count(0) != self.empty_clusters
        ):
            raise ValueError("cluster sizes do not match evaluation counts")
        _check_groups(self.group_mse)
        return self


class RunMetrics(ContractModel):
    """Versioned run metrics with explicit holdout evaluation status."""

    schema_version: Literal[1] = 1
    run_key: str
    config_hash: str
    manifest_hash: str
    init_hash: str
    phase: Literal["pilot", "main"]
    train: EvaluationMetrics
    validation: EvaluationMetrics
    test: EvaluationMetrics | None
    test_reason: Literal["not_evaluated"] | None
    timing: TimingBreakdown
    stats: dict[str, JSONValue]

    @model_validator(mode="after")
    def check_test_policy(self) -> Self:
        """Forbid pilot test results and require a reason for omission."""
        if self.phase == "pilot" and self.test is not None:
            raise ValueError("pilot must not contain test evaluation")
        if (self.test is None) != (self.test_reason == "not_evaluated"):
            raise ValueError("missing test requires not_evaluated reason")
        return self


def _check_json(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _check_json(item)
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    raise ValueError("value is not JSON compatible")


def _check_matrix(value: npt.NDArray[np.float32], name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.ndim != 2
        or not value.flags.c_contiguous
        or not all(value.shape)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a nonempty, finite, C-contiguous float32 matrix")


def _check_int_vector(value: npt.NDArray[np.int64], name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.int64
        or value.ndim != 1
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{name} must be a C-contiguous int64 vector")


@dataclass(frozen=True)
class PreparedDataset:
    """Splits of finite C-contiguous float32 vectors shaped (rows, dimension).

    Group arrays, when present, are C-contiguous int64 vectors with one
    entry per corresponding split row. Hashes identify processed data/split.
    """

    train: npt.NDArray[np.float32]
    validation: npt.NDArray[np.float32]
    test: npt.NDArray[np.float32]
    train_groups: npt.NDArray[np.int64] | None
    validation_groups: npt.NDArray[np.int64] | None
    test_groups: npt.NDArray[np.int64] | None
    n_train: int
    split_hash: str
    data_hash: str
    manifest_path: Path

    def __post_init__(self) -> None:
        """Validate split dimensions, group alignment, and train count."""
        for name in ("train", "validation", "test"):
            array = getattr(self, name)
            _check_matrix(array, name)
            if array.shape[1] != self.train.shape[1]:
                raise ValueError("all splits must have the same dimension")
            groups = getattr(self, f"{name}_groups")
            if groups is not None:
                _check_int_vector(groups, f"{name}_groups")
                if len(groups) != len(array):
                    raise ValueError("group rows must match split rows")
        if self.n_train != len(self.train) or not self.split_hash or not self.data_hash:
            raise ValueError("n_train and data/split hashes must match the dataset")


@dataclass(frozen=True)
class TrainingResult:
    """Training output; centroids are finite float32 (k, dimension) rows."""

    centroids: npt.NDArray[np.float32]
    timing: TimingBreakdown
    stats: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        """Validate centroids and finite JSON statistics."""
        _check_matrix(self.centroids, "centroids")
        _check_json(self.stats)


@dataclass(frozen=True)
class Evaluation:
    """Exact MSE with int64 cluster sizes shaped (k,) and optional groups."""

    mse: float
    n_points: int
    empty_clusters: int
    cluster_sizes: npt.NDArray[np.int64]
    group_mse: Mapping[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        """Validate finite MSE and cluster/group counts."""
        _check_int_vector(self.cluster_sizes, "cluster_sizes")
        if (
            not math.isfinite(self.mse)
            or self.mse < 0
            or self.n_points <= 0
            or len(self.cluster_sizes) == 0
            or np.any(self.cluster_sizes < 0)
            or int(self.cluster_sizes.sum()) != self.n_points
            or int(np.count_nonzero(self.cluster_sizes == 0)) != self.empty_clusters
        ):
            raise ValueError("invalid evaluation or cluster counts")
        _check_groups(self.group_mse)


def derive_seed(seed: int, stream_name: str) -> int:
    """Derive a stable signed-safe seed for a named random stream.

    Parameters
    ----------
    seed : int
        Experiment seed.
    stream_name : str
        UTF-8 stream identifier.

    Returns
    -------
    int
        SHA256-derived integer in the range 0 to 2**31 - 1.
    """
    digest = hashlib.sha256(f"emeans-v1:{seed}:{stream_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
