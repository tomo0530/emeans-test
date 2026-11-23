"""Pure eight-GPU sentinel overlap qualification against hand-built windows."""

from copy import deepcopy
from typing import cast

import pytest
from project_module.e_means.contracts import JSONValue
from project_module.e_means.sentinel_analysis import audit_concurrent_sentinel

_EXPECTED_UUIDS = [f"GPU-{index}" for index in range(8)]


def _records() -> list[dict[str, JSONValue]]:
    return [
        {
            "gpu_uuid": uuid,
            "input_hash": "same-data",
            "init_hash": "same-init",
            "timing": {
                "metadata": {
                    "train_started_at_ns": 1_000_000_000 + index * 10,
                    "train_finished_at_ns": 1_000_000_100 + index * 10,
                    "train_started_monotonic_ns": 500_000 + index * 10,
                    "train_finished_monotonic_ns": 500_100 + index * 10,
                }
            },
        }
        for index, uuid in enumerate(_EXPECTED_UUIDS)
    ]


def _metadata(record: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return cast("dict[str, JSONValue]", cast("dict[str, JSONValue]", record["timing"])["metadata"])


def test_common_overlap_start_skew_and_fraction_are_hand_computed() -> None:
    records = _records()
    before = deepcopy(records)

    result = audit_concurrent_sentinel(
        records, expected_gpu_uuids=_EXPECTED_UUIDS, group_status=True
    )

    assert result["qualified"] is True
    assert result["reasons"] == []
    wall = cast("dict[str, JSONValue]", result["wall_clock"])
    monotonic = cast("dict[str, JSONValue]", result["monotonic_clock"])
    assert wall["common_window_ns"] == 30
    assert wall["overlap_ns"] == 30
    assert wall["start_skew_ns"] == 70
    assert wall["min_duration_ns"] == 100
    assert wall["overlap_fraction_of_min_duration"] == 0.3
    assert monotonic["common_window_ns"] == 30
    assert records == before


@pytest.mark.parametrize("mode", ["duplicate", "missing", "unexpected", "invalid_expected"])
def test_exact_eight_unique_expected_gpu_uuids_are_required(mode: str) -> None:
    records = _records()
    expected = _EXPECTED_UUIDS.copy()
    if mode == "duplicate":
        records[-1]["gpu_uuid"] = records[0]["gpu_uuid"]
    elif mode == "missing":
        records.pop()
    elif mode == "unexpected":
        records[-1]["gpu_uuid"] = "GPU-other"
    else:
        expected[-1] = expected[0]

    result = audit_concurrent_sentinel(records, expected_gpu_uuids=expected, group_status=True)

    assert result["qualified"] is False
    assert result["reasons"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_started_at_ns", float("nan")),
        ("train_finished_at_ns", float("inf")),
        ("train_finished_at_ns", 1_000_000_000),
        ("train_finished_monotonic_ns", 500_000),
        ("train_started_monotonic_ns", None),
    ],
)
def test_nonfinite_missing_or_nonpositive_windows_are_not_qualified(
    field: str, value: JSONValue
) -> None:
    records = _records()
    _metadata(records[0])[field] = value

    result = audit_concurrent_sentinel(
        records, expected_gpu_uuids=_EXPECTED_UUIDS, group_status=True
    )

    assert result["qualified"] is False
    assert result["reasons"]


@pytest.mark.parametrize("clock", ["wall", "monotonic"])
def test_all_eight_training_windows_must_have_positive_common_overlap(clock: str) -> None:
    records = _records()
    metadata = _metadata(records[-1])
    start, end = (
        ("train_started_at_ns", "train_finished_at_ns")
        if clock == "wall"
        else ("train_started_monotonic_ns", "train_finished_monotonic_ns")
    )
    metadata[start] = cast("int", metadata[start]) + 1000
    metadata[end] = cast("int", metadata[end]) + 1000

    result = audit_concurrent_sentinel(
        records, expected_gpu_uuids=_EXPECTED_UUIDS, group_status=True
    )

    assert result["qualified"] is False
    assert f"no_{clock}_overlap" in cast("list[JSONValue]", result["reasons"])


@pytest.mark.parametrize(
    "status", [False, {"completed": True, "aborted": True}, {"completed": False, "aborted": False}]
)
def test_aborted_or_incomplete_group_is_not_qualified(status: bool | dict[str, JSONValue]) -> None:
    records = _records()

    result = audit_concurrent_sentinel(
        records, expected_gpu_uuids=_EXPECTED_UUIDS, group_status=status
    )

    assert result["qualified"] is False
    assert result["reasons"]


@pytest.mark.parametrize("field", ["input_hash", "init_hash"])
def test_present_input_and_initialization_hashes_must_match_all_trials(field: str) -> None:
    records = _records()
    records[-1][field] = "different"

    result = audit_concurrent_sentinel(
        records, expected_gpu_uuids=_EXPECTED_UUIDS, group_status=True
    )

    assert result["qualified"] is False
    assert f"{field}_mismatch" in cast("list[JSONValue]", result["reasons"])


def test_explicit_completed_group_json_is_accepted() -> None:
    result = audit_concurrent_sentinel(
        _records(),
        expected_gpu_uuids=_EXPECTED_UUIDS,
        group_status={"completed": True, "aborted": False},
    )

    assert result["qualified"] is True
