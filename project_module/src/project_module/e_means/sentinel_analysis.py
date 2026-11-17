"""Pure qualification of actual eight-GPU training-window overlap."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from project_module.e_means.contracts import JSONValue

_TIMESTAMP_LIMIT = 2**63
_HASH_FIELDS = ("input_hash", "data_hash", "manifest_hash", "init_hash")


def _group_reason(group_status: bool | Mapping[str, JSONValue]) -> str | None:
    if isinstance(group_status, bool):
        return None if group_status else "group_aborted_or_incomplete"
    if group_status.get("aborted") is True:
        return "group_aborted"
    if group_status.get("completed") is True and group_status.get("aborted") is False:
        return None
    return "group_incomplete"


def _timestamp(value: JSONValue) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < _TIMESTAMP_LIMIT


def _window(
    records: Sequence[Mapping[str, JSONValue]],
    start_key: str,
    finish_key: str,
) -> dict[str, JSONValue] | None:
    if len(records) != 8:
        return None
    starts: list[int] = []
    finishes: list[int] = []
    for record in records:
        timing = record.get("timing")
        metadata = timing.get("metadata") if isinstance(timing, dict) else None
        if not isinstance(metadata, dict):
            return None
        start, finish = metadata.get(start_key), metadata.get(finish_key)
        if not _timestamp(start) or not _timestamp(finish):
            return None
        start_ns, finish_ns = cast("int", start), cast("int", finish)
        if finish_ns <= start_ns:
            return None
        starts.append(start_ns)
        finishes.append(finish_ns)
    common_start, common_finish = max(starts), min(finishes)
    common_window = common_finish - common_start
    minimum_duration = min(end - start for start, end in zip(starts, finishes, strict=True))
    overlap = max(0, common_window)
    return {
        "common_start_ns": common_start,
        "common_finish_ns": common_finish,
        "common_window_ns": common_window,
        "overlap_ns": overlap,
        "start_skew_ns": max(starts) - min(starts),
        "min_duration_ns": minimum_duration,
        "overlap_fraction_of_min_duration": overlap / minimum_duration,
    }


def audit_concurrent_sentinel(
    records: Sequence[Mapping[str, JSONValue]],
    *,
    expected_gpu_uuids: Sequence[str],
    group_status: bool | Mapping[str, JSONValue],
) -> Mapping[str, JSONValue]:
    """Qualify actual overlap of eight exact GPU training participants.

    Parameters
    ----------
    records : Sequence[Mapping[str, JSONValue]]
        One mapping per trial containing ``gpu_uuid`` and ``timing.metadata``.
        Metadata must contain integer nanoseconds ``train_started_at_ns``,
        ``train_finished_at_ns``, ``train_started_monotonic_ns`` and
        ``train_finished_monotonic_ns``, each in [0, 2**63). Both clock windows
        must be positive per trial. Optional input_hash/data_hash/manifest_hash/
        init_hash fields, when present in any record, must be nonempty identical
        strings in all eight records. This function reads no artifact or state.
    expected_gpu_uuids : Sequence[str]
        Exactly eight distinct nonempty GPU UUIDs selected for this group.
    group_status : bool or Mapping[str, JSONValue]
        Explicit caller-supplied qualification of the coordination generation.
        True means completed without abort. A mapping must state
        ``completed=True`` and ``aborted=False``; aborted/incomplete groups fail.

    Returns
    -------
    Mapping[str, JSONValue]
        Finite JSON with qualified, reasons, participant/UUID diagnostics,
        verified_hashes, and wall_clock/monotonic_clock window diagnostics.
        Each clock reports signed min(finish)-max(start), nonnegative overlap,
        start skew and overlap/minimum individual duration. Both common windows
        must be strictly positive. No speed or interference threshold is applied.
    """
    reasons: list[JSONValue] = []
    group_reason = _group_reason(group_status)
    if group_reason:
        reasons.append(group_reason)
    expected = list(expected_gpu_uuids)
    if len(expected) != 8 or len(set(expected)) != 8 or not all(expected):
        reasons.append("invalid_expected_gpu_uuids")
    observed = [record.get("gpu_uuid") for record in records]
    uuids = [uuid for uuid in observed if isinstance(uuid, str) and uuid]
    if len(records) != 8:
        reasons.append("participant_count")
    if len(uuids) != len(records):
        reasons.append("invalid_gpu_uuid")
    duplicates = sorted(uuid for uuid, count in Counter(uuids).items() if count > 1)
    if duplicates:
        reasons.append("duplicate_gpu_uuid")
    if set(uuids) != set(expected):
        reasons.append("gpu_uuid_set_mismatch")
    verified_hashes: dict[str, JSONValue] = {}
    for field in _HASH_FIELDS:
        if not any(field in record for record in records):
            continue
        values = [record.get(field) for record in records]
        if not all(isinstance(value, str) and value for value in values):
            reasons.append(f"{field}_missing_or_invalid")
        elif len(set(cast("list[str]", values))) != 1:
            reasons.append(f"{field}_mismatch")
        else:
            verified_hashes[field] = values[0]
    wall = _window(records, "train_started_at_ns", "train_finished_at_ns")
    monotonic = _window(records, "train_started_monotonic_ns", "train_finished_monotonic_ns")
    for name, window in (("wall", wall), ("monotonic", monotonic)):
        if window is None:
            reasons.append(f"invalid_{name}_window")
        elif cast("int", window["common_window_ns"]) <= 0:
            reasons.append(f"no_{name}_overlap")
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "participant_count": len(records),
        "expected_gpu_uuids": cast("list[JSONValue]", expected),
        "observed_gpu_uuids": cast("list[JSONValue]", uuids),
        "missing_gpu_uuids": cast("list[JSONValue]", sorted(set(expected) - set(uuids))),
        "unexpected_gpu_uuids": cast("list[JSONValue]", sorted(set(uuids) - set(expected))),
        "duplicate_gpu_uuids": cast("list[JSONValue]", duplicates),
        "verified_hashes": verified_hashes,
        "wall_clock": wall,
        "monotonic_clock": monotonic,
    }
