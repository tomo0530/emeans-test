"""Fail-closed single-host sentinel rendezvous over atomic files and flocks."""

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_RELEASE_LEAD_NS = 50_000_000
_POLL_SECONDS = 0.02


class SentinelCoordinationError(RuntimeError):
    """A sentinel generation cannot safely be used for qualification."""


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _name(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class SentinelGroup:
    """Coordinate one immutable generation and its single-use member claim."""

    def __init__(
        self,
        root: Path,
        group_id: str,
        members: list[str],
        member: str,
        timeout: float,
        descriptor: dict[str, Any],
    ) -> None:
        if (
            not group_id
            or len(set(members)) != len(members)
            or member not in members
            or timeout <= 0
        ):
            raise ValueError("invalid sentinel group")
        self.directory = root / ".emeans-sentinel" / _name(group_id)
        self.members = sorted(members)
        self.member = member
        self.timeout = timeout
        self.descriptor = {
            "group_id": group_id,
            "members": self.members,
            "plan": descriptor,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "release_lead_ns": _RELEASE_LEAD_NS,
        }
        self._alive: Any = None

    @contextmanager
    def _state(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "state.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _abort_locked(self, reason: str) -> None:
        if (self.directory / "complete.json").exists():
            return
        path = self.directory / "abort.json"
        if not path.exists():
            _atomic(path, {"reason": reason})

    def abort(self, reason: str) -> None:
        """Permanently invalidate this generation for all members."""
        with self._state():
            self._abort_locked(reason)

    def assert_active(self) -> None:
        """Detect an aborted generation or a claimed member that died."""
        with self._state():
            self._check_locked()

    def _check_locked(self) -> None:
        abort = self.directory / "abort.json"
        if abort.exists():
            reason = json.loads(abort.read_bytes()).get("reason", "aborted")
            raise SentinelCoordinationError(f"sentinel group aborted: {reason}")
        for member in self.members:
            claim = self.directory / "claims" / f"{_name(member)}.json"
            if not claim.exists():
                continue
            with (self.directory / "liveness" / f"{_name(member)}.lock").open("a+b") as live:
                try:
                    fcntl.flock(live, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                self._abort_locked("participant_not_alive")
                raise SentinelCoordinationError("sentinel participant is no longer alive")

    def join(self) -> None:
        """Claim this member once and hold its liveness lock until close."""
        with self._state():
            if (self.directory / "complete.json").exists():
                raise SentinelCoordinationError("sentinel generation is already complete")
            self._check_locked()
            descriptor = self.directory / "descriptor.json"
            encoded = _encoded(self.descriptor)
            if descriptor.exists() and descriptor.read_bytes() != encoded:
                self._abort_locked("descriptor_mismatch")
                raise SentinelCoordinationError("sentinel descriptor mismatch")
            if not descriptor.exists():
                _atomic(descriptor, self.descriptor)
            claim = self.directory / "claims" / f"{_name(self.member)}.json"
            if claim.exists():
                self._abort_locked("duplicate_claim")
                raise SentinelCoordinationError("sentinel member already claimed")
            live_path = self.directory / "liveness" / f"{_name(self.member)}.lock"
            live_path.parent.mkdir(parents=True, exist_ok=True)
            self._alive = live_path.open("a+b")
            try:
                fcntl.flock(self._alive, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._alive.close()
                self._alive = None
                self._abort_locked("duplicate_live_member")
                raise SentinelCoordinationError("sentinel member already active") from exc
            claim.parent.mkdir(parents=True, exist_ok=True)
            try:
                claim_fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(claim_fd, "wb") as stream:
                    stream.write(_encoded({"member": self.member}))
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError as exc:
                self._alive.close()
                self._alive = None
                self._abort_locked("duplicate_claim")
                raise SentinelCoordinationError("sentinel member already claimed") from exc

    def wait(self, trial: dict[str, Any], inputs: dict[str, Any]) -> float:
        """Wait for identical trial/input arrivals and the common release time."""
        start = time.monotonic()
        deadline = start + self.timeout
        trial_hash = hashlib.sha256(_encoded(trial)).hexdigest()
        directory = self.directory / "trials" / trial_hash
        arrival = directory / f"arrival-{_name(self.member)}.json"
        with self._state():
            self._check_locked()
            if not (self.directory / "claims" / f"{_name(self.member)}.json").exists():
                self._abort_locked("unclaimed_member")
                raise SentinelCoordinationError("sentinel member has no claim")
            if arrival.exists():
                self._abort_locked("duplicate_arrival")
                raise SentinelCoordinationError("sentinel duplicate arrival")
            _atomic(arrival, {"member": self.member, "trial": trial, "inputs": inputs})
        release = directory / "release.json"
        while True:
            with self._state():
                self._check_locked()
                arrivals = [directory / f"arrival-{_name(member)}.json" for member in self.members]
                if all(path.exists() for path in arrivals):
                    entries = [json.loads(path.read_bytes()) for path in arrivals]
                    if any(
                        item != {"member": member, "trial": trial, "inputs": inputs}
                        for item, member in zip(entries, self.members, strict=True)
                    ):
                        self._abort_locked("input_mismatch")
                        raise SentinelCoordinationError("sentinel input mismatch")
                    if not release.exists():
                        _atomic(
                            release,
                            {"release_at_monotonic_ns": time.monotonic_ns() + _RELEASE_LEAD_NS},
                        )
                target = (
                    json.loads(release.read_bytes())["release_at_monotonic_ns"]
                    if release.exists()
                    else None
                )
                if target is not None and time.monotonic_ns() >= target:
                    return time.monotonic() - start
                if time.monotonic() >= deadline:
                    self._abort_locked("sentinel_barrier_timeout")
                    raise SentinelCoordinationError("sentinel barrier timeout")
            remaining = (
                _POLL_SECONDS
                if target is None
                else min(_POLL_SECONDS, max(0.0, (target - time.monotonic_ns()) / 1e9))
            )
            time.sleep(remaining)

    def finish(self) -> None:
        """Wait until every member has completed the generation successfully."""
        deadline = time.monotonic() + self.timeout
        done = self.directory / "done" / f"{_name(self.member)}.json"
        complete = self.directory / "complete.json"
        with self._state():
            self._check_locked()
            _atomic(done, {"member": self.member})
        while True:
            with self._state():
                if complete.exists():
                    return
                self._check_locked()
                if all(
                    (self.directory / "done" / f"{_name(member)}.json").exists()
                    for member in self.members
                ):
                    _atomic(complete, {"members": self.members})
                    return
                if time.monotonic() >= deadline:
                    self._abort_locked("sentinel_completion_timeout")
                    raise SentinelCoordinationError("sentinel completion timeout")
            time.sleep(_POLL_SECONDS)

    def close(self) -> None:
        """Release this process's liveness lock."""
        if self._alive is not None:
            self._alive.close()
            self._alive = None


def inspect_group(root: Path, group_id: str) -> dict[str, Any]:
    """Read a generation's completion, membership, and immutable descriptor hash."""
    directory = root / ".emeans-sentinel" / _name(group_id)
    descriptor_path = directory / "descriptor.json"
    if not descriptor_path.is_file():
        return {
            "status": "missing",
            "members": [],
            "completed_members": [],
            "descriptor_hash": None,
            "abort_reason": None,
        }
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    members = descriptor["members"]
    complete = directory / "complete.json"
    abort = directory / "abort.json"
    return {
        "status": "complete"
        if complete.exists()
        else "aborted"
        if abort.exists()
        else "incomplete",
        "members": members,
        "completed_members": [
            member for member in members if (directory / "done" / f"{_name(member)}.json").exists()
        ],
        "descriptor_hash": hashlib.sha256(descriptor_bytes).hexdigest(),
        "abort_reason": json.loads(abort.read_bytes()).get("reason") if abort.exists() else None,
    }
