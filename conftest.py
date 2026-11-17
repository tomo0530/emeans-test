from __future__ import annotations

import warnings

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        warnings.warn(
            "====== NO TESTS COLLECTED. But tests should be implemented. ======",
            stacklevel=2,
        )
        session.exitstatus = pytest.ExitCode.OK
