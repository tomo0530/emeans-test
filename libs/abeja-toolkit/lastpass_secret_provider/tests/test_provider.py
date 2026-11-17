from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from lastpass_secret_provider import (
    LastPassCommandTimeoutError,
    LastPassEntryNotFoundError,
    LastPassExecutionError,
    LastPassNotLoggedInError,
    LastPassSecretProvider,
)


def _completed(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["lpass"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_get_password_returns_secret_str() -> None:
    provider = LastPassSecretProvider(command_runner=lambda _: _completed(stdout="pw\n"))

    secret = provider.get_password("my-entry")

    assert secret.get_secret_value() == "pw"


def test_get_field_returns_secret_str() -> None:
    provider = LastPassSecretProvider(command_runner=lambda _: _completed(stdout="token\n"))

    secret = provider.get_field("my-entry", "token")

    assert secret.get_secret_value() == "token"


def test_get_field_uses_expected_command() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed(stdout="token\n")

    provider = LastPassSecretProvider(command_runner=runner)
    provider.get_field("my-entry", "token")

    assert commands == [["show", "--field", "token", "--", "my-entry"]]


def test_get_field_standard_username_uses_expected_command() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed(stdout="user\n")

    provider = LastPassSecretProvider(command_runner=runner)
    provider.get_field("my-entry", "username")

    assert commands == [["show", "--username", "--", "my-entry"]]


def test_raises_not_logged_in_error() -> None:
    provider = LastPassSecretProvider(
        command_runner=lambda _: _completed(
            returncode=1,
            stderr="Error: Not logged in. Try logging in with `lpass login`.",
        ),
    )

    with pytest.raises(LastPassNotLoggedInError):
        provider.get_password("my-entry")


def test_raises_entry_not_found_error() -> None:
    provider = LastPassSecretProvider(
        command_runner=lambda _: _completed(
            returncode=1,
            stderr="Error: Could not find specified account(s).",
        ),
    )

    with pytest.raises(LastPassEntryNotFoundError):
        provider.get_password("missing-entry")


def test_raises_execution_error_without_secret_in_message() -> None:
    leaked_secret = "super-secret-token"
    provider = LastPassSecretProvider(
        command_runner=lambda _: _completed(
            returncode=1,
            stderr=f"command failed: {leaked_secret}",
        ),
    )

    with pytest.raises(LastPassExecutionError) as exc_info:
        provider.get_password("my-entry")

    assert leaked_secret not in str(exc_info.value)


def test_timeout_error_is_mapped() -> None:
    def timeout_runner(_: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["lpass"], timeout=1.0)

    provider = LastPassSecretProvider(command_runner=timeout_runner)

    with pytest.raises(LastPassCommandTimeoutError):
        provider.get_password("my-entry")


def test_empty_input_raises_value_error() -> None:
    provider = LastPassSecretProvider(command_runner=lambda _: _completed(stdout="value\n"))

    with pytest.raises(ValueError):
        provider.get_password(" ")
    with pytest.raises(ValueError):
        provider.get_field("my-entry", "")


def test_subprocess_run_uses_safe_defaults() -> None:
    provider = LastPassSecretProvider(timeout=7.5)

    with patch("lastpass_secret_provider.provider.subprocess.run") as mock_run:
        mock_run.return_value = _completed(stdout="pw\n")
        provider.get_password("my-entry")

    mock_run.assert_called_once_with(
        ["lpass", "show", "--password", "--", "my-entry"],
        capture_output=True,
        check=False,
        text=True,
        shell=False,
        timeout=7.5,
    )
