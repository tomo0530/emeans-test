from __future__ import annotations

import subprocess
from typing import Callable

from pydantic import SecretStr

from .exceptions import (
    LastPassCommandTimeoutError,
    LastPassEntryNotFoundError,
    LastPassExecutionError,
    LastPassNotLoggedInError,
)

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class LastPassSecretProvider:
    """
    Safe wrapper for fetching secret values via LastPass CLI (`lpass`).

    This provider centralizes LastPass CLI execution with safer defaults and
    typed exception mapping. Public APIs return `pydantic.SecretStr` so callers
    can avoid passing plain-text secrets around application boundaries.
    """

    _NOT_LOGGED_IN_PATTERNS = (
        "not logged in",
        "try logging in with",
        "invalid session",
        "login required",
    )
    _ENTRY_NOT_FOUND_PATTERNS = (
        "could not find specified account",
        "did not match any accounts",
        "no entries found",
        "no such entry",
    )
    _STANDARD_FIELD_OPTIONS = {
        "username": "--username",
        "password": "--password",
        "url": "--url",
        "notes": "--notes",
        "id": "--id",
        "name": "--name",
    }

    def __init__(
        self,
        executable: str = "lpass",
        timeout: float = 10.0,
        command_runner: CommandRunner | None = None,
    ) -> None:
        """
        Initialize a LastPass secret provider.

        Parameters
        ----------
        executable : str, default "lpass"
            Command name or absolute path to LastPass CLI executable.
        timeout : float, default 10.0
            Timeout in seconds for each LastPass CLI invocation.
            Must be greater than 0.
        command_runner : CommandRunner | None, default None
            Optional command runner mainly for testing. If provided, it must
            accept command tokens (without executable) and return
            `subprocess.CompletedProcess[str]`.
            When omitted, provider uses internal subprocess execution with
            `shell=False`.

        Raises
        ------
        ValueError
            If `timeout` is less than or equal to 0.
        """
        if timeout <= 0:
            msg = "timeout must be greater than 0."
            raise ValueError(msg)
        self._executable = executable
        self._timeout = timeout
        self._command_runner = command_runner or self._run_lpass_command

    def get_password(self, entry_name: str) -> SecretStr:
        """
        Retrieve password from a LastPass entry.

        Parameters
        ----------
        entry_name : str
            LastPass entry name.

        Returns
        -------
        SecretStr
            Password wrapped by `pydantic.SecretStr`.

        Raises
        ------
        ValueError
            If `entry_name` is an empty string or only whitespace.
        LastPassNotLoggedInError
            If the current LastPass session is not logged in.
        LastPassEntryNotFoundError
            If the specified entry does not exist.
        LastPassCommandTimeoutError
            If the CLI invocation exceeds the configured timeout.
        LastPassExecutionError
            For other CLI execution failures.
        """
        self._validate_non_empty_input("entry_name", entry_name)
        # Use `--` to ensure `entry_name` is treated as a positional argument,
        # even if it begins with `-`, avoiding option injection.
        return self._read_secret(["show", "--password", "--", entry_name])

    def get_field(self, entry_name: str, field_name: str) -> SecretStr:
        """
        Retrieve a specific field value from a LastPass entry.

        Parameters
        ----------
        entry_name : str
            LastPass entry name.
        field_name : str
            Field name to retrieve.
            Standard LastPass fields (`username`, `password`, `url`,
            `notes`, `id`, `name`) are fetched via dedicated `lpass show`
            options. Other values are fetched via `--field`.

        Returns
        -------
        SecretStr
            Field value wrapped by `pydantic.SecretStr`.

        Raises
        ------
        ValueError
            If `entry_name` or `field_name` is empty or only whitespace.
        LastPassNotLoggedInError
            If the current LastPass session is not logged in.
        LastPassEntryNotFoundError
            If the specified entry does not exist.
        LastPassCommandTimeoutError
            If the CLI invocation exceeds the configured timeout.
        LastPassExecutionError
            For other CLI execution failures.
        """
        self._validate_non_empty_input("entry_name", entry_name)
        self._validate_non_empty_input("field_name", field_name)
        normalized_field_name = field_name.lower()
        standard_field_option = self._STANDARD_FIELD_OPTIONS.get(normalized_field_name)
        if standard_field_option is not None:
            # Use `--` to ensure `entry_name` is treated as a positional argument,
            # even if it begins with `-`, avoiding option injection.
            return self._read_secret(["show", standard_field_option, "--", entry_name])
        return self._read_secret(["show", "--field", field_name, "--", entry_name])

    def _read_secret(self, command: list[str]) -> SecretStr:
        completed = self._execute(command)
        if completed.returncode != 0:
            self._raise_for_failed_command(completed)
        return SecretStr(completed.stdout.rstrip("\r\n"))

    def _execute(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self._command_runner(command)
        except subprocess.TimeoutExpired as err:
            raise LastPassCommandTimeoutError("LastPass CLI command timed out.") from err
        except OSError as err:
            raise LastPassExecutionError("LastPass CLI could not be executed.") from err

    def _run_lpass_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._executable, *command],
            capture_output=True,
            check=False,
            text=True,
            shell=False,
            timeout=self._timeout,
        )

    def _raise_for_failed_command(
        self,
        completed: subprocess.CompletedProcess[str],
    ) -> None:
        diagnostic = self._get_normalized_diagnostic(completed)
        if self._contains_pattern(diagnostic, self._NOT_LOGGED_IN_PATTERNS):
            raise LastPassNotLoggedInError("LastPass CLI is not logged in.")
        if self._contains_pattern(diagnostic, self._ENTRY_NOT_FOUND_PATTERNS):
            raise LastPassEntryNotFoundError("Requested LastPass entry was not found.")
        raise LastPassExecutionError(
            f"LastPass CLI command failed with exit code {completed.returncode}.",
        )

    @staticmethod
    def _get_normalized_diagnostic(
        completed: subprocess.CompletedProcess[str],
    ) -> str:
        output = f"{completed.stderr}\n{completed.stdout}"
        return output.lower()

    @staticmethod
    def _contains_pattern(text: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _validate_non_empty_input(name: str, value: str) -> None:
        if not value.strip():
            msg = f"{name} must be a non-empty string."
            raise ValueError(msg)
