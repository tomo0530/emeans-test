class LastPassSecretProviderError(Exception):
    """Base class for all LastPass secret provider exceptions."""


class LastPassExecutionError(LastPassSecretProviderError):
    """Raised when LastPass CLI execution fails."""


class LastPassCommandTimeoutError(LastPassExecutionError):
    """Raised when LastPass CLI execution times out."""


class LastPassNotLoggedInError(LastPassExecutionError):
    """Raised when LastPass CLI is not logged in."""


class LastPassEntryNotFoundError(LastPassExecutionError):
    """Raised when a requested LastPass entry is not found."""
