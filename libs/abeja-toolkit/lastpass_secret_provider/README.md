# LastPass Secret Provider

Python wrapper for safely retrieving secrets from LastPass CLI (`lpass`).

## Description

`lastpass_secret_provider` centralizes `lpass` execution and standardizes:

- secret retrieval APIs (`get_password`, `get_field`)
- secure subprocess execution settings
- typed secret return value with `pydantic.SecretStr`
- predictable exception handling for common failure patterns

This package is designed for projects that need to fetch secrets from LastPass
without scattering `subprocess` calls across the codebase.

### Key Features

- **Simple API**: `get_password(entry_name)` and `get_field(entry_name, field_name)`
- **Secret-safe return type**: returns `pydantic.SecretStr`
- **Safer subprocess defaults**: `shell=False`, timeout, captured output
- **Typed exceptions**: distinguish login state, missing entries, timeout, and other failures
- **Testability**: injectable command runner for unit tests

## Prerequisites

Before using this library:

1. Install LastPass CLI (`lpass`) on your environment.
2. Login to LastPass:

```shell
lpass login <your-email>
```

3. Confirm that your target entry exists:

```shell
lpass ls
```

## Installation

```shell
uv add --editable /path/to/abeja-toolkit/lastpass_secret_provider
```

Or with pip:

```shell
pip install -e /path/to/abeja-toolkit/lastpass_secret_provider
```

## Basic Usage

```python
from lastpass_secret_provider import (
    LastPassEntryNotFoundError,
    LastPassNotLoggedInError,
    LastPassSecretProvider,
)

provider = LastPassSecretProvider(timeout=10.0)

try:
    password = provider.get_password("example/service-account")
    token = provider.get_field("example/service-account", "token")
except LastPassNotLoggedInError:
    print("Please login with `lpass login` first.")
except LastPassEntryNotFoundError:
    print("Requested LastPass entry was not found.")
else:
    print(password.get_secret_value())
    print(token.get_secret_value())
```

## API Reference

### `LastPassSecretProvider`

```python
from lastpass_secret_provider import LastPassSecretProvider

provider = LastPassSecretProvider(
    executable="lpass",
    timeout=10.0,
)
```

Parameters:

- `executable` (`str`, default: `"lpass"`): command name/path for LastPass CLI
- `timeout` (`float`, default: `10.0`): timeout in seconds for each CLI execution
- `command_runner` (`Callable | None`, default: `None`): injected runner for testing

### `get_password(entry_name: str) -> SecretStr`

Fetches password from the specified LastPass entry.

```python
password = provider.get_password("infra/prod/db")
plain_password = password.get_secret_value()
```

### `get_field(entry_name: str, field_name: str) -> SecretStr`

Fetches a specific field from the specified LastPass entry.

```python
api_token = provider.get_field("infra/prod/api", "token")
plain_api_token = api_token.get_secret_value()
```

## Error Handling

```python
from lastpass_secret_provider import (
    LastPassCommandTimeoutError,
    LastPassEntryNotFoundError,
    LastPassExecutionError,
    LastPassNotLoggedInError,
)

try:
    value = provider.get_field("infra/prod/api", "token")
except LastPassNotLoggedInError:
    # lpass login is required
    ...
except LastPassEntryNotFoundError:
    # target entry does not exist
    ...
except LastPassCommandTimeoutError:
    # command exceeded timeout
    ...
except LastPassExecutionError:
    # other execution failure (including lpass missing)
    ...
```

Exception classes:

- `LastPassSecretProviderError`: base exception
- `LastPassExecutionError`: generic CLI execution failure
- `LastPassCommandTimeoutError`: command timed out
- `LastPassNotLoggedInError`: not logged in to LastPass
- `LastPassEntryNotFoundError`: entry not found

## Security Notes

- Secret values are returned as `SecretStr` instead of plain strings.
- The subprocess is executed with `shell=False`.
- Command output is not embedded into exception messages.
- Secret values are not included in raised exception messages.

## Testing

### Unit Tests

```shell
uv run --package lastpass-secret-provider pytest lastpass_secret_provider/tests
```

### Integration Testing (E2E)

Use a real LastPass account and a dedicated test entry.

1. Prepare a dedicated LastPass entry for E2E (example entry name: `abeja-toolkit/e2e/lastpass-secret-provider`).
   Create/update the entry with CLI:

   ```shell
   lpass login <your-email>

   lpass add abeja-toolkit/e2e/lastpass-secret-provider-test
   # Name: abeja-toolkit/e2e/lastpass-secret-provider-test
   # URL: http://
   # Username:
   # Password:dummy-password
   # Notes:    # Add notes below this line.
   
   lpass add --note-type=ssn abeja-toolkit/e2e/lastpass-secret-provider-test-note
   # Name: abeja-toolkit/e2e/lastpass-secret-provider-test-note
   # NoteType: Social Security
   # SecretNote: dummy-secret-note
   # Name:
   # Number:
   # Notes:    # Add notes below this line.
   ```
2. Verify the entry can be read from CLI:

   ```shell
   lpass show --password abeja-toolkit/e2e/lastpass-secret-provider-test
   lpass show --field SecretNote abeja-toolkit/e2e/lastpass-secret-provider-test-note
   ```

   Both commands should print the expected values from step 1.

3. Run the following E2E script:

   ```shell
   uv run python -c "from lastpass_secret_provider import LastPassSecretProvider; \
   provider = LastPassSecretProvider(); \
   assert provider.get_password('abeja-toolkit/e2e/lastpass-secret-provider-test').get_secret_value() == 'dummy-password'; \
   assert provider.get_field('abeja-toolkit/e2e/lastpass-secret-provider-test-note', 'SecretNote').get_secret_value() == 'dummy-secret-note'; \
   print('E2E test passed.')"
   ```

4. Confirm `E2E test passed.` is printed.

Notes:

- Run E2E against a test-only entry, not production credentials.
- Do not commit real secret values to code or CI logs.
- `get_field()` supports standard LastPass fields (`username`, `password`, `url`, `notes`, `id`, `name`) via dedicated `lpass show` options.
- Custom fields via `--field` are typically available on secure note entries.
