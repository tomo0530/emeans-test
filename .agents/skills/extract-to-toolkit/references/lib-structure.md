# Standard lib-structure for a new lib under libs/

Reference for Phase 5 of the `extract-to-toolkit` skill.
Describes the required directory layout and file templates when
creating a new package under `libs/`.

> The new lib goes at `libs/<lib_name>/` — **not** inside
> `libs/abeja-toolkit/`, which `make sync-tools` wipes and re-clones.
> See the destination note at the top of `SKILL.md`.

## Directory layout

```
libs/<lib_name>/
  src/
    <lib_name>/
      __init__.py
      py.typed            # empty marker file (PEP 561)
      <module files>
  tests/
    __init__.py
  pyproject.toml
  README.md
```

Naming rule, matching the underscore-directory convention of the synced
packages under `libs/abeja-toolkit/` (e.g. the directory
`lastpass_secret_provider` ships distribution name
`lastpass-secret-provider`):
- `<lib_name>` (underscore) — the top-level directory under `libs/`
  **and** the `src/<lib_name>/` import package. Same string, both places.
- `<lib-name>` (kebab-case) — the PyPI-style distribution name. Used
  only inside `pyproject.toml`'s `[project] name` and wherever another
  project declares this lib as a dependency. Never used as a directory
  name.

Example: lib name "gcs-client" → directory and import package
`gcs_client`, distribution name `gcs-client`.

---

## pyproject.toml template

```toml
[project]
name = "<lib-name>"
version = "0.1.0"
description = ""
readme = "README.md"
requires-python = ">=3.10,<3.14"
dependencies = [
    # list third-party packages inferred from imports
    # add version constraints once confirmed with the user
]

[dependency-groups]
dev = [
    "mypy>=1.15.0",
    "pytest>=8.4.2",
    "pytest-cov>=6.2.1",
    "ruff>=0.11.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/<lib_name>"]
```

Replace `<lib-name>` and `<lib_name>` with the actual names.
Do **not** add `[tool.uv.sources]` to the lib's own `pyproject.toml`;
workspace source resolution is done from the root only.

---

## __init__.py convention

Re-export the public API so callers use `from <lib_name> import X`
rather than deep module paths:

```python
from .<module> import ClassA, function_b

__all__ = ["ClassA", "function_b"]
```

---

## README.md content

Keep it minimal:

```markdown
# <lib-name>

Brief one-line description.

## Install

This package is part of the `dsg-project-template` uv workspace, so it
doesn't need to be published to PyPI. Each consuming project must still
declare it explicitly — see "Source project pyproject.toml changes"
below.

## Usage

\```python
from <lib_name> import ...
\```
```

---

## Root pyproject.toml changes

After creating the lib directory, append the new lib to the existing
`[tool.uv.workspace] members` list. Don't reproduce or reorder the
entries already there (they change over time) — just add one line:

```toml
# In [tool.uv.workspace], keep the existing members and add:
    "libs/<lib_name>",   # ← underscore directory, directly under libs/
```

---

## Source project pyproject.toml changes

In each project that uses the extracted lib, add to `[project] dependencies`:

```toml
dependencies = [
    "<lib-name>",
]
```

And add to `[tool.uv.sources]`:

```toml
[tool.uv.sources]
<lib-name> = { workspace = true }
```

Note: `<lib-name>` in dependency declarations uses hyphens (PyPI
convention), not underscores. Python import uses underscores.

---

## py.typed

Create an empty `py.typed` file in the package root to declare PEP 561
compliance (tells mypy that this package ships inline types):

```
libs/<lib_name>/src/<lib_name>/py.typed   # empty file
```
