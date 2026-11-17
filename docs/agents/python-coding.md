# Python Coding Standards

Detailed Python coding rules for AI agents. AGENTS.md §6 indexes this
file. Apply these when writing or modifying Python in this codebase.

## Package management (`uv` only)

Environment-level rules (Docker default, approval before `uv add`,
dependency cooldowns) are in AGENTS.md §4. Code-level conventions:

* `uv add <pkg>` / `uv add --dev <pkg>` / `uv lock --upgrade-package <pkg>`.
* **Do NOT** use `pip`, `uv pip install`, or `<pkg>@latest` syntax.

## Type hints

* Python 3.10+ modern syntax: `list[str]`, `dict[str, int]`,
  `set[int]`, `tuple[int, int]`, `tuple[int, ...]`, `int | None`.
* Annotate every function argument and return value.
* Narrow `Optional` types with explicit `None` checks.
* **Avoid** `typing.List`, `typing.Dict`, `typing.Set`,
  `typing.Tuple`, `typing.Optional`.

## Data models

* Prefer **Pydantic** over raw `dict`.
* `dataclass` is acceptable for simple, internal structures.
* When ML types (`torch.Tensor`, `np.ndarray`, `pd.DataFrame`) are
  unavoidable, see [ML and Data Guidelines](./ml-and-data.md) for
  required docstring conventions.

## Docstrings & comments

* **Docstrings**: NumPy style on all public APIs. Document arguments,
  returns, and raised exceptions.
* **Comments**: add only when they explain intent, branching rationale,
  fallback priority, threshold logic, or domain constraints. Do **not**
  comment obvious code.

## Function calls

* Use keyword arguments for readability, especially on multi-parameter
  calls.

## Naming

* Variables / functions: `snake_case`.
* Classes: `PascalCase`.
* Constants: `UPPER_SNAKE_CASE`.
* Private members start with `_`.
* Handler functions use the `handle_` prefix.

## Structure & style

* **Early returns** (guard clauses) to keep nesting shallow.
* **Functional style** where it does not hurt readability.
* **DRY**: extract duplicated logic into reusable components.
* **Explicit**: do not silently fall back for invalid states or
  unsupported modes — raise explicit errors instead. Normal argument
  defaults and well-documented config defaults are fine.
* **Imports at the top of the file**. Deferred or function-local
  imports are allowed only for optional dependencies or to break an
  import cycle.
* **No nested functions** unless there is a specific reason.

## Runtime behavior

* Log milestones, branch decisions, retries, and failures in
  non-trivial execution paths so long-running benchmarks and data
  pipelines remain observable.
