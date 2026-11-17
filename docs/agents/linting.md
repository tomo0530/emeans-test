# Linting, Formatting, and Type Checking

Detailed quality-check rules for AI agents. AGENTS.md §6 indexes this
file. Apply these before committing or completing a task.

## Commands

* `make all-lint [path]` — runs format-check + type-check + lint in one
  pass (matching the verification order below).
* `make lint [path]` — runs `ruff check`.
* `make format [path]` — runs `ruff format --check`.
* `make type-check [path]` — runs `mypy`.
* `make fix [path]` — auto-fixes lint and formatting only; it does
  **not** type-check.

## Configuration

* [`ruff.toml`](../../ruff.toml): line-length **100**, Python 3.10+,
  rules `E`, `W`, `F`, `B`, `I` enabled.
* [`mypy.ini`](../../mypy.ini): strict mode
  (`disallow_untyped_defs`, `warn_return_any`,
  `warn_unreachable`, etc.). Excludes: top-level `libs/` (anchored at
  repo root — `project_module/libs/` is still checked) and any
  `tests/` directory at any depth.

## Verification order

Before committing or marking a task complete, verify in this order:

1. **format** — `make format` to check; `make format-fix` to
   auto-correct.
2. **type-check** — `make type-check`.
3. **lint** — `make lint` to check; `make lint-fix` to auto-correct.

Run all checks together with `make all-lint`. If formatting or lint
checks fail, `make fix` auto-corrects both at once.

## CI

* [`lint.yaml`](../../.github/workflows/lint.yaml) runs `ruff` and
  `mypy` against changed packages on PR `opened` / `synchronize` events
  and on push to `main`.
* If CI fails after a local pass, ensure your venv is in sync:
  `uv sync --all-groups`.
