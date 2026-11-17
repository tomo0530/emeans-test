# `project_module/` — DS Logic Rules

Code under this directory is the handoff unit from DS to ENG. It must
follow the DS integration contract
([`docs/ds_integration_contract.md`](../docs/ds_integration_contract.md)).

Hard rules (see the contract for details and examples):

* Define `Input` / `Output` / `Config` as Pydantic models; expose the
  logic as `invoke(input_data: Input) -> Output`, using the shared
  `BaseInvokable` / `BaseFactory` from `project_module.base`.
* No untyped fields — bare `dict[str, Any]` is forbidden. If a value
  cannot be Pydantic-typed (Tensor / ndarray / DataFrame), document
  dtype, shape, and meaning in the docstring
  (see [`docs/agents/ml-and-data.md`](../docs/agents/ml-and-data.md)).
* `invoke()` has no side effects: no DB writes, file saves, or global
  state changes.
* Constructors receive initialised objects only — never resolve env
  vars, file paths, or connections inside the logic.
* Environment-dependent I/O (files, DB, S3, external APIs) goes behind
  an abstract interface with a verification-env implementation; ENG
  supplies the production implementation.

To extract logic from a notebook, use `/ds-extract`. Before handing
off to ENG, run `/ds-contract-check` (other agents: read
[`.claude/skills/`](../.claude/skills/) `*/SKILL.md` directly).
