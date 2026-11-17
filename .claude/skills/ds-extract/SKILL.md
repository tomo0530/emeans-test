---
description: Extract DS verification logic from a notebook into a project_module/ component that conforms to the DS integration contract (docs/ds_integration_contract.md). Use when notebook exploration is complete and logic is ready to hand off to ENG or promote to project_module/.
disable-model-invocation: true
---

Extract DS verification logic from a notebook into a `project_module/`
component that conforms to the DS integration contract
(`docs/ds_integration_contract.md`).

## Inputs expected from the user

The user should provide one of:

- A notebook file path (e.g. `experimentation/foo.ipynb`)
- A cell range or specific cells to extract
- A pasted code block from a notebook

If not provided, ask before proceeding.

## Steps

1. **Read the source.**
   Read the notebook or code block the user pointed to. Identify the
   single logical unit to extract (one `invoke`-worthy operation).
   If multiple independent units are present, list them and ask the
   user which to extract first.

2. **Define schemas.**
   - Define `{Name}Input(BaseModel)` with typed, constrained fields.
   - Define `{Name}Output(BaseModel)` with typed fields.
   - Define `{Name}Config(BaseModel)` for logic-specific parameters
     (thresholds, model names, flags). Do not put env-specific values
     here.
   - For fields that cannot be Pydantic-typed (e.g. `torch.Tensor`,
     `np.ndarray`, `pd.DataFrame`), keep the type hint and add a
     docstring that documents dtype, shape, axis semantics, or required
     columns. Never use bare `dict[str, Any]` without explanation.
     If such a type must live inside a Pydantic model, set
     `model_config = ConfigDict(arbitrary_types_allowed=True)` on that
     model (Pydantic rejects unknown types at class-definition time
     otherwise); if not, keep it outside the Pydantic boundary as a
     documented typed argument.

3. **Implement `invoke`.**
   - Create `{Name}Logic(BaseInvokable[{Name}Input, {Name}Output])`,
     importing `BaseInvokable` from `project_module.base` (already
     shipped with the template; the class definitions shown in the
     contract document are the spec — import, do not copy them).
   - Move the core logic into `invoke(self, input_data: {Name}Input)
     -> {Name}Output`.
   - Constructor receives only initialised objects (model, config);
     it must not resolve file paths, DB connections, or env vars.
   - Keep `invoke` free of side effects: no DB writes, no file saves,
     no global state changes.

4. **Define dependency interfaces (if needed).**
   - If the logic loads a model, reads external data, or calls an
     external API, define a `Base{Dep}(BaseInvokable)` abstract
     interface for each dependency.
   - Provide a local/file-based concrete implementation for the
     verification environment (e.g. `Local{Dep}`).
   - Document where the production implementation must be supplied by
     ENG.

5. **Create a `BaseFactory` (if dependencies exist).**
   - Define `{Name}Factory(BaseFactory[{Name}Config, {Name}Logic])`.
   - `build(config)` wires dependencies; it does not know about S3,
     DB, or Secrets Manager — those come via injected lower-level
     `BaseInvokable` implementations.

6. **Write tests.**
   - Create `project_module/tests/test_{name}.py` with:
     - Contract tests: valid input accepted, invalid input rejected,
       output matches schema.
     - At least one happy-path unit test for `invoke`.
     - When applicable: a `Config`-variation test (different config →
       different output) and boundary / undecidable / low-confidence
       cases (contract section 12.2).
     - Fixtures under `fixtures/{name}/` so extractions of multiple
       components do not overwrite each other. Use `sample_input.json`
       / `sample_output.json` only when the I/O is JSON-serializable;
       for Tensor / ndarray / DataFrame payloads use a format that
       preserves dtype / shape / index (`.npz`, `.parquet`, `.pt`) or
       an in-test fixture builder, and assert those properties in the
       tests.
   - Report which section-12 test items were skipped as not applicable
     and why.

7. **Write a README section.**
   Add or update `project_module/README.md` (or a per-module
   `README.md`) with:
   - What the logic does
   - How to call it (minimal code example)
   - Input / output specs
   - Config fields
   - Known constraints
   - Points that ENG must resolve for the production environment

8. **Present the plan before writing.**
   Show the user:
   - Proposed file paths
   - Schema field list (name, type, constraints)
   - Which dependencies need abstract interfaces
   - Any assumptions made about the notebook logic

   Wait for explicit approval before writing files.

9. **Write files and confirm.**
   After approval, create the files. Run `make all-lint` and
   `make test` to verify formatting, types, and that the tests pass.
   Report any issues.

## Output file layout

Follows the template's src-layout (`project_module/src/project_module/`,
tests under `project_module/tests/` per `pytest.ini`).

```text
project_module/
  src/project_module/
    base.py            # shared BaseInvokable / BaseFactory (shipped with the template)
    {name}/
      __init__.py
      logic.py         # {Name}Logic
      schemas.py       # {Name}Input, {Name}Output, {Name}Config
      dependencies.py  # Base{Dep} + local implementations
  tests/
    test_{name}.py
    fixtures/
      {name}/
        sample_input.json   # or .npz / .parquet / .pt for non-JSON I/O
        sample_output.json
```

## Reference

`docs/ds_integration_contract.md` — full contract specification.
