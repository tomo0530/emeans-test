---
description: Check whether existing DS logic in project_module/ conforms to the DS integration contract (docs/ds_integration_contract.md). Use before handing code to ENG, before a review meeting, or when revisiting an older component that predates the contract. Also offers to refactor non-compliant parts.
disable-model-invocation: true
---

Check whether existing DS logic in `project_module/` conforms to the
DS integration contract (`docs/ds_integration_contract.md`), then
offer to refactor non-compliant parts.

## Inputs expected from the user

- A file path, directory, or module name to check.

If not provided, ask before proceeding.

## Steps

### Phase 1 — Check

Read the target files and evaluate each item in the checklist below.
Report findings as a table: ✅ pass / ❌ fail / ⚠️ partial / N/A not
applicable.

**Checklist (items 1-13 are the section-14 handoff checklist; items
14-16 are code-level rules from the referenced sections)**

| # | Item | Ref |
|---|---|---|
| 1 | `Input` Pydantic model defined | §14 |
| 2 | `Output` Pydantic model defined | §14 |
| 3 | `Config` Pydantic model defined | §14 |
| 4 | `invoke()` public signature present | §14 |
| 5 | Invocation granularity (record / batch / file) is clear | §14 |
| 6 | Sample input available | §14 |
| 7 | Sample output available | §14 |
| 8 | Invalid-input behaviour is documented | §14 |
| 9 | Undecidable / low-confidence representation is documented | §14 |
| 10 | Abstract dependency interfaces + verification-env implementations present — only when the logic has external I/O or env-dependent dependencies; N/A for pure logic (§15.1) | §14, §6 |
| 11 | Artifact save/load method documented — N/A if no artifact | §14, §9 |
| 12 | Known constraints documented | §14 |
| 13 | Minimum tests present | §14, §12 |
| 14 | Non-Pydantic types (Tensor, ndarray, DataFrame) have shape/dtype docstrings — N/A if all fields are Pydantic-typed | §5.2, `docs/agents/ml-and-data.md` |
| 15 | `invoke()` is free of side effects (no DB writes, file saves, global state) | §10.2 |
| 16 | Constructor does not resolve env vars, file paths, or connections | §6 |

After reporting, ask the user whether to proceed with refactoring.
Stop here if the user only wanted a check.

### Phase 2 — Refactor (requires user confirmation)

For each ❌ or ⚠️ item:

1. **Show the proposed change** as a diff or before/after snippet.
2. **Explain why** the change is needed (reference the contract section).
3. **Ask for confirmation** on the full set of changes before writing.
   If any proposed change is ambiguous or requires domain knowledge
   the model cannot infer (e.g. value constraints, axis semantics),
   flag it explicitly and ask the user.
4. **Apply** approved changes.
5. Run `make all-lint` and `make test` after writing (refactoring is
   structural only, so existing tests must still pass). Fix any issues
   before reporting done.

## Important constraints

- Do not change logic behaviour. Refactoring is structural only:
  adding type wrappers, extracting schemas, adding docstrings,
  separating side effects.
- If a behavioural decision is required (e.g. what error to raise on
  invalid input), surface it to the user rather than guessing.
- Do not add `InputType` / `OutputType` class variables unless the
  user confirms API-registration or registry use is planned.

## Reference

`docs/ds_integration_contract.md` — full contract specification.
