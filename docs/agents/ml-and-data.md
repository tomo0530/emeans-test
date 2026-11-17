# ML and Data Guidelines

Detailed ML, data, and notebook rules for AI agents. AGENTS.md §6
indexes this file. Apply these on any ML, data pipeline, or notebook
task.

## Leakage prevention

* Prevent target, future, and duplicate leakage.
* Be explicit about temporal ordering, group boundaries, and
  aggregation granularity.

## Evaluation strategy

* Choose train / validation / test splits and CV strategy (stratify,
  time-aware, group-aware) that match the data-generating process.
* For ML tests, use small synthetic data and clear separation between
  training-like and evaluation-like inputs.

## Reproducibility

* Pin seeds, environments, data versions, and configs so any
  experiment can be reproduced later.
* Use deterministic randomness; for numerical comparison use
  tolerances rather than exact floating-point equality.

## Interface contracts for non-Pydantic types

When input or output types cannot be expressed as Pydantic models
(`torch.Tensor`, `np.ndarray`, `pd.DataFrame`), compensate with
explicit docstrings on the public API:

* **Tensors / arrays**: document dtype, shape, and the meaning of
  each axis (`batch_size`, `seq_len`, `feature_dim`, etc.).
* **NumPy annotations**: `np.ndarray` is the runtime class; for type
  annotations prefer `numpy.typing.NDArray` with an explicit dtype
  (e.g. `npt.NDArray[np.float32]`). Shape and axis semantics cannot be
  encoded in the annotation, so they still go in the docstring.
* **DataFrames**: document required columns, their dtypes, and any
  index assumptions.
* **`dict[str, Any]`**: document every expected key with its type
  and semantics. Bare `dict[str, Any]` without explanation is
  forbidden.

When such a type must be stored inside a Pydantic model field, set
`model_config = ConfigDict(arbitrary_types_allowed=True)` on that
model — Pydantic rejects unknown types at class-definition time
otherwise.

Example:

```python
def invoke(self, x: torch.Tensor) -> torch.Tensor:
    """Run inference.

    Parameters
    ----------
    x : torch.Tensor
        Float32 tensor of shape ``(batch_size, seq_len, feature_dim)``.
        ``feature_dim`` must match the model's input size.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(batch_size, n_classes)`` — raw
        logits (not softmax-normalized).
    """
```

This makes the interface contract explicit even when type hints alone
cannot carry the full specification.

## Code separation

* Keep exploration code under `experimentation/`.
* Promote reusable preprocessing, feature engineering, and evaluation
  logic into `project_module/` or `libs/`.
* Do not import from `experimentation/` in production / library code.

## Notebook discipline

* Notebooks must run top-to-bottom without manual intervention.
* **Do not commit cell outputs.** Use `make reset-notebook [path]` to
  strip them before committing.
* Do not import notebooks from production code.

## Traceability

Record enough metadata to trace each result back to its inputs:

* data version / snapshot ID
* code SHA (and branch)
* config / hyperparameters
* model artifact path or hash
