# `project_module/` — Research Package Rules

Apply the root instructions and [experiment design](../docs/plan/README.md).
The `e_means` package is a research runner and evaluation library,
not a production DS-to-ENG handoff.

* Use typed function contracts and frozen configuration models.
  Annotate arrays with `numpy.typing.NDArray` and explicit dtypes;
  document shape, axes, and finite-value assumptions.
* Separate numerical training/evaluation from persistence and
  orchestration. `data.py` and `runner.py` may perform specified I/O;
  adapters return centroids and metrics through the shared types.
* Resolve secrets at the CLI boundary and pass initialized resources
  inward. Persist only the environment allowlist, never token values.
* GPU imports may be lazy at adapter/worker boundaries so CPU tests
  work without Torch/CUDA and device selection happens first.
* Test the control algorithm with small numerical oracles and
  independent evaluation. Keep benchmarks separate from unit tests.

Apply the [DS integration contract](../docs/ds_integration_contract.md)
at a future production handoff boundary. Research functions do not
need unused BaseInvokable or factory layers from the template.
