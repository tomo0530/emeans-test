[日本語 README](./README.md)

# emeans-test

This repository benchmarks e-means against Lloyd, MiniBatch, and Elkan k-means. The evaluation measures the relationship between clustering quality (MSE) and training time, using public text and image embeddings plus controlled synthetic data.

The main evaluation completed 9,520 trials across ten real-data and twelve synthetic conditions, using six methods and ten seeds. The [Japanese results report](docs/report/README.md) includes all 22 quality–time curves, attainment and group-loss tables, and the endpoint reproducibility checks.

The Torch methods ran on a single V100 with a shared native FP32 deterministic profile. The [main protocol](docs/report/protocol/README.md) records how interference testing selected that cohort and covers the stock-default reproducibility and sensitivity results.

The primary time axis remains warm training time. The secondary `e2e_seconds` uses the versioned `resident-input-pipeline-v1` scope: it starts immediately before the measured adapter call and ends after centroid save/fsync. It excludes input loading, initialization, evaluation, and ledger I/O; it is not a load-inclusive end-to-end metric.

The target runtime is CPython `>=3.14,<3.15` with uv, in the existing GPU-enabled Docker environment. GPU commands should use `uv run --extra gpu`; set `UV_PROJECT_ENVIRONMENT="$PWD/.venv"` to keep uv pointed at this repository's environment. Public Hugging Face data needs no additional API key; `HF_TOKEN` is optional.

Experiment design, protocol, runtime/artifact details, and the implementation plan are indexed in the [Japanese design documents](docs/plan/README.md). The main README contains setup, CPU/GPU test commands, config names, and CLI examples in Japanese.

For reproducible restarts, use a frozen commit/worktree and keep parent CPU affinity (including taskset/numactl usage) the same across phases. Do not commit, edit, re-prepare data, or synchronize dependencies while a phase is running; use `UV_NO_SYNC=1` or `uv run --no-sync --extra gpu` with the frozen environment.
