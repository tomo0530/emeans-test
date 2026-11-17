[日本語 README](./README.md)

# emeans-test

This repository benchmarks e-means against Lloyd, MiniBatch, and Elkan k-means. Its main result will be the relationship between clustering quality (MSE) and training time, using public text and image embeddings plus controlled synthetic data.

The B1–B3 implementation gate is approved by Opus and Sol. The profile source has passed 199 CPU tests, 10 GPU tests, and lint; its formal runner also passed a two-fresh-process check. The eight-GPU qualification, all-condition preflight, interference gate, and pilot remain pending, and no main benchmark performance results are available. See the [implementation progress](docs/plan/implementation-progress.md) for current status.

The planned main evaluation keeps 22 cells: 10 real-data and 12 synthetic conditions, with 10 seeds per cell and the existing six methods and budgets. The two Torch methods are planned to share a V100 native-FP32 reproducibility profile. This is not the upstream Torch default profile, so stock-default S0 reproducibility failures will remain a separate finding and the main GPU curves will not be generalized to stock-default speed or quality.

The profile evidence, limits, and pending qualification steps are documented in the [GPU reproducibility profile record](docs/research/2026-11-18-gpu-reproducibility.md).

The primary time axis remains warm training time. The secondary `e2e_seconds` uses the versioned `resident-input-pipeline-v1` scope: it starts immediately before the measured adapter call and ends after centroid save/fsync. It excludes input loading, initialization, evaluation, and ledger I/O; it is not a load-inclusive end-to-end metric.

The target runtime is CPython `>=3.14,<3.15` with uv, in the existing GPU-enabled Docker environment. GPU commands should use `uv run --extra gpu`; set `UV_PROJECT_ENVIRONMENT="$PWD/.venv"` to keep uv pointed at this repository's environment. Public Hugging Face data needs no additional API key; `HF_TOKEN` is optional.

Experiment design, protocol, runtime/artifact details, and the implementation plan are indexed in the [Japanese design documents](docs/plan/README.md). The main README contains setup, CPU/GPU test commands, config names, and CLI examples in Japanese.

For reproducible restarts, use a frozen commit/worktree and keep parent CPU affinity (including taskset/numactl usage) the same across phases. Do not commit, edit, re-prepare data, or synchronize dependencies while a phase is running; use `UV_NO_SYNC=1` or `uv run --no-sync --extra gpu` with the frozen environment.
