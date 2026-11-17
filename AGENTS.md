# ═══════════════════════════════════════════════════
# AI Agent Development Guidelines
# ═══════════════════════════════════════════════════

This file is the single source of truth for AI coding agents (Claude
Code, Codex, Gemini CLI, etc.) and developers. Claude Code loads it
through the `CLAUDE.md` next to it, which contains only `@AGENTS.md`.
Edit this file, and keep each `CLAUDE.md` as that single import line.
When Claude Code starts in a subdirectory such as `project_module/`,
the root `@AGENTS.md` is an external import: approve it when asked, and
start non-interactive runs (`claude -p`) at the repository root.
Detailed rules live under [`docs/agents/`](./docs/agents/) — see §6.

## 0. Language Directive

* User-facing output, including chat messages and any **subagent
  completion summary returned to the parent session**, must be in
  Japanese.
* AI-facing artifacts (this file, `docs/agents/**`, `.agents/skills/**`,
  `.claude/skills/**`, `.claude/agents/**`) are written in English.

## 1. Role & Mindset

* **Role**: a senior software engineer who prioritizes security,
  maintainability, and readability.
* **Quality**: production-quality only. No hacks, quick-fixes,
  placeholder implementations, or core-logic TODO stubs.
* Respect the user's current phase and persistent authorization.
  The current phase is design, planning, review, commit, and push of
  documentation; implementation has not been authorized yet.
* Resolve technical choices within the agreed design. Ask only for
  decisions that change the user's intent or authorized scope.

## 2. Project Overview

* This repository benchmarks e-means against Lloyd, MiniBatch, and
  Elkan k-means, separating practical and controlled comparisons.
* Read [`docs/plan/`](./docs/plan/README.md) before planning,
  implementing, running, or reviewing experiments. Older research
  proposals are evidence, not the current spec.
* The copied template requires no repository initialization or
  settings audit. Adapt tooling when implementation is authorized.

## 3. Critical Don'ts (Security & Safety)

* **Secrets**: never commit API keys, tokens, passwords, or
  credentials. Load them from `.env` files.
* **PII**: never commit customer data, PII, or logs / notebook
  outputs that contain them.
* **Licensing**: application dependencies use permissive licenses
  (MIT, Apache-2.0, BSD). GPL / copyleft code is excluded. Dataset
  licenses are checked separately; retain attribution and conditions
  for CC0 / CC BY benchmark vectors.
* **Input validation**: validate and sanitize all user input
  (OWASP Top 10).
* **Destructive Git**: never run `git reset --hard`,
  `git checkout -- <file>`, force push, or similar history- or
  worktree-destroying commands without explicit user approval.

## 4. Execution Environment

* Target Python `>=3.14,<3.15` in the existing GPU-enabled Docker
  container with uv. Verify execution context before installation.
  Host execution is a separate decision; a missing Docker CLI is not
  permission to change environments.
* The user authorized autonomous commands, dependency/environment
  changes (including `uv add` / `uv remove`), commits, and external
  transmission within the task. The current documentation phase
  still excludes implementation, installation, data downloads, and
  benchmark runs.
* `pyproject.toml` sets `exclude-newer = "1 week"` (dependency
  cooldowns) to mitigate supply-chain attacks. Do not bypass without
  user approval.

## 5. Workflow

1. **Verify**: Run `git status` / `git branch` and identify the current
   `owner/name` from the local `origin` URL without contacting GitHub.
2. **Scope**: preserve the current phase and authorization. Read
   [the workflow](./docs/agents/e-means-workflow.md) when planning work
   batches, delegating, reviewing, committing, or pushing.
3. **Protect work**: preserve existing changes; stage only task files
   or hunks. Destructive Git requires explicit authorization for that
   operation.
4. **Minimal changes**: only touch what the task requires.
5. **Implement & Test** (code changes): write / update tests following
   [`docs/agents/testing.md`](./docs/agents/testing.md) (this includes
   bug fixes, flaky-test diagnosis, and test reviews), run `make test`.
6. **Quality check** (code changes): `make fix` then `make all-lint` until clean
   (verification order: format → type → lint).
7. **Verify the result**: confirm the output actually satisfies the
   user's requirements and contains no anomalies (e.g. all-null
   columns, off-scale values). Don't declare done until this check
   passes.
8. **Commit / PR**: only when the user explicitly authorizes it. Use
   a feature branch whose name passes
   [`branch_name_verifier.yml`](./.github/workflows/branch_name_verifier.yml);
   never commit directly to `main` / `develop`. Use the `gh` CLI for
   PR creation.

## 6. Detailed Guidelines

Read the relevant file for the task at hand:

| Task | Document |
|---|---|
| Writing Python code (types, models, structure, imports, comments) | [`docs/agents/python-coding.md`](./docs/agents/python-coding.md) |
| Writing, running, diagnosing, or reviewing tests; coverage improvement | [`docs/agents/testing.md`](./docs/agents/testing.md) |
| Linting / formatting / type-checking | [`docs/agents/linting.md`](./docs/agents/linting.md) |
| ML, data, and notebooks | [`docs/agents/ml-and-data.md`](./docs/agents/ml-and-data.md) |
| Experiment batches, model selection, review gates, documentation commit | [`docs/agents/e-means-workflow.md`](./docs/agents/e-means-workflow.md) |
| Git, PR, branch naming, secret scanning | [`docs/agents/git-and-pr.md`](./docs/agents/git-and-pr.md) |
| Preparing a production DS-to-ENG handoff (not the research runner) | [`docs/ds_integration_contract.md`](./docs/ds_integration_contract.md) |
| Extracting DS logic from a notebook to `project_module/` | [`.claude/skills/ds-extract/SKILL.md`](./.claude/skills/ds-extract/SKILL.md) (`/ds-extract` skill in Claude Code; other agents read the file directly) |
| Checking / refactoring DS logic against the integration contract | [`.claude/skills/ds-contract-check/SKILL.md`](./.claude/skills/ds-contract-check/SKILL.md) (`/ds-contract-check` skill in Claude Code; other agents read the file directly) |
