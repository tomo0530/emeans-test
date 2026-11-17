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
* **Consensus First**: in interactive sessions, present a plan and
  obtain explicit user approval before generating or modifying code.
  **If anything is ambiguous, ask the user — do not guess.** In
  non-interactive automation (CI, batch jobs), execute only the scope
  encoded in the job input.

## 2. Project Overview

<!-- #### Project-specific overview — replace this block upon adoption #### -->
> [!NOTE]
> **TEMPLATE PLACEHOLDER** — fill this in when adopting the template
> and remove the notice.

* See [`README.md`](./README.md) for directory layout, setup steps,
  and dev container tips.
* This repository is a `uv` workspace: the root project plus
  `project_module/` and `libs/abeja-toolkit/*`.

## 3. Critical Don'ts (Security & Safety)

* **Secrets**: never commit API keys, tokens, passwords, or
  credentials. Load them from `.env` files.
* **PII**: never commit customer data, PII, or logs / notebook
  outputs that contain them.
* **Licensing**: use only permissive licenses (MIT, Apache-2.0, BSD).
  GPL / copyleft is forbidden.
* **Input validation**: validate and sanitize all user input
  (OWASP Top 10).
* **Destructive Git**: never run `git reset --hard`,
  `git checkout -- <file>`, force push, or similar history- or
  worktree-destroying commands without explicit user approval.

## 4. Execution Environment

* Run `uv` / `python` / `make` targets **inside the Docker dev
  container by default** (`make docker-up-cpu` or
  `make docker-up-gpu`, then `make docker-shell`). Host-only execution
  requires the user's explicit confirmation.
* Adding / removing dependencies modifies the environment — ask the
  user before `uv add` / `uv remove`.
* `pyproject.toml` sets `exclude-newer = "1 week"` (dependency
  cooldowns) to mitigate supply-chain attacks. Do not bypass without
  user approval.

## 5. Workflow

1. **Verify**: Run `git status` / `git branch`, identify the current
   `owner/name` from the local `origin` URL without contacting GitHub,
   and read only
   [`.github/repository-initialization.yml`](./.github/repository-initialization.yml)
   for the initial repository-settings check. For an ordinary task,
   require schema version `2` and proceed without reading the detailed
   settings guide only when either:
   * `status` is `completed`, `repository` matches the current
     `owner/name`, and the non-empty `completed_audit_revision` equals
     `required_audit_revision`; or
   * the current repository is `abeja-inc/dsg-project-template` and its
     copied default marker has a non-empty `required_audit_revision`,
     `status: pending`, and empty `repository` and
     `completed_audit_revision` fields.

   Treat a missing status file, an unsupported schema, or any values not
   matching those cases—including `pending` outside the template
   exception, a repository mismatch, or stale revisions—as incomplete.
   Stop before ordinary task work, report the reason, and wait for
   explicit approval to initialize. Do not read
   [`docs/agents/recommended-repository-settings.md`](./docs/agents/recommended-repository-settings.md)
   until that approval is given. Read it immediately when initialization
   is approved or the user explicitly requests a fresh audit, even when
   the local status is otherwise valid. Do not call GitHub settings APIs
   during the initial local check or before initialization approval.
2. **Plan & Align**: apply Consensus First (§1) and wait for approval.
3. **Permission check**: ask before environment-modifying or
   destructive commands.
4. **Minimal changes**: only touch what the task requires.
5. **Implement & Test**: write / update tests following
   [`docs/agents/testing.md`](./docs/agents/testing.md) (this includes
   bug fixes, flaky-test diagnosis, and test reviews), run `make test`.
6. **Quality check**: `make fix` then `make all-lint` until clean
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
| Repository initialization and settings audit | [`docs/agents/recommended-repository-settings.md`](./docs/agents/recommended-repository-settings.md) |
| Git, PR, branch naming, secret scanning | [`docs/agents/git-and-pr.md`](./docs/agents/git-and-pr.md) |
| Writing DS logic in `project_module/` / handing it off to ENG (integration contract) | [`docs/ds_integration_contract.md`](./docs/ds_integration_contract.md) |
| Extracting DS logic from a notebook to `project_module/` | [`.claude/skills/ds-extract/SKILL.md`](./.claude/skills/ds-extract/SKILL.md) (`/ds-extract` skill in Claude Code; other agents read the file directly) |
| Checking / refactoring DS logic against the integration contract | [`.claude/skills/ds-contract-check/SKILL.md`](./.claude/skills/ds-contract-check/SKILL.md) (`/ds-contract-check` skill in Claude Code; other agents read the file directly) |
