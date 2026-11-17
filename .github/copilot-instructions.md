# GitHub Copilot Instructions

## Language

- Write every code review comment and chat response in Japanese (日本語).
- Keep feedback concise, objective, and constructive.
- For each point, state the issue, explain why it matters, then suggest a concrete fix.

## How this file fits in

- This file is the review-facing subset of `AGENTS.md` (the single source of
  agent instructions) and `docs/agents/`, restated so it stays self-contained.
- Copilot code review on github.com reads this file, not `AGENTS.md`, so the
  review-critical rules are repeated here.
- The IDE and the coding agent also load `AGENTS.md`; keep the two consistent.

## Code review focus

Review in this priority order:

- **Secrets**: flag any API key, password, token, or credential in code, logs, or
  notebook output; these load from `.env` and are never committed.
- **PII & customer data**: flag any that reaches code, logs, or notebook output.
  Anonymize it, replace it with dummy data, or remove it — do not move it into `.env`.
- **Correctness**: behavioral bugs, missing error handling, unhandled edge cases, resource leaks.
- **Maintainability**: unclear names, missing docstrings, comments that should explain *why*.
- **Tests**: confirm new features and bug fixes ship with tests that cover the main cases.
- **Licensing**: dependencies use permissive licenses (MIT, Apache-2.0, BSD); flag GPL or other copyleft.
- **Input validation**: user input is validated and sanitized (OWASP Top 10).

Leave formatting and basic type style to the linters (`ruff`, `mypy`); focus on logic and design.

## Suggesting reuse and modularization

- Raise this only when a change looks clearly reusable across projects — not on every review.
- When the shared value is obvious, suggest extracting the logic into the shared
  `abeja-toolkit` library.
- When a whole workflow is repeatable, suggest packaging it as a skill in `abeja-skills`.
- Prefer an existing standard library or installed dependency before proposing a new helper.

## Security guardrails

- Keep secrets and credentials out of the repo; load them from `.env`.
- Keep PII and customer data out of code, logs, and committed notebooks; anonymize or remove it.
- Treat destructive Git operations (`git reset --hard`, force push) as changes that need explicit approval.

## When implementing from an issue (coding agent)

- Make the smallest change that resolves the issue; avoid unrelated refactoring.
- Add or update tests for every new feature or bug fix.
- Run the checks inside the Docker dev container by default — `make fix`, then
  `make all-lint`, then `make test`; host execution needs explicit confirmation.
- Use a branch name accepted by `.github/workflows/branch_name_verifier.yml`, and
  commit to that feature branch rather than `main` or `develop`.

<!-- TEMPLATE NOTICE (dsg-project-template): Keep this file self-contained and
     under ~4,000 characters — Copilot code review reads only the first 4,000.
     Adjust the `make` targets if your project uses different commands, then
     delete this notice. -->
