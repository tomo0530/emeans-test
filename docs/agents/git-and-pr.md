# Git, PR, and Secret Scanning

Detailed Git and PR rules for AI agents. AGENTS.md §6 indexes this
file. Apply these when committing, pushing, or creating PRs.

## Branch naming

* Use a feature branch whose name passes
  [`branch_name_verifier.yml`](../../.github/workflows/branch_name_verifier.yml).
* Standard prefixes (see `README.md` for the full convention):
  `feat/` (or `feature/`), `fix/`, `doc/`, `change/` / `rename/`,
  `refactor/` / `style/` / `chore/`, `test/`, `dep/` / `dependabot/`,
  `release/`.
* **Never** commit directly to `main` / `develop`.

## Commits

* Commit only when the user explicitly authorizes it.
* Keep commits focused on a single logical change.
* Destructive Git commands (`git reset --hard`, force push, etc.) are
  prohibited without explicit approval — see AGENTS.md §3.

## Pull requests

* Use the `gh` CLI. **Not installed by default** — uncomment the
  GitHub CLI block in [`context/Dockerfile`](../../context/Dockerfile)
  and rebuild the dev container, or install it on the host
  (`brew install gh` or the official release).
* Follow the repository's
  [`pull_request_template.md`](../../.github/pull_request_template.md)
  when creating PRs.
* Do **not** force-push to shared branches without approval.

## Secret scanning (`gitleaks`)

### Local scan

* `make secret-scan [path]` scans the working tree using
  [`.gitleaks-local.toml`](../../.gitleaks-local.toml). Default path
  is `.`.

### Pre-push hook

* [`.githooks/pre-push`](../../.githooks/pre-push) runs `gitleaks git`
  on every push.
* `make setup-hooks` installs it; `context/entrypoint.sh` auto-installs
  it on first dev container start.

### CI

* [`secret-scan.yaml`](../../.github/workflows/secret-scan.yaml) scans
  on PR and on push to `main` / `develop`.

### Handling false positives

Priority order:

1. **Inline `# gitleaks:allow`** — most specific.
2. **`.gitleaks.toml`** — structural rules shared by CI and local.
   **Never** add path exclusions here; they would suppress CI
   detection too.
3. **`.gitleaks-local.toml`** — path exclusions used only by
   `make secret-scan` (local scope, never by CI).
4. **`.gitleaksignore`** — fingerprint-based, last resort (fragile
   under squash merge).
