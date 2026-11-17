# Recommended Repository Settings

Read this guide only after the user approves repository initialization
or explicitly requests a fresh repository-settings audit. The initial
local status decision is self-contained in `AGENTS.md` and
`.github/repository-initialization.yml`; ordinary tasks on a valid,
initialized repository must not load this guide or call GitHub settings
APIs.

The canonical audit revision is the status file's
`required_audit_revision`. Increment that value whenever this guide's
required audit checks change. A completed audit records the same value
in `completed_audit_revision`, allowing stale audits to be detected from
the local status file alone.

## Audit entry conditions

Enter this procedure only when one of these conditions is true:

* The initial check defined in `AGENTS.md` stopped because initialization
  was incomplete, and the user then explicitly approved initialization.
* The user explicitly requested a fresh audit. This request takes
  precedence over a valid completed status and the template-repository
  exception.

Before either condition is met, do not read the remainder of this guide,
audit GitHub settings, or change repository settings.

An agent must never determine that initialization is unnecessary. It may
record `status: completed` without an audit only after an explicit user
instruction that a human project owner has determined initialization to
be absolutely unnecessary. Record the rationale and approver in the
non-secret `notes` field.

Expected status values:

```yaml
schema_version: 2
required_audit_revision: recommended-repository-settings-v1
completed_audit_revision:
status: pending # pending | completed
repository:
completed_at:
completed_by:
reference: docs/agents/recommended-repository-settings.md
notes: "Template default. Repositories created from this template must audit settings before setting status to completed."
```

Completed status example:

```yaml
schema_version: 2
required_audit_revision: recommended-repository-settings-v1
completed_audit_revision: recommended-repository-settings-v1
status: completed
repository: owner/name
completed_at: "YYYY-MM-DDTHH:MM:SSZ"
completed_by: github-login-or-team
reference: docs/agents/recommended-repository-settings.md
notes: "Optional non-secret notes."
```

## Fast-path decision scenarios

The initial check in `AGENTS.md` must produce these outcomes without
reading this guide:

| Scenario | Initial action |
|---|---|
| Completed, repository matches, revisions match | Proceed with the ordinary task |
| `status: pending` outside the template exception | Stop and request initialization approval |
| Status file missing or schema unsupported | Stop and request initialization approval |
| Repository mismatch | Stop and request initialization approval |
| Required and completed revisions differ or are empty | Stop and request initialization approval |
| Template repository with its exact schema-v2 pending marker | Proceed with the ordinary task |
| Explicit fresh-audit request | Read this guide and run the audit |

Repository-settings changes such as team access, rulesets, Actions
permissions, and secrets are repository administration actions and must
not be made before approval.

## Baseline commands

These commands require `gh` authentication and repository admin or
maintain-level permissions for some endpoints.

```bash
repo=$(gh repo view --json nameWithOwner --jq '.nameWithOwner')

gh api "repos/$repo" \
  --jq '{default_branch,delete_branch_on_merge,visibility,security_and_analysis}'

gh api "repos/$repo/teams" \
  --paginate \
  --jq '.[] | [.name, .slug, .permission] | @tsv'

gh api "repos/$repo/collaborators" \
  --paginate \
  --jq '.[] | [.login, .role_name] | @tsv'

gh api "repos/$repo/rulesets" \
  --paginate \
  --jq '.[] | [.id, .name, .target, .enforcement] | @tsv'

gh api "repos/$repo/actions/permissions"
gh api "repos/$repo/actions/permissions/workflow"
gh api "repos/$repo/actions/secrets" \
  --paginate \
  --jq '.secrets[]?.name'
```

Inspect each active branch ruleset from the ruleset summary:

```bash
gh api "repos/$repo/rulesets/<ruleset-id>"
```

If a settings endpoint returns `Resource not accessible by integration`
or `Not Found`, confirm the viewer's repository role before treating
the setting as absent.

## Required checklist

Repository access:

* Required teams are granted access. Include the `DataScience` team
  with the project-approved permission level unless the project
  responsible owner explicitly approves an exception; record that
  exception in the non-secret `notes` field when marking the audit
  completed.
* Required individual collaborators and bot accounts are present with
  the minimum necessary permission.
* Team access is checked with `repos/$repo/teams`; collaborators alone
  cannot verify that the required team itself has been granted access,
  because the list aggregates access from multiple grant sources.

Branch rulesets:

* `main` has an active GitHub Ruleset.
* `develop` has an active GitHub Ruleset when the project uses the
  default `main` / `develop` workflow.
* Prefer GitHub Rulesets over classic branch protection. A repository
  can be protected even when the classic branch protection API returns
  `Branch not protected`.
* Protected branch rulesets include `deletion`,
  `non_fast_forward`, `pull_request`, `copilot_code_review`, and
  `required_status_checks` where applicable.
* Pull requests require at least one approving review.
* Stale approvals are dismissed when new commits are pushed.
* Branch deletion and force pushes are blocked.

Required checks:

* Required status checks include the standard CI checks used by this
  template where applicable: `gitleaks`, `python_lint / run_lint`, and
  `python_test / run_tests`.
* Docker lint runs as a matrix job for changed Dockerfiles. Do not
  register matrix-expanded Docker lint contexts as required status
  checks; add a stable aggregate context when one is introduced.
* Required status checks use strict branch update behavior when the
  project requires branches to be up to date before merge.

GitHub Actions:

* Actions are enabled for the repository.
* Default workflow token permissions are read-only.
* GitHub Actions cannot approve pull request reviews.
* Repository secrets required by enabled workflows are configured. For
  Slack notifications, configure `SLACK_WEBHOOK_URL`.

Copilot review:

* Copilot code review is enabled in the applicable branch rulesets.
* Decide per branch whether draft pull requests should receive Copilot
  review.

## Setup guide

Access and collaborators:

* GitHub UI: `Settings` -> `Collaborators and teams`.
* Add required teams first, then add individual collaborators only when
  team membership is insufficient.
* Use the minimum permission that allows the expected work.

Branch rulesets:

* GitHub UI: `Settings` -> `Rules` -> `Rulesets`.
* Create active branch rulesets for `refs/heads/main` and, when used,
  `refs/heads/develop`.
* Enable deletion protection, non-fast-forward protection, pull request
  requirements, Copilot code review, and required status checks.
* Add required status check names after the corresponding workflows
  have run at least once in the target repository.

Actions:

* GitHub UI: `Settings` -> `Actions` -> `General`.
* Set workflow permissions to read-only by default.
* Disable GitHub Actions approval of pull request reviews.
* Review allowed actions policy before enabling stricter restrictions.

Secrets:

* GitHub UI: `Settings` -> `Secrets and variables` -> `Actions`.
* Add repository secrets required by enabled workflows.
* Never write secret values into issues, pull requests, docs, or the
  initialization status file.

After setup:

* Re-run the baseline commands.
* Confirm every required checklist item is satisfied.
* If the user approves file changes, update
  `.github/repository-initialization.yml` to the completed status
  example for the target repository. Copy `required_audit_revision` to
  `completed_audit_revision`, and record the target repository,
  completion timestamp, and verifier. Never store secret values in the
  status file.

## Project-dependent settings

Decide these per project instead of treating them as universal
requirements:

* Whether to add `CODEOWNERS` and require code owner review.
* Whether to enable GitHub native secret scanning, push protection, or
  code security features in addition to the repository's gitleaks
  checks.
* Whether to restrict allowed GitHub Actions or require SHA pinning for
  all actions. Some workflows intentionally call trusted reusable
  workflows by branch.
* Whether to enable Discussions, Wiki, Projects, and repository topics.
* Which merge methods are allowed (`merge`, `squash`, `rebase`) and
  whether merged branches are deleted automatically.
