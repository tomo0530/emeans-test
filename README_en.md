[Japanese](./README.md)

# `dsg-project-template`

<!-- #### Please replace this with the official project name or repository name if needed. -->

## About this project template

<!-- #### This section explains `dsg-project-template`. You can delete it after starting your own project. #### -->

For how to create and use a project repository from `dsg-project-template`,
see the Japanese guide
[`dsg-project-template を使った DS 検証案件の進め方`](./docs/how-to-use-dsg-project-template.md).
After creating a new repository from this template, verify GitHub repository
settings by following
[`docs/agents/recommended-repository-settings.md`](./docs/agents/recommended-repository-settings.md).

Contributions to `dsg-project-template` are welcome.
Please refer to the [Development Guide](https://github.com/abeja-inc/dsg-project-template/wiki/Development-Guide) and create an Issue / Pull Request.

The following part of this README is intended to be directly reused as the project’s README.
The content assumes `dsg-project-template` by default, but feel free to edit it as needed (especially this section can be removed without any problem).

Comments written as `<!-- #### SOMETHING #### -->` are guidelines for writing the README.

↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

## Overview  <!-- Required -->

<!-- ####  
Describe concisely, in a few sentences, the problem this project/repository aims to solve and its expected outcomes.
If there is a Notion project page, please add a link here.
Examples:
- "Provides a full batch pipeline to automatically generate XX reports for the YY business."
- "Experimental repository for demand forecasting project for Company Foo. Achieved xx% performance."
#### -->

### Technologies / Methods Used  <!-- Optional -->

<!-- ####
List the main (or characteristic) technology stack and methods used in this project.
If there are no particularly notable technologies, you may delete this section.
(Examples: language / framework / library / middleware / infrastructure, etc.)
Example:
- Python 3.11
- PyTorch / TensorFlow / scikit-learn
- FastAPI / uvicorn
- ...
#### -->

## Architecture <!-- Optional -->

<!-- ####
Describe information that helps understand the overall system architecture or the full picture of the experimental environment.
If there is no particular major architecture information, this section can be deleted.

- If possible, include a diagram or a mermaid diagram (configuration of main components and integrations with external services).
- If using text only, explain the relationships between external services, batch jobs, APIs, etc. in bullet points.
#### -->

## Setup Instructions  <!-- Required -->

<!-- ####
Write down the steps to build the development/experimental environment from scratch.
Aim for "If you follow these steps exactly, the environment setup will be complete."
Commands should be in a copy-paste-ready form as much as possible.
#### -->

### Requirements  <!-- Required -->

<!-- ####
List the prerequisites required to run this project.
Below is an example from `dsg-project-template`. Modify according to your project-specific requirements.
#### -->

- `docker compose`
- `git`
- `make`

<!-- #### Add more items as necessary. #### -->

### How to Setup  <!-- Required -->

<!-- ####
Describe the concrete steps for setting up the environment in order.
Below is the simplest expected setup flow for `dsg-project-template`.
Please edit/add according to your project-specific setup procedure.

In particular, be sure to cover the following if applicable:
- Configuration of gcloud SDK / AWS CLI / ABEJA CLI, etc.
- Creation of environment variable files (.env, etc.)
- Commands for initial startup and migrations
- Customization details of Dockerfile in context/ and docker-compose.yml
#### -->

1. Clone the repository: `git clone <repository_url>`

2. Move into the working directory: `cd <repository_name>`

3. Edit `.env.container` and configure optional build flags.

   ```bash
   ${EDITOR:-vi} .env.container
   ```

   Default values vary per flag. Edit `.env.container` as needed.

   - `WITH_NODE` (default: `false`): Enable Node.js / npm
   - `WITH_OPENCV` (default: `false`): Enable extra dependencies for OpenCV (`libgl1`, `libglib2.0-0`)
   - `WITH_GH` (default: `false`): Enable GitHub CLI (`gh`)
   - `WITH_GITLEAKS` (default: `true`): Enable `gitleaks`
   - `WITH_LASTPASS` (default: `false`): Enable `lastpass-cli`
   - `WITH_VIM` (default: `false`): Enable `vim-tiny`

   If you want to add your own apt-based package to the container, add its runtime
   dependency to the local runtime manifest file that is copied into
   `/opt/runtime-manifests/` during the build (these manifest files are aggregated
   and installed in the `runtime-assembly` stage of `context/Dockerfile`).
   In this template, optional apt packages are intentionally managed through these
   runtime manifests so the `final` stage can stay focused on user/environment wiring.

4. Create the container: `make docker-up-cpu`

   (If you want to use CUDA, use `make docker-up-gpu` instead.)

   On first startup, `context/entrypoint.sh` automatically runs:
   - `git config --local core.hooksPath .githooks` to enable the repository-managed hook only when the effective `core.hooksPath` is unset
   - `uv sync --all-groups` to synchronize dependencies

   *Note*: If `core.hooksPath` is already set to another value in local, global, or system config, the container leaves it unchanged.
   If you work only on the host without using the container, or if you want this repository to switch to `.githooks`, run `make setup-hooks` manually.

   *Note*: `context/Dockerfile` uses `useradd -l` to avoid abnormal growth of
   `lastlog` / `faillog` when the host UID is large. If you run an SSH server or
   similar services inside the container and need login monitoring or auditing,
   remove this `-l`.

   <details><summary>Specifying the Python version</summary>

   Specify the Python version in `docker-compose.yml` under `context/<cpu or gpu>`.

   </details>

   <details><summary>JupyterLab port settings</summary>

   If you use JupyterLab, uncomment the `ports` section and align the port number specified in the `start-jupyter` command in the Makefile with the port setting in `docker-compose.yml`.
   By default, port 8888 is used. When running on a shared machine such as a DGX, check for available ports and change the host-side port in `docker-compose.yml` accordingly.
   (Also change the port specified in the `start-jupyter` command accordingly.)

   </details>

4. Enter the container.

   For normal development work, use the default user:

   ```bash
   make docker-shell
   ```

   If you need root privileges for OS-level package installation such as `apt-get install`, use:

   ```bash
   make docker-shell-as-root
   ```

   However, limit root usage to OS-level package installation. For Python dependencies, use `uv add` / `uv sync` as the normal user.

5. (If needed, inside the container) re-sync Python dependencies: `uv sync`

   Normally, `uv sync --all-groups` runs automatically when the container starts,
   so this step is needed only when you want to re-sync dependencies manually.
   In this template, the container uses `.venv_container` (configured via `UV_PROJECT_ENVIRONMENT`) instead of `.venv`
   to avoid sharing a virtual environment with the host machine (and to prevent disk pressure on environments where Docker volumes
   may be placed on non-RAID disks such as DGX).

   ```bash
   uv sync  # Sync all groups (in general, run this)
   uv sync --dev  # Sync only dev group packages
   uv sync --no-dev  # Sync all groups except dev
   uv sync --group <group_name1> --no-group <group_name2>  # Include/exclude specific groups as needed
   ```

#### (Optional) Setting up an analysis environment

<details><summary>VSCode</summary>

1. Connect VSCode to the Docker container.

   Use the `Dev Containers` extension in VSCode and attach VSCode to the running container.

2. VSCode setup

   **Inside the container**, run the following commands to install the minimum set of recommended VSCode extensions:

   ```bash
   # For stable VSCode
   make setup-vscode

   # For VSCode Insiders
   make setup-vscode-insiders
   ```

</details>

<details><summary>JupyterLab</summary>

1. On a local environment, run the following command to start JupyterLab:

   ```bash
   make start-jupyter
   ```

2. After startup, open your browser and access [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab).

   If you changed the port number, replace 8888 in the URL accordingly.
   When running on a shared machine such as a DGX, do SSH port forwarding as shown below before accessing the URL:

   ```bash
   ssh -N -L 8888:localhost:{PORT_NUMBER_ON_DGX} dgx1a
   ```

   If your host OS is Windows, create `%USERPROFILE%/.wslconfig` and add the following content (then shutdown WSL once):

   ```
   [wsl2]
   localhostForwarding=true
   ```

3. When you finish using JupyterLab, shut it down.

   You can shut it down from the JupyterLab GUI, or you can stop the container entirely using:

   ```bash
   make docker-stop
   ```

   If you only want to stop JupyterLab, you can instead run:

   ```bash
   make stop-jupyter
   ```

</details>

## Datasets Description  <!-- Required if data is used -->

<!-- ####
This section is mandatory for projects that use data. If the project does not handle any data, you may delete this section.

Describe how to prepare the data needed for reproducible experiments.
At a minimum, please include the following aspects:
- Data source / how to obtain the data
- Version, acquisition date, or other identifiers of the data
- Placement path (local / S3 / ABEJA Datalake, etc.)
- How anonymization, masking, and access permissions are handled
- If pseudo-data / sample data can be used as a substitute, how to create it
#### -->

## How to Run  <!-- Optional -->

<!-- ####
For each main process provided by this project (batch, training, inference, API server, etc.), write representative commands for running them.
If there are no particularly major processes, this section can be deleted.

Examples:
- Preprocessing: `python -m project_module.xxxx.preprocess`
- Training: `python -m project_module.xxxx.train`
- Inference: `python -m project_module.xxxx.inference`

Examples:
- Start API server: `make xxxx`
#### -->

## Reproduction Scenario (Smoke Test)  <!-- Required -->

<!-- ####
Describe the minimum steps that can be performed in about 5–10 minutes after the project ends to verify that "it works as described in the README".
If this project is not a pure experiment, you can delete this section, but in general it is recommended to provide it.

Examples:
1. Prepare sample data.
2. Run the sequence of commands for preprocessing + training + inference.
3. Describe how to check the expected output (files / logs / metrics, etc.).

Example:
1. Obtain xxx.csv from hogehoge and place it under `experimentation/dataset/`.
2. Run `experimentation/notebooks/main.ipynb`.
3. Confirm that `output.md` is generated under `experimentation/outputs/`.
#### -->

## Development

<!-- ####
Describe information related to developing this project.
Below is an example from `dsg-project-template`. Modify as needed to match your project structure.
#### -->

### Instruction files for AI coding agents

- The single source of truth for shared instructions to agents such as Claude Code, Codex, and Gemini CLI is the root `AGENTS.md` (additional instructions for `project_module/` live in `project_module/AGENTS.md`).
  To change the instructions, edit `AGENTS.md` and keep the `CLAUDE.md` in the same directory as the single line `@AGENTS.md`.
- When Claude Code starts in a subdirectory such as `project_module/`, the root `AGENTS.md` is treated as an external import and an approval dialog appears.
  Approve it; otherwise the shared root instructions are not loaded.
- Non-interactive runs such as `claude -p` cannot approve the import, so the shared root instructions are not loaded. Run them from the repository root.

### Branch naming

The expected branch name format in this repository is `prefix/short-summary` or `prefix-short-summary`.
The prefix has the following meanings:

- `feat/` or `feature/` … New feature
- `fix/` … Bug fix
- `doc/` … Documentation update
- `change/` / `rename/` … Spec changes / breaking changes
- `refactor/` / `style/` / `chore/` … Refactoring / formatting / misc updates
- `test/` … Test code updates
- `dep/` / `dependabot/` … Dependency updates
- `release/` … For release branches (only `release/**`)
- `copilot/` … Automated handling by GitHub Copilot

Whether a branch name follows this format is automatically checked by the [`Branch Name Verifier`](.github/workflows/branch_name_verifier.yml) workflow when a PR is opened against `main`, `develop`, or `release/**`.

<details><summary>For branch names that follow the above format, labels are automatically added to PRs by [`PR Auto Labeler`](.github/workflows/pr-auto-labeler.yml). </summary>

- `feat/` or `feature/`: `feature`
- `fix/`: `bug`
- `doc/`: `documentation`
- `change/` / `rename/`: `change`
- `refactor/` / `style/` / `chore/`: `other`
- `test/`: `enhancement` or `other`
- `dep/` / `dependabot/`: `dependencies`
- `copilot/`: `other`

In addition, if `.github/workflows/**` is changed, the `cicd` label is automatically added based on file matching.
Also, when `pyproject.toml` / `poetry.lock` / `uv.lock` / `requirements*.txt` are changed, the `dependencies` label is automatically added.

</details>

### How to run tests and static analysis  <!-- Required -->

<!-- ####
Describe commands for tests and static analysis tools that are prepared to maintain quality.
Below is an example from `dsg-project-template`. Modify to match your project structure.
#### -->

* Run tests (`pytest`): `make test` or `make test <path-to-tests>`

  For configuration, see [pytest.ini](./pytest.ini).

  Note:
  In standard pytest behavior, if no tests are collected, the exit code is 5.
  In this project, a pytest hook in `./conftest.py` treats this case as success (exit code 0) with a warning message.
  If you want to change this behavior, edit the hook in `./conftest.py`.
  Ref: [conftest.py: sharing fixtures across multiple files](https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files)

* Lint / format / type check (`ruff`, `mypy`): `make all-lint <directory or file>`

  Individual commands are as follows:

  * Lint: `make lint <directory or file>`
  * Format check: `make format <directory or file>`
  * Type check: `make type-check <directory or file>`
  * Lint auto-fix: `make lint-fix <directory or file>`
  * Format auto-fix: `make format-fix <directory or file>`
  * Lint + format auto-fix: `make fix <directory or file>`

  For lint/formatter configuration, see [ruff.toml](./ruff.toml). For type checking configuration, see [mypy.ini](./mypy.ini).
  By default, `libs/` is excluded from `ruff`, and `tests/` is excluded from `mypy`.
  Also, `tests/` is not fully excluded from `ruff`; only `E402` is configured via per-file-ignore.
  If you want to exclude specific libraries under `libs/` from `mypy`, update `exclude` in `mypy.ini`.
  If you want to change the exact inclusion or exclusion rules, update `ruff.toml` and `mypy.ini`.

* Secret scanning (`gitleaks`): `make secret-scan` or `make secret-scan <path>`

  Detects hardcoded API keys, tokens, and other secrets in your source code.
  This check runs automatically on PRs and on pushes to `main` / `develop` via GitHub Actions, but you can also run it locally before committing.

  Gitleaks must be installed beforehand (recommended version: 8.30.0 or later):
  - macOS: `brew install gitleaks`
  - Linux / Windows: Download the binary from [GitHub Releases](https://github.com/gitleaks/gitleaks/releases) (see [official README](https://github.com/gitleaks/gitleaks#installing) for details)
  - Inside Docker container: Included in the default image. If you are using an older image, rebuild it (`make docker-build-cpu` or `make docker-build-gpu`)

  When the container starts, `context/entrypoint.sh` automatically sets
  `core.hooksPath=.githooks` only when the effective `core.hooksPath` is unset,
  so `.githooks/pre-push` runs `gitleaks git` before `git push`. If another hook
  path is already configured, the container leaves it unchanged. If you do not
  use the container, or if you want this repository to switch to `.githooks`,
  you can enable the same hook manually with `make setup-hooks`. The local hook
  can be skipped with `git push --no-verify`, but the CI `secret-scan` still runs.
  To reproduce the exact `pre-push` check locally, use
  `gitleaks git --config .gitleaks.toml --redact --log-opts='<commit>^!'`.
  `make secret-scan [path]` is a broader working-tree scan that uses
  `.gitleaks-local.toml`, so it does not exactly match the hook behavior.

  For configuration, see [.gitleaks.toml](./.gitleaks.toml) (CI) and [.gitleaks-local.toml](./.gitleaks-local.toml) (local).
  If you encounter false positives, handle them in this order:
  1. `gitleaks:allow` (inline comment for per-line exclusion)
  2. `.gitleaks.toml` (structural exclusion shared by CI and local. **Do NOT add directory path exclusions here.**)
  3. `.gitleaks-local.toml` (directory path exclusions for local `make secret-scan` only)
  4. `.gitleaksignore` (fingerprint-based exclusion, last resort)

  **Recommended**: Add `secret-scan / gitleaks` as a Required status check in Settings > Branches > Branch protection rules to block merging PRs that contain detected secrets.

### Updating dependencies

To update project dependencies, run the following commands:

```bash
uv add <package_name>  # Add a package
uv add <package_name> --group <group_name>  # Add a package to a specific group
uv add <package_name> --dev  # Add a package to the dev group
uv add --editable <path_to_package>  # Add a local package in editable mode
uv remove <package_name>  # Remove a package
```

For details, see:
[https://docs.astral.sh/uv/concepts/projects/dependencies/](https://docs.astral.sh/uv/concepts/projects/dependencies/)

*Note*: If you want to update dependencies for `project_module`, move to `./project_module` and run `uv add`, etc. there.

### Checking dependencies for vulnerabilities

Run the following commands **inside the container** to check the dependencies recorded in `uv.lock` (including transitive ones) for known vulnerabilities.
If `uv.lock` does not exist, the dependencies are resolved first.

```bash
uv audit --preview-features audit-command  # Check all dependency groups
uv audit --preview-features audit-command --no-dev  # Check all groups except dev
```

- As of uv 0.12, `uv audit` is experimental, so pass `--preview-features audit-command` (it also runs without the flag, but prints a warning)
- When vulnerabilities are found, the affected packages and fixed versions are listed, and the command exits with a non-zero status
- For details, see [https://docs.astral.sh/uv/reference/cli/#uv-audit](https://docs.astral.sh/uv/reference/cli/#uv-audit)

### CI/CD

* [branch_name_verifier.yml](.github/workflows/branch_name_verifier.yml): Workflow definition that checks branch name rules
* [finding_changed_package.yml](.github/workflows/finding_changed_package.yml): Workflow that detects changed Python packages. Used by [lint.yaml](.github/workflows/lint.yaml) and [test.yaml](.github/workflows/test.yaml).
* [lint.yaml](.github/workflows/lint.yaml): Workflow definition that runs lint/formatter automatically. Runs on PRs and pushes to `main`.
* [pr-auto-labeler.yml](.github/workflows/pr-auto-labeler.yml): Workflow definition for PR Auto Labeler, which labels PRs based on their branch names. Runs when PRs are opened.
* [release-drafter.yml](.github/workflows/release-drafter.yml): Workflow definition that updates the draft release notes with Release Drafter. Runs on pushes to `main`.
* [slack.yaml](.github/workflows/slack.yaml): Workflow definition that notifies Slack when Issues / PRs are created
* [test.yaml](.github/workflows/test.yaml): Workflow definition that runs tests automatically. Runs on PRs and pushes to `main`.
* [secret-scan.yaml](.github/workflows/secret-scan.yaml): Workflow definition for secret scanning with Gitleaks. Runs on PRs and pushes to `main` / `develop`.

Below are CI/CD workflows managed for `dsg-project-template` itself:

* [create-release-pr.yml](.github/workflows/create-release-pr.yml): Workflow definition that is triggered manually via `workflow_dispatch`, creates a `release/{version}` branch from a specified source branch such as `develop`, and opens a release PR to `main`
  * Summary: Accepts `version` / `source_branch` / `target_branch`, then creates the release branch and PR while skipping safely if the branch or an open PR already exists.
  * How to use it outside `dsg-project-template`:
    * You can use it as-is.
    * However, the `v1.2.3` version-format validation runs only when `github.repository` is `abeja-inc/dsg-project-template` and is skipped for other repositories.
    * If your release flow differs, update the default `source_branch` / `target_branch` values and the validation condition as needed.
* [sync-main-to-develop.yml](.github/workflows/sync-main-to-develop.yml): Workflow definition that is triggered by pushes to `main` and automatically opens a sync PR from `main` to `develop`
  * Summary: After a push to `main`, it checks whether an open sync PR already exists and, if not, creates exactly one PR back to `develop`.
  * How to use it outside `dsg-project-template`:
    * By default, it runs only when `github.repository == 'abeja-inc/dsg-project-template'`.
    * To use it in another repository, remove that condition or replace it with your own repository name.
    * If needed, change the source and target branches (default: `main` → `develop`) to match your branching model.

### Updating [Abeja Toolkit](https://github.com/abeja-inc/abeja-toolkit)

<details><summary>You can sync to the latest version of abeja-toolkit or add new tools with the following commands.</summary>

```shell
chmod +x ./scripts/sync_tools.sh  # Run once only
./scripts/sync_tools.sh [<tool1> <tool2> ...] [--version <version>] [--debug]
```

Alternatively, you can use `make`:

```shell
make sync-tools [<tool1> <tool2> ...] [VERSION=<version:branch or tag>] [DEBUG=1]
```

</details>

### Updating [`dsg-project-template`](https://github.com/abeja-inc/dsg-project-template)

<details><summary>You can sync `dsg-project-template` to the latest version with the following commands.</summary>

```shell
chmod +x ./scripts/sync_dsg-project-template.sh  # Run once only
./scripts/sync_dsg-project-template.sh [--version <version:branch or tag>] [--debug]
```

Or you can use `make`:

```shell
make sync-template [VERSION=<version:branch or tag>] [DEBUG=1]
```

For details, see `./scripts/sync_dsg-project-template.sh --help`.

</details>

## Directory structure  <!-- Required -->

<!-- ####
Describe the directory structure of this repository and briefly explain the role of major directories/modules.
Below is an example from `dsg-project-template`. Modify according to your project structure.
You may also use an excerpt from the `tree` command output.
#### -->

```text
.githooks/
  pre-push                      # Git pre-push hook for Gitleaks
.github/
  pull_request_template.md      # PR template settings
  workflows/
    lint.yaml                   # Automatic lint / formatter
    test.yaml                   # Automatic test execution
    secret-scan.yaml            # Secret scanning (Gitleaks)
    ...
.vscode/                        # VSCode Dev Container / recommended settings
  settings.json                 # VSCode workspace settings. formatOnSave is off by default.
context/                        # Dockerfile for analysis/experiments and docker-compose.yml for CPU / GPU environments
  Dockerfile                    
  .hadolint.yaml
  cpu/docker-compose.yml        # CPU container settings
  gpu/docker-compose.yml        # GPU container settings
experimentation/
  dataset/                      # Dataset location
  latest/                       # Configs and results of latest experiments
  notebooks/                    # Notebooks for experiments
  outputs/                      # Experiment outputs
libs/
  abeja-toolkit/                # ABEJA internal shared tools
    lastpass_secret_provider/   # Tool to retrieve secrets from LastPass
    spreadsheet/                # Tool to read/write spreadsheet data
project_module/
  tests/                        # Unit tests etc.; expects filenames in the format test_*.py
  src/project_module/           # Project-specific logic implementation; put developed tools under this directory
  pyproject.toml                # Dependency definitions for project_module
  README.md                     # Usage of project_module
scripts/                        # Helper scripts for syncing templates/tools, etc.
  sync_dsg-project-template.sh  # dsg-project-template sync script
  sync_tools.sh                 # abeja-toolkit sync script
.gitignore
Makefile                        # Entry point for main commands
conftest.py                     # pytest common settings and hooks
mypy.ini
pyproject.toml                  # Dependencies for analysis / experiments / development
pytest.ini
README.md                       # This repository’s README
ruff.toml
.gitleaks.toml                  # Gitleaks configuration (CI, secret scanning)
.gitleaks-local.toml            # Gitleaks configuration (local, with directory exclusions)
uv.lock
```

## Tips

<details><summary>Access GitHub from Docker containers on DGX / Linux servers</summary>

This section applies only to **DGX / Linux servers**. macOS + Docker Desktop and similar
environments are out of scope.
ssh-agent must already be enabled. For details, see
[ssh-agent のすすめ / Recommended ssh-agent](https://www.notion.so/abejainc/ssh-agent-Recommended-ssh-agent-1956fbf10b72806c96c1e98cb613bf3f?source=copy_link).

1. Log in to the DGX with `ssh <dgx-host>` and confirm that `SSH_AUTH_SOCK` is available.
   (Assumes `ForwardAgent yes` is enabled in `~/.ssh/config`.)

   ```bash
   ssh <dgx-host>
   echo $SSH_AUTH_SOCK
   ls -l $SSH_AUTH_SOCK
   ssh-add -l
   ```

2. Uncomment the ssh-agent forwarding settings in
   `context/cpu/docker-compose.yml` or `context/gpu/docker-compose.yml`
   (both `volumes` and `environment`).

3. Start the container.

   ```bash
   make docker-up-cpu  # or make docker-up-gpu
   ```

4. Confirm GitHub access from inside the container.

   ```bash
   echo $SSH_AUTH_SOCK
   ssh-add -l
   ssh -T git@github.com
   ```

Note: If `SSH_AUTH_SOCK` is unset or the socket does not exist,
`docker compose up` fails with `bind source path does not exist`.
Leave the entries commented out in that case.

</details>

<details><summary>Use `~/.ssh` inside the container</summary>

If you want to use `~/.ssh` directly inside the container, uncomment the
`.ssh` mount settings in `context/cpu/docker-compose.yml` or
`context/gpu/docker-compose.yml`.

On DGX / Linux servers, it is recommended to use ssh-agent forwarding
instead of mounting `.ssh` directly into the container. Follow the previous
`Access GitHub from Docker containers on DGX / Linux servers` tip.

1. Uncomment the `.ssh` mount settings in
   `context/cpu/docker-compose.yml` or `context/gpu/docker-compose.yml`.

2. Start the container.

   ```bash
   make docker-up-cpu  # or make docker-up-gpu
   ```

3. Confirm that `~/.ssh` is visible inside the container.

   ```bash
   ls -la ~/.ssh
   ```

Note: If you do not need `~/.ssh` inside the container, keep the `.ssh`
mount settings commented out.

Note: If you uncomment the `.ssh` mount settings while `~/.ssh` does not
exist on the host, `docker compose up` and similar commands fail. In that
case, create `~/.ssh` first and try again.

</details>

<details><summary>Install GitHub CLI (`gh`) in the dev container</summary>

`gh` is disabled by default. Build with `WITH_GH=true` only when needed, and
the image will install `gh` by downloading a `.deb` from
`https://github.com/cli/cli/releases/`.
Set the version via `GH_VERSION` in `context/cpu/docker-compose.yml` or
`context/gpu/docker-compose.yml`.

In particular, on **DGX / Linux servers, it is recommended to first enable
ssh-agent inside the container by following the previous
`Access GitHub from Docker containers on DGX / Linux servers` tip**.
`gh` also performs SSH-based Git operations against GitHub, so having
ssh-agent working in advance reduces authentication issues.

1. If needed, update `GH_VERSION` in `context/cpu/docker-compose.yml` or
   `context/gpu/docker-compose.yml`.
2. If you changed `GH_VERSION`, also update
   `GH_EXPECTED_SHA256_AMD64` / `GH_EXPECTED_SHA256_ARM64` in
   `context/cpu/docker-compose.yml` or `context/gpu/docker-compose.yml`.
3. Rebuild the image with `WITH_GH=true`.

   ```bash
   WITH_GH=true make docker-build-cpu  # or WITH_GH=true make docker-build-gpu
   ```

4. Start the container.

   ```bash
   make docker-up-cpu  # or make docker-up-gpu
   ```

5. Authenticate GitHub CLI inside the container.

   ```bash
   gh --version
   gh auth login --git-protocol ssh --skip-ssh-key
   gh auth status
   ```

   `gh` authentication is not persisted. If the container is recreated,
   run `gh auth login --git-protocol ssh --skip-ssh-key` again when needed.

6. Optionally verify GitHub operations.

   ```bash
   gh repo clone <owner>/<repo>
   gh pr status
   ```

</details>

<details><summary>Use lastpass-cli (lpass) to retrieve secrets</summary>

With [lastpass-cli](https://github.com/lastpass/lastpass-cli), you can retrieve passwords and secrets stored in LastPass from inside the container.

**Installation**

Set `WITH_LASTPASS=true` in `.env.container` and rebuild the container.

```bash
# .env.container
WITH_LASTPASS=true
```

```bash
make docker-build-cpu  # or make docker-build-gpu for GPU environments
```

**Logging in and retrieving secrets**

Run the following commands inside the container:

```bash
# Log in to LastPass (enter your email address)
lpass login <your-lastpass-email>

# Retrieve a password example
lpass show --password "<entry-name>"

# Retrieve a specific field example
lpass show --field="<field-name>" "<entry-name>"

# Log out
lpass logout
```

For details, see the [lastpass-cli official documentation](https://github.com/lastpass/lastpass-cli).

</details>

<details><summary>Remove notebook outputs</summary>

If notebook cell outputs contain confidential information, it is recommended to remove the outputs before pushing to GitHub.
You can bulk-remove notebook outputs under a given directory with the following command:

```bash
make reset-notebook /path/to/directory/
```

</details>

<details><summary>Notify Slack when Issues / PRs are created</summary>

1. Access Slack App settings and create a bot for notifications.
2. In the app management screen, select `Incoming Webhooks` from the left navigation menu and enable webhooks.
3. Scroll down the page and submit a request to create a webhook (organization admin approval may be required).
4. Once approved, create a webhook.
5. Set the webhook URL as `SLACK_WEBHOOK_URL` in the repository’s GitHub Repository Secrets.

</details>

## Troubleshooting

* [Case: Error occurs when running `make docker-up-cpu` or `make docker-up-gpu` because `~/.gitconfig` or `~/.ssh` does not exist](https://github.com/abeja-inc/dsg-project-template/wiki/Troubleshooting)

* [Case: Error `useradd: user 'root' already exists` when running `sudo make docker-up-cpu` or `sudo make docker-up-gpu`](https://github.com/abeja-inc/dsg-project-template/wiki/Troubleshooting)

* [Case: When executing `make docker-up-cpu` or `make docker-up-gpu`, an error like `invalid project name "_hoge-username-userid". must consist only of lowercase alphanumeric characters, hyphens and underscores as well as start with a letter or number` occurs and the container cannot be built](https://github.com/abeja-inc/dsg-project-template/wiki/Troubleshooting)

## Limitations / Known Issues

<!-- ####
Describe assumptions, limitations, and known issues that users should be aware of.
Examples:
- Memory usage increases for large-scale datasets.
- Accuracy is not sufficient for certain parameter settings.
#### -->

## License / Contacts

<!-- ####
Describe license information for dependent libraries/models (when publishing externally), and list primary owners and reviewers for this project, along with contact information.
#### -->

## Related Links

<!-- ####
Add links to related Slack channels / Notion pages, other external documents, and repositories for shared platforms related to this project.
#### -->
