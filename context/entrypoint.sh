#!/usr/bin/env bash
set -euo pipefail

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  repo_root="$(git rev-parse --show-toplevel)"
  hooks_path="${repo_root}/.githooks/pre-push"
  effective_hooks_path="$(git config --get core.hooksPath || true)"

  if [[ -f "${hooks_path}" ]]; then
    chmod +x "${hooks_path}"

    if [[ -z "${effective_hooks_path}" ]]; then
      git config --local core.hooksPath .githooks
      echo "Configured Git hooks path to '.githooks'."
    elif [[ "${effective_hooks_path}" != ".githooks" ]]; then
      echo "Git core.hooksPath is already set to '${effective_hooks_path}'. Leaving it unchanged."
    fi
  fi
fi

if [[ -f "pyproject.toml" ]]; then
  uv sync --all-groups || {
    echo "uv sync failed. Aborting container startup." >&2
    exit 1
  }
else
  echo "No pyproject.toml found in the current working directory: $(pwd). Skipping 'uv sync'. Ensure your project directory is mounted into the container and that pyproject.toml is located at the root of that directory."
fi

# Execute the original command passed to the container.
exec "$@"
