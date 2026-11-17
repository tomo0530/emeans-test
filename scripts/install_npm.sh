#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /.dockerenv ]]; then
  echo "❌ Error: Run this command inside the Docker container."
  exit 1
fi

NVM_VERSION="v0.39.7"
if [[ ! -d "${HOME}/.nvm" ]]; then
  curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash
fi

export NVM_DIR="${HOME}/.nvm"
if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  . "${NVM_DIR}/nvm.sh"
fi

ORIGINAL_PATH="${PATH}"
nvm install node
nvm alias default node
nvm use default >/dev/null

NODE_VERSION="$(nvm version default)"
if [[ "${NODE_VERSION}" == "N/A" ]]; then
  echo "❌ Failed to resolve Node.js version via nvm." >&2
  exit 1
fi

NODE_BIN_DIR="${NVM_DIR}/versions/node/${NODE_VERSION}/bin"
LINK_DIRS="${HOME}/.local/bin"

symlink_in_path_dir() {
  IFS=':' read -r -a path_dirs <<< "${ORIGINAL_PATH}"
  for entry in "${path_dirs[@]}"; do
    [[ -z "${entry}" ]] && entry="."
    [[ "${entry}" == "${HOME}/.local/bin" ]] && continue
    case "${entry}" in
      "${HOME}"/*)
        if [[ -d "${entry}" && -w "${entry}" ]]; then
          LINK_DIRS+=" ${entry}"
          break
        fi
        if [[ ! -e "${entry}" ]]; then
          LINK_DIRS+=" ${entry}"
          break
        fi
        ;;
    esac
  done
}

symlink_in_path_dir

for target_dir in ${LINK_DIRS}; do
  [[ "${target_dir}" == "${NODE_BIN_DIR}" ]] && continue
  install -d "${target_dir}"
  for tool in node npm npx corepack; do
    if [[ -x "${NODE_BIN_DIR}/${tool}" ]]; then
      ln -sf "${NODE_BIN_DIR}/${tool}" "${target_dir}/${tool}"
    fi
  done
done

for rc_file in "${HOME}/.bashrc" "${HOME}/.profile"; do
  [[ -f "${rc_file}" ]] || touch "${rc_file}"
  if ! grep -qs '\\$HOME/.local/bin' "${rc_file}" \
    && ! grep -qs "${HOME}/.local/bin" "${rc_file}"; then
    printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "${rc_file}"
  fi
done

hash -r
printf '\n✅ install-npm: Run "source ~/.profile" or "exec bash -l" to use npm immediately in this shell.\n'
