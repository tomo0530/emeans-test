# Global Arguments
#
# NOTE: The CPU image SBOM job in .github/workflows/export_sbom.yaml builds
# this Dockerfile with the ARG defaults as-is (no compose build args). Keep
# every package-affecting ARG default in this file (BASE_IMAGE, *_VERSION,
# WITH_* toggles) equal to the build args in context/cpu/docker-compose.yml,
# or the exported SBOM will no longer match the image developers run.
ARG BASE_IMAGE=ubuntu:24.04@sha256:0d39fcc8335d6d74d5502f6df2d30119ff4790ebbb60b364818d5112d9e3e932
ARG UV_VERSION=0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff

# Stage structure
#
# Inheritance graph (FROM):
#   base
#   ├─ base-python
#   │  └─ runtime-assembly
#   │     └─ final
#   ├─ lib-supplier-common
#   │  ├─ node-supplier
#   │  ├─ gh-supplier
#   │  └─ gitleaks-supplier
#   └─ source-build-common
#      └─ lpass-builder
#
# Artifact / dependency graph (COPY --from):
#   uv ─────────────▶ base-python
#   node-supplier ─▶ runtime-assembly
#   gh-supplier ───▶ runtime-assembly
#   gitleaks-supplier ─▶ runtime-assembly
#   lpass-builder ─▶ runtime-assembly
#
# Notes:
# - `uv` is a supplier stage and is not part of the inheritance chain.
# - `base` is the shared PoC / verification environment base. It intentionally
#   includes tools commonly needed inside the final container, such as `git`,
#   `make`, and a minimal native-build toolchain for ad-hoc dependency experiments.
# - Optional supplier stages produce a mini rootfs under `/artifacts/` so
#   `runtime-assembly` can merge them with a single `COPY --from=... /artifacts/ /`.
# - Runtime dependency manifests are emitted under
#   `/artifacts/opt/runtime-manifests/<tool>.txt`. Disabled stages still emit
#   empty manifest files, so `runtime-assembly` can aggregate them unconditionally.
# - `runtime-assembly` only merges prepared artifacts and installs declared
#   runtime dependencies. Small manifest-only toggles stay local there.

# ==========================================
# uv
# ==========================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ==========================================
# base
# ==========================================
FROM ${BASE_IMAGE} AS base

# Environment variables
ENV LANG=C.UTF-8
ENV PATH=/usr/local/bin:$PATH
ENV TZ=Asia/Tokyo

# Install packages that should remain available in the final PoC / verification
# container. This includes Python runtime libraries, shell utilities, and a
# minimal native-build toolchain because users may add or evaluate dependencies
# interactively inside the container.
# Docker does not invalidate this layer when the apt repositories publish
# updates, so cached builds keep the old packages. Bump this value to force the
# apt layer (including ca-certificates) to be rebuilt when a security update
# must reach existing build caches. Keep it out of the compose build args so the
# Dockerfile default stays the single source.
# 2026-10-23: ca-certificates 20260816ubuntu0.24.04.1 (USN-8944) distrusts the
# compromised emSign ECC Root CA - G3.
ARG APT_REFRESH=2026-10-23
RUN echo "apt refresh: ${APT_REFRESH}" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends \
        bash-completion \
        ca-certificates \
        curl \
        gcc \
        git \
        libc6-dev \
        netbase \
        libssl-dev \
        make \
        openssh-client \
        tzdata \
        unzip \
        wget \
        xz-utils \
        zip \
        zsh \
        libbz2-1.0 \
        libexpat1 \
        libffi8 \
        liblzma5 \
        libncursesw6 \
        libreadline8 \
        libsqlite3-0 \
        libssl3 \
        zlib1g \
        libgdbm6 \
        libuuid1 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# base-python
# ==========================================
FROM base AS base-python

ARG PYTHON_VERSION=3.14.9

# Make uv-managed Python placement explicit
# NOTE: https://docs.astral.sh/uv/reference/storage/#python-versions
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
# NOTE: https://docs.astral.sh/uv/reference/storage/#python-executables
ENV UV_PYTHON_BIN_DIR=/usr/local/bin

COPY --from=uv /uv /usr/local/bin/uv

# Install Python and verify the installed executables.
RUN uv python install ${PYTHON_VERSION} --default \
    && which python \
    && which python3 \
    && python --version \
    && python3 --version

# ==========================================
# lib-supplier-common
# ==========================================
FROM base AS lib-supplier-common

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==========================================
# source-build-common
# ==========================================
FROM base AS source-build-common

ARG ENABLE_SOURCE_BUILD=auto
ARG WITH_LASTPASS=false

# Preparation stage for source-build-heavy optional tools.
# - ENABLE_SOURCE_BUILD=true: always install source-build dependencies.
# - ENABLE_SOURCE_BUILD=false: never install source-build dependencies.
# - ENABLE_SOURCE_BUILD=auto (default): install when any source-build tool is
#   enabled (currently WITH_LASTPASS=true).
# Why this branch exists:
# - This Dockerfile unconditionally references supplier stages from
#   `runtime-assembly` via `COPY --from=...`, so shared stages are still
#   evaluated even when optional tools are disabled.
# - Without this guard, the heavy apt install in this stage would run even when
#   every source-build tool flag is false.
# - When adding a new source-build tool flag (e.g., WITH_XXX), update only the
#   auto-detection block below.
RUN set -eu; \
    enable_source_build="${ENABLE_SOURCE_BUILD}"; \
    if [ "${enable_source_build}" = "auto" ]; then \
        enable_source_build="false"; \
        if [ "${WITH_LASTPASS}" = "true" ]; then \
            enable_source_build="true"; \
        fi; \
    fi; \
    case "${enable_source_build}" in \
        true) \
            apt-get update \
            && DEBIAN_FRONTEND=noninteractive \
                apt-get install -y --no-install-recommends \
                build-essential \
                cmake \
                dpkg-dev \
                libcurl4-openssl-dev \
                libxml2-dev \
                pkg-config \
            && apt-get clean \
            && rm -rf /var/lib/apt/lists/* \
            ;; \
        false) \
            ;; \
        *) \
            echo "ENABLE_SOURCE_BUILD must be true, false, or auto." >&2 \
            && exit 1 \
            ;; \
    esac

# ==========================================
# node-supplier
# ==========================================
FROM lib-supplier-common AS node-supplier
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG WITH_NODE=false
# If you change this, also update the following corresponding expected SHA256 values below.
ARG NODE_VERSION=24.22.1
# v24.22.1 # See https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt
ARG NODE_EXPECTED_SHA256_AMD64=6c172a9ab6718f7b801fb5a7bd486af2ee5e32758086a04b84fcce2aa77095e6  
ARG NODE_EXPECTED_SHA256_ARM64=dbabf7a1cbf947e6059ffad2c27b80041db97dea1e553b62afdee7f13fca336e
ARG TARGETARCH

# Prepare a Node.js runtime rootfs artifact from the official tarball.
RUN mkdir -p /artifacts/usr/local /artifacts/opt/runtime-manifests \
    && : > /artifacts/opt/runtime-manifests/node.txt \
    && if [ "${WITH_NODE}" = "true" ]; then \
        resolved_targetarch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
        case "${resolved_targetarch}" in \
            x86_64) resolved_targetarch="amd64" ;; \
            aarch64) resolved_targetarch="arm64" ;; \
        esac; \
        case "${resolved_targetarch}" in \
            amd64) node_arch="x64"; expected_sha256="${NODE_EXPECTED_SHA256_AMD64}" ;; \
            arm64) node_arch="arm64"; expected_sha256="${NODE_EXPECTED_SHA256_ARM64}" ;; \
            *) echo "Unsupported TARGETARCH for Node.js: ${resolved_targetarch}" >&2; exit 1 ;; \
        esac; \
        node_dist="node-v${NODE_VERSION}-linux-${node_arch}"; \
        node_base_url="https://nodejs.org/dist/v${NODE_VERSION}"; \
        node_tarball="/tmp/${node_dist}.tar.xz"; \
        curl -fsSL "${node_base_url}/${node_dist}.tar.xz" -o "${node_tarball}" \
        && printf '%s  %s\n' "${expected_sha256}" "${node_tarball}" | sha256sum -c - \
        && tar -xJf "${node_tarball}" --strip-components=1 -C /artifacts/usr/local \
        && rm -f "${node_tarball}"; \
    fi

# ==========================================
# gh-supplier
# ==========================================
FROM lib-supplier-common AS gh-supplier
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG WITH_GH=false
# If you change this, also update the following corresponding expected SHA256 values below.
ARG GH_VERSION=2.98.0
# v2.98.0 # See https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_checksums.txt
ARG GH_EXPECTED_SHA256_AMD64=f65a3fa2fa0eb2e97c445ee3f5e087a40aae03b64847f45a8f13805e504535d6
ARG GH_EXPECTED_SHA256_ARM64=bbc4ac7964c2a091fd555cd1758d10a7cfcfdc472e405f0b0fb958f05d535cb6
ARG TARGETARCH

# Prepare an expanded GitHub CLI rootfs artifact from the release `.deb`.
# NOTE: `dpkg-deb -x "${gh_deb_tmp}" /artifacts` expands the `.deb` contents under `/artifacts/`
# while preserving the directory structure, so `COPY --from=gh-supplier /artifacts/ /` can merge it directly.
RUN mkdir -p /artifacts /artifacts/opt/runtime-manifests \
    && : > /artifacts/opt/runtime-manifests/gh.txt \
    && if [ "${WITH_GH}" = "true" ]; then \
        gh_target_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
        case "${gh_target_arch}" in \
            x86_64) gh_target_arch="amd64" ;; \
            aarch64) gh_target_arch="arm64" ;; \
        esac; \
        case "${gh_target_arch}" in \
            amd64) gh_arch="amd64"; gh_expected_sha256="${GH_EXPECTED_SHA256_AMD64}" ;; \
            arm64) gh_arch="arm64"; gh_expected_sha256="${GH_EXPECTED_SHA256_ARM64}" ;; \
            *) echo "Unsupported TARGETARCH for GitHub CLI: ${gh_target_arch}" >&2; exit 1 ;; \
        esac; \
        gh_pkg="gh_${GH_VERSION}_linux_${gh_arch}.deb"; \
        gh_url="https://github.com/cli/cli/releases/download/v${GH_VERSION}/${gh_pkg}"; \
        gh_deb_tmp="/tmp/${gh_pkg}"; \
        curl -fsSL "${gh_url}" -o "${gh_deb_tmp}" \
        && printf '%s  %s\n' "${gh_expected_sha256}" "${gh_deb_tmp}" | sha256sum -c - \
        && dpkg-deb -x "${gh_deb_tmp}" /artifacts \
        && rm -f "${gh_deb_tmp}"; \
    fi

# ==========================================
# gitleaks-supplier
# ==========================================
FROM lib-supplier-common AS gitleaks-supplier
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG WITH_GITLEAKS=true
# If you change this, also update the following corresponding expected SHA256 values below.
ARG GITLEAKS_VERSION=8.30.0
# v8.30.0 # See https://github.com/gitleaks/gitleaks/releases
ARG GITLEAKS_EXPECTED_SHA256_AMD64=79a3ab579b53f71efd634f3aaf7e04a0fa0cf206b7ed434638d1547a2470a66e
ARG GITLEAKS_EXPECTED_SHA256_ARM64=b4cbbb6ddf7d1b2a603088cd03a4e3f7ce48ee7fd449b51f7de6ee2906f5fa2f
ARG TARGETARCH

# Prepare a gitleaks runtime rootfs artifact from the official tarball.
RUN mkdir -p /artifacts/usr/local/bin /artifacts/opt/runtime-manifests \
    && : > /artifacts/opt/runtime-manifests/gitleaks.txt \
    && if [ "${WITH_GITLEAKS}" = "true" ]; then \
        gitleaks_target_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
        case "${gitleaks_target_arch}" in \
            x86_64) gitleaks_target_arch="amd64" ;; \
            aarch64) gitleaks_target_arch="arm64" ;; \
        esac; \
        case "${gitleaks_target_arch}" in \
            amd64) gitleaks_arch="x64"; gitleaks_expected_sha256="${GITLEAKS_EXPECTED_SHA256_AMD64}" ;; \
            arm64) gitleaks_arch="arm64"; gitleaks_expected_sha256="${GITLEAKS_EXPECTED_SHA256_ARM64}" ;; \
            *) echo "Unsupported TARGETARCH for gitleaks: ${gitleaks_target_arch}" >&2; exit 1 ;; \
        esac; \
        archive_name="gitleaks_${GITLEAKS_VERSION}_linux_${gitleaks_arch}.tar.gz"; \
        archive_tmp="/tmp/${archive_name}"; \
        curl -sSfL \
            "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/${archive_name}" \
            -o "${archive_tmp}" \
        && printf '%s  %s\n' "${gitleaks_expected_sha256}" "${archive_tmp}" | sha256sum -c - \
        && tar -xzf "${archive_tmp}" -C /artifacts/usr/local/bin gitleaks \
        && rm -f "${archive_tmp}" \
        && /artifacts/usr/local/bin/gitleaks version; \
    fi

# ==========================================
# lpass-builder
# ==========================================
FROM source-build-common AS lpass-builder

ARG WITH_LASTPASS=false
ARG LASTPASS_CLI_VERSION=feaeacf1549da07a646505cc531e63d6dfdc0f3f

RUN mkdir -p /artifacts/usr/local/bin /artifacts/opt/runtime-manifests \
    && : > /artifacts/opt/runtime-manifests/lpass.txt \
    && if [ "${WITH_LASTPASS}" = "true" ]; then \
        git init /tmp/lastpass-cli \
        && git -C /tmp/lastpass-cli remote add origin https://github.com/lastpass/lastpass-cli.git \
        && git -C /tmp/lastpass-cli fetch --depth 1 origin "${LASTPASS_CLI_VERSION}" \
        && git -C /tmp/lastpass-cli checkout --detach FETCH_HEAD \
        && make -C /tmp/lastpass-cli \
        && install -Dm755 /tmp/lastpass-cli/build/lpass /artifacts/usr/local/bin/lpass \
        && printf '%s\n' libcurl4 libxml2 libssl3 >> /artifacts/opt/runtime-manifests/lpass.txt \
        && rm -rf /tmp/lastpass-cli; \
    fi

# ==========================================
# runtime-assembly
# ==========================================
FROM base-python AS runtime-assembly

ARG WITH_OPENCV=false
ARG WITH_VIM=false

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=gitleaks-supplier /artifacts/ /
COPY --from=node-supplier /artifacts/ /
COPY --from=gh-supplier /artifacts/ /
COPY --from=lpass-builder /artifacts/ /

# If you want to add more optional runtime dependencies here,
# add `printf '%s\n' <package-name> >> /opt/runtime-manifests/local.txt;` after `: > /opt/runtime-manifests/local.txt`.
# If a dependency belongs to a supplier-provided tool, update the corresponding
# manifest file in that supplier stage as well.
RUN set -euo pipefail; \
    mkdir -p /opt/runtime-manifests; \
    : > /opt/runtime-manifests/local.txt; \
    if [ "${WITH_OPENCV}" = "true" ]; then \
        printf '%s\n' libgl1 libglib2.0-0 >> /opt/runtime-manifests/local.txt; \
    fi; \
    if [ "${WITH_VIM}" = "true" ]; then \
        printf '%s\n' vim-tiny >> /opt/runtime-manifests/local.txt; \
    fi; \
    cat /opt/runtime-manifests/*.txt | sed '/^$/d' | sort -u > /opt/all-runtime-packages.txt; \
    if [ -s /opt/all-runtime-packages.txt ]; then \
        mapfile -t runtime_packages < /opt/all-runtime-packages.txt; \
        echo "Resolved runtime packages:"; \
        cat /opt/all-runtime-packages.txt; \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${runtime_packages[@]}"; \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    rm -rf /opt/runtime-manifests /opt/all-runtime-packages.txt

# ==========================================
# final
# ==========================================
FROM runtime-assembly AS final

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USERNAME=ubuntu
ARG USER_UID=1001
ARG USER_GID=1001
ARG USER_GROUP_IDS=1001

# ==========================================
# Create a non-root user whose UID matches the host user to reduce
# file-permission mismatches on bind-mounted volumes.
# ==========================================
# NOTE: Use hadolint ignore to split uv installation and useradd command into separated layers.
# NOTE: --no-log-init avoids huge lastlog/faillog files when the host UID is large.
# hadolint ignore=DL3059
RUN if id -u ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi; \
    if getent group "${USER_GID}" >/dev/null 2>&1; then \
        primary_group="$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        groupadd -g "${USER_GID}" "${USERNAME}"; \
        primary_group="${USERNAME}"; \
    fi; \
    useradd -l --uid "${USER_UID}" -g "${primary_group}" -m "${USERNAME}"; \
    if [ -n "${USER_GROUP_IDS}" ]; then \
        echo "${USER_GROUP_IDS}" | tr ',' '\n' | while read -r gid; do \
            [ -n "$gid" ] || continue; \
            if getent group "$gid" >/dev/null 2>&1; then \
                group_name="$(getent group "$gid" | cut -d: -f1)"; \
            else \
                group_name="grp_$gid"; \
                groupadd -g "$gid" "$group_name"; \
            fi; \
            usermod -aG "$group_name" ${USERNAME}; \
        done; \
    fi

COPY --chown=${USER_UID}:${USER_GID} --chmod=755 entrypoint.sh /usr/local/bin/uv-entrypoint.sh

USER ${USERNAME}

# Reconfigure uv-managed Python placement for non-root user
# NOTE: https://docs.astral.sh/uv/reference/storage/#python-versions
ENV UV_PYTHON_INSTALL_DIR=/home/${USERNAME}/.local/share/uv/python
# NOTE: https://docs.astral.sh/uv/reference/storage/#python-executables
ENV UV_PYTHON_BIN_DIR=/home/${USERNAME}/.local/bin
ENV PATH=/home/${USERNAME}/.local/bin:$PATH

ENTRYPOINT ["/usr/local/bin/uv-entrypoint.sh"]
CMD ["/bin/bash"]
WORKDIR /home/${USERNAME}/
