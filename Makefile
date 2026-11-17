.PHONY: docker-up-cpu \
	docker-up-gpu \
	docker-down \
	docker-down-rmi \
	docker-stop \
	docker-build-cpu \
	docker-build-gpu \
	docker-build-cpu-no-cache \
	docker-build-gpu-no-cache \
	docker-build-up-cpu \
	docker-build-up-gpu \
	docker-shell \
	docker-shell-as-root \
	docker-rmi \
	reset-notebook \
	setup-vscode \
	setup-vscode-insiders \
	setup-hooks \
	start-jupyter \
	stop-jupyter \
	all-lint \
	lint \
	format \
	type-check \
	lint-fix \
	format-fix \
	fix \
	test \
	to-notion \
	sync-template \
	sync-tools \
	install-npm \
	init \
	help \
	secret-scan

# const
_TRUTHY_VALUES := 1 y yes Yes YES true True TRUE
CONTAINER_ENV_FILE := .env.container

# Define variables for the project
export USER_NAME := $(shell id -un 2>/dev/null || whoami 2>/dev/null || echo user)
export USER_UID := $(shell id -u 2>/dev/null || echo 0)
export USER_GID := $(shell id -g 2>/dev/null || echo 0)
export USER_GROUP_IDS := $(shell id -G 2>/dev/null | tr ' ' ',')
export SANITIZED_USER := $(subst .,-,$(USER_NAME))
export PROJECT_DIR := $(shell basename $(shell pwd))
export PARENT_DIR := $(shell dirname $(shell pwd))
export PROJECT := $(shell basename $(shell pwd) | tr '[:upper:]' '[:lower:]')
export PROJECT_NAME := $(PROJECT)-$(SANITIZED_USER)-$(USER_UID)
export IMAGE_NAME := $(PROJECT):$(SANITIZED_USER)_$(USER_UID)
export CONTAINER_NAME := $(PROJECT)_$(SANITIZED_USER)_$(USER_UID)
export JUPYTER_PORT := 8888
export JUPYTER_HOST := 0.0.0.0  # NOTE: this will be used in the docker container, not on the host machine.
export JUPYTER_TOKEN ?=

# Directory
SCRIPTS_DIR := ./scripts
CONTEXT_DIR := ./context

# Define Docker Macros
DOCKER := docker
DOCKER_COMPOSE := $(DOCKER) compose -p $(PROJECT_NAME)
DOCKER_COMPOSE_CPU := RUNTIME=cpu $(DOCKER_COMPOSE) -f $(CONTEXT_DIR)/cpu/docker-compose.yml
DOCKER_COMPOSE_GPU := RUNTIME=gpu $(DOCKER_COMPOSE) -f $(CONTEXT_DIR)/gpu/docker-compose.yml
DOCKER_EXEC := $(DOCKER) exec $(CONTAINER_NAME)
DOCKER_EXEC_IT := $(DOCKER) exec -it $(CONTAINER_NAME)

# Python Macros
PYTHON_ENV_RUNNER := uv run
PYTHON := $(PYTHON_ENV_RUNNER) python
PYTHON_TEST_RUNNER := $(PYTHON_ENV_RUNNER) pytest
PYTHON_LINT_RUNNER := $(PYTHON_ENV_RUNNER) ruff check --config ./ruff.toml
PYTHON_LINT_FIX_RUNNER := $(PYTHON_LINT_RUNNER) --fix
PYTHON_FORMAT_FIX_RUNNER := $(PYTHON_ENV_RUNNER) ruff format --config ./ruff.toml
PYTHON_FORMAT_RUNNER := $(PYTHON_FORMAT_FIX_RUNNER) --check
PYTHON_TYPE_CHECK_RUNNER := $(PYTHON_ENV_RUNNER) mypy --config-file ./mypy.ini

# Security Macros
GITLEAKS := gitleaks

# -------------------------------------------------------------------
# make targets
# -------------------------------------------------------------------

# init
init: ## Initialize the project directory
# Implement any necessary initialization steps here
# (e.g., creating virtual environments, installing dependencies, create initial files, etc.)
	@echo "Project initialized. Please run 'make docker-up-cpu' or 'make docker-up-gpu' to start the Docker container."

# Docker targets
docker-up-cpu: ## Start the Docker container with CPU runtime
	@$(DOCKER_COMPOSE_CPU) --env-file $(CONTAINER_ENV_FILE) up -d

docker-up-gpu: ## Start the Docker container with GPU runtime
	@$(DOCKER_COMPOSE_GPU) --env-file $(CONTAINER_ENV_FILE) up -d

docker-down: ## Stop and remove the Docker container
	@$(DOCKER_COMPOSE) down

docker-down-rmi: ## Stop and remove the Docker container and its image
	@$(DOCKER_COMPOSE) down --rmi all

docker-stop: ## Stop the Docker container without removing it
	@$(DOCKER_COMPOSE) stop

docker-build-cpu: ## Build the Docker image for CPU runtime
	@$(DOCKER_COMPOSE_CPU) --env-file $(CONTAINER_ENV_FILE) build

docker-build-gpu: ## Build the Docker image for GPU runtime
	@$(DOCKER_COMPOSE_GPU) --env-file $(CONTAINER_ENV_FILE) build

docker-build-cpu-no-cache: ## Build the Docker image for CPU runtime without cache
	@$(DOCKER_COMPOSE_CPU) --env-file $(CONTAINER_ENV_FILE) build --no-cache

docker-build-gpu-no-cache: ## Build the Docker image for GPU runtime without cache
	@$(DOCKER_COMPOSE_GPU) --env-file $(CONTAINER_ENV_FILE) build --no-cache

docker-build-up-cpu: ## Build the Docker image and start the Docker container with CPU runtime
	@$(DOCKER_COMPOSE_CPU) --env-file $(CONTAINER_ENV_FILE) up -d --build

docker-build-up-gpu: ## Build the Docker image and start the Docker container with GPU runtime
	@$(DOCKER_COMPOSE_GPU) --env-file $(CONTAINER_ENV_FILE) up -d --build

docker-shell: ## Open a shell in the running Docker container
	@if ! $(DOCKER) ps --filter "name=^$(CONTAINER_NAME)$$" --format "{{.Names}}" | grep -q $(CONTAINER_NAME); then \
		echo "❌ Error: Docker container '$(CONTAINER_NAME)' is not running. " \
		"Please run 'make docker-up-cpu' or 'make docker-up-gpu' to start it." >&2; \
		exit 1; \
	else \
		$(DOCKER_EXEC_IT) /bin/bash; \
	fi

docker-shell-as-root: ## Open a root shell in the running Docker container
	@if ! $(DOCKER) ps --filter "name=^$(CONTAINER_NAME)$$" --format "{{.Names}}" | grep -q $(CONTAINER_NAME); then \
		echo "❌ Error: Docker container '$(CONTAINER_NAME)' is not running. " \
		"Please run 'make docker-up-cpu' or 'make docker-up-gpu' to start it." >&2; \
		exit 1; \
	else \
		$(DOCKER) exec -it -u root $(CONTAINER_NAME) /bin/bash; \
	fi

docker-rmi: ## Remove the Docker image
	@echo "🗑️ Removing Docker image '$(IMAGE_NAME)'..."; \
	if $(DOCKER) images --format "{{.Repository}}:{{.Tag}}" | grep -q "^$(IMAGE_NAME)$$"; then \
		$(DOCKER) rmi $(IMAGE_NAME) \
		&& echo "✅ Docker image '$(IMAGE_NAME)' has been removed." \
		|| echo "❌ Failed to remove Docker image '$(IMAGE_NAME)'. " \
		"Did you stop and delete the container by 'make docker-down'?"; \
	else \
		echo "❌ Docker image '$(IMAGE_NAME)' does not exist."; \
	fi

# Jupyter targets
start-jupyter: ## Start Jupyter Lab in the Docker container
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval arg1 := $(if $(word 1, $(args)), $(word 1, $(args)),.))
	@$(DOCKER_EXEC) /bin/bash -c \
		"$(PYTHON_ENV_RUNNER) ipython kernel install --user --name=$(PROJECT) \
		&& $(PYTHON_ENV_RUNNER) jupyter lab --no-browser --port $(JUPYTER_PORT) --ip=$(JUPYTER_HOST) --allow-root --NotebookApp.token='$(JUPYTER_TOKEN)' --notebook-dir='$(arg1)'"

stop-jupyter: ## Stop Jupyter Lab in the Docker container
	@$(DOCKER_EXEC) /bin/bash -c \
		"pkill -f jupyter"

reset-notebook: ## Clear output of all Jupyter notebooks in the specified directory (Usage: make reset-notebook <directory>. Default is current directory)
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval arg1 := $(if $(word 1, $(args)), $(word 1, $(args)),.))
	@find "$(arg1)" -name "*.ipynb" -exec $(PYTHON_ENV_RUNNER) jupyter nbconvert --clear-output --inplace {} +

# VSCode targets
setup-vscode: ## Install recommended VSCode extensions for Python development
	@code --install-extension ms-python.python
	@code --install-extension ms-python.mypy-type-checker
	@code --install-extension charliermarsh.ruff
	@code --install-extension GitHub.copilot
	@code --install-extension ms-toolsai.jupyter

setup-vscode-insiders: ## Install recommended VSCode Insiders extensions for Python development
	@code-insiders --install-extension ms-python.python
	@code-insiders --install-extension ms-python.mypy-type-checker
	@code-insiders --install-extension charliermarsh.ruff
	@code-insiders --install-extension GitHub.copilot
	@code-insiders --install-extension ms-toolsai.jupyter

# Python Linting, Formatting, Type Checking targets

all-lint: ## Run lint, format, and type-check (Usage: make all-lint <file_or_directory>. Default is current directory)
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(MAKE) --no-print-directory lint $(args)
	$(MAKE) --no-print-directory format $(args)
	$(MAKE) --no-print-directory type-check $(args)

lint: ## Run linters (Usage: make lint <file_or_directory>. Default is current directory). Note: This target only checks linting issues. If you want to automatically fix linting issues, please use 'make lint-fix'
	@echo "⚠️ Warning: 'make lint' is different from the previous version of 'make lint'."
	@echo "If you want to use the old behavior, please run 'make all-lint' instead."
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(PYTHON_LINT_RUNNER) $(args)

format: ## Run formatters in check mode (Usage: make format <file_or_directory>. Default is current directory). Note: This target only checks formatting. If you want to automatically fix formatting issues, please use 'make format-fix'
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(PYTHON_FORMAT_RUNNER) $(args)

type-check: ## Run type checking (Usage: make type-check <file_or_directory>. Default is current directory)
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval args := $(if $(strip $(args)),$(args),.))
	$(PYTHON_TYPE_CHECK_RUNNER) $(args)

lint-fix: ## Run linters in fix mode (Usage: make lint-fix <file_or_directory>. Default is current directory)
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(PYTHON_LINT_FIX_RUNNER) $(args)

format-fix: ## Run formatters in fix mode (Usage: make format-fix <file_or_directory>. Default is current directory)
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(PYTHON_FORMAT_FIX_RUNNER) $(args)

fix: ## Run linters and formatters in fix mode (Usage: make fix <file_or_directory>. Default is current directory)
	$(MAKE) lint-fix $(filter-out $@,$(MAKECMDGOALS))
	$(MAKE) format-fix $(filter-out $@,$(MAKECMDGOALS))

test: ## Run tests (Usage: make test <test_path> or make test <directory_with_tests>)
	$(PYTHON_TEST_RUNNER) $(filter-out $@,$(MAKECMDGOALS))

# Security targets
setup-hooks: ## Configure Git to use repository-managed hooks (.githooks)
	@if [ ! -f .githooks/pre-push ]; then \
		echo "❌ Error: .githooks/pre-push was not found." >&2; \
		exit 1; \
	fi
	@git config core.hooksPath .githooks
	@chmod +x .githooks/pre-push
	@echo "✅ Configured Git hooks path to '.githooks'."

secret-scan: ## Scan working tree paths for secrets (broader than pre-push; default path is .)
	@if ! command -v $(GITLEAKS) >/dev/null 2>&1; then \
		echo "❌ Error: 'gitleaks' is not installed." >&2; \
		echo "  Please install it first:" >&2; \
		echo "    macOS:   brew install gitleaks" >&2; \
		echo "    Linux:   See https://github.com/gitleaks/gitleaks#installing" >&2; \
		echo "    Docker:  Rebuild the dev image from context/Dockerfile (make docker-build-cpu or make docker-build-gpu)" >&2; \
		exit 1; \
	fi
	$(eval args := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval args := $(if $(strip $(args)),$(args),.))
	$(GITLEAKS) dir --config $(CURDIR)/.gitleaks-local.toml --redact --verbose $(args)

# npm targets
install-npm: ## Install npm
	@if [ ! -x $(SCRIPTS_DIR)/install_npm.sh ]; then \
		echo "❌ Error: Make $(SCRIPTS_DIR)/install_npm.sh executable (e.g., run 'chmod +x $(SCRIPTS_DIR)/install_npm.sh')."; \
		exit 1; \
	fi
	@$(SCRIPTS_DIR)/install_npm.sh

sync-template: SYNC_TEMPLATE_FLAGS := \
		$(if $(VERSION),--version $(VERSION)) \
		$(if $(DEBUG),--debug) \
		$(if $(NO_BRANCH), --no-branch)
sync-template: ## Sync the DSG project template (Usage: make sync-template [VERSION=<version>] [DEBUG=1] [NO_BRANCH=1])
	@echo "Syncing the DSG project template..."
	$(SCRIPTS_DIR)/sync_dsg-project-template.sh $(SYNC_TEMPLATE_FLAGS)


sync-tools: SYNC_TOOLS_FLAGS := \
		$(if $(VERSION),--version $(VERSION)) \
		$(if $(DEBUG),--debug)
sync-tools: ## Sync the abeja-toolkit tools (Usage: make sync-tools [VERSION=<version>] [DEBUG=1])
	@echo "Syncing abeja-toolkit tools..."
	$(SCRIPTS_DIR)/sync_tools.sh $(filter-out $@,$(MAKECMDGOALS)) $(SYNC_TOOLS_FLAGS)

help: ## Show help (auto-generated)
	@awk 'BEGIN{FS=":.*## "}; /^[a-zA-Z0-9_-]+:.*## /{printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

%:
	@:

# -------------------------------------------------------------------
# Deprecation helpers
# -------------------------------------------------------------------

# 1) Replacement exists: Warn and forward to the new target
# Usage: $(eval $(call DEPRECATE_FORWARD,old-target,new-target[,forward_pseudo_args]))
define DEPRECATE_FORWARD
.PHONY: $(1)
$(1):
	@echo "⚠️  Warning: '$(1)' is deprecated. Use 'make $(2)' instead." >&2
	@$(MAKE) --no-print-directory $(2) \
	    $(if $(filter $(_TRUTHY_VALUES),$(3)),$(filter-out $(1) $(2),$(MAKECMDGOALS)),)
endef

# 2) No longer needed: Warn and do nothing (considered successful)
# Usage: $(eval $(call DEPRECATE_DROP,old-target,reason))
define DEPRECATE_DROP
.PHONY: $(1)
$(1):
	@echo "⚠️  Warning: '$(1)' is deprecated and no longer necessary. $(2)" >&2
	@:
endef

# 3) No longer needed: Warn and fail (useful for CI detection)
# Usage: $(eval $(call DEPRECATE_FAIL,old-target,reason))
define DEPRECATE_FAIL
.PHONY: $(1)
$(1):
	@echo "⚠️  Warning: '$(1)' is deprecated. $(2)" >&2
	@exit 2
endef

_TO_NOTION_DEPRECATION_REASON := notebook_exporter / notion_extension are no longer bundled by this template. \
	To export a notebook to Notion ask a coding agent to upload it or run \
	'make sync-tools notebook_exporter notion_extension' and then \
	'$(PYTHON) ./libs/abeja-toolkit/notebook_exporter/export.py notion -nb \
	<notebook_path> --title <notion_page_title>'.

# 削除（noop）
$(eval $(call DEPRECATE_DROP,to-notion,$(_TO_NOTION_DEPRECATION_REASON)))
$(eval $(call DEPRECATE_DROP,init-runtime-cpu,""))
$(eval $(call DEPRECATE_DROP,init-runtime-gpu,""))
$(eval $(call DEPRECATE_DROP,init-working-dir,""))
$(eval $(call DEPRECATE_DROP,init-base,""))
$(eval $(call DEPRECATE_DROP,init-cpu,""))
$(eval $(call DEPRECATE_DROP,init-gpu,""))

# 代替（forward）
$(eval $(call DEPRECATE_FORWARD,init-docker-cpu,docker-up-cpu))
$(eval $(call DEPRECATE_FORWARD,init-docker-gpu,docker-up-gpu))
$(eval $(call DEPRECATE_FORWARD,ruff-check,lint,1))
$(eval $(call DEPRECATE_FORWARD,ruff-format,format,1))
$(eval $(call DEPRECATE_FORWARD,mypy-check,type-check,1))
$(eval $(call DEPRECATE_FORWARD,ruff-fix,fix,1))
