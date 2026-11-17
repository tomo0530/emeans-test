---
name: extract-to-toolkit
description: >-
  Scans the current project's project_module/ implementation, scores each
  module for extractability into libs/, and performs the actual
  extraction (file move, import rewrite, workspace registration).
  Run this skill from within a client project repo built on dsg-project-template.
  Use when the user wants to identify or move reusable code to libs/,
  mentions "共通化", "ライブラリ化", or "toolkit への移行". Also use this skill
  proactively whenever the user talks about project_module/ growing large or
  hard to navigate, asks where a utility/client/adapter file should live, wants
  to avoid duplicating an API client or helper across projects, or asks
  "this could probably be shared, right?" about a piece of code — even if they
  never say "extract" or name libs/ explicitly.
metadata:
  author: abeja-inc
  version: "1.0"
---

# Extract to Toolkit

Identifies and moves extractable modules from this project's
`project_module/` into `libs/<lib_name>/` as reusable packages.
Runs entirely within a single client project repo.

**Destination is `libs/<lib_name>/`, a project-local lib — not
`libs/abeja-toolkit/`.** `libs/abeja-toolkit/` is a mirror of the
separate `abeja-inc/abeja-toolkit` repo that `make sync-tools` wipes
with `rm -rf` and re-clones, so anything placed there is lost on the
next sync. Contributing to the shared `abeja-toolkit` is a separate
upstream PR and out of scope for this skill.

## Prerequisites

- `uv` workspace at the repo root with `pyproject.toml`.
- `project_module/src/<pkg>/` exists and contains implemented code.
- `make fix`, `make all-lint`, and `make test` available (or
  `uv run ruff format`, `uv run ruff check`, `uv run mypy`,
  `uv run pytest`).

## Instructions

### Phase 1: Setup

1. Locate the Python package root under `project_module/src/`:
   list the **immediate** subdirectories of `project_module/src/` (do
   not recurse) and keep the ones that directly contain an
   `__init__.py`. Nested subpackages (e.g. `<pkg>/utils/__init__.py`)
   also have an `__init__.py` but are not candidates at this step —
   only look one level deep.
   - If exactly one candidate exists, that directory is the
     **package root**.
   - If more than one candidate exists, list them and ask the user
     which one to analyze. Do not guess.
   - If `$ARGUMENTS` is given, treat it as an override path to the
     package root and skip the search above.
2. If the package root is empty (only `__init__.py`, no other `.py`
   files), tell the user there is nothing to analyze and stop.
3. Confirm the package root to the user before proceeding.

---

### Phase 2: Scan

Recursively collect all `.py` files under the package root
(excluding `tests/`, `__pycache__`, `*.pyc`).

For each `.py` file record:
- **rel_path**: path relative to the package root (e.g. `utils/http.py`)
- **dotted_path**: `rel_path` converted to a dotted module path, the
  form that actually appears in import statements (e.g. `utils/http.py`
  → `utils.http`; `utils/__init__.py` → `utils`)
- **dir_path**: its directory component (e.g. `utils`)
- **imports**: every `import X`, `import X as Y`, and
  `from X import Y` statement
- **symbols**: top-level function and class names
- **internal_importers**: other `.py` files in `project_module/` that
  import this module, in **any** form. Text/regex matching misses
  cases, so parse each file's AST (`ast.parse`, then walk `ast.Import`
  and `ast.ImportFrom` nodes) and resolve each import to an absolute
  module path (accounting for `ImportFrom.level` for relative imports).
  A file is an importer if any of these resolves to this module or its
  parent package + this module's name. All of these forms must be
  caught (using `utils/gcs.py`, dotted_path `utils.gcs`, as examples):
  - `from <pkg>.utils.gcs import upload_file` — symbol import
  - `from <pkg>.utils import gcs` — **module** imported from its parent
  - `import <pkg>.utils.gcs` / `import <pkg>.utils.gcs as g`
  - `from .gcs import upload_file` / `from . import gcs` — relative,
    from a sibling in the same package
  - `from ..utils.gcs import upload_file` / `from ..utils import gcs` —
    relative, from a sibling package at any depth

Also scan `libs/` (both the synced `abeja-toolkit/*` packages and any
project-local libs) and note existing lib names and their exported
symbols — used later to flag overlap.

---

### Phase 3: Analyze

Score every `.py` file on three axes (0–3 each, max 9). Base every
score on what the code actually does and depends on — **never on
directory names, filenames, or naming conventions.** Each client
project names things differently (including in Japanese or ad-hoc
conventions), so a directory called `utils` may contain business logic
just as easily as a directory called `domain` may contain a generic
helper. Read the code itself. Directory/file names may be weak
supporting context, but must never decide a score on their own.

**Layer score (0–3)** — how domain-independent the code's actual
*role* is. Ask: "If you stripped away this project's business context,
would this code still make sense and work unchanged in a completely
unrelated project?"
- **3** — infrastructure / low-level utility with no awareness of this
  project's domain (e.g. a thin wrapper around an HTTP client, a cloud
  SDK, generic date/string/file helpers).
- **2** — an abstraction or service skeleton that could be reused with
  adaptation, but shaped around this project's specific use case.
- **1** — encodes this project's business rules, ML feature
  definitions, or process-specific logic.

**Coupling score (0–3)** — how tied the code's *dependencies* are to
this project. Ask: "Would this file work if copied into a fresh
project with only its imports installed, and no other file from this
project available?"
- **3** — no dependency on any other file in this project (stdlib /
  third-party imports only).
- **2** — depends on sibling files in this project, but those siblings
  are themselves reusable (would also score ≥ 2 here) — the whole
  chain could move together.
- **1** — depends on a mix of reusable and business-specific code.
- **0** — directly depends on this project's business objects
  (entities, aggregates, ML feature classes, etc.).

**Genericity score (0–3)** — whether the code's *name and content*
reveal this project's specific business, and whether it's actually
reused elsewhere. Ask: "Could someone unfamiliar with this client
mistake this for a generic library module? Is it actually used in more
than one place?"
- **3** — domain-neutral name and content, reused by 3+ other files.
- **2** — domain-neutral but limited reuse (1–2 importers), or clearly
  general-purpose code even if only used once so far.
- **1** — ambiguous name/content, limited or no reuse yet.
- **0** — name or content clearly reveals project-specific business
  concepts. Flag these to the user regardless of the other scores.

When any axis is genuinely ambiguous, say so in the report instead of
guessing — see [references/layer-rubric.md](references/layer-rubric.md)
for worked examples.

**Total extractability score** = layer + coupling + genericity.
Only present candidates with total score ≥ 4.

---

### Phase 4: Report

Present a ranked table (highest score first). The `L/C/G` column shows
the layer / coupling / genericity subscores that make up the total —
not the directory the file lives in:

```
| Score | File | L/C/G | Imported by | Key imports | Note |
|-------|------|-------|-------------|-------------|------|
| 9/9   | utils/gcs.py | 3/3/3 | pipeline.py, loader.py, exporter.py | google-cloud-storage | — |
| 7/9   | clients/bq.py | 3/2/2 | pipeline.py | google-cloud-bigquery | similar to existing spreadsheet lib |
```

In the **Note** column, flag:
- Files whose name contains a domain-specific noun (likely harder to extract)
- Modules whose symbols overlap with an existing package under `libs/`

Then ask:
> "上記の候補から抽出するファイルを選択してください。
> 複数選択可。グループ化して1つのライブラリにする場合はその旨を教えてください。
> また、各ライブラリ名（kebab-case）を指定してください。"

Wait for explicit user selection before proceeding.

---

### Phase 5: Extract

For each approved group (one group = one new lib):

#### 5-1. Create lib directory

Create the new lib at `libs/<lib_name>/` — directly under `libs/`, **not**
inside `libs/abeja-toolkit/` (see the destination note at the top of this
file for why: `abeja-toolkit/` is wiped by `make sync-tools`).

Two names are in play, matching the underscore-directory convention used
by the synced packages under `libs/abeja-toolkit/` (e.g. the directory
`lastpass_secret_provider` on disk ships
`[project] name = "lastpass-secret-provider"` inside its `pyproject.toml`):
- `<lib_name>` — underscore, used for **both** the top-level directory
  under `libs/` and the `src/<lib_name>/` import package.
- `<lib-name>` — kebab-case, used **only** for the PyPI-style
  distribution name: `[project] name` in the new `pyproject.toml`, and
  wherever another project declares it as a dependency.

Read [references/lib-structure.md](references/lib-structure.md) for the
standard layout. Create:

```
libs/<lib_name>/
  src/<lib_name>/
    __init__.py
    py.typed
    <extracted files>
  tests/
    __init__.py
  pyproject.toml
  README.md
```

#### 5-2. Copy and adapt code

1. Copy source files into `src/<lib_name>/`.
2. Remove any imports that were project-specific and are no longer needed.
3. If a file imported another file also being extracted in the same group,
   update that import to use the new package name.
4. Do **not** add logic or refactor beyond what is needed for the move.
5. Write `src/<lib_name>/__init__.py` to re-export every top-level
   symbol (function, class) from the moved modules that any importer in
   `project_module/` will need — see the `__init__.py convention` in
   [references/lib-structure.md](references/lib-structure.md). This is
   required: 5-4 rewrites importers to `from <lib_name> import X`, which
   only works if `X` is re-exported from `__init__.py`.

#### 5-3. Write pyproject.toml

Use the template in [references/lib-structure.md](references/lib-structure.md).

Infer `dependencies` from the third-party imports found in Phase 2, but
**do not derive the PyPI distribution name mechanically from the
import path** — the import name (the top-level package you `import`)
and the distribution name (what you declare as a dependency) frequently
diverge. `from google.cloud import storage` and `from google.cloud
import bigquery` share the same `google.cloud` import path but are
separate distributions (`google-cloud-storage`, `google-cloud-bigquery`);
`import yaml` is `PyYAML`; `import cv2` is `opencv-python`.

Resolve each import to its distribution name in this order, and only
fall through when a step can't answer:
1. Reuse the exact name already declared in `project_module/pyproject.toml`
   or the root `pyproject.toml` for that same import.
2. Cross-check the root `uv.lock` — it records the resolved
   distribution names already in this workspace.
3. Map the top-level import package to its distribution via
   `importlib.metadata.packages_distributions()` in the project's
   environment (this is the authoritative import-name → distribution
   mapping for what's actually installed).
4. If still unresolved, propose a name and confirm it with the user
   before writing it. Don't silently guess.

Do **not** guess version constraints — omit them and note that the user
should pin them.

#### 5-4. Remove originals and rewrite imports

1. Delete the original `.py` file(s) from `project_module/`.
2. Rewrite every importer found in Phase 2. Work from the AST (the same
   `ast.Import` / `ast.ImportFrom` nodes), not a text search — the
   importing form determines the correct rewrite, and a plain `import`
   is referenced by its qualified name throughout the file body, not
   just at the import line. For a moved module `utils/gcs.py` becoming
   package `<lib_name>`:
   - **Symbol import** — `from <pkg>.utils.gcs import X`,
     `from .gcs import X`, `from ..utils.gcs import X` →
     `from <lib_name> import X` (works because 5-2 re-exports `X`).
   - **Module imported from its parent** — `from <pkg>.utils import gcs`,
     `from . import gcs`, `from ..utils import gcs` →
     `import <lib_name> as gcs` (preserve the name `gcs` the file
     already uses, so `gcs.upload_file(...)` call sites keep working).
   - **Plain module import** — `import <pkg>.utils.gcs` or
     `... as g` → `import <lib_name>` / `import <lib_name> as g`, and
     rewrite every qualified use in the file body
     (`<pkg>.utils.gcs.upload_file` → `<lib_name>.upload_file`, or via
     the alias) to match.
   After rewriting, grep the file for the old package/module path to
   confirm no reference to the deleted module remains.
3. Add `<lib-name>` (kebab-case distribution name) to
   `project_module/pyproject.toml`:
   - Under `[project] dependencies`
   - Under `[tool.uv.sources]` as `{ workspace = true }`

#### 5-5. Register in workspace

In the repo root `pyproject.toml`, add
`"libs/<lib_name>"` (underscore directory, directly under `libs/`) to
`[tool.uv.workspace] members`.

#### 5-6. Quality check

Run in order — fix all errors before proceeding to the next step:
```bash
make fix          # ruff format + ruff check --fix
make all-lint     # ruff check, mypy
make test         # catch import-rewrite or missing-reference regressions
```

If `make` targets are unavailable:
```bash
uv run ruff format . --config ./ruff.toml
uv run ruff check . --config ./ruff.toml --fix
uv run mypy . --config-file ./mypy.ini
uv run pytest
```

Two gaps these repo-wide commands leave, both of which you must cover
explicitly:
- **`make all-lint` only covers format / type-check / lint.** It won't
  catch a missed import rewrite or a reference to a file that was just
  deleted — only the test suite (`make test`) exercises the changed
  code paths.
- **The repo-root config may exclude `libs/` from the repo-wide
  checks** — at time of writing `ruff.toml` excludes `libs`, so the new
  lib's ruff pass is skipped by `make all-lint` / `make fix`. Check the
  current `ruff.toml` and `mypy.ini` `exclude` settings, and for
  whatever the repo-wide run skips, lint / type-check the new lib
  explicitly, scoped to it. It ships its own ruff / mypy / pytest dev
  deps, so run them against it directly:
  ```bash
  uv run ruff check libs/<lib_name>
  uv run ruff format --check libs/<lib_name>
  # run from inside the lib so any repo-root exclude doesn't apply
  uv run --directory libs/<lib_name> mypy src
  # and run the lib's own tests if any were added
  uv run --directory libs/<lib_name> pytest
  ```

Report any remaining errors to the user before declaring completion.

#### 5-7. Report completion

Summarize:
- New lib location and files moved
- Import rewrites applied
- Any manual follow-up needed (version pinning, missing type stubs, etc.)

---

## Examples

### Example 1: No arguments (typical usage)

```
User: /extract-to-toolkit

Phase 1: Found package root at project_module/src/my_project/.
Phase 2: Scanned 14 files.
Phase 3: Scored 14 files, 4 candidates with score ≥ 4.
Phase 4: Presenting ranked table.
User: utils/gcs.py を "gcs-client" として抽出してください。
Phase 5: Creating libs/gcs_client/ (distribution name: gcs-client) ...
```

### Example 2: Grouping related files

```
User: clients/bq.py と clients/bq_utils.py をまとめて "bigquery-client" にしてください。
Phase 5: Both files moved to libs/bigquery_client/src/bigquery_client/ (distribution name: bigquery-client).
```
