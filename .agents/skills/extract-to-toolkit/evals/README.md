# Evals for `extract-to-toolkit`

`evals.json` defines 3 test prompts against the dummy project in
`fixtures/dummy_project_module/`. This fixture is not a real
`project_module/`, since `extract-to-toolkit` mutates the repo it runs in
(moves files, rewrites imports, edits `pyproject.toml`) — running it
against this skill's own repo would be destructive.

## Running an eval manually

1. Create a scratch copy of a `dsg-project-template`-based repo (or a
   throwaway worktree).
2. Copy `fixtures/dummy_project_module/` to
   `project_module/src/dummy_project_module/` in that scratch repo.
3. Run the skill (`/extract-to-toolkit`) with the eval's `prompt`.
4. Check the output against `expectations`.

See `skill-creator`'s eval workflow (spawn with-skill + baseline subagents,
grade against `expectations`, review via `eval-viewer/generate_review.py`)
for a more rigorous run — this fixture is written to be compatible with
that flow once `skill-creator` is installed.
