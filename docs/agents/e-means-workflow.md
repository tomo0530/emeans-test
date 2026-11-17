# e-means Planning and Execution Workflow

Read this when batching, delegating, reviewing, or preparing Git changes.
The [design and plan](../plan/README.md) owns experiment values/interfaces.

## Scope and authorization

The current request authorizes documentation, reviews, commits, and a
final push after approval. Implementation is explicitly deferred.
Autonomous dependency/environment changes and commands are authorized
for later implementation; do not ask for the same permission again.
Keep cooldowns and preserve unrelated work.

## Batches and ownership

Freeze shared contracts once, then parallelize data, adapters, and
evaluation/analysis with disjoint file ownership. Integrate and
self-check the whole batch before requesting review. Combine small
investigations, fixes, evidence, and commits into coherent milestones.

## Model selection

Check current supported IDs at dispatch. The current mapping is
Luna=`gpt-6-luna`, Terra=`gpt-6.1-terra`, Sol=`gpt-6.1-sol`,
Astra=`gpt-6-astra`.

| Work | Model and effort |
| --- | --- |
| Bounded mechanical docs / artifact checks | Luna, xhigh |
| Implementation and integration | Terra, xhigh |
| Statistics / scientific validity / independent verification | Sol, xhigh |
| Architecture / difficult experimental decisions | Astra, medium/high/xhigh according to difficulty |

Luna/Terra/Sol always use xhigh. Choose Astra medium for bounded
analysis, high for cross-module decisions, and xhigh for the hardest
scientific/design review. Use the authenticated Claude CLI for exact
Fable/Opus selections when the native subagent tool does not offer them.

## Review gates

For this design/plan commit, require explicit approval from Claude
Fable xhigh and fresh Astra xhigh. Protocol/evaluation/execution safety
also require final Opus xhigh and GPT-6.1 Sol xhigh approval. Review
the complete set in parallel with explicit scope and bounded requests.

For later implementation, protocol freeze, and scientific conclusions,
Opus xhigh plus Sol xhigh are the final gate. Send judgment calls and
proposed corrections/improvements to both in one consolidated
discussion. Complex new design ideas may additionally consult Astra
high plus Fable high. An early opinion does not approve later docs.

Merge findings, resolve facts from primary evidence, and apply one
correction batch. Recheck changed scope with the required reviewers;
avoid rereviewing unrelated completed work. Keep verdicts, requested
models, review scope, and material resolutions in one review record.
Timeouts and empty/error output are not approval.

## Git and verification

Use a topic branch and explicit staging allowlist. Exclude
`skills-lock.json`, `.agents/`, and `.claude/` from commits.
Preserve existing
dependency/container/skill/library edits. Check staged content,
whitespace, links, and secret scanning before the documentation commit.
After all four required documentation approvals, push once at the end
and verify the remote head. Remove only disposable
worktrees created for this task. Create a PR only when requested.

For code batches, run meaningful tests and format -> type -> lint before
the final gate. For this documentation batch, run document checks and
secret scanning without installing the experiment for make targets.
