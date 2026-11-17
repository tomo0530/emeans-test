# Testing Standards

The entry point for all test work in this repository — writing, running,
diagnosing, and reviewing tests, and coverage improvement. AGENTS.md §6
indexes this file. It holds both the mechanical conventions and the
judgment layer on top: what deserves a test, which technique to use, and
what the first move is for each shape of testing task.

The flow is always: preflight → pick a task mode (and read its
procedure) → work, using the judgment sections below as the mode needs
them → self-check → run → hand off. Review-only is the exception: it
ends at the report, and its section in `testing/task-modes.md` lists
which steps it may run and which it must not.

Detail that is only needed conditionally lives one level down:

| Document | Read it when |
|---|---|
| [`testing/task-modes.md`](testing/task-modes.md) | you have picked a task mode below and are about to start |
| [`testing/design-friction.md`](testing/design-friction.md) | a test is hard to write and you're tempted to work around it |
| [`testing/ds-numerical-testing.md`](testing/ds-numerical-testing.md) | the target touches numpy, pandas, randomness, or model artefacts |

Out of scope for this guide: extracting code from a notebook into
`project_module/` — this guide applies after the code has been promoted
to a Python module. Also out of scope: judging model accuracy or loss
(model evaluation, not unit testing), and investigating CI
infrastructure failures beyond running the tests locally.

## What a test is for

A test earns its place by letting someone change the code later without
fear. Three consequences that cut against instinct:

- **A test suite is a liability as well as an asset.** Every test must be
  updated on refactors, read when behaviour is unclear, and diagnosed when it
  fails. Deleting a test whose value doesn't cover that cost is a legitimate
  outcome — say so when you find one.
- **Coverage shows where tests are absent, never whether tests are good.**
  An assertion-free test reaches 100% branch coverage. `pytest.ini` prints
  coverage on every run; don't quote it as evidence of quality — report
  which behaviour is now pinned down instead.
- **A test that fails while behaviour is still correct is worse than no
  test.** It trains the team to ignore failures, which disables the suite as
  a safety net even for the parts that are right.

## Evaluate a test's value

Use four lenses when deciding whether to add, retain, or improve a test:

| Lens | Ask |
|---|---|
| **Protection against regressions** | Does it exercise meaningful production behaviour and fail for a realistic bug? |
| **Resistance to refactoring** | Does it stay green when implementation details change but observable behaviour does not? |
| **Fast feedback** | Is it quick and deterministic enough to run as part of the normal development loop? |
| **Maintainability** | Can a reader understand its scenario, and can it run without costly external setup? |

Treat resistance to refactoring as a gate: a test that raises false alarms
cannot remain trusted. Balance regression protection against feedback speed,
then minimise maintenance cost. When reviewing a suite, report these four
properties rather than using coverage as a quality score.

## Preflight

Before writing or changing any test:

1. **`git status`** — note the branch and any uncommitted user changes;
   never overwrite or revert them, and keep your edits separable from
   theirs.
2. **Read the ground rules**: this file, plus `pytest.ini`, the root
   `conftest.py`, and the target package's `pyproject.toml`. The marker
   exclusions and the no-tests-collected trap (see "Run it") all originate
   there.
3. **Find the existing tests** for the target package and follow their
   naming and fixture conventions instead of inventing parallel ones.
4. **Run the relevant tests once** (inside the dev container) to establish
   a baseline. A pre-existing failure is context to report, not something
   to silently absorb into your change.
5. **A new dependency is an environment change** — if a test seems to need
   one, ask before any `uv add` (`AGENTS.md` §4).

## Choose the task mode

As a baseline, new behaviour ships with tests and bug fixes ship with a
regression test; the modes below define how, and the classification in
the next section decides the granularity. Different requests need a
different first move:

| The request looks like | Mode | First move |
|---|---|---|
| "implement this" — behaviour that does not exist yet | New behavior | pick one observable behaviour; plan tests and implementation together (order flexible, evidence required) |
| "add tests for this" — code that already works, incl. promoted notebook code | Existing code | pin current behaviour with tests; prove they can fail via an approved mutation check |
| "this is broken, fix it" | Bug fix | reproduce the bug as a failing regression test before touching code |
| untested old code needs restructuring | Legacy refactor | pin current behaviour with characterization tests |
| "improve coverage" | Coverage | treat the report as a candidate map, not a score |
| "this test fails sometimes" | Flaky diagnosis | reproduce first; suspect shared state first, but not only it |
| "are these tests any good?" / "review, don't change" | Review-only | read and grade; change nothing until asked |

Say which mode you picked, then read that mode's procedure in
[`testing/task-modes.md`](testing/task-modes.md) before starting.

## Decide whether the code deserves a unit test

Place the target on two axes — how complex or business-important it is, and
how many collaborators it drags in:

| | **few collaborators** | **many collaborators** |
|---|---|---|
| **complex / important** | **Domain logic or algorithm.** Unit test thoroughly — the highest return available in this repo. | **Over-complicated code.** Don't test it as it stands; split it first (see design friction), then test the extracted logic. |
| **simple** | **Trivial code.** Don't test. A passthrough `__init__` or one-line accessor has nowhere to hide a bug. | **Controller / orchestration.** A few broader tests of the wiring, not unit tests. Mark them `@pytest.mark.integrate` only when they genuinely need an external service — that is what the marker means in `pytest.ini`; in-process wiring tests stay in the normal suite. |

When asked to "add tests" for a whole module, classify the pieces first and
say which quadrant each falls in. Proposing to skip the trivial ones is part
of doing the job well.

## Pick the verification technique

Prefer, in this order — the cost of keeping a test alive rises sharply down
the list:

1. **Output-based** — call it, assert on the return value. Nothing to couple
   to. Always reach for this first.
2. **State-based** — run it, then assert on the resulting state of the object
   or a real collaborator.
3. **Communication-based** — assert that a collaborator was called. This is
   legitimate exactly when the outgoing interaction *is* the contract: the
   payload sent to an external API, the message published, the row written
   to an external store. Calls that merely *fetch inputs* are implementation
   detail — pinning them down blocks precisely the refactors tests exist to
   enable.

If the code can't be tested output-based, that is usually a fact about the
code rather than about testing. Before working around it, read
[`testing/design-friction.md`](testing/design-friction.md): the workaround
you're reaching for names a design flaw, and any production-code fix needs
the user's approval first.

## Write it — pytest conventions

Structure every test as **Arrange / Act / Assert** with visible separation
between the phases. On top of that:

- **One behaviour per test.** Two Act phases, an `if`, or a `for` is two
  tests wearing a trenchcoat — split them, or use
  `pytest.mark.parametrize(..., ids=[...])` so each case reports on its own.
- **Name the behaviour as a fact** — skimming the names should teach a
  reader what the module guarantees.
- **Call the subject `sut`** when several objects appear in one test, so the
  subject stands out from its collaborators; with one object, use its
  natural name.
- **Substitute dependencies in this order**: real objects → lightweight
  in-process fakes → `monkeypatch`. Introduce `Mock()` / `MagicMock()` only
  when replacing an external boundary (network, third-party SDK, file
  system) is the only practical option, and prefer the `mocker` fixture
  from `pytest-mock` (already in dev dependencies) over
  `unittest.mock.patch` — it cleans up between tests automatically.
- **Shared setup goes into named factory helpers**
  (`_make_client_with_cached_token()`), not into autouse fixtures the
  reader can't see from the test body. Fixtures requested by name in the
  signature (`tmp_path`, `monkeypatch`, `mocker`) are explicit and fine.
  Extract only once the duplication is real.
- **Assert every observable result of the one behaviour** — but a dozen
  field-level asserts usually means the production code is missing a value
  object with its own equality.
- **Error paths**: `pytest.raises(SomeError, match="...")` — `match` pins
  the message a caller will actually see.
- **Async tests use `anyio`**, not `pytest-asyncio`.
- **Hygiene against shared state**: use `tmp_path` and
  `monkeypatch.setenv` so cleanup is automatic, and never rely on
  collection order. Avoid `subprocess` and `time.sleep()` in unit tests;
  prefer deterministic events or polling boundaries.

### Example

```python
def test_applies_ten_percent_discount_at_ten_units():
    # Arrange
    unit_price = 10.0

    # Act
    total = apply_volume_discount(unit_price, quantity=10)

    # Assert — 10 * 10.0 * 0.9 worked out by hand, not recomputed from
    # the production formula
    assert total == pytest.approx(90.0)


def test_rejects_negative_quantity():
    with pytest.raises(ValueError, match="quantity must be non-negative"):
        apply_volume_discount(unit_price=10.0, quantity=-1)
```

The name states a fact, the phases are visible, the expected value is
independent of the implementation, and the error test pins the message a
caller will actually see (`match=`).

## DS code: numerical, random, DataFrame, model artefacts

When the target touches numpy, pandas, randomness, probabilistic
behaviour, or trained models, read
[`testing/ds-numerical-testing.md`](testing/ds-numerical-testing.md)
before writing assertions. It decides: how randomness is injected, what to
assert when no exact answer exists, tolerance-based float and array
comparison, DataFrame schema contracts, whether a model-artefact test is
normal, `slow`, or `integrate`, and how to build synthetic fixture data —
real customer data is never committed (`AGENTS.md` §3).

## Self-check before handing over

- **Would it survive a refactor?** Renaming a private helper, reordering
  internal calls, or swapping an internal data structure must not fail this
  test. If it would, revisit the verification technique or the design.
- **Has it been seen failing?** Bug-fix mode — and New behavior when the
  test was written first — produces this evidence as the failing run.
  For tests written against already-working code
  (existing-code, characterization, coverage), the check is a temporary
  mutation — but breaking production code even briefly is a
  production-code change, so get the user's OK first unless the approved
  plan already includes a mutation check. When it is approved, does not
  overlap pre-existing user changes, and can be restored exactly: record
  the current diff, break the behaviour minimally, watch the test fail,
  restore the original text without destructive Git commands, and confirm
  the diff is back to its expected state. Otherwise skip this check and
  say why. A test never seen failing buys confidence it hasn't earned.
- **Does the failure message identify the problem?** Read what pytest
  printed on the failing run; if it doesn't point at the cause, add
  `match=` or split the assertion.
- **Is the expected value independent of the implementation?** A test that
  recomputes the production formula agrees with the code even when the code
  is wrong.

## Run it

Per `AGENTS.md` §4, run inside the dev container:

```bash
make test                                    # default test paths
make test project_module/tests/test_foo.py   # or a single file / directory
make fix && make all-lint
```

In Review-only mode, run only what leaves the worktree untouched:
executing the existing tests and `make all-lint` is fine, but `make fix`
rewrites files — leave it, and any edit, until the user asks for
changes.

`make test` runs `uv run pytest`. Two repo-specific traps:

- **`make test` passing does not mean tests ran.** The root `conftest.py`
  turns pytest's "no tests collected" (exit code 5) into success with only
  a warning — see the root `conftest.py` to change this behaviour. Check
  the collected count in the output before reporting green.
- **`pytest.ini` runs `-m "not integrate and not slow"` by default.** A
  marked test is excluded from the normal run and from CI. That is usually
  what you want — but say so explicitly when you add one, so nobody assumes
  it is being exercised.

## Hand off the result

Structure the final report like this (omit rows that don't apply, but say
so rather than dropping them silently):

```markdown
## Test work summary
- Mode: <task mode>
- Classification: <domain logic / trivial / orchestration / over-complicated;
  anything deliberately left untested, and why>
- Behaviours protected: <behaviour → verification technique used>
- RED evidence: <why the failing run failed> → GREEN: <result>
  (bug fix and test-first new behaviour; for other modes, the
  mutation-check result or why it was skipped)
- Design findings: <production-code issues found; approval still needed?>
- Commands: <commands run>; collected: <N>; result: <pass/fail>
- Excluded from default run: <integrate/slow tests, or "none">
- Unresolved risks: <unreproduced flakes, pre-existing failures,
  skipped self-checks, or "none">
```

## CI

[`test_on_pr_push.yaml`](../../.github/workflows/test_on_pr_push.yaml)
runs `pytest` on changed packages on PR `opened` / `synchronize` events
and on push to `main`.
