# Task modes — detailed procedures

Read the section matching the mode chosen in `testing.md`'s "Choose the
task mode" table. Each mode exists because a different first move prevents
a different failure.

## New behavior

The behaviour does not exist yet — an ordinary implementation request.
This repo does not mandate test-first; it mandates that the tests are
planned with the implementation and that they end up with proof they can
detect a failure.

1. Pick one observable behaviour and agree the plan (Consensus First,
   `AGENTS.md` §1), including which tests will protect the behaviour and
   in which order you will write them.
2. Choose the order by the work's nature. When the expected behaviour is
   already well specified, writing the failing test first is the
   cheapest evidence — run it and confirm it fails *because the
   behaviour is missing* (an `ImportError` or fixture typo is a broken
   test, not a meaningful failure). When the shape is still exploratory,
   implement first and then pin the result following the Existing code
   steps below.
3. Whichever order you chose, end with the evidence: the failing run if
   the test came first, or an approved mutation check (see "Has it been
   seen failing?" in `testing.md`'s self-check) if it came after.
4. Finish with every relevant test green, and state in the handoff which
   order you used and what the evidence was.

## Existing code

The code already works — often logic promoted from a notebook after
exploratory verification — and the problem is that nothing pins its
behaviour down. The tests' job is to fix the current, correct behaviour
in place so later changes can be made without fear.

1. Classify the pieces first (`testing.md`'s two-axis table) and say which
   parts are trivial enough to skip.
2. Write tests asserting what the code observably does. Work expected
   values out independently — by hand, or from outputs already verified
   during exploration — never by re-running the implementation inside
   the test.
3. There is no RED step: the code is correct, so new tests start green.
   That also means no test has yet proven it can fail. Verify detection
   with a temporary mutation check where approved (see "Has it been seen
   failing?" in `testing.md`'s self-check), and say so when you skip it.
4. When something is hard to test, read `design-friction.md` before
   reaching for a workaround — friction here usually means the promoted
   code kept notebook habits (I/O mixed with logic, hidden randomness).

If the user also wants the code restructured, that is Legacy refactor —
the same pinning step followed by behaviour-preserving refactoring.

## Bug fix

1. Write the smallest regression test that reproduces the bug.
2. Run it **before touching production code** and confirm it fails for the
   reported reason — the same exception, the same wrong value. A test
   failing for an unrelated reason proves nothing about the bug.
3. Fix the root cause, and only the root cause.
4. Re-run the regression test and the surrounding tests.

## Legacy refactor

For code with no tests and no reliable spec, current behaviour is the spec
until proven otherwise.

1. Probe what the code actually does today — small inputs, edge cases,
   surprising branches.
2. Pin the behaviour that must survive with **characterization tests**:
   assert what the code *does now*, not what you wish it did.
3. Where current behaviour looks accidental or wrong, surface the
   discrepancy to the user as a question. Don't fold a silent behaviour
   change into a refactor.
4. Refactor in small steps with the characterization tests staying green.

## Coverage improvement

Coverage is a map of what is untested, never a score to maximise.

1. Read the coverage report as a list of candidates.
2. For each uncovered region ask: what realistic bug could hide here?
3. Where there is one, test it through public observable behaviour.
4. Where there isn't — trivial passthroughs, `__repr__`, generated code —
   skip it deliberately and say so. Never add an assertion-free or
   implementation-mirroring test just to reach a line.
5. Report which behaviours are now protected, not the percentage.

## Flaky test diagnosis

1. Reproduce before theorising: exact command, same environment, repeated
   runs; alone vs. its file vs. the package vs. the full suite. Note that
   `pytest.ini` sets `--ff`, so ordering changes after a failure.
2. A pass-alone / fail-in-suite difference makes **shared state** the
   first suspect — but only the first. Work down the list: module-level
   mutables, caches and singletons; collection order; environment
   variables and CWD; wall clock, timezone, locale; RNG without an
   injected seed; fixed filesystem paths; network and external resources;
   concurrency; a tolerance too tight for the numerics; import-time side
   effects.
3. Fix the root cause. Retries, sleeps, and loosened assertions hide the
   symptom and bill the debugging time to whoever hits it next.
4. If you cannot reproduce it, report exactly what you tried and what
   evidence is missing — don't guess-fix.

## Review-only

The user asked for judgment, not edits.

1. Read the suite and its configuration **without changing any file**.
2. Grade each finding against the four lenses.
3. Report severity, evidence (`file:line`), impact, and the recommended
   action for each.
4. Change nothing until the user explicitly asks — where review ends and
   rewriting begins is their call.

This mode overrides the guide's common flow wherever that flow would
write to the worktree:

- Read-only verification is fine: running the existing tests, and
  checks that do not rewrite files (`make all-lint`).
- Anything that writes waits for an explicit request or approval:
  `make fix`, temporary mutation checks, and edits of any kind.
- In the handoff, state that no files were changed and list only the
  read-only checks you ran.
