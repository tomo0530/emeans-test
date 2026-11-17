# When the code fights back — design friction catalogue

Read this when a test is hard to write: you want to patch something
private, the setup balloons, or there is no clean way to assert. Test
friction is diagnostic — the workaround you're reaching for names the
design flaw.

| You catch yourself wanting to… | What it actually means | Do this instead |
|---|---|---|
| test a private function directly, or patch one | a whole responsibility is hiding inside the unit (missing abstraction) | test it through existing public behaviour, or extract the responsibility into its own unit — but don't widen the package's public API just to give a test a handle; visibility is a design decision, not a testing convenience |
| make an attribute public just so a test can read it | the test is reaching for state as a proxy for behaviour | assert on the behaviour that state produces |
| partially mock the object under test (`mocker.patch.object(sut, "_fetch")`) | the object both decides and performs I/O — two responsibilities | split it: a pure function that computes, and a thin adapter that fetches |
| write a three-line Act phase to make one thing happen | the API leaks implementation details, and any caller could break the invariant by forgetting a step | fold the steps into one operation that maintains the invariant itself |
| copy a formula out of the production code to build the expected value | the test has no independent source of truth, so it agrees with the code even when the code is wrong | hardcode expected values worked out from the spec or by hand |
| add an `is_test` / `debug` flag to production code | test concerns are leaking into production | inject the varying behaviour as a dependency (a `Protocol` plus a test implementation) |
| patch `datetime.now()` or reach for `freezegun` | time is an implicit dependency | pass the timestamp in as a value; inject a clock only if threading a value through is genuinely impractical |
| stub out five things to get one test to run | the unit sits too deep in a dependency chain | move the logic outward to where it needs nothing, or accept it is a controller and cover the wiring instead |

Every "do this instead" that touches production code is a bigger commitment
than adding a test file: raise it with the user first, as a choice with the
cost of each option. If the design cannot change right now, don't build the
brittle partial-mock test anyway as a workaround — cover what is coverable
cleanly and report the constraint as a finding.
