# Testing numerical, random, and DataFrame code

Read this file when the code under test touches numpy arrays, pandas
DataFrames, random number generation, probabilistic algorithms, or trained
model artefacts. Each section changes a concrete decision: how randomness
enters the test, what to assert when no exact answer exists, how to compare
floats and frames, and which pytest marker a model-artefact test gets.

## Randomness: inject the generator

Take `rng: np.random.Generator` as a parameter and pass
`np.random.default_rng(seed)` from the test.

Avoid `np.random.seed()` and the legacy global `np.random` state in
general. The problem is one hidden, process-wide RNG shared by every
test:

- a test that does **not** seed it depends on however many draws earlier
  tests happened to consume — its values change with collection order;
- a test that **does** call `np.random.seed()` gets deterministic values
  for itself, but silently resets the state that every later test
  observes.

Either way, tests interfere with each other through state none of them
owns. An injected `Generator` gives each test its own isolated stream,
and the parameter makes the stochastic dependency visible in the
function's signature.

pandas APIs take the same injection through their `random_state` parameter
(`df.sample(random_state=rng)`); a `Generator` passes through unchanged.

If the production code currently reaches for the global `np.random` module
or a bare `df.sample()`, that is design friction — see
[`design-friction.md`](design-friction.md): propose threading a `Generator`
(or `random_state`) parameter through, and get user approval before
changing production code.

## When the exact value is unknowable

For stochastic or numerically involved code, prefer these, roughly in
order:

1. **Invariants** — properties any correct implementation satisfies:
   probabilities sum to 1, output length equals input length, no `NaN`
   appears that wasn't in the input, every resampled value comes from the
   input set.
2. **Boundary and degenerate inputs** — empty, single element,
   all-identical values: cases where the answer *is* knowable exactly
   (the bootstrap mean of `[x]` is `x`).
3. **Metamorphic relationships** — a controlled change to the input with a
   predictable effect on the output: doubling every weight preserves a
   ranking, permuting rows leaves an aggregate unchanged, adding a
   constant shifts the mean by exactly that constant.
4. **Constructed known-answer cases** — inputs designed by hand so the
   correct output is computable on paper.
5. **Independent naive reference** — a slow, obviously-correct
   implementation (plain loops, no vectorisation) compared against within
   tolerance. Keep it inside the test module and let it share nothing with
   production code, or it stops being independent.
6. **Golden files** — last resort. A stored snapshot cannot tell an
   improvement from a regression, and every intentional change forces a
   regeneration nobody can review by eye.

## Comparing floats and arrays

- Scalars and small structures: `pytest.approx`.
- Arrays: `np.testing.assert_allclose`. Plain `==` on floats fails for
  representation reasons, and `assert_array_equal` is right only for
  integer or exact-by-construction outputs.
- Assert **shape, then dtype, then values**. A shape bug reported as a
  value mismatch produces a wall of numbers instead of a one-line
  diagnosis.
- Choose `rtol` / `atol` from the numerics, not by widening until green:
  float64 carries ~1e-16 relative error per operation, so a short
  computation justifies `rtol=1e-12`, a long accumulation maybe `1e-8`.
  If the test only passes with a loose tolerance, the implementation is
  telling you it is numerically unstable — investigate instead of widening.
- Set `atol` explicitly when expected values include exact zeros: a purely
  relative tolerance compares against zero and fails on harmless noise.
- Prefer **non-square fixtures** with distinct values: a `(2, 3)` input
  catches transposed axes that a `(3, 3)` input hides, and distinct values
  catch mixed-up elements that constant fills let through.

## DataFrames: the schema is part of the contract

- Assert columns, dtypes, and `NaN` handling explicitly. A dtype that
  silently drifts from `int64` to `float64` (a classic `NaN` side effect)
  breaks downstream code without any value being "wrong".
- `pd.testing.assert_frame_equal` matches by **label, not position**.
  `check_like=True` only forgives column/index *ordering* — under a
  default `RangeIndex`, row order still is the label, so it does not make
  row order irrelevant. When order is not part of the contract, sort both
  sides by a key column first.
- Decide what a `NaN` in the input must produce — propagated, dropped, or
  an error — and assert that decision; it is the part of the contract most
  often left accidental.

## Trained models and artefacts

Unit tests cover the **plumbing**, never the quality:

- preprocessing output shapes and dtypes,
- threshold and boundary behaviour around the decision point,
- save/load round-trip,
- a clear error on a missing or corrupt artefact.

Accuracy, loss, and metric values are model *evaluation* — a different
activity with different tooling. A unit test asserting accuracy fails on
every retrain without any code being wrong.

For logic *around* a model, stub the model behind a `Protocol` that returns
fixed scores, so the surrounding code tests output-based with no artefact
at all.

### Which marker a model test gets

`pytest.ini` excludes `integrate` and `slow` from the default run and CI.
Choose by cost and external dependency — not by whether a "real" object is
involved:

| Situation | Marker |
|---|---|
| Small committed artefact (a tiny synthetic `.joblib`), loads fast | none — normal suite |
| Heavy local artefact or a long in-test fit | `@pytest.mark.slow` |
| Needs a registry, download, credential, or running service | `@pytest.mark.integrate` |

A local file is not an external service: `integrate` means "needs the
outside world". Whenever you add either marker, state in the handoff that
the test is excluded from default runs and CI, so nobody assumes it is
being exercised.

## Fixture data is synthetic only

Never commit real customer data, even scrubbed — `AGENTS.md` §3 forbids it
regardless of PII removal. While the real data is still in front of you,
record its schema, dtypes, and the edge cases you observe (nulls, boundary
values, odd encodings, duplicate keys), then synthesise a handful of rows
that reproduce exactly those properties. A small builder function with
keyword defaults (`_make_orders(n=3, with_nulls=False)`) keeps the
synthetic data readable and lets each test state only what it cares about.

## Not worth testing

- model quality metrics (evaluation, not unit testing),
- third-party library behaviour in general — test *your* assumption about
  a library only where a silent change would hurt (a dtype coercion, a
  join that must stay one-to-one),
- plot rendering,
- exploratory notebook cells.
