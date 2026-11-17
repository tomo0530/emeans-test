# Extractability Scoring — Worked Examples

Supplementary material for Phase 3 of the `extract-to-toolkit` skill.
The scoring questions themselves live in `SKILL.md` — read that first.
This file exists to calibrate judgment on ambiguous cases; it is not a
lookup table. **Directory and file names are never the basis for a
score** — every example below is judged by what the code does, and the
directory name is mentioned only to show it can mislead.

## Why directory names are not used

`project_module/` starts empty in this template, and every client team
names its own directories (in English, Japanese, or a mix). A file
under `utils/` full of this client's billing rules is not extractable;
a file under `domain/` that's actually a stateless date-formatting
helper is. Scoring must follow the code, not the folder it happens to
sit in.

## Worked examples

### A file that looks domain-specific but scores high

`billing/gcs_upload.py` (directory name suggests billing domain) —
but the file only wraps `google-cloud-storage` upload/download calls
with no billing-specific logic, types, or constants. Score high on all
three axes: the directory name is a red herring.

### A file that looks generic but scores low

`utils/pricing_rules.py` (directory name suggests generic utility) —
but the file hardcodes this client's discount tiers and tax logic.
Layer and genericity should both be low despite the `utils/` location.

### Coupling chains

If `clients/bq.py` imports `clients/bq_utils.py`, first judge
`bq_utils.py` on its own merits. If it only does generic string/date
formatting with no project-specific logic, `bq.py`'s coupling score
should reflect that the dependency is itself extractable (score 2, not
1) — the two files can be grouped and extracted together.

### Third-party wrapper with a domain-flavored name

A class named `InvoiceGcsClient` that only calls generic
`google-cloud-storage` methods and takes a bucket/blob name as
arguments (never touching an `Invoice` type) — the name is
domain-flavored but the content is fully generic. Score based on
content: this is a genericity judgment call, and if uncertain, flag it
to the user rather than guessing from the name alone.

### Genuinely domain-coupled code

A function that accepts a project-specific `ContractEntity` or
`SalesFeature` object as a parameter, or that hardcodes business
constants (vendor names, tax rates, feature thresholds tuned for this
client's data) is domain-coupled regardless of where it lives or what
it's named. Score coupling 0 and layer 1.

## When to flag instead of scoring confidently

- The file mixes a generic helper with one or two business-specific
  branches (e.g. a generic retry wrapper that special-cases one
  client-specific error code). Score conservatively and note the mixed
  nature in the Phase 4 report.
- A class or function's true reusability depends on data you can't see
  from static reading (e.g. it's generic in structure but every
  observed caller passes project-specific values). Say so — don't
  invent certainty.
- The module's symbols look similar to an existing package under
  `libs/` (a synced `abeja-toolkit/*` package or a project-local lib).
  Note the overlap in the report instead of silently scoring it as a
  duplicate or an original.
