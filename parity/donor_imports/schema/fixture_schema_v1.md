# Fixture schema v1 — human description

Each fixture is a single JSON file conforming to `fixture_schema_v1.json`.

## Top-level keys

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int — must be `1` | Bumps on incompatible format changes. |
| `fixture_id` | string `[a-z0-9_-]+` | Stable identifier, used in `manifest.json` and the tolerance table. |
| `generator` | object | The producing implementation. See below. |
| `data` | object | The deterministic `(X, Y)` input. |
| `expected` | object | Per-field expected outputs with tolerances. |
| `comparison_policy` | object | How the comparator aligns components, resolves signs, and looks up tolerances. |

## `generator`

| Key | Meaning |
|---|---|
| `name` | e.g. `sklearn.cross_decomposition.PLSRegression`. |
| `version` | The library version that produced the fixture (e.g. `1.4.2`). |
| `git_revision_sha` | Short SHA of the generator repo at production time, or `unknown`. |
| `params` | Free-form object of constructor arguments — copied verbatim into the fixture for audit. |

## `data`

`X` and `Y` are dense matrices, each with `{ shape, layout, dtype, rng_seed?, values }`.

- `shape` is `[rows, cols]` (always 2-D, even for univariate `Y` which has `cols == 1`).
- `layout` is `row_major` or `col_major`. Phase 0 fixtures use row-major exclusively.
- `dtype` is `f64` or `f32`. Phase 0 uses `f64`.
- `rng_seed` is included when the matrix was generated via a deterministic RNG (every synthetic Phase 0 fixture sets it).
- `values` is a flat array of `rows * cols` numbers in the declared layout.

## `expected`

Free-form object whose keys are field names (`coefficients`, `intercept`, `x_mean`, `x_scale`, `y_mean`, `y_scale`, `weights_W`, `loadings_P`, `y_loadings_Q`, `rotations_R`, `predict_train`, `scores_T`, `y_scores_U`, …). Each value is an `expected_array` with:

| Key | Meaning |
|---|---|
| `shape` | Up to 3-D. |
| `values` | Flat array of the appropriate length. |
| `abs_tol` | Absolute tolerance for this field — applied per element. |
| `rel_tol` | Relative tolerance — applied as `\|a - b\| <= rel_tol * max(\|a\|, \|b\|, eps)`. |
| `sign_invariant` | If true, the comparator first flips signs along the relevant axis so each component direction has its largest-magnitude element positive (handles the PLS `±w` ambiguity). |

If a fixture omits one of the absolute tolerances or relative tolerances for a given field, the comparator looks them up by name in `parity/tolerances.md` using the `tolerance_table_row` key from `comparison_policy`.

## `comparison_policy`

| Key | Meaning |
|---|---|
| `components_alignment` | `first-k-prefix` (PLS / SIMPLS / NIPALS — components are taken in the order produced), `rotation-aware` (PLS canonical — rotations may need to be matched up), or `exact` (every component matches by index unconditionally). |
| `sign_resolver` | `max_abs_element_positive` or `none`. Applied to every field tagged `sign_invariant: true`. |
| `tolerance_table_row` | A `parity/tolerances.md` key like `sklearn/PLSRegression/nipals`. Used to resolve any tolerance not specified inline. |

## Determinism

Every Phase 0 fixture is deterministic: same library versions + same seeds + same `git_revision_sha` produce a byte-identical JSON. `manifest.json` records the SHA-256 of each fixture so the parity-gate workflow detects accidental drift.
