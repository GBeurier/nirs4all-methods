# pls4all

`pls4all` is the slim, PLS-only subset of **nirs4all-methods** — a thin Python
binding over the portable `libn4m` C ABI (a C++17 PLS / NIRS engine). The wheel
bundles the `libn4m` shared library, so `pip install pls4all` is self-contained;
no separate native build is required. For the full method surface (preprocessing,
selectors, diagnostics, augmenters, …) install the `nirs4all-methods` package
instead and import it as `n4m` — both load the same `libn4m`.

The binding loads `libn4m` with `ctypes.CDLL` (so the GIL is released during
native calls) and exposes:

- `version()` / `abi_version()` introspection,
- a Pythonic `Context` and `Config` (RAII lifecycle wrappers),
- the PLS fit/predict surface and a scikit-learn-compatible
  `pls4all.sklearn.PLSRegression` (and the other PLS-family estimators),
- a typed `Pls4allError` raised on any non-OK status, carrying the
  context's `last_error` message.

## Quick start

```python
import numpy as np
import pls4all
from pls4all.sklearn import PLSRegression

print(pls4all.version())      # e.g. "1.4.1+abi.2.24.0"
print(pls4all.abi_version())  # (2, 24, 0)

rng = np.random.default_rng(0)
X = rng.standard_normal((40, 12))
y = X @ rng.standard_normal(12)

model = PLSRegression(n_components=5).fit(X, y)
print(model.predict(X).shape)  # (40,)
```

Low-level lifecycle, if you need it:

```python
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.n_components = 5
    # ... drive a fit through the C ABI ...
```

`scikit-learn` is an optional dependency (only `pls4all.sklearn` needs it); the
core `import pls4all` works with NumPy alone.

## Verified affine-model migration

`pls4all.export_linear_predictor_n4mm(...)` is the public, PREDICT-only
migration boundary for a converter that has already verified a source model's
affine equation.  It accepts finite, row-major coefficients with shape
`(n_features, n_targets)` and a target-length intercept, then returns a raw
N4MM payload.  It neither reads pickle/joblib data nor retrains a model.

```python
from pls4all import export_linear_predictor_n4mm

payload = export_linear_predictor_n4mm(
    coefficients=[[0.5], [-1.0]],
    intercept=[0.25],
    source_training_samples=120,
)
assert payload[:4] == b"N4MM"
```

The resulting imported-linear-predictor model supports native prediction only;
it deliberately does not claim to reconstruct a source PLS latent transform.
Callers remain responsible for attesting that the source model is exactly
`intercept + X @ coefficients` before conversion.

## Pure-native estimator selection in the full package

The full `nirs4all-methods` distribution (imported as `n4m`) exposes the native
single-level finetuning driver. It is not part of the slim `pls4all` namespace.
Build the validation plan and search space through public owning objects:

```python
import numpy as np

from n4m.model_selection import (
    Algorithm,
    Metric,
    Sampler,
    SearchSpace,
    ValidationPlan,
    finetune_estimator,
)

rng = np.random.default_rng(42)
X = rng.normal(size=(60, 12))
y = X @ rng.normal(size=12) + rng.normal(scale=0.02, size=60)
fold_ids = np.arange(X.shape[0], dtype=np.int32) % 5

with ValidationPlan.from_fold_ids(fold_ids) as plan:
    with SearchSpace() as space:
        space.add_int("n_components", 1, 10)
        result = finetune_estimator(
            Algorithm.PLS_REGRESSION,
            X,
            y,
            plan,
            space,
            n_trials=30,
            sampler=Sampler.TPE,
            metric=Metric.RMSE,
            seed=42,
            timeout_seconds=20.0,
        )

print(result.best_params)       # for example: {"n_components": 6}
print(result.best_score)
print(result.timed_out)         # True only for a successful partial timeout
print(len(result.trials))       # owning completed/failed trial snapshots
```

The eligible algorithms are `PLS_REGRESSION`, `PLS_CANONICAL`, `PLS_SVD`,
`OPLS`, `SPARSE_PLS` and `PCR`. The five non-sparse routes require the exact
keyword `n_components`, a non-log integer axis. `SPARSE_PLS` accepts any
non-empty subset of `n_components` and `sparsity_lambda`; omitted axes retain
the native defaults `2` and `0.0`. `sparsity_lambda` must stay in `[0, 1)` and
may be declared with `space.add_float(..., log=True)` when `low > 0`.

This call selects a configuration and returns an owning trace; it does **not**
fit a final estimator on all rows. A timeout before any completion raises
`N4MError` with `CANCELLED`. A later timeout returns the best partial result
with `result.timed_out is True` and fewer trials than
`result.requested_trials`. See
[`docs/methods/optimization.md`](../../docs/methods/optimization.md#pure-native-estimator-selection)
for the exact schema, status and validation-plan contracts.

## Explainability in the full package

`n4m.explain` (full `nirs4all-methods` distribution, not the slim `pls4all`
namespace) wraps the native `n4m/explain.h` attribution kernels. All numerics
run in libn4m; the binding only marshals arrays and drives the native ask/tell
session, so any Python predictor works:

```python
import numpy as np
from n4m import explain as ex

# X and background live in the feature space the predictor consumes, e.g. raw
# spectra of a complete fitted pipeline.
groups, n_groups = ex.groups_fixed_width(X.shape[1], 16)   # optional grouping
result = ex.explain(
    pipeline.predict,            # receives C-contiguous float64 (rows, n_features)
    X,
    background=background,
    feature_group=groups,
    n_groups=n_groups,
    method="permutation_shap",   # exact_shap | permutation_shap | kernel_shap | lime | occlusion
    baseline="background",       # background | mean | interpolate
    n_permutations=10,
    seed=0,
    progress=lambda done, total: print(f"{done}/{total}"),
)
result.attributions               # (n_rows, n_groups, n_outputs)
result.expected, result.predictions
result.exactness, result.additivity_max_abs_error
mean_abs, mean_signed = ex.summarize(result.attributions, output_index=0)
```

`predict_batch` returns `(rows,)` or `(rows, n_outputs)` finite predictions in
whatever output space the host chose (target units, a class probability, a
decision score); the binding never relabels it. `n_outputs` is inferred from
one probe row when omitted.

| Kind | `exactness` | Additive | `group_aggregation` |
| --- | --- | --- | --- |
| `exact_shap` (groups <= 16) | `exact` | yes | `group_game` |
| `permutation_shap` (antithetic; `std_error` = Monte-Carlo SE) | `sampled` | yes | `group_game` |
| `kernel_shap` | `sampled`, `exact` when the budget covers all 2^G - 2 coalitions | yes | `group_game` |
| `lime` | `surrogate` | no | `group_game` |
| `occlusion` | `perturbation` | no | `group_game` |
| `linear_shap(X, coefficients, reference=..., intercept=...)` | `exact` (interventional) | yes | `none`, or `group_game` with groups |
| `tree_shap(X, TreeEnsemble(...))` (path-dependent) | `exact` | yes | `none`, or `sum_of_feature_values` with groups |

`TreeEnsemble` is a generic flat layout (concatenated nodes, per-tree
offsets, local child indices, cover and leaf values, `decision_type` 0 for
`x <= threshold` or 1 for `x < threshold`, optional `base_offset` and
`tree_scale`). Converting scikit-learn, XGBoost or LightGBM models into it is
the host's job (the `nirs4all` SDK). `groups_from_ranges(n_features, ranges)`
turns half-open feature-index ranges (for example chemical bands already
mapped to indices) into a partition and reports the source range of each
group (`-1` for uncovered runs). `settings` uses the snake_case keys of the
`nirs4all.explanation.v1` envelope; hosts add axis, output and sample-index
metadata. `bench/explain/bench_explain.py` times every explainer over spectral
widths, group widths and background sizes.

## Relation to nirs4all conformal learning and robustness

The Python bindings are thin translation layers over the `libn4m` C ABI. They
expose native kernels, scikit-learn-compatible PLS estimators and the
single-estimator `finetune_estimator(...)` selection trace, but they do not own
the higher-level `nirs4all` statistical lifecycle.

Use `nirs4all.run(tuning=...)` when the workflow needs final winner projection,
workspace persistence, `.n4a` packaging, optional `nirs4all.calibrate()` /
`predict_calibrated()` semantics, `CalibratedRunResult`, or
`nirs4all.robustness()` / `RobustnessReport` artifacts. Those guarantees and
invalidation reasons are produced by `nirs4all`, not by `n4m` or `pls4all`.

Binding consumers must not reimplement tuning/conformal/robustness guarantees
from raw native predictions. A UI, Studio panel or alternate language binding
that wants to display conformal intervals or robustness summaries should consume
the public `nirs4all` artifact/schema surfaces instead of inferring guarantees
locally from `finetune_estimator(...)` or `PLSRegression.predict(...)`.

## Loading `libn4m`

The bundled wheel ships `libn4m` inside `pls4all/lib/`, found automatically. For
development against a local build the loader searches, in order:

1. `$PLS4ALL_LIB_PATH` — explicit path to `libn4m` for this package,
2. `$N4M_LIB_PATH` — shared `libn4m` override honoured by both `pls4all` and `n4m`,
3. `pls4all/lib/libn4m*` next to the installed package (wheel layout),
4. `<repo-root>/build/dev-release/cpp/src/libn4m*` (developer convenience),
5. the standard system search path (`LD_LIBRARY_PATH`, macOS rpath, Windows `PATH`).

## Building `libn4m` from source (developers)

```bash
cmake --preset dev-release
cmake --build --preset dev-release --parallel
```

This produces `build/dev-release/cpp/src/libn4m.so` (`.dylib` / `.dll` on
macOS / Windows), which the loader rules above pick up.

See <https://github.com/GBeurier/nirs4all-methods> for the full project.

### SNV scalar contract

`n4m.transform.scatter.SNV` and generated `n4m.roles.SNV` share strict private
scalar admission: flags are Python/NumPy booleans; `ddof` is a nonboolean
Integral with `operator.index` in `[0, INT32_MAX]`. NumPy integer scalars and
IntEnum are admitted; timedelta64, strings, fractional/whole floats and booleans
as ddof are refused before native parameter setters. Legacy SNV revalidates
current attributes before fit/transform, recreates changed handles, and applies
`set_params` atomically, including without sklearn. The unused unexported
`n4m._impl.native.snv` function is removed; use the public SNV classes.

When scaling nonempty data, `ddof < n_features` is required by libn4m before
output writes. Centering-only/no-op accepts any valid nonnegative ddof;
uncentered scaling uses SD about the row mean. Valid constant rows retain
their zero-SD fallback. `pls4all` accepts canonical Solver tokens including
`kernel_algorithm` and keeps `kernel` and existing aliases/defaults unchanged.
