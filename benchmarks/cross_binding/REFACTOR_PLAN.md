# Cross-binding bench: registry-driven dispatch refactor

## Why

`bench_cpp.py` is the parity reference for the entire matrix but its dispatch
cascade in `_common.py:dispatch_pls4all_fit()` only knows about ~25 of the 71
canonical methods. Every other bench script (python_tier1, python_tier2,
r_tier1, r_tier2, matlab_tier1, matlab_tier2) has the same disease: a
hand-coded `switch`/`if` table that covered the catalog as it was 6 months
ago.

Today the registry (`benchmarks/parity_timing/registry.py`) is the single
source of truth — every method declares its canonical pls4all invocation as
`MethodSpec.pls4all_fn`. The only bench script that uses the registry is
`bench_registry_pls4all.py`, which gets 70/71 algos. The rest fall back to
a generic `fit_fn(ctx, cfg, X, y2d)` that misses the per-algo extra args
(`X_target`, `block_sizes`, `n_unique_per_block`, …) and crashes with
`TypeError` or `RuntimeError: no _methods.<algo>_fit available`.

Goal: every bench script routes through the registry (Python directly, R
and MATLAB via shared spec tables), so every pls4all column is populated
for every canonical method × cell × build.

## Target column layout in the rendered docs

| pls4all column | Bench script | Build sweep |
|----------------|-------------|-------------|
| `pls4all.cpp.native`         | `bench_pls4all_native.py` | dev-release |
| `pls4all.cpp.blas`           | `bench_pls4all_native.py` | blas-on |
| `pls4all.cpp.omp`            | `bench_pls4all_native.py` | omp-on |
| `pls4all.cpp.blas+omp`       | `bench_pls4all_native.py` | blas-omp |
| `pls4all.python`             | `bench_pls4all_sklearn.py` (idiomatic class API) | default build |
| `pls4all.R.direct`           | `bench_pls4all_r_direct.R` (registry-driven via R dispatcher) | default build |
| `pls4all.R.formula`          | `bench_pls4all_r_formula.R` (formula API where it exists, else `—` on purpose) | default build |
| `pls4all.matlab.procedural`  | `bench_pls4all_matlab.m` (procedural functions) | default build |
| `pls4all.matlab.classdef`    | `bench_pls4all_matlab_classdef.m` (where a class exists) | default build |

Total: 9 columns, plus the external-reference columns (sklearn, pls, ropls,
mixOmics, plsregress, ikpls, and the per-method registry references).

## Architectural changes

### 1. Shared per-binding spec table (registry-side)

Extend `MethodSpec` to carry per-binding hooks that are themselves source of
truth:

```python
@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str
    pls4all_fn: Callable[..., Any]        # already exists
    cell_params: dict

    # NEW — declarative binding handles. None means "not wired for this
    # binding yet"; the bench script records the row with FAIL reason
    # "binding handle not declared", *not* a generic TypeError.
    sklearn_estimator: Optional[Callable[..., Any]] = None       # () -> est
    r_dispatcher_algo: Optional[str] = None                       # for pls4all_method(algo, …)
    r_formula_fn: Optional[str] = None                            # name of R fn used with y ~ . formula
    matlab_proc_fn: Optional[str] = None                          # e.g. "sparse_simpls"
    matlab_class: Optional[str] = None                            # e.g. "PLSRegression"

    # NEW — extra args feeder. Returns kwargs to splat into pls4all_fn
    # AND a dict ready to send to R/MATLAB. Today `adapted_params` in
    # `bench_registry_common.py` already builds the Python form for us;
    # we extend it to emit a JSON-ready dict for the other languages.
    binding_extras_fn: Optional[Callable[..., dict]] = None
```

These hooks are optional; the only mandatory one is `pls4all_fn` (already
there). When a hook is `None` the bench script for that binding emits
`ok=False, reason="not bound"` so the rendered cell shows `—` honestly
instead of crashing.

### 2. `bench_pls4all_native.py` — replaces `bench_cpp.py` + `bench_python_tier1.py`

Both old scripts do the exact same thing (Python ctypes call into
`libn4m.so` with a Config + Context). The only difference between them and
`bench_registry_pls4all.py` is which dispatcher they used.

```
bench_pls4all_native.py
└── method = registry.load_method(algo)
    method.pls4all_fn(ctx, cfg, X, Y, **adapted_params, **benchmark_inputs)
```

Same env-var contract (`PLS4ALL_LIB_DIR` → the 4 libn4m builds). One script
× 4 build columns in the matrix = 4 cpp.* columns, all populated.

Drop `_common.dispatch_pls4all_fit()` entirely.

### 3. `bench_pls4all_sklearn.py` — replaces `bench_python_tier2.py`

```
sklearn_class = method.sklearn_estimator(method, n, p, nc)
sklearn_class.fit(X, y); sklearn_class.predict(X)
```

For algos where no sklearn class exists, the lambda is None → script emits
`reason="no sklearn estimator declared"` and the cell renders `—`.

### 4. `bench_pls4all_r_direct.R` — replaces `bench_r_tier1.R`

```r
res <- pls4all::pls4all_method(method$r_dispatcher_algo %||% algo,
                                xy$X, xy$y, nc,
                                params = method$r_params)
```

The R C-dispatcher (`r_p4a_dispatch_fit`) handles 45 cases today. Where the
registry's `r_dispatcher_algo` is None, the R script emits `reason="no R
direct handle"`.

`r_params` is a small per-method list in the registry (`r_params={"alpha":
0.5}` for ECR, `r_params={"gamma": 0.5}` for CPPLS, etc.). The Python side
serialises it to JSON in env var `BENCH_R_PARAMS` and R parses it on entry
via `jsonlite::fromJSON`.

### 5. `bench_pls4all_r_formula.R` — replaces `bench_r_tier2.R`

```r
df <- as.data.frame(xy$X); df$y <- xy$y
fn_name <- method$r_formula_fn
fn <- get(fn_name, envir = asNamespace("pls4all"))
fit <- do.call(fn, c(list(y ~ ., df), method$r_formula_extras))
predict(fit, newdata = df)
```

Formula coverage stays smaller (≈15 methods) but the script no longer needs
hand-maintenance — adding a new formula wrapper in the R package + setting
`r_formula_fn` in the registry is enough.

### 6. `bench_pls4all_matlab.m` — replaces `bench_matlab_tier1.m`

```matlab
fn_name = getenv("BENCH_MATLAB_FN");          % set from registry
fn = str2func(sprintf("pls4all.%s", fn_name));
res = fn(X, y, nc, varargin{:});
preds = res.predictions(:);
```

The orchestrator builds `BENCH_MATLAB_FN` and `BENCH_MATLAB_EXTRAS` from
`MethodSpec.matlab_proc_fn` and `binding_extras_fn`. MATLAB extras are
passed as a small JSON env var, parsed via `jsondecode`. 63/71 methods
already have a `+pls4all/<algo>.m`; the registry's `matlab_proc_fn` field
captures the exact name (handles aliases like `pls → sparse_simpls`).

### 7. `bench_pls4all_matlab_classdef.m` — replaces `bench_matlab_tier2.m`

Same logic, but uses `MethodSpec.matlab_class` to instantiate
`pls4all.<Class>(n_components, ...)` and call `.fit(X,y); .predict(X)`.
Coverage is ~25 classes; the rest render `—`.

### 8. Orchestrator BACKENDS update

```python
BACKENDS = [
    ("native",            "bench_pls4all_native.py",            "Python",  "registry",  "n4m_core"),
    ("sklearn_idiomatic", "bench_pls4all_sklearn.py",           "Python",  "sklearn",   "pls4all_binding"),
    ("r_direct",          "bench_pls4all_r_direct.R",           "R",       "direct",    "pls4all_binding"),
    ("r_formula",         "bench_pls4all_r_formula.R",          "R",       "formula",   "pls4all_binding"),
    ("matlab_procedural", "bench_pls4all_matlab.m",             "MATLAB",  "procedural","pls4all_binding"),
    ("matlab_classdef",   "bench_pls4all_matlab_classdef.m",    "MATLAB",  "classdef",  "pls4all_binding"),
    # externals unchanged …
]
```

The `cpp` / `python_tier1` / `python_tier2` / `r_tier1` / `r_tier2` /
`matlab_tier1` / `matlab_tier2` rows are retired. The corresponding entries
in `full_matrix.csv` need a one-shot migration pass (rename + dedupe).

### 9. Doc renderer (`render_docs.py`) column mapping

Update `BACKEND_DISPLAY` so the new backend names map to the column titles
listed in the target layout (cpp.native, cpp.blas, …, python, R.direct,
R.formula, matlab.procedural, matlab.classdef). The libn4m-build × backend
matrix used by the renderer already understands per-build expansion; we
just point the new `native` backend at it.

## File-by-file change list

```
benchmarks/parity_timing/registry.py
    + MethodSpec fields: sklearn_estimator, r_dispatcher_algo,
      r_formula_fn, matlab_proc_fn, matlab_class, binding_extras_fn
    + populate per-METHOD (71 entries) with the right handles; today
      the per-method blocks already capture this info in their wrappers,
      we just need to pull it up to the spec level
    + helper `adapted_params_for_binding(method, lang, n, p, nc, seed)`
      that returns ready-to-serialise Python / R / MATLAB dicts

benchmarks/cross_binding/orchestrator.py
    + BACKENDS list refactored (6 entries instead of 14)
    + env var plumbing: BENCH_R_PARAMS, BENCH_MATLAB_FN,
      BENCH_MATLAB_EXTRAS, BENCH_SKLEARN_CLASS

benchmarks/cross_binding/scripts/
    + bench_pls4all_native.py             (new, replaces cpp + python_tier1)
    + bench_pls4all_sklearn.py            (rewrite of python_tier2)
    + bench_pls4all_r_direct.R            (rewrite of r_tier1)
    + bench_pls4all_r_formula.R           (rewrite of r_tier2)
    + bench_pls4all_matlab.m              (rewrite of matlab_tier1)
    + bench_pls4all_matlab_classdef.m     (rewrite of matlab_tier2)
    + bench_registry_common.py            (extend with adapted_params_for_binding)
    − bench_cpp.py                        (delete)
    − bench_python_tier1.py               (delete)
    − bench_python_tier2.py               (delete)
    − bench_r_tier1.R                     (delete)
    − bench_r_tier2.R                     (delete)
    − bench_matlab_tier1.m                (delete)
    − bench_matlab_tier2.m                (delete)
    − _common.dispatch_pls4all_fit        (delete from _common.py)

benchmarks/cross_binding/results/full_matrix.csv
    + one-shot rename pass: cpp → native (with libp4a_build kept),
      python_tier1 → native (dedupe against cpp), python_tier2 → sklearn_idiomatic, etc.

docs/_extras/build_landing.py
    + BACKEND_DISPLAY entries for new names
    + remove obsolete entries

benchmarks/cross_binding/render_docs.py
    + same backend renaming, column ordering update

docs/benchmarks/cross_binding{,_threads}.md
    + regenerated, includes the new column layout
```

## Validation strategy

1. Unit-ish smoke from CLI before the long sweep:
   ```bash
   python benchmarks/cross_binding/orchestrator.py \
       --algorithms pls cppls di_pls mb_pls so_pls pls_lda gpr_pls \
       --sizes 100x50 --threads 1 --n-runs 1 \
       --libn4m-build blas-omp --reference-backends none \
       --out-csv /tmp/smoke.csv
   ```
   Expect: every pls4all backend OK on every of the 7 algos; no `TypeError`
   / `no _methods.X_fit available` rows.

2. Per-binding diff against the existing registry column to catch
   regressions (sklearn and registry should be numerically identical for
   the methods they share — same kernel, same params).

3. Full re-run at 100x50 / 500x50 / 100x500 / 500x500 × t=1,3,10 ×
   all-cpu builds, with `--resume-existing` left ON so the orchestrator
   reuses the cells where the binding handle is genuinely absent (those
   FAIL rows stay stable).

4. Re-render the docs, confirm:
   - Every algo has the 4 cpp.* columns populated (or all four `—` if
     the method genuinely has no C-side fit, which today is empty set).
   - sklearn_idiomatic / R.direct / matlab.procedural columns are
     populated for the methods declaring those handles, and `—` (with the
     "handle not declared" reason traceable in the FAIL string) for
     the rest. No more `TypeError` failures.

## Risks / open questions

- Some sklearn classes accept a different parameter set than the raw
  `_fit` entry point. The registry's binding handle carries the estimator
  factory + kwargs.

- The R-formula and MATLAB-classdef paths will deliberately have wide `—`
  regions because the corresponding APIs in the pls4all bindings haven't
  been generalised to every algo. That is by design and visible in the
  dashboard.

## Codex review fixes folded in (2026-05-18)

- **Stale params**: the legacy `_common.dispatch_pls4all_fit` cascade uses
  different parameter values than `MethodSpec.pls4all_fn` (e.g.
  `ridge_lambda=1.0` vs registry `0.5`; `robust max_irls_iter=20` vs `5`;
  `weighted_pls sample_weights = ones(n)` vs `abs(y)+0.5`). The existing
  `full_matrix.csv` rows for cpp/python_tier1 are **not just incomplete,
  they encode wrong-params timing**. ⇒ **No CSV migration**; full rerun
  after refactor.
- **MethodSpec shape**: nest the binding handles in a typed
  `CrossBindingSpec` dataclass instead of six loose `Optional` fields on
  `MethodSpec`. `MethodSpec.cross_binding: CrossBindingSpec` keeps the
  parity-gate surface clean.
- **Per-backend build sweep**: today every non-external backend sweeps all
  libn4m builds. Only `native` should expand to 4 builds; sklearn / R /
  MATLAB backends force `blas-omp`. Add explicit `build_sweep: bool` on
  the backend tuple in `BACKENDS` and gate the build loop on it.
- **Array extras serialisation**: R/MATLAB cannot regenerate NumPy PCG64
  arrays exactly. Pass scalar/list extras via env JSON, but write
  `X_target`, `sample_weights`, `y_labels`, `group_assignment` to per-run
  `.npy` sidecars under `data/.bench_extras/` and pass paths in env.
- **Reference rename in `compute_parity`**: today the parity ref hunt
  hard-codes `cpp` / `python_tier1`. Rename both candidates to `native`
  and prefer the same `blas-omp` build.
- **`prediction_key`**: `bench_registry_pls4all.py` hard-codes
  `"predictions"`. The new `bench_pls4all_native.py` must use
  `method.prediction_key` so selector methods that emit
  `selected_indices` parity correctly.
- **Lazy R-package probing in `registry.py`**: today the import does
  Rscript subprocess calls to probe installed packages. Most cells don't
  need that. Wrap in a lazy proxy so non-R scripts don't pay the startup
  cost.
- **Docs surface**: also touch `docs/_extras/build_methods.py`,
  `docs/benchmarks/methodology.md`, and column labels in
  `render_docs.py` and `build_landing.py`.

## Execution sequence (revised)

1. **Registry side**: add `CrossBindingSpec` dataclass; add
   `cross_binding: CrossBindingSpec` field to `MethodSpec`; populate the
   71 entries (this is the bulk; can be split per category — regression /
   selection / diagnostics).
2. **Add `adapted_extras_for_binding(method, lang, n, p, nc, seed)`** to
   `bench_registry_common.py`: returns `(env_dict, sidecar_paths)`.
3. **Add new bench scripts side-by-side**, no orchestrator change yet:
   `bench_pls4all_native.py`, `bench_pls4all_sklearn.py`,
   `bench_pls4all_r_direct.R`, `bench_pls4all_r_formula.R`,
   `bench_pls4all_matlab.m`, `bench_pls4all_matlab_classdef.m`.
4. **Smoke each new script** at `--algorithms pls cppls di_pls so_pls
   weighted_pls bve_select pls_diagnostic_dmodx --sizes 100x50
   --threads 1 --n-runs 1 --libn4m-build blas-omp` against `/tmp/smoke.csv`.
   Confirm OK rows for the algos declaring the handle, "not bound" rows
   for the others.
5. **Orchestrator switch**: add `build_sweep` to backend tuple, swap to
   new BACKENDS list, rename `compute_parity` reference candidates,
   delete old script entries.
6. **Renderer + landing**: update `BACKEND_DISPLAY` / column order /
   labels in `render_docs.py` and `build_landing.py`; refresh
   `methodology.md`.
7. **Delete old scripts** (`bench_cpp.py`, `bench_python_tier1.py`,
   `bench_python_tier2.py`, `bench_r_tier1.R`, `bench_r_tier2.R`,
   `bench_matlab_tier1.m`, `bench_matlab_tier2.m`) and drop
   `_common.dispatch_pls4all_fit`.
8. **Wipe stale CSV**: archive `results/full_matrix.csv` to
   `results/full_matrix.legacy-pre-registry.csv`, start fresh.
9. **Clean rerun** at `100x50 / 500x50 / 100x500 / 500x500 × t=1,3,10 ×
   all-cpu builds`.
10. **Re-render docs + bench-data.json**, commit, push (release/m2-cran +
    cherry-pick to main), trigger Pages.

Estimated effort: ~6–8 h focused work + ~2–3 h bench wall-clock. Step 1
(registry population) is the long pole; the bench scripts themselves are
~60 lines each.
