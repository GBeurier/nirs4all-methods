# Selector Cross-Binding Parity — Investigation

Branch: `release/m2-cran`. All file paths absolute; line numbers reflect the
files at the time of investigation.

## 1. The C kernel's notion of a validation plan

The C kernel does **not** synthesise a validation plan internally. Every
selector that needs cross-validated scoring takes a fully formed
`p4a_validation_plan_t*` as an input argument. The plan structure
(`/home/delete/nirs4all/pls4all/cpp/include/pls4all/p4a.h:249,666-703`) holds
`n_samples` plus a list of `(train_indices, test_indices)` folds. Folds are
populated exclusively from the binding side via
`p4a_validation_plan_set_n_samples` and `p4a_validation_plan_add_fold`
(`cpp/src/c_api/c_api_validation_plan.cpp:52-84`).

`cross_validate_regression` (`cpp/src/core/cross_validation.cpp:209-248`)
iterates `plan.folds` verbatim; nothing is shuffled or seeded by libp4a.
This is also visible in the per-selector call: e.g. `bipls_selection.cpp:211-231`
calls `::pls4all::core::cross_validate_regression(ctx, cfg, view, Y, plan, cv)`
and uses `cv.metrics.rmse` to score each candidate interval set. Selectors
that maintain their own internal RNG (GA, PSO, random_frog, scars, irf, iriv,
randomization) seed it with the explicit `uint64_t seed` argument from their
public signature (`cpp/include/pls4all/p4a.h:1629-1668,1688-1702,1726-1738`);
that seed is propagated identically by every binding, so the RNG is portable.

**Conclusion:** the only source of binding-dependent non-determinism in the
selector family is the `p4a_validation_plan_t` the binding hands in.

## 2. Python registry plan construction

`benchmarks/parity_timing/registry.py:1508-1524` defines `_build_default_plan`:

```python
def _build_default_plan(n_samples, n_folds=3, seed=0):
    plan = pls4all.ValidationPlan(); plan.n_samples = n_samples
    rng = np.random.default_rng(seed); idx = np.arange(n_samples); rng.shuffle(idx)
    fold_size = max(1, n_samples // n_folds)
    for f in range(n_folds):
        start = f*fold_size
        end = (f+1)*fold_size if f < n_folds-1 else n_samples
        test = idx[start:end]; train = np.setdiff1d(idx, test, assume_unique=False)
        plan.add_fold([int(x) for x in train], [int(x) for x in test])
```

That is **3 PCG64-shuffled folds with `seed=0`**. Every plan-taking selector
adapter (`_interval_select_pls4all` line 1673, `_bipls_select_pls4all` 1692,
`_uve_select_pls4all` 1746, `_cars_select_pls4all` 1777, `_scars_select_pls4all`
1820, `_ga_select_pls4all` 1841, `_shaving_select_pls4all` 1865,
`_bve_select_pls4all` 1885, `_irf_select_pls4all` 1923, `_t2_select_pls4all`
1960, `_emcuve_select_pls4all` 2006, `_rep_select_pls4all` 2047,
`_ipw_select_pls4all` 2067, `_st_select_pls4all` 2088, plus
`_vissa_select_pls4all` 1086, `_pso_select_pls4all` 1110,
`_stability_select_pls4all` 1730, `_random_frog_select_pls4all` 1796) calls
`_build_default_plan(X.shape[0])` with default `n_folds=3, seed=0`. The only
exception is `_iriv_select_pls4all` (line 1913) which passes
`n_folds=int(fold)` from `cell_params["fold"]` (default `3`).

`bench_cpp.py:63` and `bench_python_tier1.py:43` both reach selectors through
`method.pls4all_fn`, so the `cpp`, `python_tier1`, and `registry_pls4all`
columns of the matrix share that exact 3-shuffled plan.

## 3. R dispatcher

`bindings/r/pls4all/src/r_dispatch.c:329-358` defines `make_default_plan`:

```c
static p4a_validation_plan_t* make_default_plan(int n, int requested_folds) {
    int k = requested_folds; if (k > n/2) k = n/2; if (k < 2) k = 2;
    int fold_size = (n + k - 1) / k;
    for (int f = 0; f < k; ++f) {
        int t0 = f*fold_size; int t1 = t0+fold_size; if (t1>n) t1=n;
        for (int i = 0; i < n; ++i) {
            if (i >= t0 && i < t1) test[te++] = i;
            else                    train[ti++] = i;
        }
        p4a_validation_plan_add_fold(plan, train, ti, test, te);
    }
}
```

This is **5 contiguous (un-shuffled) folds**: `DISPATCH_PLAN_SELECT` at lines
892-1039 always passes `make_default_plan(n, 5)` (e.g. cars 894, interval 907,
stability 913, uve 918, random_frog 924, scars 934, ga 942, pso 952, vissa 964,
shaving 975, bve 982, emcuve 988, bipls 996, sipls 1002, rep 1008, ipw 1015,
iriv 1023, irf 1030). `t2_select` and `st_select` (the two with non-uniform
parameter shapes) get the same `make_default_plan(n, 5)` at lines 1047 and
1062. The dispatcher never reads `params$n_folds`, `params$plan_seed`, or any
per-fold index list.

## 4. MATLAB MEX dispatcher

`bindings/matlab/mex/p4a_method_fit_mex.c:239-272` is the same algorithm as
the R one — **5 contiguous folds**, no shuffle:

```c
for (int f = 0; f < n_folds; ++f) {
    int t_start = f * fold_size; int t_end = t_start + fold_size; ...
    for (int i = 0; i < n; ++i) {
        if (i >= t_start && i < t_end) test[te++] = i;
        else                            train[ti++] = i;
    }
    p4a_validation_plan_add_fold(plan, train, ti, test, te);
}
```

Every plan-taking branch invokes `make_default_plan(n, 5)` (lines 872, 885,
897, 910, 927, 942, 959, 978, 998, 1017, 1030, 1048, 1084, 1107, 1120, 1134,
1149, 1167, 1180, 1198 — 21 sites). MATLAB tier-2 (`bench_matlab_tier2.m:24-42`)
deliberately rejects every selector (`SUPPORTED` list at line 26 covers only
21 regressor algos), so the only MATLAB path in scope is tier-1 → the MEX
dispatcher.

## 5. Python sklearn classes

`bindings/python/src/pls4all/sklearn/_selection.py:37-54` defines
`_default_plan` which is **bit-identical to the registry's**: same
`np.random.default_rng(seed).shuffle(idx)`, same `fold_size = max(1, n//k)`,
same train/test boundary logic. Each estimator wraps it through
`_open_plan(n, n_folds, seed)` (line 215).

Per-class defaults (lines 245-851):

* `StabilitySelector`, `CARSSelector`, `RandomFrogSelector`, `SCARSSelector`,
  `GASelector`, `PSOSelector`, `VISSASelector`, `IPWSelector` → `n_folds=3,
  seed=0` (overridable in ctor).
* `UVESelector` → `n_folds=3`, plan seed `= noise_seed` (line 286).
* `ShavingSelector`, `BVESelector`, `REPSelector`, `IntervalSelector`,
  `BiPLSSelector`, `SiPLSSelector` → `n_folds=3, plan_seed=0` hard-wired.
* `STSelector` (line 718) and `T2Selector` (line 847) → `n_folds=3, plan_seed=0`
  hard-wired.

`bench_python_tier2.py` (line 182-203) just instantiates the class with the
registry's `cell_params` (no `n_folds`/`seed` override). So the **sklearn
column already matches the registry plan when both default to
`(n_folds=3, seed=0)`**. The diff against the registry should be **0** for
those selectors; the divergence is between **registry+sklearn** (3 shuffled)
vs **r_tier1+matlab_tier1** (5 contiguous).

## 6. C ABI supports user-supplied plans

Yes — by design. The plan handle (`p4a_validation_plan_t`) is opaque but the
binding sees `p4a_validation_plan_create`,
`p4a_validation_plan_set_n_samples(plan, int64_t)`,
`p4a_validation_plan_add_fold(plan, const int64_t* train, int64_t n_train,
const int64_t* test, int64_t n_test)`, and `*_destroy`
(`cpp/include/pls4all/p4a.h:677-703`; implementation
`cpp/src/c_api/c_api_validation_plan.cpp:29-95`). The R dispatcher and the
MATLAB MEX both already link this API; they currently choose to build the
plan locally rather than read it from `params`.

## Conclusion — feasibility

**Bit-identical parity is feasible for every selector**, including the
stochastic ones (GA, PSO, VISSA, random_frog, scars, irf, randomization),
because:

1. The C kernel's only externalised non-determinism for selection is the
   `ValidationPlan*` passed in. The kernel never re-shuffles, never reseeds.
2. Algorithmic RNG (GA / PSO / random_frog / scars / irf / vissa /
   randomization) is seeded by an explicit `uint64_t seed` argument that all
   bindings already plumb through `cell_params["seed"]` →
   `BENCH_R_PARAMS_JSON` → `get_u64(params, "seed", 0)` in R/MATLAB. No
   binding uses the host RNG. The Python `_*_select_pls4all` adapters in
   `registry.py` pass `seed=int(seed)` (lines 1104, 1129, 1814, 1835, 1859,
   1917, 1939, 2022, 2043, 2082).
3. The fold partition is fully expressible via `add_fold(train, test)`. Two
   options are open:

   **Option A (preferred — minimum delta):** Change all bindings to construct
   the **same** plan the registry uses (3 PCG64-shuffled folds, seed 0). The
   Python `_default_plan` is already deterministic and reproducible in C if
   we re-implement the shuffle inline. PCG64 isn't trivially portable, so a
   safer canonical is a **shuffle-less 3-fold contiguous plan** built
   uniformly by all four dispatchers. Convert the registry adapter to
   `_build_default_plan(n, n_folds=3, seed=None)` with `seed=None` skipping
   the shuffle; convert R/MATLAB `make_default_plan(n, requested_folds)` to
   default `requested_folds=3` and keep their contiguous layout; convert the
   sklearn `_default_plan` to skip the shuffle when `seed is None` and
   default `n_folds=3, seed=None`. This is the cheapest invariant.

   **Option B (more work, more flexible):** Plumb the plan through
   `BENCH_R_PARAMS_JSON` as an optional pair `("plan_folds": k,
   "plan_test_indices": [[…],[…],[…]])`. R/MATLAB dispatchers would call
   `p4a_validation_plan_add_fold` per entry instead of `make_default_plan`.
   This preserves the registry's current shuffle.

   File-count estimate for **Option A**:
   * `benchmarks/parity_timing/registry.py` — 1 function, ~5 lines (drop the
     `rng.shuffle` when `seed is None` and switch the call sites' default).
   * `bindings/python/src/pls4all/sklearn/_selection.py` — 1 helper +
     ~20 constructor signatures to change defaults. Tightest churn.
   * `bindings/r/pls4all/src/r_dispatch.c` — change `make_default_plan(n, 5)`
     → `make_default_plan(n, 3)` at all 21 call sites (lines 894, 907, 913,
     918, 924, 934, 942, 952, 964, 975, 982, 988, 996, 1002, 1008, 1015,
     1023, 1030, 1047, 1062, plus the now-trivial `requested_folds`
     parameter). 1-line per site, plus 1 line in the helper.
   * `bindings/matlab/mex/p4a_method_fit_mex.c` — same 21 call sites
     (`make_default_plan(n, 5)` → `make_default_plan(n, 3)`).

   File-count estimate for **Option B**: ~50 lines in `registry.py`
   (serialise `plan_test_indices`), ~30 lines in `r_dispatch.c`
   (parse `params$plan_test_indices`, call `p4a_validation_plan_add_fold` per
   fold), ~30 lines in MEX, and ~10 lines in sklearn classes.

**Residual non-parity floor.** After unifying the plan, the algorithmic RNG
of GA/PSO/VISSA/random_frog/scars/irf/randomization is bit-portable because
it's all integer LCG state (`ga_selection.cpp:267-637` — `random_bounded`
uses a `uint64_t state = seed` mixed-state walker; analogous code in
`pso_selection.cpp`, `random_frog_selection.cpp`, `scars_selection.cpp`,
`irf_selection.cpp`, `randomization_selection.cpp`). BLAS reductions in the
underlying PLS regression are the only remaining floating-point variability
across builds, but the non-selector parity rows already prove that gives 0.0
max-abs-diff for these dataset sizes — so selector masks (which are integer
0/1 decisions made by comparing CV-RMSEs that match to BLAS precision) will
also collapse to 0.0.

**No selector is fundamentally non-portable.** Even `random_frog_select` and
`pso_select` are deterministic given identical `(plan, seed, params)`: their
RNG is a pure integer mixer with no `std::random_device` / OS entropy.
Verified by reading `random_frog_selection.cpp` and `pso_selection.cpp` —
neither pulls from `/dev/urandom` or `std::random_device`, and the only
`<random>`-style header inclusion is in `irf_selection.cpp` (still seeded
exclusively from the user `seed` argument).

The simplest path forward is Option A. The riskiest selectors are those
whose final RMSE comparison is very close across candidate sets (e.g.
`interval_select` with small `interval_step`, `bipls_select` near the
minimum-intervals boundary): under a different plan they pick different
"winners", which flips bits of the mask. Once the plan is unified those
ties resolve identically.
