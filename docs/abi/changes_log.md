# ABI — Changes Log

## 2026-06-03 — ABI 1.11.0: direct (closed-form) Ridge regression

One additive public symbol (ABI MINOR bump 1.10.0 → 1.11.0), backward-compatible
(no signature/layout change, nothing removed):

- `n4m_ridge_fit(n4m_context_t*, const n4m_config_t*, const n4m_matrix_view_t* X,
  const n4m_matrix_view_t* Y, const double* lambdas, int64_t n_lambdas,
  n4m_method_result_t** out_result)`

This is a **genuine closed-form** multi-output Ridge — `beta = (Xc'Xc + lambda I)^-1
Xc'Yc` on column-centered X/Y with `intercept = y_mean - x_mean.beta` (the penalty is
not applied to the intercept, for `sklearn.linear_model.Ridge` parity). It is distinct
from the pre-existing `n4m_ridge_pls_fit` (ridge-augmented SIMPLS, rank-truncated by
`n_components`). The solver is chosen automatically by shape (PRIMAL augmented-QR for
p ≤ n, DUAL Gram-on-samples for p > n; identical coefficients up to round-off).

Declared with `N4M_API` in `cpp/include/n4m/pls.h` (after `n4m_continuum_regression_fit`),
implemented in `cpp/src/c_api/c_api_method_result.cpp` over the new core kernel
`cpp/src/core/ridge.cpp`. Result keys: `coefficients` (p×q), `intercept` (1×q),
`x_mean`, `x_scale` (1×p), `y_mean` (1×q), `predictions` (n×q), scalar `rmse`,
scalar `lambda`.

Snapshots regenerated for all three platforms via
`scripts/regen_abi_snapshots.sh --derive` (linux from the lib; macos/windows derived
= linux minus the `N4M_1` version node). `n4m_ridge_fit` is present in
`cpp/abi/expected_symbols_{linux,macos,windows}.txt`. Header
`N4M_ABI_VERSION_MINOR` and `bindings/python/src/n4m/_ffi.py:ABI_VERSION_MINOR`
both bumped 10 → 11; `bump_version.sh --check` is green (project version unchanged
at 0.98.0).

## 2026-06-03 — macOS/Windows snapshot correction + cross-platform gate enforced

No ABI surface change (still ABI 1.10.0). This is an **audit-trail and CI
correction**: the 2026-05-30 entry below claimed "Snapshots regenerated for all
three platforms", but `expected_symbols_{macos,windows}.txt` were in fact a stale,
truncated copy of an old Linux `nm -D` dump — 500 lines, carrying the Linux-only
`@@N4M_1` version tag (which macOS `nm -gU` / Windows `dumpbin` never emit), and
missing ~171 symbols (the whole selection / method-result / aom / config family).
They were also **not diffed by CI** on macOS/Windows (only Linux was fail-closed).

Corrected here:

- `expected_symbols_{macos,windows}.txt` regenerated to the real 671-symbol set
  — identical to the Linux `n4m_*` names minus the Linux-only `N4M_1` version
  node (the only legitimate cross-platform difference).
- `.github/workflows/abi-check.yml` now diffs the committed snapshot **fail-closed
  on all three platforms** (macOS `diff`, Windows `Compare-Object` set comparison),
  with `LC_ALL=C`-pinned sorts so ordering is reproducible.
- Added a SONAME / RPATH-RUNPATH linkage gate to the Linux job (asserts
  `SONAME == libn4m.so.1` and no baked-in absolute search path).
- Added `scripts/regen_abi_snapshots.sh` — the single canonical regenerator
  (`--check` for CI/pre-commit, `--derive` to produce the macOS/Windows files
  from the Linux snapshot when only a Linux box is available).

## 2026-05-18 — Linux export baseline for ABI 1.16.0

`build/dev-release/cpp/src/libn4m.so.1.16.0` exports 27 additional
`n4m_*` symbols compared with the previous Linux baseline. Each added symbol is
declared with `N4M_API` in the public header `cpp/include/pls4all/p4a.h`, so the
Linux ABI gate now treats them as intentional public additions:

- `n4m_method_result_get_int64_vector`
- `n4m_mb_pls_fit`, `n4m_lw_pls_fit`, `n4m_pls_lda_fit`,
  `n4m_pls_logistic_fit`, `n4m_aom_preprocess_fit`
- `n4m_variable_select_rank`, `n4m_interval_select`,
  `n4m_stability_select`, `n4m_uve_select`, `n4m_spa_select`,
  `n4m_cars_select`, `n4m_random_frog_select`, `n4m_scars_select`,
  `n4m_ga_select`, `n4m_shaving_select`, `n4m_bve_select`,
  `n4m_t2_select`, `n4m_wvc_select`, `n4m_wvc_threshold_select`,
  `n4m_emcuve_select`, `n4m_randomization_select`, `n4m_bipls_select`,
  `n4m_sipls_select`, `n4m_rep_select`, `n4m_ipw_select`, `n4m_st_select`

## 2026-05-30 — ABI 1.10.0: additive RNG-kind config selector

Two additive public symbols (ABI MINOR bump 1.9.0 → 1.10.0), backward-compatible
(no signature/layout change, nothing removed):

- `n4m_config_set_rng_kind(n4m_config_t*, n4m_rng_kind_t)`
- `n4m_config_get_rng_kind(const n4m_config_t*, n4m_rng_kind_t*)`

New enum `n4m_rng_kind_t` { `N4M_RNG_SPLITMIX64`=0 (default), `N4M_RNG_PCG64`=1,
`N4M_RNG_MT_R`=2, `N4M_RNG_NUMPY_MT`=3 } selects the RNG engine a stochastic
method draws from, so its output can match an external reference library's exact
RNG (numpy default_rng / base R / numpy RandomState) for parity. Default
SPLITMIX64 reproduces n4m's historical streams bit-for-bit — leaving it unset
changes nothing. Snapshots regenerated for all three platforms
(expected_symbols_{linux,macos,windows}.txt). Engines verified bit-exact:
docs/dev/RNG_TIER0_INVENTORY.md, cpp/tests/test_rng_engine.cpp.
