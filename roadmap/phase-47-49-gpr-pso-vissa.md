# Phases 47-49 — GPR-on-PLS, PSO-PLS, VISSA-PLS

**Status**: shipped 2026-05-16 · tags `phase-47-gpr-on-pls`,
`phase-48-pso-pls`, `phase-49-vissa-pls` · release
`0.97.0+abi.1.14.0` (ABI minor bumped — three new additive symbols).

## Goal

Add three methods previously marked deferred in §22 (GPR-on-PLS) and
§14.4 (PSO-PLS, VISSA-PLS) of `Overview.md`. All three follow the
canonical method-add pattern (C++ core → C ABI wrapper → Python shim →
parity registry); no homemade NumPy mirrors. Public ABI stays
backwards-compatible (minor bump only).

## Phase 47 — GPR-on-PLS (Gaussian Process Regression on PLS scores)

Two-stage regression:

1. SIMPLS PLS produces rotation R (p × k) and centring vectors.
2. Gaussian Process on the training scores T = (X − x_mean) @ R with
   RBF kernel `K(t_i, t_j) = exp(-‖t_i − t_j‖² / (2·ℓ²))` and diagonal
   noise variance `σ²·I`. Solve `(K + σ²·I) α = y_c` via scalar
   Cholesky.

### GP head reuse (anticipating GPR-on-AOMPLS)

The GP solve lives in a standalone callable
`pls4all::core::fit_gp_on_scores(...)` that takes pre-computed scores T
and centred y. A future GPR-on-AOMPLS module reuses this primitive
unchanged — only the score-generating frontend (AOM-PLS instead of
SIMPLS) differs.

### Parity gate

External reference: `sklearn.gaussian_process.GaussianProcessRegressor`
with `kernel = RBF(length_scale=ℓ, length_scale_bounds="fixed") +
WhiteKernel(noise_level=σ², noise_level_bounds="fixed")`, `optimizer=
None`, `normalize_y=False`. Adapter `GprPlsSklearnReference` factors
out the PLS rotation-convention mismatch by re-running pls4all to
extract the same T_train, then fitting sklearn GP on that T.

Result: `rmse_rel = 2.3e-10` against the sklearn reference, well
inside the `1e-8` tolerance.

### `noise_level` convention

`noise_level` is the diagonal noise **variance** (σ², not σ),
matching sklearn `WhiteKernel(noise_level=...)`. The C header doc
and Python docstring both call this out explicitly (Codex review
flagged the ambiguity pre-ship).

### Single-target only

Phase 47 ships GPR-on-PLS for Y with `cols == 1`. Multi-target GP
needs either independent GPs per target or a block-structured kernel
Cholesky — different complexity envelope, deferred.

## Phase 48 — PSO-PLS (Binary Particle Swarm Optimisation)

Variable selection via binary PSO (Kennedy & Eberhart 1997). Each
particle holds a binary mask `x ∈ {0,1}^p`; velocity is continuous and
updated by inertia + cognitive + social terms; position is
re-binarised via a sigmoid stochastic threshold:

```
v_id ← w·v_id + c1·r1·(pbest_id − x_id) + c2·r2·(gbest_d − x_id)
v_id ← clip(v_id, -v_max, +v_max)
x_id ← 1 if rand() < σ(v_id) else 0
```

Fitness is `subset_rmse` (CV-RMSE on the selected subset using
`cross_validate_regression`). Default hyperparameters follow the
Clerc-Kennedy 2002 constriction analysis (`w=0.729, c1=c2=1.494,
v_max=4.0`).

### Repair step

After binarisation, `repair_mask` flips random zero bits to ones
until `popcount(mask) ≥ n_components`. This guarantees PLS on the
subset is well-defined. Repair bits are counted in
`inclusion_frequencies` so the diagnostic reflects the
actually-evaluated mask (Codex review caught a pre-fix undercount
where repair bits were missed).

### Parity gate

Paper-only — Kennedy & Eberhart (1997). `pyswarms.discrete.BinaryPSO`
uses a different RNG advance order so bit-equivalent parity is
impossible. The deterministic splitmix64 seed gives reproducible
output for a given `seed` parameter.

## Phase 49 — VISSA-PLS (Variable Iterative Space Shrinkage Approach)

Weighted Binary Matrix Sampling iterative shrinkage (Deng et al.
2014). Initial inclusion probability `w_j = 0.5`; per iteration:

1. Sample `n_submodels` binary masks via Bernoulli(w_j) per feature.
2. Skip masks with popcount `< n_components`.
3. Compute CV-RMSE per submodel.
4. Sort ascending, keep top `K = round(n_submodels * ratio_kept)`.
5. Update `w_j ← mean(mask[j])` across the kept submodels.
6. Clamp `w_j` to `[floor, 1-floor]` to preserve exploration.

After all iterations, `selected_indices = {j : w_j > threshold}`.

### Risks handled

- Degenerate iteration (all submodels below `n_components`): return
  `P4A_ERR_NUMERICAL_FAILURE` with a clear error pointing the user
  at `floor_probability` or `n_submodels`.
- Probability collapse: the `floor_probability` clamp (default 0.01)
  prevents `w_j` from collapsing to 0 or 1 prematurely.
- Sequential PRNG: single splitmix64 stream advanced per Bernoulli
  draw in fixed `(submodel, feature)` order — fully deterministic
  per seed.

### Parity gate

Paper-only — Deng, B., Yun, Y., Liang, Y. (2014) Anal. Chim. Acta
838:27-40. No widely installable port (authors' MATLAB code is in
supplementary materials only).

## Method-add pattern recap

For each method, files touched:

```
cpp/src/core/<name>.{hpp,cpp}              new
cpp/src/c_api/c_api_method_result.cpp      added wrapper
cpp/include/pls4all/p4a.h                  added decl
cpp/src/CMakeLists.txt                     added source
cpp/abi/expected_symbols_linux.txt         added symbol
bindings/python/src/pls4all/_methods.py    added ctypes proto + shim
bindings/python/src/pls4all/__init__.py    added re-export
benchmarks/parity_timing/registry.py       added driver + MethodSpec
```

All three methods are exported under the `p4a_*` symbol prefix, no
manual entry needed in `cpp/src/c_api/p4a_linux.map` (the wildcard
`p4a_*;` rule handles them).

## Codex review findings

1. **PSO `inc_count` undercount** (ship blocker, fixed): incrementing
   inclusion count BEFORE the repair step missed repair-added bits.
   Fixed by moving the increment to the post-repair sync loop.
2. **GPR `noise_level` convention ambiguity** (doc, fixed): clarified
   in both the C header and the Python docstring that `noise_level`
   is the diagonal noise variance (σ², not σ).

## Release

- Library version: `0.96.0` → `0.97.0` (additive: three new methods).
- ABI version: `1.13.0` → `1.14.0` (three new public symbols).
- Tags (one per phase): `phase-47-gpr-on-pls`, `phase-48-pso-pls`,
  `phase-49-vissa-pls`.

## Backlog

- **GPR-on-AOMPLS**: reuse `fit_gp_on_scores` after an AOM-PLS frontend.
- **PSO multi-restart / niching**: defer to Phase 48b if profiling
  shows the single-swarm hits local optima on real NIRS datasets.
- **VISSA full-paper defaults**: the Deng paper uses N_iter=50,
  Q=1000; the parity-gate cell uses 10/60 for runtime. Document the
  defaults for production callers.
