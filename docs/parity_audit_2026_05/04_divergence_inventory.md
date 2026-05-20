# Divergence inventory at strict 1e-3 threshold

Date: 2026-05-20

Scope: local workspace state after the Sphinx/dashboard rebuild. This report
uses a strict product threshold of `reference_parity_rmse_rel <= 1e-3`.
That threshold is intentionally stricter than many current
`MethodSpec.rmse_rel_tol` values.

## Sources audited

| Source | Role | Count |
|---|---:|---:|
| `benchmarks/cross_binding/results/full_matrix.csv` | canonical raw matrix | 27,456 rows |
| `dashboard_refresh_*.csv` | dashboard deltas | 10,506 rows |
| builder-selected CSV rows | `build_landing.py` dedupe/prefer rules | 11,688 rows |
| `docs/_static/bench-data.json` | rendered dashboard payload | 584 rows, 11,785 cells |
| `docs/_build/html/_static/bench-data.json` | local Sphinx payload | same SHA as static payload |

Payload SHA:

```text
380c65b543f8d86cfc0d5496967c788f44ea0d9ba67fb604683ffdab670e6ad5  docs/_static/bench-data.json
380c65b543f8d86cfc0d5496967c788f44ea0d9ba67fb604683ffdab670e6ad5  docs/_build/html/_static/bench-data.json
```

Subagents used:

| Agent | Focus | Integrated result |
|---|---|---|
| Heisenberg | raw data, `.npy` oracle recovery, BLAS/OMP isolation | yes |
| Singer | dashboard/orchestrator/comparator semantics | yes |
| Ampere | method-family cause audit | yes |

## Executive verdict

This is not a binding-parity BLAS+OMP drift problem in the current data.
`binding_parity_max_diff` is clean:

| Universe | finite binding comparisons | `>1e-6` | `>1e-3` | max |
|---|---:|---:|---:|---:|
| raw `full_matrix.csv` | 9,883 | 0 | 0 | `9.404368839e-08` |
| dashboard-selected CSV rows | 3,639 | 0 | 0 | `9.404368839e-08` |

The real ship blocker is the reference gate. Many rows are far above
`1e-3`, but still marked as passing because current method tolerances are
often qualitative (`0.5`, `1`, `2`, `5`, `10`). In other words, the current
dashboard `exact` label often means "within MethodSpec tolerance", not
"numerically close".

| Universe | finite reference comparisons | `>1e-3` | `>0.1` | `>1` | max | currently marked pass | currently marked fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw recorded `full_matrix.csv` | 6,284 | 3,550 | 2,449 | 994 | `5.721719` | 3,269 | 281 |
| raw + recoverable `.npy` oracles | 6,566 | 3,718 | 2,617 | 994 | `5.721719` | 3,413 | 305 |
| dashboard-selected CSV rows | 2,619 | 1,403 | 1,015 | 415 | `10.295630` | 1,228 | 175 |
| rendered dashboard payload | 3,232 numeric cells | 428 | 309 | 126 | `3.559026` | n/a | n/a |

Rendered dashboard cells above `1e-3` are all `divergence_basis=reference`.
There are zero rendered binding numeric cells above `1e-3`.

There is also a concrete payload inconsistency: four
`shaving_select 100x2500 t1` R cells render `reference_parity=divergent`
while the displayed numeric divergence is `0` with
`divergence_basis=binding`. That is a dashboard merge/staleness bug, not a
valid "binding exact but reference divergent means zero divergence" result.

## Bug display vs real bugs

### Display and reporting bugs

1. The dashboard legend is wrong for the reference gate. It says
   `exact <= 1e-6`, `drift <= 1e-3`, `divergent > 1e-3`, but
   `reference_parity_code()` currently uses `MethodSpec.rmse_rel_tol`.
   A value like `rmse_rel=0.5` can be displayed as `exact` when the method
   tolerance is `5`.

2. Divergence mode mixes two different metrics:
   `binding_parity_max_diff` for pls4all bindings, and
   `reference_parity_rmse_rel` for C++/core/external rows. The UI exposes
   this as one "divergence" surface, which makes reference drift look like
   binding drift.

3. The dashboard over-emphasizes `blas+omp` because non-C++ columns are
   deduped by preferring the `blas-omp` row, and some native/BLAS/OMP rows
   are `no_compare` even when `.npy` oracles exist.

4. 168 strict divergences are hidden behind `reference oracle missing`.
   Those rows are recoverable from existing prediction/oracle `.npy` files.

5. Four `shaving_select 100x2500 t1` R cells carry a divergent reference
   verdict from `dashboard_refresh_partial_current.csv` (`rmse_rel=10.2956`,
   `prediction_seed=1234567908`), but the rendered divergence field kept the
   binding-gate value `0`. The display therefore understates the numeric
   failure exactly where the icon is red.

### Real product problems

1. At least 3,718 strict observable reference divergences are above `1e-3`
   in the raw matrix plus recoverable oracles.

2. 3,413 of those strict divergences still pass the current gate because
   the tolerance policy is too permissive for release quality.

3. Several method families have `rmse_rel > 1`, which is not numerical
   noise. These are either wrong method implementations, wrong reference
   contracts, wrong compared output artifacts, stochastic seed mismatch,
   or invalid test/oracle plumbing.

4. Current dashboard-selected rows still contain 175 strict divergences that
   also fail the existing MethodSpec tolerance. These are hard failures even
   before applying the new `1e-3` release rule.

## BLAS+OMP map

Raw recorded reference rows above `1e-3` by build:

| Build | rows `>1e-3` |
|---|---:|
| `blas-omp` | 1,024 |
| `blas-on` | 847 |
| `omp-on` | 840 |
| `dev-release` | 839 |

Raw plus `.npy`-recoverable rows above `1e-3` by build:

| Build | rows `>1e-3` |
|---|---:|
| `blas-omp` | 1,024 |
| `blas-on` | 903 |
| `omp-on` | 896 |
| `dev-release` | 895 |

Dashboard-selected C++ reference stats:

| C++ build | finite refs | max | median | `>1e-3` | `>0.1` | `>1` |
|---|---:|---:|---:|---:|---:|---:|
| `dev-release` | 165 | `3.55903` | `0.0613157` | 99 | 74 | 29 |
| `blas-on` | 168 | `3.55903` | `0.0561467` | 101 | 74 | 29 |
| `omp-on` | 165 | `3.55903` | `0.0613157` | 99 | 74 | 29 |
| `blas-omp` | 186 | `3.55903` | `0.0587312` | 114 | 84 | 36 |

Important interpretation:

- Binding gate: no BLAS+OMP issue found.
- Reference gate: `blas+omp` has more visible rows mostly because it has
  more finite comparisons and is preferred by dashboard dedupe for
  non-C++ rows.
- Same-seed inter-build check found zero cases where `blas-omp` is worse
  than the other builds by more than `1e-3`.
- Apparent `random_frog_select 100x50 t1` BLAS+OMP worsening disappears
  when seed identity is enforced; the compared rows used different
  `prediction_seed` values.

## Hidden oracle problem

Raw `full_matrix.csv` has 3,480 rows marked:

```text
reference oracle missing; run canonical reference backend
```

But 282 of those rows can be recomputed locally from existing `.npy` files.
Among those recoverable rows, 168 are strict divergences above `1e-3`.

Recoverable hidden strict divergences:

| Method | hidden `>1e-3` rows |
|---|---:|
| `cars_select` | 24 |
| `ipw_select` | 24 |
| `randomization_select` | 24 |
| `shaving_select` | 24 |
| `spa_select` | 24 |
| `variable_select_vip` | 24 |
| `wvc_threshold_select` | 24 |

All 168 hidden strict divergences are in `dev-release`, `blas-on`, and
`omp-on` rows, 56 per build. This is one reason `blas+omp` looks worse in
the dashboard: some non-`blas+omp` divergences are hidden as `no_compare`.

## Dashboard-selected hard failures

These are rows in the current dashboard-selected CSV universe where:

```text
reference_parity_rmse_rel > 1e-3
and reference_parity_ok == false
```

They are failures even under the current loose MethodSpec tolerances.

| Method | rows | max rmse_rel | `>0.1` | `>1` | dominant sizes/threads | dominant kinds |
|---|---:|---:|---:|---:|---|---|
| `shaving_select` | 54 | `10.2956` | 54 | 30 | `100x500 t1`, `1000x500 t1`, `100x50 t8` | binding/core |
| `bve_select` | 24 | `1.73205` | 24 | 24 | `100x50 t8`, `100x50 t1` | binding/core |
| `uve_select` | 24 | `2.91548` | 24 | 24 | `100x50 t8`, `100x50 t1` | binding/core |
| `di_pls` | 21 | `0.0206365` | 0 | 0 | `100x50 t8`, `100x50 t1` | binding/core |
| `emcuve_select` | 8 | `1.63299` | 8 | 8 | `100x50 t1` | binding/core |
| `pls_glm` | 8 | `1.35918` | 8 | 8 | `100x50 t1` | binding/core |
| `rosa` | 8 | `1.34988` | 8 | 8 | `100x50 t1` | binding/core |
| `so_pls` | 8 | `1.37393` | 8 | 8 | `100x50 t1` | binding/core |
| `ipw_select` | 4 | `1.73205` | 4 | 4 | `100x2500 t1` | binding |
| `ridge_pls` | 4 | `0.987332` | 4 | 0 | multiple | external |
| `gpr_pls` | 3 | `0.0624596` | 0 | 0 | multiple | external |
| `pcr` | 2 | `1.51142` | 2 | 1 | `1000x500` | external |

## Dashboard-selected strict divergences by method

This is the current local dashboard source after `build_landing.py`
dedupe/prefer rules. `visible dashboard >1e-3` is what the rendered
payload exposes numerically in divergence mode.

| Method | rows `>1e-3` | max rmse_rel | `>0.1` | `>1` | current fail | current pass | binding rows | core rows | external rows | visible dashboard `>1e-3` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse_simpls` | 69 | `0.0241374` | 0 | 0 | 0 | 69 | 48 | 15 | 6 | 21 |
| `shaving_select` | 54 | `10.2956` | 54 | 30 | 54 | 0 | 38 | 16 | 0 | 16 |
| `random_subspace_pls` | 52 | `1.2514` | 52 | 6 | 0 | 52 | 32 | 20 | 0 | 20 |
| `bagging_pls` | 50 | `0.26332` | 13 | 0 | 0 | 50 | 34 | 16 | 0 | 16 |
| `boosting_pls` | 50 | `0.493559` | 50 | 0 | 0 | 50 | 34 | 16 | 0 | 16 |
| `n_pls` | 47 | `5.72172` | 47 | 47 | 0 | 47 | 35 | 12 | 0 | 12 |
| `wvc_threshold_select` | 47 | `0.384775` | 47 | 0 | 0 | 47 | 35 | 12 | 0 | 12 |
| `pls_lda` | 44 | `1.04322` | 44 | 31 | 0 | 44 | 28 | 16 | 0 | 16 |
| `o2pls` | 43 | `0.354652` | 43 | 0 | 0 | 43 | 31 | 12 | 0 | 12 |
| `one_se_rule` | 43 | `0.671317` | 43 | 0 | 0 | 43 | 31 | 12 | 0 | 12 |
| `pls_qda` | 43 | `3.2919` | 43 | 43 | 0 | 43 | 27 | 16 | 0 | 16 |
| `mb_pls` | 42 | `0.0613157` | 0 | 0 | 0 | 42 | 26 | 16 | 0 | 16 |
| `pls_logistic` | 42 | `0.873213` | 42 | 0 | 0 | 42 | 27 | 15 | 0 | 15 |
| `ipw_select` | 41 | `1.73205` | 41 | 4 | 4 | 37 | 29 | 12 | 0 | 12 |
| `cppls` | 40 | `0.0114091` | 0 | 0 | 0 | 40 | 31 | 9 | 0 | 9 |
| `di_pls` | 40 | `0.0206365` | 0 | 0 | 21 | 19 | 33 | 7 | 0 | 7 |
| `lw_pls` | 37 | `0.0667607` | 0 | 0 | 0 | 37 | 25 | 12 | 0 | 12 |
| `on_pls` | 37 | `1.38441` | 37 | 37 | 0 | 37 | 25 | 12 | 0 | 12 |
| `weighted_pls` | 37 | `0.0021985` | 0 | 0 | 0 | 37 | 25 | 12 | 0 | 12 |
| `sparse_pls_da` | 27 | `1.5797` | 27 | 27 | 0 | 27 | 16 | 8 | 3 | 11 |
| `pso_select` | 26 | `1.01436` | 26 | 2 | 0 | 26 | 18 | 8 | 0 | 8 |
| `approximate_press` | 24 | `0.522684` | 24 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `bve_select` | 24 | `1.73205` | 24 | 24 | 24 | 0 | 16 | 8 | 0 | 8 |
| `cars_select` | 24 | `0.816497` | 24 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `continuum_regression` | 24 | `0.0617957` | 0 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `group_sparse_pls` | 24 | `0.0561467` | 0 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `kernel_pls_rbf` | 24 | `0.15695` | 24 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `pls_cox` | 24 | `0.91666` | 24 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `random_frog_select` | 24 | `1.18322` | 24 | 24 | 0 | 24 | 16 | 8 | 0 | 8 |
| `robust_pls` | 24 | `0.0548437` | 0 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `st_select` | 24 | `0.894427` | 24 | 0 | 0 | 24 | 16 | 8 | 0 | 8 |
| `stability_select` | 24 | `1.34164` | 24 | 24 | 0 | 24 | 16 | 8 | 0 | 8 |
| `uve_select` | 24 | `2.91548` | 24 | 24 | 24 | 0 | 16 | 8 | 0 | 8 |
| `variable_select_sr` | 24 | `1.18322` | 24 | 24 | 0 | 24 | 16 | 8 | 0 | 8 |
| `randomization_select` | 21 | `0.57735` | 21 | 0 | 0 | 21 | 16 | 5 | 0 | 5 |
| `spa_select` | 21 | `1` | 21 | 0 | 0 | 21 | 16 | 5 | 0 | 5 |
| `variable_select_vip` | 21 | `0.774597` | 21 | 0 | 0 | 21 | 16 | 5 | 0 | 5 |
| `interval_select` | 11 | `0.534522` | 11 | 0 | 0 | 11 | 7 | 4 | 0 | 4 |
| `iriv_select` | 11 | `0.57735` | 11 | 0 | 0 | 11 | 7 | 4 | 0 | 4 |
| `pls` | 11 | `0.0225184` | 0 | 0 | 0 | 11 | 0 | 0 | 11 | 6 |
| `vissa_select` | 11 | `1.47196` | 11 | 11 | 0 | 11 | 7 | 4 | 0 | 4 |
| `emcuve_select` | 8 | `1.63299` | 8 | 8 | 8 | 0 | 7 | 1 | 0 | 1 |
| `ga_select` | 8 | `1.20605` | 8 | 8 | 0 | 8 | 7 | 1 | 0 | 1 |
| `irf_select` | 8 | `1.13389` | 8 | 8 | 0 | 8 | 7 | 1 | 0 | 1 |
| `pls_glm` | 8 | `1.35918` | 8 | 8 | 8 | 0 | 7 | 1 | 0 | 1 |
| `rep_select` | 8 | `1.25357` | 8 | 8 | 0 | 8 | 7 | 1 | 0 | 1 |
| `rosa` | 8 | `1.34988` | 8 | 8 | 8 | 0 | 7 | 1 | 0 | 1 |
| `so_pls` | 8 | `1.37393` | 8 | 8 | 8 | 0 | 7 | 1 | 0 | 1 |
| `vip_spa_select` | 8 | `0.912871` | 8 | 0 | 0 | 8 | 7 | 1 | 0 | 1 |
| `ridge_pls` | 4 | `0.987332` | 4 | 0 | 4 | 0 | 0 | 0 | 4 | 0 |
| `gpr_pls` | 3 | `0.0624596` | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 0 |
| `pcr` | 2 | `1.51142` | 2 | 1 | 2 | 0 | 0 | 0 | 2 | 0 |

## Raw strict observable method inventory

This table is wider than the dashboard because it starts from raw
`full_matrix.csv` and adds recoverable `.npy` oracle comparisons.

| Method | strict rows `>1e-3` | max | `>0.1` | `>1` | current fail | current pass | recovered hidden |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sparse_simpls` | 193 | `0.0241374` | 0 | 0 | 0 | 193 | 0 |
| `bagging_pls` | 152 | `0.26332` | 40 | 0 | 0 | 152 | 0 |
| `boosting_pls` | 152 | `0.493559` | 152 | 0 | 0 | 152 | 0 |
| `n_pls` | 136 | `5.72172` | 136 | 136 | 0 | 136 | 0 |
| `o2pls` | 136 | `0.354652` | 136 | 0 | 0 | 136 | 0 |
| `one_se_rule` | 136 | `0.671317` | 136 | 0 | 0 | 136 | 0 |
| `pls_lda` | 128 | `1.04322` | 128 | 88 | 0 | 128 | 0 |
| `pls_qda` | 127 | `3.2919` | 127 | 127 | 0 | 127 | 0 |
| `mb_pls` | 125 | `0.0613157` | 0 | 0 | 0 | 125 | 0 |
| `random_subspace_pls` | 125 | `1.2514` | 125 | 13 | 0 | 125 | 0 |
| `pls_logistic` | 124 | `0.873213` | 124 | 0 | 0 | 124 | 0 |
| `lw_pls` | 112 | `0.0667607` | 0 | 0 | 0 | 112 | 0 |
| `on_pls` | 112 | `1.38441` | 112 | 112 | 0 | 112 | 0 |
| `weighted_pls` | 112 | `0.0021985` | 0 | 0 | 0 | 112 | 0 |
| `cppls` | 111 | `0.0114091` | 0 | 0 | 0 | 111 | 0 |
| `di_pls` | 106 | `0.0206365` | 0 | 0 | 48 | 58 | 0 |
| `sparse_pls_da` | 75 | `1.5797` | 75 | 75 | 0 | 75 | 0 |
| `approximate_press` | 72 | `0.522684` | 72 | 0 | 0 | 72 | 0 |
| `bve_select` | 72 | `1.73205` | 72 | 72 | 72 | 0 | 0 |
| `cars_select` | 72 | `0.816497` | 72 | 0 | 0 | 72 | 24 |
| `continuum_regression` | 72 | `0.0617957` | 0 | 0 | 0 | 72 | 0 |
| `group_sparse_pls` | 72 | `0.0561467` | 0 | 0 | 0 | 72 | 0 |
| `ipw_select` | 72 | `0.57735` | 72 | 0 | 0 | 72 | 24 |
| `kernel_pls_rbf` | 72 | `0.15695` | 72 | 0 | 0 | 72 | 0 |
| `pls_cox` | 72 | `0.91666` | 72 | 0 | 0 | 72 | 0 |
| `random_frog_select` | 72 | `1.18322` | 72 | 72 | 0 | 72 | 0 |
| `randomization_select` | 72 | `0.57735` | 72 | 0 | 0 | 72 | 24 |
| `robust_pls` | 72 | `0.0548437` | 0 | 0 | 0 | 72 | 0 |
| `shaving_select` | 72 | `0.816497` | 72 | 0 | 72 | 0 | 24 |
| `spa_select` | 72 | `1` | 72 | 0 | 0 | 72 | 24 |
| `st_select` | 72 | `0.894427` | 72 | 0 | 0 | 72 | 0 |
| `stability_select` | 72 | `1.34164` | 72 | 72 | 0 | 72 | 0 |
| `uve_select` | 72 | `2.91548` | 72 | 72 | 72 | 0 | 0 |
| `variable_select_sr` | 72 | `1.18322` | 72 | 72 | 0 | 72 | 0 |
| `variable_select_vip` | 72 | `0.774597` | 72 | 0 | 0 | 72 | 24 |
| `wvc_threshold_select` | 72 | `0.320256` | 72 | 0 | 0 | 72 | 24 |
| `interval_select` | 26 | `0.534522` | 26 | 0 | 0 | 26 | 0 |
| `iriv_select` | 26 | `0.57735` | 26 | 0 | 0 | 26 | 0 |
| `pso_select` | 26 | `0.884652` | 26 | 0 | 0 | 26 | 0 |
| `vissa_select` | 26 | `1.47196` | 26 | 26 | 0 | 26 | 0 |
| `pls` | 11 | `0.0225184` | 0 | 0 | 0 | 11 | 0 |
| `emcuve_select` | 8 | `1.63299` | 8 | 8 | 8 | 0 | 0 |
| `ga_select` | 8 | `1.20605` | 8 | 8 | 0 | 8 | 0 |
| `irf_select` | 8 | `1.13389` | 8 | 8 | 0 | 8 | 0 |
| `pls_glm` | 8 | `1.35918` | 8 | 8 | 8 | 0 | 0 |
| `rep_select` | 8 | `1.25357` | 8 | 8 | 0 | 8 | 0 |
| `rosa` | 8 | `1.34988` | 8 | 8 | 8 | 0 | 0 |
| `so_pls` | 8 | `1.37393` | 8 | 8 | 8 | 0 | 0 |
| `vip_spa_select` | 8 | `0.912871` | 8 | 0 | 0 | 8 | 0 |
| `ridge_pls` | 4 | `0.987332` | 4 | 0 | 4 | 0 | 0 |
| `gpr_pls` | 3 | `0.0624596` | 0 | 0 | 3 | 0 | 0 |
| `pcr` | 2 | `1.51142` | 2 | 1 | 2 | 0 | 0 |

## Fix inventory

### P0: release gate and dashboard truthfulness

1. Add a strict release gate independent of `MethodSpec.rmse_rel_tol`:
   `reference_parity_rmse_rel <= 1e-3` for every finite executable
   reference comparison, unless a method is explicitly tagged
   `qualitative_non_release`.

2. Split dashboard status fields:
   - `gate_status`: current tolerance result, e.g. `within_tol`,
     `outside_tol`, `no_compare`.
   - `numeric_severity`: strict threshold result, e.g. `strict`,
     `warning`, `fail_gt_1e3`.

3. Stop displaying `exact` for reference rows when
   `reference_parity_rmse_rel > 1e-6` or when
   `reference_parity_tolerance > 1e-3`. Use `within_tol` instead.

4. Fix dashboard legend and filters in `docs/_templates/landing.html`.
   The current text claims universal `1e-6/1e-3` semantics that do not
   match `build_landing.py`.

5. Fix dashboard cell merge rules so a `reference_parity=divergent` cell
   cannot keep `divergence_basis=binding` and numeric `divergence=0`.
   The `shaving_select 100x2500 t1` R cells are the current reproducer.

6. Stop stale `dashboard_refresh_*.csv` rows from overriding canonical
   `full_matrix.csv` rows unless the seed, schema, and oracle metadata are
   compatible.

7. Add dashboard tests for:
   - `reference_parity_ok=True`, `rmse_rel=0.1`, `tol=5` must not render
     as strict/exact.
   - `binding_parity_ok=False`, `max_diff=5e-5` must remain a binding gate
     failure even though it is below `1e-3`.
   - `reference_parity=divergent` with a finite reference RMSE must render
     the reference divergence, not a binding divergence of `0`.

### P0: oracle and seed hygiene

1. Fix oracle loading around `benchmarks/cross_binding/orchestrator.py` so
   a row is not marked `reference oracle missing` when prediction and oracle
   `.npy` files exist and are loadable.

2. Validate oracle metadata before use: algorithm, backend, `n`, `p`,
   `prediction_seed`, reference library version, and output shape.

3. Add a stale-oracle test: seed/version mismatch must invalidate the oracle
   and surface a clear note.

4. Make inter-build BLAS/OMP comparisons seed-stable. Reports must never
   compare `blas+omp` against native/BLAS/OMP rows with a different
   `prediction_seed`.

### P0: comparator and binding reference safety

1. Tighten `benchmarks/parity_timing/_comparator.py`: do not silently
   reshape arbitrary arrays when shapes differ. Allow only explicit 1D vs
   column-vector equivalence; fail matrix transpose and matrix flattening.

2. In `compute_parity()`, remove the `binding_ref = group[0]` fallback when
   no C++ or `python_tier1` pls4all reference exists. A binding gate must
   not choose an external row as its reference.

3. Add tests for missing binding references and shape mismatches.

### P1: method-family corrections

These need method-by-method diagnosis after the dashboard/gate bugs above are
fixed, because some are real algorithm bugs and some are wrong comparison
contracts.

| Family | Methods | Likely issue to inspect first |
|---|---|---|
| hard selector failures | `bve_select`, `uve_select`, `shaving_select`, `emcuve_select`, `ipw_select` | selected-mask semantics, stochastic seed, top-k/count agreement |
| hidden selector drifts | `cars_select`, `spa_select`, `variable_select_vip`, `randomization_select`, `wvc_threshold_select` | oracle recovery plus deterministic seed/options |
| classification/GLM | `pls_lda`, `pls_qda`, `pls_logistic`, `pls_glm`, `sparse_pls_da` | label/probability vs score artifact, classifier defaults |
| multiblock/tensor | `n_pls`, `o2pls`, `on_pls`, `mb_pls`, `rosa`, `so_pls` | output artifact mismatch and deflation/scaling conventions |
| ensembles | `bagging_pls`, `boosting_pls`, `random_subspace_pls` | stochastic control and aggregation semantics |
| core numeric drift | `sparse_simpls`, `cppls`, `weighted_pls`, `lw_pls`, `robust_pls`, `kernel_pls_rbf`, `continuum_regression` | preprocessing/scaling/deflation convention |
| external/proxy rows | `ridge_pls`, `gpr_pls`, `pcr` | decide whether proxy externals are release-blocking references |

Method-specific first hypotheses from code inspection:

| Method | Current diagnosis | First files to inspect |
|---|---|---|
| `bve_select` | C++ appears to do greedy CV-RMSE elimination, while the R `plsVarSel` reference uses a different VIP/cutoff-style selection contract. Decide whether to implement `plsVarSel` semantics or change the declared oracle. | `cpp/src/core/bve_selection.cpp`, `benchmarks/parity_timing/registry.py`, `benchmarks/cross_binding/scripts/bench_registry_common.py` |
| `uve_select` | C++ stability score/noise threshold and R `mcuve_pls` do not appear seed/criterion equivalent. `bench_registry_common.py` also rewrites noise parameters. | `cpp/src/core/uve_selection.cpp`, `cpp/src/core/stability_selection.cpp`, `benchmarks/parity_timing/registry.py` |
| `shaving_select` | C++ ranks by max absolute coefficient; R reference uses `plsVarSel::shaving(method='SR')` with selectivity ratio and internal CV. Dashboard also has stale/merged `100x2500` R failures. | `cpp/src/core/shaving_selection.cpp`, `benchmarks/parity_timing/registry.py` |
| `emcuve_select` | Likely inherits UVE mismatch plus ensemble seed differences between C++ and R reference. | `cpp/src/core/emcuve_selection.cpp`, `benchmarks/cross_binding/scripts/bench_registry_reference.py` |
| `di_pls` | Smaller but real drift. First suspect is centering/rescale target and DI penalty formulation vs `diPLSlib(..., rescale="Target")`. | `cpp/src/core/model.cpp`, `benchmarks/parity_timing/registry.py` |
| `pls_glm` | C++ implementation appears closer to centered linear PLS than a real GLM/IRLS link, while `plsRglm` is a true PLS-GLM reference. | `cpp/src/core/extra_pls.cpp`, `benchmarks/parity_timing/registry.py` |
| `so_pls` | Coefficients may be kept in orthogonalized block space without a reliable back-transform to raw block space. | `cpp/src/core/multiblock_extensions.cpp`, `benchmarks/parity_timing/registry.py` |
| `rosa` | Coefficient accumulation and block deflation look simplified relative to `multiblock::rosa`; raw-space coefficient reconstruction needs review. | `cpp/src/core/multiblock_extensions.cpp`, `benchmarks/parity_timing/registry.py` |

### P2: regeneration and guardrails

1. Regenerate `full_matrix.csv` after the oracle/seed fixes, not just
   `bench-data.json`.

2. Rebuild Sphinx and assert static/build payload SHA equality.

3. Add a CI check that fails when any release-mode finite
   `reference_parity_rmse_rel > 1e-3` remains outside an explicit
   non-release allowlist.

4. Add a dashboard snapshot test that counts strict numeric severities, so
   future Sphinx rebuilds cannot silently turn hard divergences into green
   `exact` cells.

## Immediate correction order

1. Fix dashboard semantics and stale merge first. This removes the false
   "binding drift" reading and makes every `>1e-3` cell visible as such.

2. Fix oracle recovery and metadata validation. This will expose hidden
   native/BLAS/OMP divergences and should reduce the apparent BLAS+OMP bias.

3. Enforce seed-stable inter-build comparison before drawing any BLAS/OMP
   conclusion.

4. Turn the strict `1e-3` release rule on in report/CI mode.

5. Attack hard method failures first: `shaving_select`, `bve_select`,
   `uve_select`, `di_pls`, `emcuve_select`, `pls_glm`, `rosa`, `so_pls`,
   then the selector hidden-drift group.

## 2026-05-20 implementation update

Applied after this inventory:

| Area | Status |
|---|---|
| Dashboard/reference policy | `reference_parity` now uses the strict release rule: `exact <= 1e-6`, `drift <= 1e-3`, `divergent > 1e-3`. MethodSpec qualitative tolerances no longer turn `>1e-3` into green/exact. |
| Divergence display | Reference divergence is shown first for every row with gate-2 metadata. Binding diff is only a legacy fallback. This fixes `so_pls`/`rosa` and `shaving_select` cells that previously showed reference failure with `divergence=0`. |
| Method pages | `docs/_extras/build_methods.py` now uses the same strict reference policy as the dashboard. |
| Orchestrator gate | Gate 2 now compares with `min(MethodSpec.rmse_rel_tol, 1e-3)`. |
| Prediction/oracle hygiene | Added metadata validation for predictions and stored oracles; stale seed/shape/backend metadata becomes a gate failure instead of a numeric comparison. New prediction caches are moved to seed-stable filenames. |
| Binding gate safety | Removed fallback to arbitrary `group[0]`; if no C++/python binding reference exists, the binding gate reports `binding reference missing`. |
| Comparator safety | Shape mismatch no longer reshapes arbitrary same-size matrices; only 1-D vs row/column vector quirks are accepted. |
| `di_pls` | C++ path updated. Live reference check against `diPLSlib` on `100x50 seed1234567894`: `rmse_rel=8.3157e-15` strict pass. Existing snapshot oracle for that seed is stale and must be regenerated. |
| `pls_glm` | C++ now runs a real score-space GLM/IRLS and no longer crashes on the registry smoke cell, but live comparison vs `plsRglm` is still not strict (`rmse_rel ~4.1e6`). Remains a real method-parity issue. |
| `bve_select` | C++ selector aligned to `plsVarSel::bve_pls` VIP semantics; local `100x50` oracle check is strict exact. |
| `uve_select` / `emcuve_select` | C++ selectors moved closer to `plsVarSel::mcuve_pls`; still mask-level divergences (`~0.7` / `~0.8`) and not strict release passes. |
| `shaving_select` | C++ moved to SR-shaving semantics; local parity remains non-strict due to remaining CV/segment differences. |

Post-enforcement local dashboard payload:

| Metric | Count |
|---|---:|
| `reference_parity=divergent` | 1,389 |
| `reference_parity=drift` | 74 |
| finite divergence cells | 2,303 |
| finite divergence cells `>1e-3` | 1,389 |

Important: these dashboard counts still come from existing CSV/oracle
artifacts. A full `full_matrix.csv` regeneration is required before using the
counts as the new post-fix truth.
