# Merge Roadmap — `pls4all` + `nirs4all-methods` → unified **`nirs4all-methods`**

**Status**: post-Codex revision 1 · **Author**: Claude (Opus 4.7) · **Date**: 2026-05-21
**Revision log**: see §12 at the end for Codex's punch list and how each item is addressed.
**Repos in scope**:
- `https://github.com/GBeurier/pls4all` (base) — 73 PLS methods, 24 variable selection, 14 bindings tier-1
- `https://github.com/GBeurier/nirs4all-methods` (donor) — 118 preprocessing/augmentation/splitter/filter/utility operators, tree structure, Python+R bindings

---

## 0. Executive summary

The two repos are **architecturally identical** (portable C++17 core, single stable C ABI, opaque-handle ops, zero mandatory deps, Codex+Opus review workflow) but **functionally disjoint**:

| Domain                            | Lives in today  | Methods | Tree status |
|-----------------------------------|-----------------|--------:|-------------|
| Preprocessing (53 ops)            | `nirs4all-methods` | 53   | ✅ subcategorised (`baselines/`, `derivatives/`, `scaling/`, `scatter/`, `signal_conversion/`, `smoothing/`, `wavelets/`, …) |
| Augmentations (39 ops)            | `nirs4all-methods` | 39   | ✅ subcategorised (`drift/`, `mixup/`, `noise/`, `scattering/`, `splines/`, `wavelength/`, …) |
| Splitters (9)                     | `nirs4all-methods` | 9    | ✅ flat (10 files) |
| Filters (11)                      | `nirs4all-methods` | 11   | ✅ flat (5 files) |
| Utilities (signal-type, T², Q, transfer metrics) | `nirs4all-methods` | 7+ | ✅ flat |
| Feature engineering (FCK)         | `nirs4all-methods` | 1+   | ✅ `specialized/` |
| **PLS regression models** (NIPALS, SIMPLS, kernel, wide-kernel, SVD, power, randomized SVD, PCR, canonical, PLSSVD, OPLS, OPLS-DA, sparse, MIR-PLS, ridge, robust Huber, weighted, continuum, CPPLS, N-PLS, kernel-RBF, O2-PLS, recursive, missing-aware NIPALS, …) | `pls4all` | ~30 | ❌ flat (`model.cpp` is 180 KB) |
| **PLS classification** (PLS-DA, PLS-LDA, PLS-QDA, PLS-logistic, PLS-Cox, sparse PLS-DA, group-sparse, fused-sparse) | `pls4all` | 8 | ❌ flat |
| **Multi-block / local / ensembles** (MB-PLS, LW-PLS, bagging, boosting, random-subspace, ROSA, SO-PLS, ON-PLS) | `pls4all` | 8 | ❌ flat |
| **Variable selection** (24 ops: VIP, SR, SPA, CARS, GA, PSO, VISSA, IRIV, IRF, shaving, BVE, REP, IPW, ST, T2, WVC, EMCUVE, randomization, biPLS, siPLS, interval, stability, UVE, random frog, SCARS, VIP-SPA) | `pls4all` | 24 | ❌ flat |
| **Diagnostics & monitoring** (Hotelling T², Q residuals, DModX, monitoring with alarms, approximate PRESS, one-SE) | `pls4all` | 6 | ❌ flat |
| **AOM-PLS / POP-PLS** (the scientific differentiator) | `pls4all` | 4 | ❌ flat |
| **Calibration transfer** (PDS, DS) | `pls4all` | 2 | ❌ flat |
| **GPR-on-PLS / GLM / Cox** | `pls4all` | 3 | ❌ flat |

**Key observation.** The user's phrase _"ce sont les mêmes"_ is true at the **architectural** level (same C ABI mental model, same CMake presets, same Codex+Opus review workflow) but false at the **scope** level. The repos do not duplicate methods; they cover complementary halves of the NIRS pipeline. The merge therefore is **not a deduplication** — it is a **unification with reorganisation**.

**The decision.** Use **`pls4all` as the merge-base** (more code, more bindings, more parity infrastructure, more mature benchmark dashboard) and **port `nirs4all-methods` into it** while simultaneously **reorganising both halves into the `nirs4all-methods` tree**. Then rename the resulting repo to `nirs4all-methods` and reduce `pls4all` to a CRAN/PyPI-publishable **slim subset** carved from a method manifest.

---

## 1. Goals and non-goals

### Goals

1. **Single repository, single C ABI**, single CMake build, single parity gate, single benchmark dashboard.
2. **Tree-organised** source so finding `airpls.c` or `aom_selection.cpp` is O(category) rather than O(N).
3. **Namespaced** ABI symbols (`n4m_preprocessing_*`, `n4m_models_pls_*`, `n4m_selection_*`, `n4m_diagnostics_*`, `n4m_augmentation_*`, `n4m_splitter_*`, `n4m_filter_*`, `n4m_utility_*`) so the surface is self-documenting and grep-friendly.
4. **Subset publishability.** `pls4all` continues to exist as a slim package on **PyPI** (`pip install pls4all`) and **CRAN** (`install.packages("pls4all")`), built from the merged source via a method manifest that selects only the PLS/selection/AOM/POP/diagnostics families. Same opaque-handle ABI, same parity contracts, just a smaller shared library and a smaller method catalog.
5. **Standardised parity gate** that treats every category uniformly: pinned reference library + checked-in fixture + tolerance row + divergence registry entry. No more "parity for preprocessing" vs "parity for PLS models" with different runners.
6. **Benchmark dashboard with hierarchy categories** — keep `pls4all`'s richer dashboard (118 KB `cross_binding.md`, 829 KB `bench-data.json`, threaded multi-binding matrix), add tree-aware category filters reflecting the new namespaces.
7. **Tier-1 binding catalog stays at 14** (python, r, matlab/octave, julia, js/wasm, go, rust, dotnet, ruby, lua, nim, jni, android) — every binding gets the full method set after the merge.
8. **External-library subset packages anticipated**: a manifest format that lets us publish, in addition to `pls4all`, future slim packages such as `nirs4all-preprocessing`, `nirs4all-augmentation`, `nirs4all-selection`, `aompls`, `aug4all` (depending on adoption signal), each carved from the same merged source.

### Non-goals

- **Not** rewriting any method. Both halves are already implemented and parity-gated.
- **Not** changing the C ABI mental model — opaque handles, error-buffer-per-context, stride-aware matrix views, dual versioning (ABI + project), Codex+Opus review — all carry forward.
- **Not** dropping any binding. The Python/R/MATLAB tier-1 surface stays; secondary bindings stay at PoC level.
- **Not** preserving `p4a_*` symbol names indefinitely — see §4 on the ABI rename.

---

## 2. Target tree

```
nirs4all-methods/
├─ cpp/
│  ├─ include/n4m/
│  │  ├─ n4m.h                        # umbrella (#include "n4m/preprocessing.h"…)
│  │  ├─ n4m_version.h                # single SOR for ABI & project semver
│  │  ├─ n4m_export.h.in              # __declspec / visibility shim
│  │  ├─ preprocessing.h              # n4m_preprocessing_* (~53 ops)
│  │  ├─ augmentation.h               # n4m_augmentation_* (~39 ops)
│  │  ├─ splitters.h                  # n4m_splitter_* (~9 ops)
│  │  ├─ filters.h                    # n4m_filter_* (~11 ops)
│  │  ├─ utilities.h                  # n4m_signal_*, n4m_metric_*, n4m_transfer_*
│  │  ├─ models.h                     # n4m_pls_*, n4m_opls_*, n4m_pcr_*, n4m_mbpls_*, n4m_lwpls_*, n4m_pls_da_*, …
│  │  ├─ selection.h                  # n4m_selection_* (~24 selectors)
│  │  ├─ diagnostics.h                # n4m_diag_* (T2, Q, DModX, PRESS, monitoring)
│  │  ├─ aom_pop.h                    # n4m_aom_*, n4m_pop_*
│  │  ├─ transfer.h                   # n4m_calib_transfer_pds, _ds
│  │  └─ context.h                    # n4m_context_t, error buffer, RNG, backend selection
│  └─ src/
│     ├─ core/
│     │  ├─ common/                   # linalg.{c,hpp}, banded_solver, bspline, svd, pca_helper, rng_pcg64, wavelet_kernels, _vendored/
│     │  ├─ preprocessing/
│     │  │  ├─ baselines/             # airpls, arpls, asls, beads, detrend, snip, …
│     │  │  ├─ derivatives/           # derivate, first_derivative, second_derivative, norris_williams, savitzky_golay
│     │  │  ├─ feature_selection/     # flexible_pca, flexible_svd
│     │  │  ├─ orthogonalization/     # epo, osc
│     │  │  ├─ resampling/            # crop, kbins_discretizer, range_discretizer, resample_transformer, resampler
│     │  │  ├─ scaling/               # baseline, log_transform, normalize, simple_scale, autoscale, pareto
│     │  │  ├─ scatter/               # area_normalization, emsc, local_snv, msc, robust_snv, snv
│     │  │  ├─ signal_conversion/     # fraction_to_percent, from_absorbance, kubelka_munk, percent_to_fraction, to_absorbance
│     │  │  ├─ smoothing/             # gaussian
│     │  │  ├─ specialized/           # fck_static
│     │  │  └─ wavelets/              # haar, wavelet, wavelet_denoise, wavelet_features, wavelet_pca
│     │  ├─ augmentation/
│     │  │  ├─ drift/                 # linear_baseline_drift, polynomial_baseline_drift
│     │  │  ├─ edge_artifacts/        # dead_band, detector_rolloff
│     │  │  ├─ environmental/         # batch_effect, path_length, instrumental_broadening
│     │  │  ├─ mixup/                 # mixup, local_mixup
│     │  │  ├─ noise/                 # gaussian_additive, heteroscedastic, multiplicative, spike
│     │  │  ├─ random/                # rng utilities for augmentation
│     │  │  ├─ scattering/            # scatter_simulation_msc
│     │  │  ├─ spectral/              # band_mask, band_perturb, channel_dropout, smooth_magnitude_warp, unsharp_spectral_mask
│     │  │  ├─ splines/               # spline_smoothing, spline_x_perturb, spline_y_perturb
│     │  │  └─ wavelength/            # wavelength_shift, wavelength_stretch, local_wavelength_warp
│     │  ├─ splitters/                # binned_strat_group_kfold, kbins_stratified, kennard_stone, kmeans, split_common, split_splitter, spxy, spxy_fold, spxy_g_fold, systematic_circular
│     │  ├─ filters/                  # composite, high_leverage, spectral_quality, x_outlier, y_outlier
│     │  ├─ utilities/                # hotelling_t2, nirs_metrics, q_residuals, signal_type_detector, transfer_metrics
│     │  ├─ models/
│     │  │  ├─ pls/                   # nipals.cpp, simpls.cpp, kernel.cpp, wide_kernel.cpp, svd.cpp, power.cpp, randomized_svd.cpp, canonical.cpp, plssvd.cpp, oscores.cpp (from extra_pls.cpp split)
│     │  │  ├─ pcr/                   # pcr.cpp
│     │  │  ├─ opls/                  # opls.cpp, opls_da.cpp, kopls.cpp (from extra_pls.cpp split)
│     │  │  ├─ sparse/                # sparse_simpls.cpp, sparse_pls_da.cpp, group_sparse.cpp, fused_sparse.cpp
│     │  │  ├─ multiblock/            # mb_pls.cpp, n_pls.cpp, o2pls.cpp, so_pls.cpp, on_pls.cpp, rosa.cpp, multiblock_extensions.cpp
│     │  │  ├─ local/                 # lw_pls.cpp, mir_pls.cpp
│     │  │  ├─ ensembles/             # bagging.cpp, boosting.cpp, random_subspace.cpp
│     │  │  ├─ specialized/           # cppls.cpp, continuum.cpp, ridge.cpp, robust_huber.cpp, weighted.cpp, recursive.cpp (from recursive_pls.cpp), missing_aware_nipals.cpp, ecr.cpp, gpr_pls.cpp, tensor_pls.cpp
│     │  │  ├─ classification/        # pls_da.cpp, pls_lda.cpp, pls_qda.cpp, pls_logistic.cpp, pls_cox.cpp
│     │  │  └─ glm/                   # pls_glm.cpp
│     │  ├─ selection/                # 24 selectors, one .cpp per algorithm (cars_selection.cpp → selection/cars.cpp, etc.)
│     │  ├─ diagnostics/              # pls_diagnostics.cpp, pls_monitoring.cpp, component_coefficients.cpp, model_selection.cpp, validation.cpp
│     │  ├─ aom_pop/                  # aom_preprocessing.cpp, aom_selection.cpp, aom_operators.cpp, operator_bank.cpp, gating_strategy.cpp
│     │  ├─ transfer/                 # calib_transfer_pds.cpp, calib_transfer_ds.cpp
│     │  ├─ pipeline/                 # pipeline.cpp split into preprocessing_pipeline.cpp + model_pipeline.cpp
│     │  ├─ context.cpp / hpp         # shared
│     │  ├─ matrix_view.{cpp,hpp}     # shared
│     │  ├─ status.{cpp,hpp}          # shared
│     │  ├─ parallel.hpp              # shared
│     │  └─ version.{cpp,hpp}         # shared
│     └─ c_api/                       # extern "C" boundary, one TU per public header
│        ├─ preprocessing_api.cpp
│        ├─ augmentation_api.cpp
│        ├─ splitter_api.cpp
│        ├─ filter_api.cpp
│        ├─ utility_api.cpp
│        ├─ models_api.cpp
│        ├─ selection_api.cpp
│        ├─ diagnostics_api.cpp
│        ├─ aom_pop_api.cpp
│        ├─ transfer_api.cpp
│        ├─ context_api.cpp
│        └─ version_api.cpp
├─ catalog/                            # **NEW** — single source of truth for method manifests
│  ├─ schema/method_v1.json            # JSON schema: id, category, family, ABI symbol, parity contract id, bench id, since, libraries
│  ├─ methods.yaml                     # full catalog (~250 entries)
│  ├─ subsets/
│  │  ├─ nirs4all_methods.yaml         # full package (all methods)
│  │  ├─ pls4all.yaml                  # PLS-only subset for the slim package
│  │  ├─ nirs4all_preprocessing.yaml   # anticipated
│  │  ├─ nirs4all_augmentation.yaml    # anticipated
│  │  ├─ nirs4all_selection.yaml       # anticipated
│  │  └─ aompls.yaml                   # anticipated micro-package (AOM-only)
│  └─ scripts/
│     ├─ validate_catalog.py           # gate every PR
│     ├─ render_pls4all_subset.py      # produce slim CMake config + slim binding sources
│     └─ render_subset.py              # generic subset renderer
├─ parity/                             # **UNIFIED**
│  ├─ schema/n4m_fixture_v1.json       # one schema for every category
│  ├─ tolerances.md                    # the merged pair-tolerance table (preprocessing rows from n4m + PLS rows from p4a)
│  ├─ divergences.json                 # documented gaps (stochastic ops, R fixture stubs, weak rows)
│  ├─ python_generator/                # the merged generator (pinned env)
│  │  ├─ requirements-lock.txt
│  │  └─ src/n4m_parity/
│  │     ├─ preprocessing/             # ports from nirs4all-methods
│  │     ├─ augmentation/              # ports from nirs4all-methods
│  │     ├─ splitters/                 # ports from nirs4all-methods
│  │     ├─ filters/                   # ports from nirs4all-methods
│  │     ├─ utilities/                 # ports from nirs4all-methods
│  │     ├─ models/                    # ports from pls4all
│  │     ├─ selection/                 # ports from pls4all
│  │     ├─ diagnostics/               # ports from pls4all
│  │     ├─ aom_pop/                   # ports from pls4all (the nirs4all/bench/AOM_v0 oracle stays the truth source)
│  │     └─ transfer/                  # ports from pls4all
│  ├─ r_generator/                     # CRAN-side generator (pls, ropls, mixOmics, plsVarSel, prospectr, mdatools, baseline, waveslim)
│  └─ fixtures/                        # ~250 JSON fixtures, one per (method, scenario, reference)
├─ benchmarks/
│  ├─ registry.py                      # **UNIFIED** — replaces both `benchmarks/cross_binding/orchestrator.py` (pls4all 60 KB) and `benchmarks/benchmark_registry.json` (n4m 10 KB). Adds `category` and `subcategory` keys mirroring the new tree.
│  ├─ cross_binding/                   # from pls4all (more mature)
│  ├─ parity_timing/                   # from pls4all
│  ├─ runners/                         # from pls4all
│  ├─ validation/                      # from n4m (validation snapshot)
│  ├─ reference_snapshots/             # from n4m
│  └─ results/                         # CSV outputs (versioned in dashboard branch only)
├─ bindings/
│  ├─ python/src/nirs4all_methods/     # **NEW** unified package
│  │  ├─ __init__.py                   # re-exports everything
│  │  ├─ preprocessing/                # sklearn-compatible classes
│  │  ├─ augmentation/                 #   "
│  │  ├─ splitters/                    #   "
│  │  ├─ filters/                      #   "
│  │  ├─ utilities/                    #   "
│  │  ├─ models/                       # PLS sklearn classes (from pls4all.sklearn)
│  │  ├─ selection/                    # variable-selection sklearn transformers
│  │  ├─ diagnostics/                  # diagnostic classes
│  │  ├─ aom_pop/                      # AOMPLSRegressor, POPPLSRegressor
│  │  ├─ transfer/                     # PDS, DS
│  │  ├─ _ffi.py                       # unified ctypes loader (n4m + ex-p4a symbols)
│  │  ├─ _context.py
│  │  ├─ _errors.py
│  │  └─ _methods.py                   # generated from catalog/methods.yaml
│  ├─ python_pls4all/                  # **NEW** slim distribution
│  │  ├─ pyproject.toml                # name = "pls4all"
│  │  └─ src/pls4all/                  # re-exports only the PLS subset of nirs4all_methods
│  ├─ r/n4m/                           # full R package
│  ├─ r_pls4all/                       # slim R package (CRAN target)
│  ├─ matlab/                          # +pls4all and +n4m co-namespaces
│  ├─ js/                              # WASM
│  └─ {go,rust,julia,ruby,dotnet,lua,nim,jni,android,octave}/  # tier-1 stubs as today
├─ docs/                               # Sphinx, merged + hierarchical category nav
├─ roadmap/                            # all phase-*.md from both repos, prefixed by category
└─ scripts/bump_version.sh, validate-symbols.py, …
```

### Naming conventions

| Layer | Convention | Example |
|---|---|---|
| Public C ABI symbol | `n4m_<category>_<operator>_<verb>` | `n4m_preprocessing_snv_transform`, `n4m_models_pls_simpls_fit`, `n4m_selection_cars_run` |
| Opaque handle typedef | `n4m_<category>_<operator>_state_t` | `n4m_models_pls_simpls_state_t` |
| Public header | `n4m/<category>.h` | `n4m/models.h` |
| Internal namespace | `n4m::<category>::<operator>` | `n4m::models::pls::simpls` |
| Python module | `nirs4all_methods.<category>.<operator>` | `nirs4all_methods.models.pls.SIMPLS` |
| R function | `n4m_<category>_<operator>` | `n4m_models_pls_simpls()` |
| MATLAB classdef | `+n4m.+<category>.<operator>` | `+n4m.+models.SimplsRegression` |
| Catalog method_id | `<category>.<subcategory>.<operator>` | `models.pls.simpls`, `preprocessing.scatter.snv` |

The `p4a_*` symbols disappear from the C ABI (see §4 on rename strategy and the `pls4all` slim package shim).

---

## 3. Branch / repo / git plan

### 3.1 Repository transitions — revised after Codex review

**Three blockers fixed compared to the v0 draft**:

1. **Naked `--allow-unrelated-histories` merge fails.** The two trees collide on dozens of filenames: `CMakeLists.txt`, `cpp/src/CMakeLists.txt`, `cpp/src/core/{context,matrix_view,status,version,linalg,parallel,context.hpp,…}`, `cpp/src/c_api/c_api_{context,matrix,version}.cpp`, `.github/workflows/*.yml`, `docs/conf.py`, etc. The donor must be imported **under a prefix** first, then moved deliberately category-by-category in dedicated PRs that resolve the shared-core collisions.
2. **GitHub-side name collision.** The donor repo at `GBeurier/nirs4all-methods` *owns the target name*. The rename of `GBeurier/pls4all` → `GBeurier/nirs4all-methods` is impossible until the donor is renamed or archived under a different slug. The fix is to **rename the donor first** to `GBeurier/nirs4all-methods-archive`, archive it, then rename `pls4all` into the freed name.
3. **`*_context_create` symbol collision.** Both repos export `n4m_context_create` (donor) *and* `p4a_context_create` (base) with overlapping semantics (error buffer per context, backend selection, RNG state). A naive `sed p4a_ → n4m_` produces a duplicate symbol. The fix is a dedicated **common-core unification** phase (new M2.5) that picks one set of `context`, `status`, `matrix_view`, `version` definitions before any source moves or symbol renames.

| Step | Repo | Action | Verified by |
|------|------|--------|-------------|
| 1 | `nirs4all-methods` (donor) | Inventory + close stale `worktree-agent-*` branches; commit or stash any local changes; tag `nirs4all-methods-final/<v>` on clean `main`. Export issues/PRs/releases via `gh api` for later import into the unified repo. | `gh repo view --json … > donor-archive-manifest.json` checked in. |
| 2 | `pls4all` (base) | Clean working tree (currently on `codex/parity-30x30-dashboard-fixes` with uncommitted artifacts); freeze on a clean tip of `main`; tag `pls4all-final/v0.97.3`. Delete or .gitignore generated benchmark artifacts that polluted earlier branches. | `git status` clean; `pls4all-final/v0.97.3` reachable; `release/m2-cran` still resolves. |
| 3 | `pls4all` | On `merge/import-donor` branch, import the donor as a subtree under a temporary prefix `_donor/nirs4all-methods/`. Both trees co-exist; nothing collides because everything donor-side is rooted under `_donor/`. Histories merged via `git subtree add --prefix=_donor/nirs4all-methods <donor-remote> main`. | `git log --all --source` shows both histories; build is still pls4all-only at this stage. |
| 4 | `pls4all` | On `merge/common-core` branch (the new M2.5): consolidate `context`, `status`, `matrix_view`, `version`, `parallel`, `linalg` into a single shared layer under `cpp/src/core/common/`. Pick the more-mature implementation per file (mostly donor's for `context.cpp`/`status.cpp` — they ship versioned ELF symbols `@@N4M_1` and machine-readable divergence registry; mostly pls4all's for `linalg.hpp` — more-tested). Delete the duplicates from `_donor/`. | Unified `n4m_context_create` / `n4m_status_*` / `n4m_matrix_view_*` build and pass both donor-fixture and pls4all-fixture suites. |
| 5 | `pls4all` | On `merge/reorg-donor` branch: lift donor sources from `_donor/nirs4all-methods/cpp/src/core/{preprocessing,augmentation,splitters,filters,utilities}/` to their final paths under `cpp/src/core/`. Mechanical `git mv` with one commit per directory so `git blame -C -C -C` continues to track the donor history. Delete the `_donor/` prefix once empty. | n4m fixtures pass; `git blame -C -C -C cpp/src/core/preprocessing/scatter/snv.c` reaches donor's original commits. |
| 6 | `pls4all` | On `merge/reorg-pls` branch: split `model.cpp` (180 KB) and `extra_pls.cpp` (80 KB) into one .cpp per algorithm under `cpp/src/core/models/{pls,opls,pcr,classification,multiblock,local,ensembles,specialized,sparse,glm}/`. The split is high-risk for blame preservation — must use **`git filter-repo --path-rename` per extracted block** when the file split is mechanical, or accept that `git blame -C -C -C` only catches the first extraction and falls back to the parent commit otherwise. Each extraction is its own commit; no functional changes in the split commits. | `git blame -C -C -C cpp/src/core/models/pls/simpls.cpp` cites the original `model.cpp` revisions, with ≥90 % of the lines attributed to a real historical commit. Audit by running `git log --follow` on a sample of 10 extracted files. |
| 7 | `pls4all` | On `merge/reorg-selectors-aom` branch: move `*_selection.cpp` (24 files) into `cpp/src/core/selection/<name>.cpp`; `aom_*.cpp` into `cpp/src/core/aom_pop/`; diagnostics, transfer, pipeline as per §2. | All fixtures green; `n4m_cli --abi-info` still lists `p4a_*` because rename hasn't happened yet. |
| 8 | `pls4all` | On `merge/abi-rename` branch: mechanical rename `p4a_*` → `n4m_models_*` / `n4m_selection_*` / `n4m_aom_pop_*` / etc. per §4. The shared-core symbols (`*_context_*`, `*_status_*`, `*_matrix_view_*`, `*_version_*`) were *already* unified to `n4m_*` in step 4, so no collisions. Tag pre-rename state as `abi-p4a-final/v0.97.3` for archeology. Symbol golden tables regenerated; format **normalised** (donor uses versioned `@@N4M_1`, base uses plain — pick one convention, document in `cpp/abi/README.md`). | `cpp/abi/expected_symbols_{linux,macos,windows}.txt` match the rename mapping in Appendix A. |
| 9 | `pls4all` | Merge all reorg+rename branches into `merge/unified`. Run full parity gate + bench smoke. | Stage 0–4 of unified parity gate passes; dashboard payload builds. |
| 10 | `nirs4all-methods` GitHub (donor) | **Rename first**: rename donor to `GBeurier/nirs4all-methods-archive`. Archive it with `NOTICE.md` redirect. Preserve issues / PRs / releases (use `gh issue list --state all --json …` + `gh pr list --state all --json …`; either keep them on the archive repo, or batch-import to the unified repo via `gh api` + new-comment-with-history-link). | Donor repo flagged archived & renamed; `gh repo view GBeurier/nirs4all-methods-archive` confirms. |
| 11 | `pls4all` GitHub (base) | Rename `GBeurier/pls4all` → `GBeurier/nirs4all-methods`. GitHub auto-redirects clones, PRs, issues. | `git remote -v` continues to work via redirect; new clones get the new URL. |
| 12 | Unified repo | Tag `merge-complete/v1.0.0-rc1`. First stable release is `v1.0.0`. ABI ladder restarts (`N4M_ABI_VERSION_MAJOR=1`) because the symbol prefix changed. | CI green; benchmark dashboard rebuilt at `gbeurier.github.io/nirs4all-methods/`. |

**Why merge into `pls4all` not the other way.** `pls4all` carries (a) the larger binding catalog (see §8 for tier breakdown), (b) much more code (180 KB `model.cpp`, 80 KB `extra_pls.cpp`, 60 KB orchestrator), (c) more parity infrastructure (R generator, MATLAB benchmarks, more documented divergence rows), (d) the published 0.97.x lineage and CRAN-side R package layout. `nirs4all-methods` carries the **organisational clarity** and the **non-PLS halves**.

**Why archive donor not delete.** All five existing publications and several internal handoff notes reference `nirs4all-methods` URLs. GitHub keeps archived repos browsable; deleting breaks DOI/citation chains. Preserve issues, PRs, releases — not just commits and a `NOTICE.md`.

### 3.2 Worktree hygiene

The `nirs4all-methods` repo currently has 11+ active `worktree-agent-*` branches. Before step 1: `git worktree list` + cleanup orphaned worktrees + force-push tags to origin so the import is complete.

---

## 4. ABI rename: `p4a_*` → `n4m_*`

**Decision.** The merged repo ships exactly one symbol prefix: `n4m_*`. No alias shim, no dual-prefix dance.

**Why.** Carrying both prefixes forever doubles the symbol audit surface (`expected_symbols_*.txt`) and confuses every binding. The `pls4all` ABI is at `v0.97.3` (pre-1.0), the rename window will never be cheaper.

**Migration for downstream pls4all users.**
- The slim `pls4all` PyPI/CRAN package (`bindings/python_pls4all/`, `bindings/r_pls4all/`) keeps the **Python/R surface identical** — `from pls4all.sklearn import PLSRegression` continues to work, the import just resolves to the merged engine.
- The **C surface** changes from `p4a_*` to `n4m_*`. The pls4all package version bumps to `1.0.0` and the C-ABI changelog `docs/abi/changes_log.md` records the rename as a hard break. Two consumers exist today (the bindings themselves, plus the CLI), both rebuild from source.
- A one-shot `scripts/migrate_p4a_to_n4m.py` (regex over headers + sources + symbol tables) is shipped in `nirs4all-methods/scripts/` for any out-of-tree consumer.

**Single source of truth.** `cpp/include/n4m/n4m_version.h` holds project + ABI semver. `scripts/bump_version.sh` already propagates; `version-sync.yml` workflow already gates drift in `pls4all`. Carry both forward.

---

## 5. Subset publishability: `pls4all` as slim package

### 5.1 Catalog-driven subset selection

```yaml
# catalog/subsets/pls4all.yaml
package: pls4all
title: "Partial Least Squares engine — PLS-only subset of nirs4all-methods"
description: |
  Slim distribution focused on PLS regression / classification, variable
  selection, AOM-PLS / POP-PLS, calibration transfer, and PLS-specific
  diagnostics. Built from the same source as nirs4all-methods.
maintainers: ["Grégory Beurier"]
license: CeCILL-2.1
abi:
  symbol_prefix: n4m
  exported_subset: true       # libnirs4all_methods_pls4all.so re-exports only PLS-related n4m_* symbols
includes:
  categories:
    - models                  # all PLS regression + classification + multiblock + local + ensembles + specialized
    - selection               # all 24 variable selection algorithms
    - diagnostics             # T², Q, DModX, monitoring, PRESS, one-SE
    - aom_pop                 # AOM-PLS, POP-PLS, AOM operators
    - transfer                # PDS, DS
  utility_subset:             # carry-along utilities required by PLS methods
    - metrics                 # RMSE, R², bias, RPD, RPIQ, MCC, F1, balanced accuracy
    - utilities.hotelling_t2
    - utilities.q_residuals
  preprocessing_subset:       # minimal preprocessing pipeline shipped historically with pls4all
    - preprocessing.scaling.autoscale
    - preprocessing.scaling.pareto
    - preprocessing.scaling.simple_scale
    - preprocessing.scatter.snv
    - preprocessing.scatter.msc
    - preprocessing.scatter.emsc
    - preprocessing.derivatives.savitzky_golay
    - preprocessing.derivatives.norris_williams
    - preprocessing.baselines.asls
    - preprocessing.baselines.detrend
    - preprocessing.wavelets.haar
    - preprocessing.orthogonalization.osc
    - preprocessing.orthogonalization.epo
  splitter_subset:            # K-fold/LOO/holdout/KS/SPXY (needed by validation)
    - splitters.kennard_stone
    - splitters.spxy
    - splitters.split_splitter
excludes:
  categories:
    - augmentation
    - filters
    - utilities.signal_type_detector
    - utilities.transfer_metrics
    - preprocessing.signal_conversion   # ToAbsorbance/KubelkaMunk are NIRS-specific
    - preprocessing.specialized          # FCK
```

### 5.2 Build system — revised after Codex review

The slim packages are **vendored-source distributions**, not "depend on the full package" or "load a filtered shared library from disk". This mirrors how the current R packages already ship (`bindings/r/pls4all/src/libp4a/` and `bindings/r/n4m/src/libn4m/`): the relevant C/C++ TUs are copied into the package's `src/` tree, built by the package's own `Makevars` / `setup.py`, and linked into a standalone shared object that the package owns. CRAN explicitly requires this — it does not accept dependencies on externally-installed `.so` files.

`render_subset.py <subset.yaml>` reads the catalog and emits, for each target binding:

1. **Vendored source tree** under `bindings/python_<subset>/src/<pkg>/_vendored/n4m/` (Python) or `bindings/r_<subset>/<pkg>/src/libn4m_<subset>/` (R): only the TUs needed for the subset's category closure (computed by the catalog dependency walker), plus the carry-along utilities and common-core (`context`, `status`, `matrix_view`, `version`).
2. **Build glue.** Python: `setup.py` + `pyproject.toml` configured to compile the vendored sources into a package-private extension (`_p4a_subset.so`) via CMake-or-setuptools; the CMake path is gated on a build-system flag for power users, the setuptools path is the default and is the only one CRAN-equivalent reviewers see. R: `Makevars` / `Makevars.win` matching the existing pattern (mirror `bindings/r/pls4all/src/Makevars` 2.6 KB and `bindings/r/n4m/src/Makevars` 1.4 KB — both already CRAN-shaped). No external shared library is referenced.
3. **Visibility script.** Only the subset's exported `n4m_*` symbols are visible (`pls4all_subset.ld` on Linux, `.def` on Windows, `-exported_symbols_list` on macOS). Anything outside the subset is hidden — defensive against the surrounding ABI but secondary to point 1 (the vendored tree only contains the subset's TUs in the first place).
4. **Native package surface** for each binding, generated from the catalog: `bindings/python_pls4all/src/pls4all/sklearn/` (the 68 sklearn classes that wrap the subset), `bindings/r_pls4all/pls4all/R/` (the formula+S3, `pls`-compat and `mdatools`-compat wrappers — already exist for pls4all in `bindings/r/pls4all/R/`).

### 5.3 Slim package wire format

```
pip install pls4all
```
installs `pls4all-X.Y.Z-cp311-cp311-linux_x86_64.whl` containing:
- the vendored subset sources compiled into a package-private `_p4a.<ext>` extension
- `pls4all/__init__.py` exposing the legacy import paths (`from pls4all.sklearn import PLSRegression`, `from pls4all.sklearn import SparseSIMPLS`, …) — backed by the vendored subset, **not** by an import of `nirs4all_methods`
- contract test suite at `pls4all/tests/legacy_imports.py` validating every legacy import path still resolves to a working sklearn-compatible estimator (acceptance criterion §10.4)

The full `nirs4all_methods` package does **not** need to be installed for `pls4all` to work. Conversely, both packages co-exist on the same Python install if the user wants both; they share no DLL.

```
install.packages("pls4all")
```
installs the CRAN-checkable R package built from the **vendored** subset source tree (`src/libn4m_pls4all/`), through `Makevars` — exactly as today's `bindings/r/pls4all/src/libp4a/` + Makevars but with the symbol prefix `n4m_*` and the additional carry-along utilities listed in `catalog/subsets/pls4all.yaml`. The current local `R CMD check --as-cran` log shows `1 WARNING + 4 NOTEs` that **must be cleaned** in M11 before CRAN submission (file size on `pls4all.so`, "no visible binding for global variable" for some R-helper symbols, vignette engine).

R package name on CRAN: **`pls4all`** (no hyphen — CRAN forbids hyphens in package names). Other anticipated R subsets use short alphanumeric names (`n4mpre`, `n4msel`) per §8.

### 5.4 Catalog dependency closure

`render_subset.py` walks the catalog to compute the transitive closure of TUs / headers / fixtures / R-registrations / Python wrappers / vendored-license files needed by the included categories. The catalog entry per method carries (the v0 schema was too thin — expanded after Codex review):

```yaml
- method_id: models.pls.simpls
  family: models.pls
  category: models
  subcategory: pls
  since_abi: "1.0.0"
  abi_symbols:                      # all C ABI symbols exposed for this method
    - n4m_models_pls_simpls_fit
    - n4m_models_pls_simpls_predict
    - n4m_models_pls_simpls_destroy
  tu:                               # translation units (built by render_subset)
    - cpp/src/core/models/pls/simpls.cpp
    - cpp/src/c_api/models_api.cpp:n4m_models_pls_simpls_*
  headers:                          # public headers needed
    - cpp/include/n4m/models.h
  carry_along_tu:                   # shared utilities pulled in (resolved by closure)
    - cpp/src/core/common/linalg.{c,hpp}
    - cpp/src/core/common/context.{cpp,hpp}
  parity:
    fixtures:
      - parity/fixtures/models_pls_simpls_basic_v1.json
      - parity/fixtures/models_pls_simpls_widekernel_v1.json
    tolerance_row: pls4all-numpy-simpls
    references:
      - library: scikit-learn
        version: "1.8.0"
        function: sklearn.cross_decomposition.PLSRegression
        doi_or_url: "https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html"
      - library: nirs4all
        version: "0.8.5"
        function: nirs4all.operators.models.SIMPLS
  bench:
    bench_id: models.pls.simpls
    registry_entry: benchmarks/parity_timing/registry.py:models.pls.simpls
  feature_flags:                    # CMake feature flags that gate this method
    - N4M_ENABLE_BLAS_BACKEND       # optional accelerated path
  vendored_licenses: []             # third-party code shipped with this method (none for SIMPLS; would list e.g. FITPACK for spline-smoothing augmenters)
  bindings:                         # idiomatic wrappers per binding
    python:
      class: nirs4all_methods.models.pls.SIMPLS
      legacy_aliases:               # used by contract tests for the slim pls4all
        - pls4all.sklearn.SIMPLS
        - pls4all.sklearn.PLSRegression  # under specific construction kwargs
    r:
      function: n4m_models_pls_simpls
      legacy_aliases:
        - pls4all::plsr
    matlab:
      classdef: +n4m.+models.SimplsRegression
  publications:                     # for the docs page bibliography
    - "Wold S., Sjostrom M., Eriksson L., 2001. PLS-regression: a basic tool of chemometrics."
```

The closure walker (`catalog/scripts/build_closure.py`) drops a `subset_manifest.json` per subset and a `bundle.tar.gz` of the source tree the binding's build picks up.

### 5.4 The same machinery generates future subset packages

| Subset YAML                          | PyPI name                   | CRAN name                | Audience |
|--------------------------------------|-----------------------------|--------------------------|----------|
| `nirs4all_methods.yaml` **(active v1.0.0)** | `nirs4all-methods` | `n4m` | the full umbrella — every method |
| `pls4all.yaml` **(active v1.0.0)** | `pls4all` | `pls4all` | PLS chemometrics, parity with R `pls`/`ropls`/`mixOmics` |
| `aompls.yaml` *(planned)*            | `aompls`                    | `aompls`                 | AOM-PLS + POP-PLS only — activates after the AOM paper landing |
| `nirs4all_preprocessing.yaml` *(planned)* | `nirs4all-preprocessing`    | `n4mpre`                 | spectra preprocessing toolbox |
| `nirs4all_augmentation.yaml` *(planned)* | `nirs4all-augmentation`     | n/a                      | data augmentation for ML training; primarily Python audience |
| `nirs4all_selection.yaml` *(planned)* | `nirs4all-selection`        | `n4msel`                 | variable selection sweeps, parity with `plsVarSel` |

All subsets share the same git history, the same ABI, the same parity gate, the same benchmark dashboard. The catalog is the only thing that varies.

---

## 6. Standardised parity gate

### 6.1 Today

| Repo | Stages | Gating | Skip policy |
|------|--------|--------|-------------|
| `nirs4all-methods` | **4 stages** today (`run_parity_gate.py`): environment lock (Stage 0), fixture determinism (Stage 1), reference parity (Stage 2), C++ parity (Stage 3). The Stage 4 "binding parity" line in donor's `parity/README.md` is **planned, not implemented** — `parity/binding_parity.py` exists (4 KB) but is not wired into the gate runner. `divergences.json` lists allowed gaps with three classes (hard / expected_skip / weak_tolerance); `--skip-policy={legacy,fail,divergences}`. | Locked refs (numpy 1.26.4, scipy 1.17.1, sklearn 1.8.0, pywt 1.8.0, pybaselines 1.2.1). | Strict by default — unlisted skip = fail. |
| `pls4all` | 4 stages, looser glue script across `parity/`, `bindings/python/scripts/`, R `test_parity.R`. Pair-tolerance table at `parity/tolerances.md` (22 KB, large), no machine-readable divergences registry — `📜` markers in the MD table identify documented-but-unwired pairs but cannot be queried by a runner. | Locked refs (numpy/sklearn/pls/ropls/mixOmics/plsVarSel/ikpls). | Mostly via tolerance loosening, occasional `📜` "documented but unwired" rows. |

### 6.2 After merge

**One runner** at `parity/python_generator/scripts/run_parity_gate.py` (port from `nirs4all-methods`, which is the more disciplined of the two). **Five stages**:

| Stage | What | Failure mode |
|-------|------|--------------|
| 0 | Environment lock matches `parity/python_generator/requirements-lock.txt` *and* `parity/r_generator/renv.lock`. | Generators refuse to regenerate fixtures from a drifted env. |
| 1 | Fixture determinism: regenerating from the locked env must produce byte-identical JSON to the manifest. | Drift = generator-side regression — must investigate before bumping pins. |
| 2 | Reference parity: each fixture's `expected_*` arrays are compared against the canonical external library's output. | Hard fail unless `divergences.json` has an entry (with rationale + planned fix). |
| 3 | C++ parity: `n4m_tests` runs fixtures through the merged libnirs4all_methods and gates predictions / coefficients / scalars within `tolerances.md`. | Hard fail. |
| 4 | Binding parity: every active binding (Python, R, then MATLAB & WASM as they land) runs the same fixtures through its FFI and compares against the C++ pass. | Hard fail. |

**Unified contract format.** Every fixture, regardless of category, conforms to `parity/schema/n4m_fixture_v1.json` — schema is broader than the v0 draft because both today's schemas (donor's `parity/schema/` and base's `parity/schema/fixture_schema_v1.{json,md}`) are PLS-shaped and lack `method_id`, `abi_symbol`, canonical reference DOI/URL, subset membership, and machine-readable tolerance metadata. The merged schema fixes that:

```jsonc
{
  "fixture_id":   "preprocessing.scatter.snv.basic_v1",
  "method_id":    "preprocessing.scatter.snv",        // joins to catalog/methods.yaml
  "abi_symbol":   "n4m_preprocessing_snv_transform", // joins to expected_symbols_*.txt
  "since_abi":    "1.0.0",
  "subsets":      ["nirs4all-methods", "pls4all"],   // joins to catalog/subsets/*.yaml
  "category":     "preprocessing",
  "subcategory": "scatter",
  "reference": {
    "library":    "nirs4all",
    "version":    "0.8.5",
    "function":   "nirs4all.operators.transforms.SNV",
    "doi_or_url": "https://github.com/nirs4all/nirs4all/blob/v0.8.5/nirs4all/operators/transforms/snv.py",
    "publication_ref": null                              // populated for methods backed by a paper
  },
  "tolerance": {
    "row":     "n4m-numpy-preprocessing",            // joins to tolerances.md
    "abs":     1e-12,
    "rel":     1e-12,
    "class":   "deterministic_kernel"                // see §6.3 table
  },
  "rng":      { "kind": "pcg64", "seed": 1234567890 },
  "backend_requirements": [],                         // ["BLAS"] | ["OPENMP"] | ["CUDA"] | [] for reference build
  "determinism_mode": "strict",                       // "strict" | "tolerant" — drives the BLAS/OpenMP/CUDA gate (see §6.4)
  "inputs":   { "X": [[…]], "y": null },
  "expected": { "transformed": [[…]] },
  "metadata": { "n_samples": 64, "n_features": 128 }
}
```

**Tolerance rows.** A merged `parity/tolerances.md` with one section per category. Existing rows from both repos are carried 1:1; the only edits are (a) `pls4all-*` row keys renamed to `n4m-*`, (b) a new column `category` for dashboard filtering.

**Divergences registry.** `parity/divergences.json` (port from `nirs4all-methods`) becomes the **only** allowed exit for any non-matching parity result. Three classes: `hard_failure`, `expected_skip`, `weak_tolerance`. Every entry must carry a tracking issue ID and a target tightening date.

**Scientific support.** Each fixture's `reference` block carries a DOI/URL field (extension to the schema). The catalog page in the docs renders bibliographic citations for every method (already done in `nirs4all-methods/docs/methods/*.md` — 118 generated pages — and partially in `pls4all/docs/methods/*.md` — 74 pages). Both directories merge into `docs/methods/<category>/<operator>.md`.

### 6.3 Quantitative agreement targets — calibrated per category

The v0 draft's `1e-12` everywhere claim was incorrect (Codex catch). The donor already records weak-tolerance zones for BEADS, IAsLS, SNIP, OSC, EPO, several transfer metrics, and most wavelet variants. The merged table reflects reality:

| Class | Examples | abs_tol | rel_tol | Comment |
|---|---|---|---|---|
| Trivial deterministic kernels | SNV, MSC, autoscale, Pareto, simple_scale, centering | 1e-12 | 1e-12 | bit-ish; tightest gate |
| Moderately deterministic kernels | EMSC, Savitzky-Golay, Norris-Williams, ASLS, RobustSNV, Detrend, KubelkaMunk | 1e-10 | 1e-10 | residual floating-point reorder from ref impl |
| Baseline solvers (iterative) | airPLS, arPLS, BEADS, IAsLS, SNIP | 1e-7 | 1e-6 | weak zone — donor `divergences.json` records these |
| Wavelets vs PyWavelets | Haar, db4, sym4, coif1 decomposition / reconstruction | 1e-9 to 1e-7 | 1e-9 to 1e-7 | depends on coefficient table source — donor weak rows |
| Orthogonalisation (supervised) | OSC, EPO | 1e-8 | 1e-7 | weak — sign / loading rotation drift |
| PLS solvers vs sklearn / NumPy mirror | SIMPLS, NIPALS, kernel, wide-kernel, SVD, power, randomized SVD | 1e-9 | 1e-9 | as in base today |
| OPLS / OPLS-DA | OPLS1, OPLS2, OPLS-DA binary / multiclass | 1e-9 | 1e-9 | predictive score column matches; orthogonal scores have rotational ambiguity (cf. `ropls-opls-vs-nirs4all-opls` row) |
| Stochastic selectors with synced PCG64 / SplitMix64 mirror | CARS, MCUVE, GA-PLS, Random Frog, SCARS, Shaving, BVE, T2, WVC | 1e-8 | 1e-8 | exact draw order possible vs. our own NumPy mirror — **not** vs upstream R libraries |
| Stochastic ops vs external (R `plsVarSel`, `auswahl`, …) | same as above | n/a | n/a | `divergences.json` "expected_skip" — upstream RNG draw order is not portable; rebuild fixtures from our mirror, cite the original library in `reference.publication_ref` |
| AOM / POP vs `nirs4all/bench/AOM_v0/aompls` (📐 oracle) | AOM operator bank, AOM selection, POP per-component | 1e-8 to 1e-10 | 1e-8 to 1e-10 | as today |
| Cross-implementation rows (sklearn vs R `pls`, mixOmics, ropls, ikpls, …) | NIPALS vs kernel, sklearn vs ropls OPLS | 1e-5 to 1e-7 | 1e-5 to 1e-6 | documented per-pair drift; the `📜` rows in current `tolerances.md` |
| Transfer metrics (CKA, Grassmann, RV, Procrustes, trustworthiness) vs ad hoc references | n4m utilities | 1e-7 to 1e-9 | 1e-7 to 1e-9 | weak — vary by which reference is canonical |
| Sklearn / parsnip / multiclass classification metrics | precision / F1 / MCC / AUC / balanced accuracy | 1e-12 | 1e-12 | deterministic |

Tightening any row (e.g. BEADS 1e-7 → 1e-9) follows the existing process: open a divergence-registry ticket, fix the offending kernel or refine the reference adapter, regenerate, gate.

### 6.4 Numerical determinism across backends

The reference build (no BLAS, no OpenMP, no CUDA) is the **single deterministic baseline**. Optional backends introduce known sources of non-bit-exact results:

| Backend | Source of drift | Policy |
|---|---|---|
| OpenMP (`N4M_ENABLE_OPENMP`) | Parallel reduction order varies thread-count-dependent | Fixtures with `determinism_mode: strict` are run **single-threaded**; `tolerant` fixtures allow multi-threaded execution with relaxed tolerances (one row per backend in `tolerances.md`: `n4m-blas-strict`, `n4m-blas-tolerant`, `n4m-omp-strict`, `n4m-omp-tolerant`, `n4m-cuda-tolerant`). |
| BLAS (`N4M_ENABLE_BLAS`) | FMA fusion, instruction order, MKL vs OpenBLAS vs Apple Accelerate | Same `strict` / `tolerant` split. CI runs the reference CPU backend as the gating one; BLAS and OpenMP are advisory matrix builds. |
| CUDA (`N4M_ENABLE_CUDA`) | Reduction order, fast-math, mixed precision | Reference build is the truth; CUDA fixtures live in their own tolerance row at `1e-5` rel for predictions, `1e-3` rel for raw latent matrices. Not part of the default `parity-gate.yml` matrix. |
| BLAS+OpenMP+CUDA composed | Compound drift | Only the reference build gates `v1.0.0`. Composed backends are exercised by `bench-nightly.yml` (already exists in pls4all) — the dashboard records both timing and parity verdict per backend triple. |

This is the missing "numerical determinism policy" that the v0 draft glossed over (Codex catch). Wire it into the schema via `backend_requirements` + `determinism_mode`.

### 6.5 Reproducible environments

| Layer | Lock file | Container | Verification |
|---|---|---|---|
| Python parity generator | `parity/python_generator/requirements-lock.txt` | `parity/python_generator/Dockerfile` produces the same environment offline | `parity/python_generator/scripts/run_parity_gate.py --stage 0` |
| R parity generator | `parity/r_generator/renv.lock` (new — donor doesn't have one yet) | `parity/r_generator/Dockerfile` | `Rscript -e 'renv::restore(); source("generate_fixtures.R")' --check` |
| Reference C++ build | `cmake --preset ci-reference` is deterministic across Linux/macOS/Windows | `dockerfiles/ci-reference-{ubuntu24,macos-arm,windows}.Dockerfile` | golden ABI symbol diff + fixture pass |
| BLAS/OpenMP build | `cmake --preset ci-blas-omp` | `dockerfiles/ci-blas-omp-ubuntu24.Dockerfile` | tolerance-relaxed fixture pass |
| CUDA build | `cmake --preset ci-cuda` | `dockerfiles/ci-cuda.Dockerfile` (CUDA 12.x base image) | nightly only |

The `pls4all` orchestrator (`benchmarks/cross_binding/orchestrator.py`, 60 KB) and runner (`run_overnight.sh`) **currently bake local paths** under `/home/delete/...` and require a locally-installed `nirs4all` / R env. M8 ports them to consume the container envs above so the dashboard can be re-run by any contributor — not just the maintainer's workstation.

---

## 7. Benchmark dashboard with hierarchy

### 7.1 Keep the `pls4all` dashboard

It already serves:
- 829 KB `bench-data.json` payload (vs 1.7 MB on n4m, which is larger because it includes more methods but lacks PLS rows)
- threaded multi-binding matrix at `docs/benchmarks/cross_binding_threads.md` (368 KB, hand-maintained format)
- a `combine_and_render.py` + `orchestrator.py` (60 KB) refactor plan checked in as `benchmarks/cross_binding/REFACTOR_PLAN.md`
- `benchmarks/parity_timing/registry.py` as canonical method catalog
- legacy fixed-reference cross-product (`REFERENCE_BACKENDS=all`)
- adaptive-runs `N_RUNS=40` with warmstart
- resumable shard runner (`run_overnight.sh`)

The `nirs4all-methods` dashboard is younger (single `orchestrator.py` 205 KB, `validate_dashboard_payload.py` 7 KB, no parity-timing registry yet).

**Decision: port the n4m method catalog into the p4a dashboard.** The dashboard infra is `pls4all`-grown; the **methods it knows about** must expand to the full unified catalog.

### 7.2 Hierarchy categories

Today, the dashboard filters by binding (python / R / matlab / …) and by library (sklearn / ikpls / pls / mixOmics / …). After merge:

| Filter axis | Old values | New values |
|---|---|---|
| `category` | (implicit, varied by row) | `preprocessing`, `augmentation`, `splitter`, `filter`, `utility`, `model.pls`, `model.opls`, `model.pcr`, `model.classification`, `model.multiblock`, `model.local`, `model.ensemble`, `model.specialized`, `selection`, `diagnostic`, `aom_pop`, `transfer` |
| `subcategory` | n/a | from `catalog/methods.yaml` (e.g. `preprocessing.scatter`, `preprocessing.derivatives`, `selection.wrapper`, `selection.ranker`, `model.pls.sparse`) |
| `binding` | python, r, matlab, octave, wasm, julia, … | unchanged |
| `subset` | n/a | `nirs4all-methods` (default), `pls4all`, `aompls`, …  — dashboard can show "this method ships in: full + pls4all + aompls" |

The dashboard UI gains:
- a left-side tree mirroring the source tree (`preprocessing/scatter/snv`, `models/pls/simpls`, `selection/cars`)
- a "ships in subset" badge column
- a verdict column unchanged (existing parity-vs-tolerance verdict logic stays as-is, already mature)
- a divergence-registry link per cell (mouseover → tolerance + divergence ID)

### 7.3 Headline table

Carry the `pls4all` README headline (the 3-row `PLS SIMPLS / AOM-PLS / POP-PLS` table) directly. Add a second headline for preprocessing once a cross-binding preprocessing benchmark lands (currently `nirs4all-methods` has only single-binding timings for SNV/MSC/SG).

---

## 8. Binding catalog and external subset libraries

### 8.1 Binding tiers after merge

The v0 draft called all 14 bindings "tier-1", which overstates reality (Codex catch). The honest tiering:

| Binding | Surface today | Tier today | Source of truth post-merge | Coverage target |
|---|---|---|---|---|
| Python (`nirs4all_methods.*`)         | sklearn-compatible + low-level FFI | **supported** (sklearn classes, n4a pickling, GridSearchCV-ready) | merged catalog | **100 %** of catalog |
| Python slim (`pls4all`)               | sklearn (68 classes) + AOM/POP low-level | **supported** | catalog/subsets/pls4all.yaml | unchanged after merge |
| R full (`n4m`)                        | `.Call` dispatch (73 methods registered from base; donor R only covers preprocessing/augmentation) | **supported (base)** + **smoke-only (donor side)** | merged catalog | **100 %** with parsnip integration |
| R slim (`pls4all`)                    | formula+S3, `pls`-compat, `mdatools`-compat | **supported** | catalog/subsets/pls4all.yaml | unchanged (with R CMD check warnings cleaned) |
| MATLAB / Octave (`+n4m`, `+pls4all`)  | MEX dispatcher + 18 classdefs       | **supported** | merged catalog | **100 %** + Octave plsregress shim |
| JS / WASM                             | `n4m.js` ESM + browser              | **smoke-only** (SIMPLS PoC in base; roadmap in donor) | merged catalog | **supported** target — Phase 25 in donor roadmap, Phase 32 in base |
| Julia                                 | `ccall`                             | **smoke-only** (SIMPLS PoC)              | merged catalog | **supported** target |
| Go                                    | cgo                                 | **smoke-only** (SIMPLS PoC)              | merged catalog | smoke-only → planned |
| Rust                                  | FFI                                 | **smoke-only** (SIMPLS PoC)              | merged catalog | smoke-only → planned |
| .NET                                  | P/Invoke                            | **smoke-only**                            | merged catalog | smoke-only → planned |
| Ruby                                  | FFI                                 | **smoke-only**                            | merged catalog | planned |
| Lua                                   | FFI                                 | **planned (stub only)**                   | merged catalog | planned |
| Nim                                   | nimterop                            | **planned (stub only)**                   | merged catalog | planned |
| JNI / Android                         | AAR predict-only                    | **smoke-only**                            | merged catalog | unchanged |

Coverage tier definitions (carried into docs):
- **supported**: idiomatic surface for the binding's audience, parity tests, sklearn-compat (Python) / parsnip-compat (R) / classdef-compat (MATLAB), packaged on PyPI/CRAN/Maven/etc., docs page maintained.
- **smoke-only**: FFI plumbing exists, one or two methods (typically SIMPLS) round-trip, no package distribution yet.
- **planned**: roadmap entry, header/stub only.

### 8.2 Anticipated external subset libraries (catalog-driven)

Per resolved question §11.2, v1.0.0 ships **only two active subsets**: the full `nirs4all-methods` and the slim `pls4all`. The rest are pre-declared in `catalog/subsets/` with `status: planned` so the catalog format is exercised but no build glue is wired yet:

| Subset | Status at v1.0.0 | When to activate | Audience |
|---|---|---|---|
| `nirs4all-methods` (full umbrella) | **active** | — | full catalog, all bindings |
| `pls4all` (PLS subset) | **active** | — | PLS chemometrics, parity with R `pls`/`ropls`/`mixOmics` |
| `aompls` | planned | After AOM paper publication / DOI request | parity-with-paper reproducibility runs |
| `nirs4all-preprocessing` (`n4mpre`)  | planned | If preprocessing-only users push back on package size | classical chemometrics, no ML training |
| `nirs4all-augmentation`              | planned | If ML practitioners ask for a pure-augmentation package | Python deep-learning audience |
| `nirs4all-selection` (`n4msel`)      | planned | If `plsVarSel` deprecates / loses upkeep | R chemometrics users on CRAN |
| `n4m-transfer`                       | planned | If calibration-transfer is requested separately | NIRS instrument vendors |

Each active subset gets:
- its own YAML in `catalog/subsets/`
- its own `bindings/python_<subset>/` + `bindings/r_<subset>/`
- its own version cadence (default §11.5: same version as full umbrella; can split later)
- its own CRAN/PyPI metadata
- shares the merged repo's ABI, parity gate, dashboard, and CI

Planned subsets only declare their `includes` / `excludes` so the catalog parity check verifies that activating them later doesn't require schema changes.

This is the **scaling answer** for external libraries: not a new repo per audience, but a new manifest per audience over the same source.

---

## 9. Phases / work breakdown — revised after Codex review

The v0 phase chain (M0–M15, ~6 weeks) was optimistic. Codex's punch list adds the common-core unification, the release-mechanics hardening, the license audit, and the dashboard portability work. Revised plan: **8–12 weeks** for a small team.

| Phase | Duration | Deliverable | Gate |
|---|---|---|---|
| **M0** — preflight | 2 days | (a) `pls4all`: clean current `codex/parity-30x30-dashboard-fixes` checkout, commit or drop generated benchmark artifacts, tag clean state `pls4all-final/v0.97.3`. (b) `nirs4all-methods`: **inventory** stale `worktree-agent-*` branches (≥11 today) into `M0/worktree-inventory.md` — one row per branch with tip commit, head author, last activity date, and a maintainer-reviewed recommendation `merge` / `close` / `keep` (per resolved question §11.3). Triage with the maintainer before any cleanup. Tag clean `nirs4all-methods-final/<v>` on `main`. (c) Export donor issues/PRs/releases via `gh api` to `donor-archive-manifest.json`. (d) Confirm CI green on both clean tips. | both `gh pr checks` green; donor manifest + worktree inventory checked in |
| **M1** — import donor | 1 day | `git subtree add --prefix=_donor/nirs4all-methods <donor> main` on a `merge/import-donor` branch of pls4all. Both trees co-exist under disjoint prefixes; no collision. | both symbol tables co-exist; ctest can build pls4all only at this stage |
| **M2** — target tree skeleton | 2 days | Create empty target directories per §2. Land `catalog/schema/method_v1.json` (expanded schema per §5.4) + initial `methods.yaml` (auto-generated from current source tree + benchmark registries) + **two active subsets at v1.0.0** (per §11.2): `subsets/nirs4all_methods.yaml` (full umbrella, all categories) and `subsets/pls4all.yaml` (all current pls4all methods). Also land `subsets/aompls.yaml`, `subsets/nirs4all_preprocessing.yaml`, `subsets/nirs4all_augmentation.yaml`, `subsets/nirs4all_selection.yaml`, `subsets/n4m_transfer.yaml` with `status: planned` (no build glue generated yet — they become active in a later release pass). Write `catalog/scripts/{validate_catalog,build_closure,render_subset}.py`. | `python catalog/scripts/validate_catalog.py` passes; the two active subsets close (full umbrella has all methods, `pls4all.yaml` matches the current pls4all catalog 1:1) |
| **M2.5** — common-core unification **(new)** | 3 days | Pick one set of `cpp/src/core/common/{context,status,matrix_view,version,linalg,parallel}.{c,cpp,hpp}` files. Donor's `context.cpp` / `status.cpp` typically win (versioned ELF symbols `@@N4M_1`, machine-readable error model); base's `linalg.hpp` may win (more-tested kernels). Delete duplicates from `_donor/`. Reconcile `n4m_status_t` and `p4a_status_t` enum values; reconcile `n4m_matrix_view_t` and `p4a_matrix_view_t` field names (`row_stride` / `col_stride` already match — verified). Reconcile context error-buffer size (4 KB on both — same). | Both donor-fixture C++ tests and pls4all-fixture C++ tests pass against the unified common-core; no duplicate symbol in the linker |
| **M3** — reorganise non-PLS (donor side) | 2 days | `git mv` donor sources from `_donor/nirs4all-methods/cpp/src/core/...` to their final paths under `cpp/src/core/`. Update `CMakeLists.txt` glob patterns. Update `parity/python_generator/src/*` imports. Settle the **`augmentations/` vs `augmentation/`** name once (target: **singular `augmentation/`** — short, mirrors Python module convention, matches sklearn `augmentation` ecosystem terms). Update donor's `_methods.py` paths. | donor fixtures still pass against the moved sources; `git blame -C -C -C` chains back to donor commits |
| **M4** — reorganise PLS (base side) | 4 days | Split `model.cpp` (180 KB, ~3800 lines) and `extra_pls.cpp` (80 KB) into one .cpp per algorithm under `cpp/src/core/models/{pls,opls,pcr,classification,multiblock,local,ensembles,specialized,sparse,glm}/`. **Blame preservation policy**: extraction commits are mechanical (no functional changes), `git blame -C -C -C -M` is the audit command; accept that the file-split commit becomes the blame target for boundary lines. Move `multiblock_extensions.cpp`, `pls_diagnostics.cpp`, `pls_monitoring.cpp`, `component_coefficients.cpp`, `model_selection.cpp`, `validation.cpp`. Move 24 `*_selection.cpp` files into `cpp/src/core/selection/<name>.cpp`. Move `aom_*.cpp` + `operator_bank.cpp` + `gating_strategy.cpp` into `cpp/src/core/aom_pop/`. Move `*_pls.cpp` (cppls, gpr, kernel, lw, mb, recursive, tensor) into appropriate model subcategories. | base fixtures pass; symbol table unchanged (still `p4a_*`); blame audit on 10 sample files shows ≥90 % lines reach a real historical commit |
| **M5** — ABI rename `p4a_*` → `n4m_*` | 2 days | Generate the rename mapping into `scripts/migrate_p4a_to_n4m.py`. Apply across headers, sources, c_api/, bindings/, fixtures, golden symbol tables, dashboards. Tag `abi-p4a-final/v0.97.3` before the rename. Normalize symbol-table format to **versioned ELF `@@N4M_1`** on Linux (per resolved question §11.4) — base symbols (currently plain) get migrated into the versioned scheme; donor symbols already match. Document the version-script convention in `cpp/abi/README.md`. macOS uses `-exported_symbols_list`, Windows uses `.def` (no equivalent of versioned symbols on those platforms). | all parity fixtures pass under new symbol names; symbol table diff matches the rename mapping; `n4m_cli --abi-info` lists 200+ symbols; `objdump -T libnirs4all_methods.so | grep ' n4m_'` shows `N4M_1` version tag on every entry |
| **M6** — unified C ABI headers | 1.5 days | Split into the 11 category headers per §2. **Wrapper-generation phase**: the v0 "one TU per public header" line was a wish, not reality (base has many `c_api_*.cpp` files following the legacy layout). The deliverable is a generator (`catalog/scripts/render_c_api.py`) that emits one `cpp/src/c_api/<category>_api.cpp` per category, deduplicating the legacy many-wrapper layout while preserving symbols. | `cpp/include/n4m/<category>.h` set complete; `cpp/src/c_api/<category>_api.cpp` set complete; no orphan symbol |
| **M7** — unified parity gate | 4 days | Port donor's 4-stage `run_parity_gate.py` to drive both halves. **Implement the missing Stage 4** (binding parity — donor has `parity/binding_parity.py` scaffolded but unwired). Merge `tolerances.md` (rename rows, calibrate per category per §6.3, add backend-specific rows per §6.4). Port base's `📜` documented-but-unwired rows into machine-readable `divergences.json` entries. Sync sklearn version across both generators (base uses 1.4, donor uses 1.8.0 — pick 1.8.0 + regenerate base fixtures; investigate any drift). | Stage 0–4 green on the merged repo; CI workflow `parity-gate.yml` updated and green; sklearn drift PR closed with documented divergence updates |
| **M8** — unified benchmark dashboard | 4 days | Bring donor methods into base's `benchmarks/cross_binding/orchestrator.py` and `benchmarks/parity_timing/registry.py`. **De-hardcode local paths** under `/home/delete/...` in the orchestrator (currently makes the dashboard maintainer-only). Wire to the M6.5 container envs. Add `category` / `subcategory` / `subset` filters to the Sphinx + dashboard UI. Carry forward `REFACTOR_PLAN.md` as Phase 26 follow-up. Split `bench-data.json` into per-category JSON files for lazy loading (size jumps from 829 KB to ~2 MB after merge). | dashboard build at `gbeurier.github.io/pls4all/` (pre-rename) shows preprocessing/augmentation/splitter/filter rows alongside PLS rows, with category filter and divergence-registry link per cell |
| **M9** — Python `nirs4all_methods` package | 3 days | Merge `bindings/python/src/pls4all/` (legacy `p4a` Python) + `bindings/python/src/n4m/` (donor) into `bindings/python/src/nirs4all_methods/`. Auto-generate `_methods.py` from `catalog/methods.yaml`. **Legacy bundle compatibility**: existing `.n4a` model bundles (`from nirs4all.sklearn import NIRSPipeline; NIRSPipeline.from_bundle("model.n4a")`) must still load — write a migration shim in `nirs4all_methods.legacy_bundle_v097` that translates the old `p4a_*` opcodes inside bundles to the new `n4m_*` ones. Contract tests at `tests/legacy_bundle_v097/`. | sklearn tests pass for both halves; legacy `.n4a` bundles from `pls4all 0.97.x` load and predict correctly |
| **M10** — Python slim `pls4all` package | 2 days | Wire `bindings/python_pls4all/` with **vendored** subset (per §5.2 — not a re-export). Wheel builds via the catalog closure walker. Contract tests at `pls4all/tests/legacy_imports.py` cover the canonical paths: `from pls4all.sklearn import PLSRegression` / `SparseSIMPLS` / `CPPLS` / `ECRegression` / `AOMPLSRegressor` / `POPPLSRegressor` / `MbPlsRegression` / `RidgePLS` / `BaggingPLS` / `GPRPLSRegression` (the README's 68-class list). | `pip install pls4all` works without `nirs4all_methods` installed; contract tests green; wheel under ~5 MB |
| **M11** — R packages: full + slim | 4 days | Wire `bindings/r/n4m/` (full) and `bindings/r_pls4all/` (slim, vendored — mirror current `bindings/r/pls4all/src/libp4a/` + Makevars pattern, just renamed). **CRAN cleanup**: fix the `1 WARNING + 4 NOTEs` in the current local R CMD check log (file size on `pls4all.so`, missing globals, vignette engine). Pass `R CMD check --as-cran` clean on R oldrel/release/devel for Linux + Windows + macOS-ARM. Final R package names: **`n4m`** (full), **`pls4all`** (slim) — no hyphens, no underscores. | both `.tar.gz` build; `R CMD check --as-cran` is **0 errors / 0 warnings / 0 notes** on `pls4all`; `n4m` is 0/0/≤1 notes |
| **M12** — MATLAB / Octave | 2 days | Extend `+n4m` to cover the merged catalog (MEX dispatcher gets ~200 entries instead of 73 — auto-generate from registry). Keep `+pls4all` as alias namespace re-exporting the PLS subset (vendored, mirrors §5.2). Octave shims preserved. | `+n4m.fit("snv", X)` + `+pls4all.fit("sparse_simpls", X, y)` smoke test |
| **M13** — secondary bindings refresh | 3 days | Update JS/WASM/Julia/Go/Rust/.NET/Ruby/Lua/Nim/JNI to import the new headers; smoke tests still pass on each. Tier classifications per §8.1 honored in docs and README. | per-binding CI green |
| **M14** — license audit **(new)** | 2 days | Audit every vendored third-party piece: FITPACK (spline-smoothing augmenters), PyWavelets coefficient tables (wavelet preprocessing), Isolation Forest / LOF / MCD (filters), Procrustes / RV (transfer metrics), `_vendored/` subdirectories under `cpp/src/core/common/` and `cpp/src/core/filters/`. Confirm CeCILL-2.1 compatibility (or document the dual-license carry-over). Land `NOTICE.md` + `THIRD_PARTY_LICENSES.md` at repo root. Populate `catalog/methods.yaml` `vendored_licenses` field per method. | audit report green; `NOTICE.md` + `THIRD_PARTY_LICENSES.md` checked in |
| **M15** — release mechanics **(new)** | 3 days | Wire `cibuildwheel` for the two PyPI packages (full + slim) across Linux x86_64 / Linux aarch64 / macOS x86_64+arm64 / Windows x86_64. Wire `auditwheel` / `delocate` / `delvewheel` per platform. Wire R-side CRAN submission rehearsals on win-builder, R-hub, macOS arm64 (Apple Silicon farm). Wire `pre-publish.yml` smoke before any tag → release. | `cibuildwheel` matrix green; R-hub run green on all three R streams (oldrel/release/devel) |
| **M16** — repo rename + donor archive | 1 day | (a) Rename `GBeurier/nirs4all-methods` (donor) to `GBeurier/nirs4all-methods-archive` and flag archived. (b) Then rename `GBeurier/pls4all` to `GBeurier/nirs4all-methods`. (c) Batch-import donor issues / PRs / releases manifest into the unified repo, keeping IDs visible via "originally from `nirs4all-methods` #N" prefix in the imported titles. Update CITATION.cff, docs URLs, badges, README links across all packages. | both GitHub Actions and ReadTheDocs builds green under the new name; archive repo flagged & redirected; imported issues searchable |
| **M17** — release `v1.0.0` | 1 day | Tag, push, publish two PyPI wheels (`nirs4all-methods 1.0.0` and `pls4all 1.0.0`) + two R sources. Submit `pls4all` to CRAN. Announce on the mailing list + Discussions + the GitHub Pages dashboard. | PyPI live; CRAN submission queued; archive repo redirect confirmed |

Total: **~38 working days** for a small team, **~8–12 weeks calendar** including Codex/Opus reviews per phase and CRAN's response latency for M17.

### Critical path

`M0 → M1 → M2 → M2.5 → (M3 ∥ M4) → M5 → M6 → M7 → M8 → M9 → M10 → M11 → M14 → M15 → M16 → M17`. M12, M13 parallelise after M6. M14 and M15 are partially in parallel — license audit must finish before release mechanics tag anything.

### Risk register — expanded after Codex review

| Risk | Severity | Mitigation |
|---|---|---|
| 180 KB `model.cpp` split degrades `git blame` quality for boundary lines | H | Mechanical extraction commits (no functional change); audit with `git blame -C -C -C -M` on a 10-file sample; accept partial loss for ≤10 % of lines and document |
| `n4m_context_create` collides with `p4a_context_create` on naive rename | H (BLOCKER, addressed) | M2.5 common-core unification phase merges the two implementations before any rename |
| GitHub name collision between donor and target | H (BLOCKER, addressed) | M16 step (a) renames donor first to free the slug |
| CRAN rejects slim `pls4all` due to existing 1 WARNING + 4 NOTEs in current local check | H | M11 cleans them before submission; CRAN rehearsals on win-builder + R-hub + macOS-ARM in M15 |
| Dashboard payload doubles in size (829 KB → ~2 MB) | M | M8 splits `bench-data.json` per category for lazy loading |
| Donor and base parity generators use different sklearn versions (1.8.0 vs 1.4) | H | M7 bumps base to sklearn 1.8.0 first; regenerate base fixtures; investigate any drift in a focused PR before continuing |
| MATLAB MEX dispatcher's 73-entry switch gets unwieldy at ~200 | L | Already auto-generated from registry; merge updates the source registry only |
| Worktree-agent branches on donor contain unmerged work | M | Inventory in M0; close stale ones with author approval |
| Numerical drift on BLAS/OpenMP/CUDA backends breaks parity tolerances | H | §6.4 tolerance rows + `determinism_mode` field in fixtures; reference build is the gating one; backends are advisory-CI |
| `.n4a` model bundles produced by `pls4all 0.97.x` stop loading after ABI rename | H | M9 legacy bundle shim translates old opcodes; contract tests in `tests/legacy_bundle_v097/` |
| Existing `from pls4all.sklearn import …` user code breaks | M | M10 contract tests cover the 68 documented sklearn classes verbatim |
| Vendored third-party code (FITPACK, PyWavelets coeff tables, IF/LOF/MCD) has license incompatibility | M | M14 license audit, `THIRD_PARTY_LICENSES.md` + `NOTICE.md` |
| CRAN reviewers reject the build system (CMake vs Makevars) | H | Slim packages ship vendored sources with Makevars only, no CMake required for the user — see §5.2 |
| Dashboard orchestrator's hardcoded `/home/delete/...` paths block contributor re-runs | M | M8 ports the orchestrator to consume the M6.5 container envs |
| Issues / PRs / releases from donor get lost in archive | M | M0 exports them; M16 step (c) batch-imports into the unified repo |
| Symbol-table format drift between platforms (versioned `@@N4M_1` vs plain) | L | M5 picks versioned ELF for Linux, documents in `cpp/abi/README.md` |
| `augmentations/` vs `augmentation/` naming inconsistency drifts into catalog IDs and docs | L | M3 settles on singular `augmentation/` once, propagates through catalog and headers |

---

## 10. Acceptance criteria for v1.0.0

1. `git clone https://github.com/GBeurier/nirs4all-methods.git` + `cmake --preset dev-release && cmake --build --preset dev-release && ctest --preset dev-release --output-on-failure` is green.
2. `parity/python_generator/scripts/run_parity_gate.py --build-dir build/dev-debug` is green: Stage 0–4, full catalog, divergences only from `divergences.json`.
3. `benchmarks/cross_binding/run_overnight.sh` produces `docs/_static/bench-data.json` covering all categories, dashboard renders with hierarchical filters.
4. `pip install -e bindings/python` exposes `nirs4all_methods.*` for all categories; `pip install -e bindings/python_pls4all` exposes `pls4all.*` for the PLS subset; both pass their own smoke suites.
5. `R CMD INSTALL bindings/r/n4m` + `R CMD INSTALL bindings/r_pls4all` both succeed; both pass `R CMD check --as-cran` on the slim one.
6. `n4m_cli --abi-info` lists every exported symbol; symbol golden tables match `cpp/abi/expected_symbols_*.txt`.
7. `catalog/scripts/validate_catalog.py` is green: every method in the source tree has a catalog entry; every catalog entry has at least one parity fixture; every subset is consistent.
8. `nirs4all-methods` GitHub repo is the active home; `pls4all` GitHub repo is renamed to it; the old `nirs4all-methods` (donor) is archived with `NOTICE.md`.
9. CITATION.cff, README badges, ReadTheDocs URLs, and PyPI/CRAN metadata all point to the unified name.
10. Codex post-review on `merge/unified` finalisation PR is "approve" (or contributor-overridden with explicit rationale in `docs/reviews/merge/codex-post.md`).

---

## 11. Open questions for the user — resolved 2026-05-21

| # | Question | Decision |
|---|---|---|
| 1 | Org for the unified repo: `GBeurier/` or new `nirs4all/` org? | **Stay on `GBeurier/`.** Final URL: `https://github.com/GBeurier/nirs4all-methods`. |
| 2 | Anticipated subsets to wire day-one: just `pls4all`, or also `aompls`? | **Two subsets only at v1.0.0**: (a) the full umbrella `nirs4all-methods` (whole catalog) and (b) the slim `pls4all` (all current pls4all methods, vendored). `aompls`, `nirs4all-preprocessing`, `nirs4all-augmentation`, `nirs4all-selection`, `n4m-transfer` stay as roadmap-only stubs in `catalog/subsets/` with a `status: planned` field, to be activated in a later release pass. |
| 3 | Donor worktree-agent branches: auto-close or inventory + review? | **Inventory + review.** Each donor `worktree-agent-*` branch is listed in `M0/worktree-inventory.md` with its tip commit, head author, and a "merge / close / keep" recommendation, then triaged with the maintainer before any cleanup. |
| 4 | Symbol-table format on Linux: versioned ELF `@@N4M_1` or plain? | **Versioned ELF `@@N4M_1`.** Documented in `cpp/abi/README.md`; the `expected_symbols_linux.txt` golden tables stay in the versioned format the donor already uses; the rename mapping in M5 normalises base symbols (currently plain) into the versioned scheme. |

### Other follow-ups (no answer needed yet — surfaced for visibility)

| # | Item | Default if no answer |
|---|---|---|
| 5 | Slim `pls4all` version coupling: same as `nirs4all-methods`, or own slower cadence? | **Default: same version** (one tag = one wheel set). Easier for users; can be split later. |
| 6 | R generator: keep `pls4all`'s adapters (`pls`, `ropls`, `mixOmics`, `plsVarSel`) + add `prospectr`/`mdatools`/`baseline`/`waveslim` for the preprocessing rows? | **Default: yes.** Land the `renv.lock` for R, port donor's preprocessing references through the R-side too. |
| 7 | License harmonisation: keep CeCILL-2.1, adopt canonical `LICENSE` from `pls4all` or from donor? | **Default: pls4all's** `LICENSE` (more recent revision); diff documented in `THIRD_PARTY_LICENSES.md`. |

(see also §12 below for the post-Codex revision log)


1. **Org for the unified repo.** Keep `GBeurier/nirs4all-methods`, or move to a `nirs4all/nirs4all-methods` org for community framing?
2. **Slim package version coupling.** Should `pls4all` slim follow `nirs4all-methods` version (always equal), or have its own slower cadence? Recommendation: same version (one tag = one wheel set), simpler for users.
3. **Anticipated subsets to wire from day one.** Just `pls4all`, or also pre-create `aompls` so the AOM paper's reproducibility doi can pin to a stable URL?
4. **R generator on the merged repo.** `nirs4all-methods` doesn't ship one yet; `pls4all` does (R `pls`, `ropls`, `mixOmics`, `plsVarSel`). Confirm we keep `pls4all`'s R generator and add prospectr/mdatools/baseline/waveslim adapters for the preprocessing rows.
5. **Worktree-agent branches** on `nirs4all-methods` (donor) — auto-close all, or inventory and review first?
6. **License harmonisation.** Both are CeCILL-2.1, but `pls4all/LICENSE` (2.2 KB) and `nirs4all-methods/LICENSE` (2.4 KB) differ slightly. Adopt one canonical file.

---

## Appendix A — Today's symbol mapping (illustrative)

Examples of what changes in the rename pass (full mapping auto-generated by `scripts/migrate_p4a_to_n4m.py`):

| Today (`pls4all`) | After (`nirs4all-methods`) |
|---|---|
| `p4a_simpls_fit` | `n4m_models_pls_simpls_fit` |
| `p4a_simpls_predict` | `n4m_models_pls_simpls_predict` |
| `p4a_pipeline_snv` | `n4m_preprocessing_snv_transform` |
| `p4a_pipeline_msc_fit` | `n4m_preprocessing_msc_fit` |
| `p4a_cars_select` | `n4m_selection_cars_run` |
| `p4a_aom_fit` | `n4m_aom_pop_aom_fit` |
| `p4a_pop_fit` | `n4m_aom_pop_pop_fit` |
| `p4a_pls_da_fit` | `n4m_models_classification_pls_da_fit` |
| `p4a_mb_pls_fit` | `n4m_models_multiblock_mb_pls_fit` |
| `p4a_pds_calibrate` | `n4m_transfer_pds_calibrate` |
| `p4a_t2_select` | `n4m_selection_t2_run` |
| `p4a_diagnostic_t2` | `n4m_diag_hotelling_t2` |
| `p4a_context_create` | `n4m_context_create` |
| `p4a_context_last_error` | `n4m_context_last_error` |
| (n4m unchanged) `n4m_signal_type_detect` | `n4m_utility_signal_type_detect` |
| (n4m unchanged) `n4m_xfilter_isolation_forest_run` | `n4m_filter_x_isolation_forest_run` |

The general rule: `n4m_<category>_<operator>_<verb>`. Existing `n4m_*` symbols that were already category-prefixed implicitly stay; those that weren't (e.g. `n4m_signal_type_detect`) get the missing `<category>` segment.

---

## Appendix B — File-count budget after merge

Estimated, based on a `find ... -type f` over both repos and a manual category-mapping pass:

| Layer | Files | Notes |
|---|---|---|
| `cpp/include/n4m/` | 11 public headers + 1 umbrella + 2 metadata | one header per category |
| `cpp/src/core/` | ~310 .c/.cpp + headers | n4m donates ~180; p4a donates ~135 (after `model.cpp` split into ~30) |
| `cpp/src/c_api/` | 12 .cpp | one per public header |
| `bindings/python/src/nirs4all_methods/` | ~60 .py | sklearn classes per category |
| `bindings/python_pls4all/src/pls4all/` | ~25 .py | slim re-exports |
| `bindings/r/n4m/R/` | ~90 .R | full catalog wrappers |
| `bindings/r_pls4all/R/` | ~35 .R | slim |
| `bindings/matlab/+n4m/` | ~80 .m | full classdefs + dispatcher |
| `bindings/matlab/+pls4all/` | ~25 .m | slim |
| `parity/fixtures/` | ~360 JSON | ~80 from p4a + ~250 from n4m + ~30 new cross-binding |
| `parity/python_generator/src/n4m_parity/` | ~50 .py | one module per category, fixture generators |
| `parity/r_generator/R/` | ~25 .R | port of p4a's R generator + new preprocessing adapters |
| `benchmarks/cross_binding/` | ~12 .py + shell | port of p4a's mature runner |
| `benchmarks/parity_timing/registry.py` | 1 | port of p4a's canonical method catalog, populated from `catalog/methods.yaml` |
| `catalog/` | ~10 YAML + ~5 .py | new, single source of truth |
| `docs/methods/<category>/` | ~250 .md | category-nested; auto-generated from catalog |
| `roadmap/<category>/` | ~40 .md | phase plans, mirrored history from both repos |

Total source files: ~1,300 (≈ 60 % from `pls4all` base + 40 % donated from `nirs4all-methods`).

---

## Appendix C — What this roadmap explicitly is NOT

- It is **not** a redesign of any algorithm. All numerics carry forward unchanged.
- It is **not** a redesign of the C ABI mental model. Opaque handles, error-buffer-per-context, dual versioning, stride-aware views, zero mandatory deps — unchanged.
- It is **not** a new dashboard. We keep `pls4all`'s mature one and extend its filter set.
- It is **not** a new parity gate. We take `nirs4all-methods`'s 5-stage runner and feed it the merged catalog.
- It is **not** a new binding layer. The 14 existing bindings keep their architecture; their generated method tables grow.

The merge is **a unification with tree reorganisation, an ABI rename, a catalog/subset layer, and uniform parity & benchmark gating**. Nothing else.

---

## 12. Post-Codex revision log

Codex reviewed the v0 draft of this document (2026-05-21) and returned a punch list. Every item below is either incorporated above, or explicitly out-of-scope with rationale.

### 12.1 Blockers — all incorporated

| Codex finding | Where addressed |
|---|---|
| M1 "naked `--allow-unrelated-histories` merge" collides on dozens of shared filenames | §3.1 step 3: import donor under `_donor/nirs4all-methods/` prefix via `git subtree add`, defer category moves to M3/M4 |
| GitHub rename collision (donor already owns the `nirs4all-methods` slug) | §3.1 step 10 renames donor first to `nirs4all-methods-archive`; step 11 then renames base into the freed slug |
| ABI rename: `p4a_context_create` collides with existing donor `n4m_context_create` | New M2.5 common-core unification phase — picks one set of `context/status/matrix_view/version/linalg/parallel` before any rename |
| CRAN plan wrong: R packages today ship vendored sources + Makevars, not "filtered shared library" | §5.2 rewritten — slim packages are **vendored-source distributions** with their own Makevars, mirroring existing `bindings/r/pls4all/src/libp4a/` |

### 12.2 Major issues — all incorporated

| Codex finding | Where addressed |
|---|---|
| Need explicit "common-core unification" phase before source moves & rename | New M2.5, see §3.1 step 4 + §9 |
| Parity gate is 4 stages today, not 5; donor's Stage 4 binding parity is scaffolded but unwired; fixture schema lacks `method_id`, `abi_symbol`, DOI/URL, subset metadata, tolerance fields | §6.1 corrected; §6.2 schema expanded; M7 implements Stage 4 |
| Tolerances too aggressive at 1e-12 for baselines / wavelets / OSC / EPO / cross-impl | §6.3 calibrated per category with explicit weak zones |
| "RNG mirrors guarantee exact draw order" is false vs upstream R libraries | §6.3 split: synced PCG64 mirrors → 1e-8; upstream R → `divergences.json` expected_skip |
| Catalog YAML too thin | §5.4 schema expanded: TU paths, headers, carry-along TU, feature flags, vendored licenses, parity fixture list, per-binding wrappers + legacy aliases, publications |
| `model.cpp` split risks blame loss | M4 policy: mechanical commits + `git blame -C -C -C -M` audit on 10-file sample |
| Dashboard hardcoded `/home/delete/...` paths and local R env | M8 includes de-hardcoding via M6.5 container envs (new §6.5) |

### 12.3 Minor issues — addressed

- M0 freezes clean tips on both repos (donor's stale worktree branches inventory, base's dirty `codex/parity-30x30-dashboard-fixes` cleaned).
- M5 normalizes symbol-table format (versioned ELF on Linux).
- §8.1 reclassifies bindings: **supported** / **smoke-only** / **planned**; the v0 "14 tier-1" line dropped.
- M11 cleans the existing `1 WARNING + 4 NOTEs` before CRAN.
- M3 settles `augmentation/` (singular) once.

### 12.4 Missing topics — added

- `.n4a` bundle migration (M9 legacy_bundle_v097 shim + contract tests).
- Legacy Python/R import contract tests (M10 `pls4all/tests/legacy_imports.py`, M11 R `predict.pls4all_fit` etc.).
- License audit (M14, new).
- Release mechanics: cibuildwheel / auditwheel / delocate / delvewheel / R oldrel-release-devel / Windows / macOS ARM (M15, new).
- Numerical determinism policy for BLAS / OpenMP / CUDA (new §6.4).
- Reproducible environments without local paths (new §6.5).
- Donor archive policy preserves issues / PRs / releases (M0 export + M16 batch-import).

### 12.5 Nits — addressed

- Count consistency: v0 mentioned "118 donor operators / ~250 entries" without grounding. Updated counts: donor has 118 generated `docs/methods/*.md` pages today, base has 74 — total ~250 catalog entries after merge accounting for cross-category methods and AOM duplicates. Donor's R DESCRIPTION 403 ABI symbols and 498-line expected-symbols files are the symbol surface counts, not method counts (one method can export ≥3 symbols: create/fit/destroy or transform/inverse_transform).
- CRAN names cannot use hyphens. M11 names them **`n4m`** (full) and **`pls4all`** (slim).
- Slim wheel **vendors** the subset (no re-export of `nirs4all_methods`) — §5.2/5.3 rewritten.
- "One TU per public header" was a wish — M6 is a deliberate wrapper-generation phase via `render_c_api.py`.
- `.n4a` is **real** — it's the nirs4all library's model bundle format (see `nirs4all` CLAUDE.md, `result.export("model.n4a")`). M9 carries the legacy bundle compat.

### 12.6 Overall verdict

Codex's verdict: "Sound strategic direction… Not executable as written. Fix the import/rename mechanics, CRAN packaging model, common-core consolidation, and machine-readable parity/catalog contracts before implementation. The estimate is optimistic; with review and packaging hardening, think closer to 8-12 weeks than 6."

All four "fix before implementation" gates are now in the plan. Timeline updated to **8–12 weeks** (§9 total).

