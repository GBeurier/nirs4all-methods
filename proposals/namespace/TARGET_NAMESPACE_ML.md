# Target namespace (ML/DL, Scheme B + Codex review) — authoritative migration contract

**Status:** FINAL. This supersedes `# Scheme B` in `NAMESPACE_PROPOSALS.md`. It is the single source
of truth for the migration. The full per-method mapping is generated and validated by
`_build_target_ml.py` → **`_target_ml_table.tsv`** (209/209, 0 collisions; see roadmap-gate
amendments at the end). Codex's reviews are archived at `codex_review_ml_table.md` (table) and
`codex_review_roadmap.md` (roadmap + final amendments).

**Authority:** chosen scheme = **ML/DL pipeline**; deltas from the proposal = the Codex change-list
(per the project review loop, Codex wins). ABI strategy = **clean break, ABI 2.0** (no runtime aliases).

---

## Final tree (208-base snapshot — see roadmap-gate amendments below for the final 209 · 12 top-level · 49 leaf)

```
n4m
├── transform (55)              fit/transform feature transforms (sklearn.preprocessing)
│   ├── scatter (12) · baseline (11) · smoothing (6) · wavelet (6) · signal_conversion (5)
│   ├── resampling (5) · scaling (4) · alignment (4)
│   ├── orthogonalization (1)   OSC only — EPO moved to domain_adaptation
│   └── specialized (1)         fck_static
├── augmentation (39)           apply-only training-time perturbations (DL convention)
│   └── instrument(7) spectral(7) scattering(6) splines(5) mixup(4) noise(4) drift(3) wavelength(3)
├── estimators (30)             supervised predictors (fit/predict)
│   ├── regression
│   │   ├── latent (6)          pls, pcr, cppls, continuum_regression, ecr, missing_aware_nipals
│   │   ├── regularized (2)     ridge, ridge_pls
│   │   ├── robust (2)          robust_pls, weighted_pls
│   │   ├── kernel (2)          kernel_pls, gpr_pls
│   │   ├── sparse (3)          sparse_simpls, fused_sparse_pls, group_sparse_pls
│   │   ├── tensor (1)          n_pls            (was tensor_pls)
│   │   ├── online (1)          recursive_pls    (was recursive)
│   │   ├── local (1)           lw_pls           (was estimators.local)
│   │   └── glm (1)             pls_glm
│   ├── classification (4)      pls_lda, pls_qda, pls_logistic, sparse_pls_da
│   ├── multiblock (6)          mb_pls, so_pls, on_pls, o2pls, rosa, mir_pls
│   └── survival (1)            pls_cox
├── feature_selection (27)      wrapper (23) · filter (2) · interval (1) · ranking (1)
├── model_selection (21)        splitters (9) · aom_search (9) · aom_campaign (2) · sweep (1)
├── domain_adaptation (9)       standardization (6) · invariant (1) · metrics (1) · orthogonalization/EPO (1)
├── outlier_detection (7)       sample-level screeners + Q/T2
├── ensemble (6)                bagging/boosting/moment_stack/random_subspace (4) · aom (2: ridge_blender, operator_pls_stack)
├── compose (5)                 aom_superblock (5)
├── metrics (5)                 scoring (3) · diagnostics (2)
├── decomposition (2)           flexible_pca, flexible_svd
├── lowlevel (1)                moments
└── utils (1)                   signal_type_detector
```

## Deltas applied from the proposal (Codex change-list)

1. **Split `estimators.regression.linear` (14)** → `regression.{latent,regularized,robust,kernel,tensor,online,local}` + fold `estimators.local.lw_pls` into `regression.local`.
2. **Split `estimators.glm_survival`** → `regression.glm` (pls_glm) + `estimators.survival` (pls_cox).
3. **EPO** → `domain_adaptation.orthogonalization`; **OSC** stays `transform.orthogonalization`.
4. **`domain_adaptation`** → `{standardization, invariant, metrics, orthogonalization}`.
5. **18 `aom_pop.*`** → `model_selection.aom_search` (9), `model_selection.aom_campaign` (2), `compose.aom_superblock` (5), `ensemble.aom` (2).
6. **`model_selection.split`** → `model_selection.splitters`.
7. **Leaf renames (clean break):** `pls_fit_simple→pls`, `kernel→kernel_pls`, `tensor_pls→n_pls`, `recursive→recursive_pls`, scaling `baseline→baseline_center`, `derivate→derivative`, `split_splitter→data_twinning`, diagnostics `model_selection→one_se_rule`.
8. **Kept** (Codex confirmed): `sparse_pls_da` & `pls_logistic` in `classification`; `flexible_pca/svd` in `decomposition`; `filters.correlation/variance` in `feature_selection.filter`.
9. **Dissolved `utils` grab-bag:** `sweep→model_selection.sweep`, `moments→lowlevel.moments`, `signal_type_detector` stays `utils`.

## API surface (three faces)

- **C ABI — CLEAN BREAK, ABI 2.0.** Rename exported symbols to canonical role names
  `n4m_<role>_<method>_<verb>` (e.g. `n4m_regression_ridge_fit`, `n4m_transform_snv_create/_transform`,
  `n4m_feature_select_cars_select`, `n4m_model_selection_kennard_stone_split`,
  `n4m_augmentation_gaussian_noise_apply`). **Delete** the terse exports (no aliases, no dead code).
  Split `n4m.h` into role headers + remove the empty stub category headers. Regenerate all three
  `cpp/abi/expected_symbols_{linux,macos,windows}.txt` + `docs/abi/changes_log.md`; bump
  `N4M_ABI_VERSION_*`. (Do not over-trust "verb encodes kind" — `_fit`/`_run` are overloaded.)
- **Python — replace flat `python.py`.** sklearn-mirroring subpackages with **public** class names
  (NOT `Native*`): `from n4m.estimators.regression.regularized import Ridge`,
  `from n4m.transform.scatter import SNV`, `from n4m.feature_selection.wrapper import CARS`,
  `from n4m.model_selection.splitters import KennardStone`, `from n4m.compose.aom_superblock import AOMRidgePLSSuperblock`,
  `from n4m.ensemble import AOMRidgeBlender`. `__init__` exposes subpackages only.
- **R — shorter singular role prefix:** `n4m_regression_ridge()`, `n4m_transform_snv()`,
  `n4m_feature_select_cars()`, `n4m_model_selection_kennard_stone()`. Keep formula/S3 wrappers idiomatic.

## Reconcile BEFORE migration (Codex risks → Phase R)

- `selection.wvc` exports both `n4m_wvc_select` and `n4m_wvc_threshold_select` while docs/bindings
  expose `wvc_threshold_select` separately — pick one canonical surface.
- Backfill the **10 methods with 0 ABI symbols** (9 Python-only AOM superblocks + moment_stack) and
  **16 `kind=null`** rows before generating role headers/symbols — decide which gain C symbols.
- Freeze canonical IDs: `n_pls`(=tensor_pls), `kernel_pls`(=kernel), `one_se_rule`(=model_selection).

## Roadmap-gate amendments (APPLIED) → 209 methods · 12 top-level · 49 leaf

Codex's roadmap review (`codex_review_roadmap.md`) required these; now applied in `_build_target_ml.py`
→ `_target_ml_table.tsv` (validated 209/209, 0 collisions):

- **WVC split** — two real public surfaces: `feature_selection.wrapper.wvc` (`n4m_wvc_select`) +
  `feature_selection.wrapper.wvc_threshold` (`n4m_wvc_threshold_select`). Cardinality **208 → 209**.
- **`utils` removed** (no top-level): `signal_type_detector` → `transform.signal_conversion`.
- **`lowlevel.moments` kept** (real sufficient-statistics substrate, multiple ABI fns) → add `lowlevel.h` to the Phase-2 header split.
- **`ensemble.aom` flattened** into `ensemble`: `aom_ridge_blender`, `aom_operator_pls_stack` (provenance in the leaf).
- `estimators.regression.glm` / `estimators.survival` confirmed as singleton growth points.
- **`_target_ml_table.tsv`'s `abi_symbol` is a SEED, not final ABI truth** — Phase R derives final C symbols from reconciled catalog `abi_symbols`; the 10 zero-ABI rows get explicit `c_surface: none`.
- **Catalog keeps stable legacy `method_id`s**; adds `namespace`, `leaf`, `fq_name`, `c_surface`, `legacy_ids` (no mass ID rename → no needless fixture/doc churn). Parity fixtures are NOT numerically regenerated unless an algorithm changed.
