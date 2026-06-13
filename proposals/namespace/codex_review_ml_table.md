**(A) Verdict**

Adopt Scheme B with changes. The ML/pipeline framing is the right primary namespace, but the current table should not be migrated as-is. The blockers are `estimators.regression.linear`, EPO/domain adaptation, AOM placement, several legacy leaf names, and the ABI alias strategy.

**(B) Specific Changes**

1. Replace `estimators.regression.linear` in [B.1/B.2](/home/delete/nirs4all/nirs4all-methods/proposals/namespace/NAMESPACE_PROPOSALS.md:495). Use:
   - `regression.latent`: `pls`, `pcr`, `cppls`, `continuum_regression`, `ecr`, `missing_aware_nipals`
   - `regression.regularized`: `ridge`, `ridge_pls`
   - `regression.robust`: `weighted_pls`, `robust_pls`
   - `regression.kernel`: `kernel_pls`, `gpr_pls`
   - `regression.tensor`: `n_pls`
   - `regression.online`: `recursive_pls`
   - `regression.local`: `lw_pls`
   Reason: `linear` is false for kernel/GPR/tensor/local/recursive methods and too broad for ML users.

2. Split `estimators.glm_survival` in [_ml_table.md](/home/delete/nirs4all/nirs4all-methods/proposals/namespace/_ml_table.md:76):
   - `pls_glm` -> `n4m.estimators.regression.glm`
   - `pls_cox` -> `n4m.estimators.survival`
   Reason: GLM and Cox survival are different task families.

3. Move `preprocessing.orthogonalization.epo` out of `transform.orthogonalization`; keep `osc` there. Put EPO under `domain_adaptation.orthogonalization`.
   Reason: OSC is generic supervised orthogonalization; EPO is normally found in calibration-transfer/domain-shift workflows.

4. Keep DS/PDS/direct-standardization under domain adaptation, but split the flat node:
   - `domain_adaptation.standardization`: `ds`, `pds`, `direct_standardization`, `piecewise_direct_standardization`, `robust_direct_standardization`, `slope_bias`
   - `domain_adaptation.invariant`: `di_pls`
   - `domain_adaptation.metrics`: `transfer_metrics`
   - `domain_adaptation.orthogonalization`: `epo`

5. Do not put all 18 `aom_pop.*` methods under `compose.*`.
   - `model_selection.aom_search`: `aom_sweep`, `aom_chain_sweep`, `aom_chain_fixed_fit`, `aom_chain_ridge_pls`, `aom_preprocessing`, `aom_pls`, `pop_pls`, `robust_hpo`, `ridge_global`
   - `model_selection.aom_campaign`: `aom_chain_screen_refit`, `aom_staged_chain_campaign`
   - `compose.aom_superblock`: `aom_pls_superblock`, `aom_ridge_pls_superblock`, `ridge_superblock`, `ridge_active_superblock`, `ridge_mkl_superblock`
   - `ensemble.aom`: `ridge_blender`, `operator_pls_stack`
   Reason: search/campaigns are model selection, superblocks are composition, stack/blend are ensembles.

6. Rename `model_selection.split` to `model_selection.splitters`, or expose splitters directly under `model_selection`. A singleton `.split` module reads oddly and collides semantically with the split verb.

7. Fix public leaf names before migration:
   - `pls_fit_simple` -> `pls`
   - `kernel` -> `kernel_pls`
   - `tensor_pls` -> `n_pls`
   - `recursive` -> `recursive_pls`
   - `baseline` under scaling -> `baseline_center`
   - `derivate` -> `derivative` or `finite_difference`
   - `split_splitter` -> `data_twinning`
   - `diagnostics.model_selection` -> `one_se_rule`
   Reason: this is the clean-break moment; do not fossilize implementation names.

8. Keep these Scheme-B placements unchanged: `sparse_pls_da` in classification, `pls_logistic` in classification, `flexible_pca/flexible_svd` in decomposition, and `filters.correlation/variance` in `feature_selection.filter`.

9. Replace `utils` as a public bucket. Put `utilities.sweep` under `model_selection.sweep`, put `utilities.moments` under a small `moments` or `lowlevel.moments` namespace, and leave only true helpers like `signal_type_detector` in `utils`.

10. Python: use the sklearn-shaped modules, but do not make `Native*` the canonical class names. `Native` is an implementation detail if every class calls libn4m. Use public names like `Ridge`, `KernelPLS`, `AOMSweepRegressor`, `AOMRidgeBlender`.

11. R: role-prefixed names are workable only if shorter and singular. Prefer `n4m_regression_ridge()`, `n4m_transform_snv()`, `n4m_feature_select_cars()` over `n4m_estimators_ridge()`. Keep formula/S3 wrappers idiomatic.

**(C) ABI Strategy**

Use a clean ABI break, not additive aliases.

The proposal’s additive-alias plan conflicts with the stated no-shim/no-dead-code philosophy and would add hundreds of public symbols to an already snapshot-gated ABI. The repo treats ABI drift as a release blocker in [CLAUDE.md](/home/delete/nirs4all/nirs4all-methods/CLAUDE.md:110) and [docs/ARCHITECTURE.md](/home/delete/nirs4all/nirs4all-methods/docs/ARCHITECTURE.md:68). Adding aliases preserves old names forever and doubles the surface bindings must test.

Recommendation: bump ABI major to `2.0.0`, rename symbols to the chosen canonical role names, delete terse old exports, regenerate all three snapshots, and update all bindings in the same migration. Also remove the stub category headers instead of keeping both category and role trees.

The “C verb suffix encodes kind” claim is only partly true. `_fit` covers estimators, calibration-transfer transformers, search methods, and survival heads; `_run` covers recursive/monitoring/sweep. Do not rely on suffix alone for role semantics.

**(D) Roadmap Risks**

- Reconcile catalog completeness before migration: `selection.wvc` currently carries both `n4m_wvc_select` and `n4m_wvc_threshold_select`, while docs/bindings expose `wvc_threshold_select` separately.
- Backfill the 10 zero-ABI methods and 16 `kind=null` rows before generating role headers/symbols.
- Freeze canonical IDs for `n_pls` vs `tensor_pls`, `kernel_pls` vs `kernel`, and `one_se_rule` vs `model_selection`.
- Regenerate ABI snapshots on Linux/macOS/Windows and update all FFI declarations together.
- Re-run parity/dashboard generation after namespace changes; benchmark data and docs currently use older method IDs.
- Treat downstream breakage as expected for ABI 2.0; ship a migration guide, not runtime aliases.