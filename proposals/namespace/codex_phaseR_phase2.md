**A. Phase R Gate**

NO-GO as-is, but the catalog data itself is correct. Required fix is small and in catalog tooling.

What verified clean:

- `methods.yaml` declares 209 methods and contains 209 actual entries.
- `_target_ml_table.tsv` has 209 unique rows; `namespace`, `leaf`, and `fq_name` match the catalog with zero mismatches.
- All 209 split files match `catalog/methods.yaml`.
- `c_surface` matches `abi_symbols` for every ABI method, and exactly 10 rows use `c_surface: "none"`.
- All 566 method C symbols are present in `cpp/abi/expected_symbols_linux.txt`; the other 136 exports are infra.
- WVC catalog split is correct in data:
  - `selection.wvc` -> `n4m_wvc_select`
  - `selection.wvc_threshold` -> `n4m_wvc_threshold_select`

Blocking issue: [catalog/abi_method_map.yaml](/home/delete/nirs4all/nirs4all-methods/catalog/abi_method_map.yaml:180) still says `selection.wvc: n4m_wvc` and has no `selection.wvc_threshold`. Today `reconcile_abi.py --check` falsely attributes both WVC symbols to `selection.wvc`. A future `reconcile_abi.py --write` would re-collapse/drop the split.

Required fix:

```yaml
selection.wvc: n4m_wvc_select
selection.wvc_threshold: n4m_wvc_threshold_select
```

Also tighten validation before Phase 2 starts. The schema `oneOf` is acceptable as a type shape, but not as a gate. Add validation that the five Phase R fields are required, `fq_name == n4m.<namespace>.<leaf>`, target-table fields match, and `c_surface == abi_symbols` unless `abi_symbols == []`, in which case `c_surface == "none"`.

**B. Locked Phase 2 C Symbol Convention**

Use the top-level namespace as the C role token. Do not use shorter semantic tokens for C. Therefore C is `n4m_estimators_ridge_fit`, not `n4m_regression_ridge_fit`. R may still choose `n4m_regression_ridge()` as an idiomatic R wrapper, but that is not the C ABI.

Role tokens:

`augmentation`, `compose`, `decomposition`, `domain_adaptation`, `ensemble`, `estimators`, `feature_selection`, `lowlevel`, `metrics`, `model_selection`, `outlier_detection`, `transform`.

Rule:

```text
old method symbol -> n4m_<top_level_role_token>_<catalog_leaf><preserved_operation_tail>
```

The deeper namespace segments are for headers/docs/bindings, not C symbol names. The generator must fail on any future `<role, leaf, tail>` collision.

Preserve operation tails byte-for-byte: `_fit`, `_predict`, `_create`, `_destroy`, `_free`, `_transform`, `_inverse_transform`, `_is_fitted`, `_select`, `_split`, `_split_fold`, `_n_splits`, `_apply`, `_run`, `_compute`, `_result_get_*`, `_output_cols`, etc. Historical irregular `n4m_pls_fit_simple` normalizes to `n4m_estimators_pls_fit`; `simple` is old implementation naming, not the canonical verb.

Worked examples:

```text
n4m_ridge_fit                          -> n4m_estimators_ridge_fit
n4m_pls_fit_simple                     -> n4m_estimators_pls_fit
n4m_pls_lda_fit                        -> n4m_estimators_pls_lda_fit
n4m_sparse_pls_da_fit                  -> n4m_estimators_sparse_pls_da_fit
n4m_pp_snv_create                      -> n4m_transform_snv_create
n4m_pp_msc_transform                   -> n4m_transform_msc_transform
n4m_pp_epo_fit                         -> n4m_domain_adaptation_epo_fit
n4m_pp_flex_pca_transform              -> n4m_decomposition_flexible_pca_transform
n4m_aug_gaussian_noise_apply           -> n4m_augmentation_gaussian_noise_apply
n4m_cars_select                        -> n4m_feature_selection_cars_select
n4m_wvc_threshold_select               -> n4m_feature_selection_wvc_threshold_select
n4m_filter_correlation_fit             -> n4m_feature_selection_correlation_fit
n4m_split_kennard_stone_split          -> n4m_model_selection_kennard_stone_split
n4m_sweep_run                          -> n4m_model_selection_sweep_run
n4m_aom_global_result_get_best_score   -> n4m_model_selection_aom_pls_result_get_best_score
n4m_aom_ridge_blender_fit              -> n4m_ensemble_aom_ridge_blender_fit
n4m_transfer_metrics_compute           -> n4m_domain_adaptation_transfer_metrics_compute
n4m_util_hotelling_t2                  -> n4m_outlier_detection_hotelling_t2
n4m_metric_rmse                        -> n4m_metrics_regression_metrics_rmse
n4m_moments_subset_compute             -> n4m_lowlevel_moments_subset_compute
```

**C. Header Set And Lockstep Files**

Role headers:

`n4m.h`, `transform.h`, `augmentation.h`, `estimators.h`, `feature_selection.h`, `model_selection.h`, `domain_adaptation.h`, `outlier_detection.h`, `ensemble.h`, `compose.h`, `metrics.h`, `decomposition.h`, `lowlevel.h`.

Second-level subheaders:

- `n4m/transform/{alignment,baseline,orthogonalization,resampling,scaling,scatter,signal_conversion,smoothing,specialized,wavelet}.h`
- `n4m/estimators/{regression,classification,multiblock,survival}.h`
- `n4m/augmentation/{drift,instrument,mixup,noise,scattering,spectral,splines,wavelength}.h`

Delete old category stubs: `aom_pop.h`, `preprocessing.h`, `models.h`, `selection.h`, `splitters.h`, `filters.h`, `diagnostics.h`, `transfer.h`, `utilities.h`, and the scaffold `context.h`. Replace current `augmentation.h` with the real role header. Delete `pls.h` from the public install set; do not leave a compatibility include.

Must change in lockstep:

- `catalog/methods.yaml`, `catalog/methods/*.yaml`, `catalog/abi_method_map.yaml`, `catalog/schema/method*.json`, `catalog/scripts/validate.py`
- `cpp/include/n4m/*.h`, plus new `cpp/include/n4m/{transform,estimators,augmentation}/*.h`
- `cpp/src/c_api/*.cpp`
- `cpp/include/n4m/n4m_version.h`: set ABI to `2.0.0`
- `cpp/src/c_api/n4m_linux.map`: `N4M_1` -> `N4M_2`
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt`
- `scripts/regen_abi_snapshots.sh`: stop hardcoding `N4M_1`; use `N4M_2` or parse current ABI node
- `.github/workflows/abi-check.yml`: SONAME `libn4m.so.2`
- `bindings/python/src/n4m/_ffi.py`, `_ffi_decls.py`, and all Python symbol-call sites under `bindings/python/src/{n4m,pls4all}`
- Active R, MATLAB/Octave, JS/WASM binding sources and generated docs/tests under `bindings/r`, `bindings/matlab`, `bindings/octave`, `bindings/js`
- `docs/abi/changes_log.md`, `docs/abi/reference.md`, `docs/MIGRATION_ABI2.md`
- `NAMESPACE_MIGRATION_LOG.md`

`cpp/src/n4m_targets.cmake` should not need a semantic edit for SONAME; it already derives `SOVERSION` from `N4M_ABI_VERSION_MAJOR`.

**D. Rename-Map Generation**

Generate, do not hand-maintain, `old_symbol -> new_symbol` from catalog:

1. Load `catalog/methods.yaml`.
2. Skip `c_surface: "none"` entirely.
3. For each method, set `role = namespace.split(".")[0]`, `leaf = method["leaf"]`.
4. For each old symbol in `c_surface`, derive the operation tail from the current symbol.
5. Emit `n4m_<role>_<leaf><tail>`.
6. Fail on ambiguous tail extraction, new-symbol collision, or any generated symbol not represented in headers/snapshots/bindings.

The 10 `c_surface: "none"` methods stay Python-only and get no C declaration, no exported symbol, no ABI snapshot entry, and no placeholder stub.

**E. Phase 2 Risks**

Top risks to watch:

- WVC generator regression unless `abi_method_map.yaml` is fixed now.
- Irregular tail extraction: `n4m_pls_fit_simple`, metric helpers, no-verb utility functions, and AOM result accessors.
- Header split source-ABI drift: public opaque handle typedefs should follow the same canonical base names; shared infra types stay as-is.
- Python/R/JS/WASM/MATLAB symbol churn must land before the branch is considered green.
- `scripts/regen_abi_snapshots.sh` and `abi-check.yml` still carry ABI-1 assumptions.
- Accidentally exporting the 10 `c_surface: none` methods.
- Treating parity fixtures as numerical changes. This phase should rename surfaces only; parity data should not be regenerated for changed numbers unless an algorithm actually changes.

No files were modified.