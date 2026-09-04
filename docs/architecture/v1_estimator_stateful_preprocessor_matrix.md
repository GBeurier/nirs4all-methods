# V1 capability matrix

This generated matrix is limited to code evidence. Native rows require a public declaration and all Linux, macOS, and Windows ABI snapshots; adapters name their FFI declarations and call path.

```bash
python3 docs/_extras/generate_v1_capability_matrix.py --check
```

Input digest (not a Git revision): `349fccdfc09a37d84c8396c35f0008b0f50c624fa2ffbaea1b13402a78b36d10`.

| Scope | Classification | Count |
| --- | --- | ---: |
| extension_boundary | refused | 1 |
| generic_model_api | native | 1 |
| model | adapter | 1 |
| model | native | 36 |
| optimization_boundary | not-promised | 1 |
| stateful_preprocessor | native | 34 |

The model universe is exactly the `category: models` catalog entries plus the generic `n4m_model_fit` algorithm routes. Runtime-extension and HPO candidates are derived from public/catalog names; they do not create an unsupported capability claim.

## Classified catalog models

`models.classification.pls_lda`, `models.classification.pls_logistic`, `models.classification.pls_qda`, `models.ensembles.bagging_pls`, `models.ensembles.boosting_pls`, `models.ensembles.moment_stack`, `models.ensembles.random_subspace_pls`, `models.heads.pls_cox`, `models.heads.pls_glm`, `models.local.lw_pls`, `models.multiblock.mb_pls`, `models.multiblock.mir_pls`, `models.multiblock.o2pls`, `models.multiblock.on_pls`, `models.multiblock.rosa`, `models.multiblock.so_pls`, `models.pls.cppls`, `models.pls.kernel`, `models.pls.pcr`, `models.pls.pls_fit_simple`, `models.regularized.continuum_regression`, `models.regularized.ridge`, `models.regularized.ridge_pls`, `models.regularized.robust_pls`, `models.regularized.weighted_pls`, `models.sparse.fused_sparse_pls`, `models.sparse.group_sparse_pls`, `models.sparse.sparse_pls_da`, `models.sparse.sparse_simpls`, `models.specialized.ecr`, `models.specialized.gpr_pls`, `models.specialized.missing_aware_nipals`, `models.specialized.recursive`, `models.specialized.tensor_pls`, `models.transfer.di_pls`, `models.transfer.ds`, `models.transfer.pds`

## Stateful preprocessors

`decomposition_flexible_pca`, `decomposition_flexible_svd`, `domain_adaptation_direct_standardization`, `domain_adaptation_epo`, `domain_adaptation_piecewise_direct_standardization`, `domain_adaptation_robust_direct_standardization`, `domain_adaptation_slope_bias`, `feature_selection_correlation`, `feature_selection_interval_generator`, `feature_selection_variance`, `outlier_detection_high_leverage`, `outlier_detection_x_outlier`, `outlier_detection_y_outlier`, `transform_baseline_center`, `transform_cow_align`, `transform_derivative`, `transform_dtw_align`, `transform_emsc`, `transform_icoshift_align`, `transform_kbins_discretizer`, `transform_local_centering`, `transform_localized_msc`, `transform_log_transform`, `transform_msc`, `transform_osc`, `transform_piecewise_msc`, `transform_piecewise_snv`, `transform_resampler`, `transform_saps`, `transform_vsn`, `transform_wavelet_pca`, `transform_wavelet_svd`, `transform_weighted_snv`, `transform_xcorr_align`
