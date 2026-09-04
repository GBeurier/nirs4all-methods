# ABI — Reference

The public C ABI is **libn4m, ABI 2.5.0** (`SONAME libn4m.so.2`). It is
`extern "C"` only — no STL, no Eigen, no smart pointers, no exceptions cross
the boundary. The single source of truth for the surface is the header tree
under `cpp/include/n4m/` and the per-platform symbol snapshots in
`cpp/abi/expected_symbols_{linux,macos,windows}.txt`.

## Header set (ABI 2.x)

The flat per-category headers were removed in the 2.0 namespace clean break.
The surface is now an umbrella header plus the 12 top-level **role** headers,
the public optimizer header and their second-level subheaders, mirroring the
`n4m.<role>` namespace tree.

- `n4m/n4m.h` — umbrella. Shared infrastructure (context, config, matrix view,
  RNG, model, pipeline, validation plan, method-result, serialization, array,
  backends) and `#include`s of every role header.
- Role headers: `n4m/transform.h`, `n4m/augmentation.h`, `n4m/estimators.h`,
  `n4m/feature_selection.h`, `n4m/model_selection.h`, `n4m/domain_adaptation.h`,
  `n4m/outlier_detection.h`, `n4m/ensemble.h`, `n4m/compose.h`, `n4m/metrics.h`,
  `n4m/decomposition.h`, `n4m/lowlevel.h`.
- `n4m/optimization.h` — optimizer and HPO configuration, observation,
  persistence and ask/tell lifecycle.
- Second-level subheaders:
  - `n4m/transform/{alignment,baseline,orthogonalization,resampling,scaling,scatter,signal_conversion,smoothing,specialized,wavelet}.h`
  - `n4m/estimators/{regression,classification,multiblock,survival}.h`
  - `n4m/augmentation/{drift,instrument,mixup,noise,scattering,spectral,splines,wavelength}.h`

The deleted headers (do **not** include them — there is no compatibility
shim): `pls.h`, `preprocessing.h`, `models.h`, `selection.h`, `splitters.h`,
`filters.h`, `diagnostics.h`, `transfer.h`, `utilities.h`, `aom_pop.h`,
`context.h`.

## Symbol convention

Every public **method** symbol follows
`n4m_<role>_<leaf><tail>`, where `role` is the top-level namespace token,
`leaf` is the catalog leaf, and `<tail>` is the operation suffix preserved
byte-for-byte from ABI 1.x (`_fit`, `_predict`, `_create`, `_destroy`,
`_transform`, `_inverse_transform`, `_is_fitted`, `_select`, `_split`,
`_apply`, `_run`, `_compute`, `_result_get_*`, …).

Examples:

| Old (ABI 1.x) | New (ABI 2.x) |
|---|---|
| `n4m_ridge_fit` | `n4m_estimators_ridge_fit` |
| `n4m_pls_fit_simple` | `n4m_estimators_pls_fit` |
| `n4m_pp_snv_create` / `_transform` | `n4m_transform_snv_create` / `_transform` |
| `n4m_pp_epo_fit` | `n4m_domain_adaptation_epo_fit` |
| `n4m_aug_gaussian_noise_apply` | `n4m_augmentation_gaussian_noise_apply` |
| `n4m_cars_select` | `n4m_feature_selection_cars_select` |
| `n4m_split_kennard_stone_split` | `n4m_model_selection_kennard_stone_split` |
| `n4m_util_hotelling_t2` | `n4m_outlier_detection_hotelling_t2` |
| `n4m_metric_rmse` | `n4m_metrics_regression_metrics_rmse` |
| `n4m_moments_subset_compute` | `n4m_lowlevel_moments_subset_compute` |

The 136 **infra** symbols (context / config / matrix view / RNG / model /
pipeline / validation plan / method-result / serialization / array / backends)
keep their ABI-1 names. The 10 Python-only methods (`c_surface: "none"` in the
catalog) export no C symbol.

The full deterministic `old_symbol → new_symbol` table is generated from
`catalog/methods.yaml` and lives at `proposals/namespace/_rename_map.tsv`
(566 method symbols, 0 collisions). The migration guide is
[MIGRATION_ABI2](../MIGRATION_ABI2.md).

## Versioning

`N4M_ABI_VERSION_{MAJOR,MINOR,PATCH}` in `cpp/include/n4m/n4m_version.h` is
`2.5.0` and is independent of the project version. `N4M_ABI_VERSION_MAJOR`
drives the SONAME (`libn4m.so.2`) and the Linux version-script node (`N4M_2`).
Bindings detect header/runtime skew with
`n4m_check_abi_compatibility(header_major, header_minor)`.

ABI `2.0.0` was the namespace-breaking baseline. Releases `2.1.0` through
`2.5.0` are additive minor revisions covering optimizer/HPO state, terminal
lifecycle, imported affine predictors, fitted-model inspection and typed N4MM
pipeline inspection. See the [changes log](changes_log.md) for the exact symbol
and structure history.
