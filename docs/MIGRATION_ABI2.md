# Migrating to ABI 2.0 — the `n4m.<role>` namespace

ABI 2.0 is a **clean break**. Every public method symbol, public C header, and
Python import path moved onto the ML/DL `n4m.<role>` namespace. There are **no
runtime aliases and no compatibility headers** — old names are gone. This guide
maps the old surface to the new one.

Authoritative sources:

- `proposals/namespace/_target_ml_table.tsv` — the per-method namespace,
  `leaf`, and `fq_name` (209 methods, 12 roles, 49 leaves).
- `proposals/namespace/_rename_map.tsv` — the deterministic
  `old_symbol → new_symbol` table (566 method symbols, 0 collisions), generated
  from `catalog/methods.yaml`.
- [ABI reference](abi/reference.md) · [ABI changes log](abi/changes_log.md).

## What changed

- **SONAME**: `libn4m.so.1` → `libn4m.so.2`. `N4M_ABI_VERSION_*` is `2.0.0`;
  the Linux version-script node went `N4M_1 → N4M_2`.
- **566 method symbols renamed**; **136 infra symbols unchanged**; **10
  Python-only methods** export no C symbol.
- **Numerics are unchanged** — this is a surface rename only. Parity fixtures
  were not regenerated.

## C symbols

Convention: `n4m_<role>_<leaf><tail>`. `role` is the top-level namespace token
(`augmentation`, `compose`, `decomposition`, `domain_adaptation`, `ensemble`,
`estimators`, `feature_selection`, `lowlevel`, `metrics`, `model_selection`,
`outlier_detection`, `transform`); `leaf` is the catalog leaf; `<tail>` is the
ABI-1 operation suffix preserved byte-for-byte (`_fit`, `_predict`, `_create`,
`_destroy`, `_transform`, `_select`, `_split`, `_apply`, `_run`, `_compute`,
`_result_get_*`, …).

| Old symbol | New symbol |
|---|---|
| `n4m_ridge_fit` | `n4m_estimators_ridge_fit` |
| `n4m_pls_fit_simple` | `n4m_estimators_pls_fit` |
| `n4m_pls_lda_fit` | `n4m_estimators_pls_lda_fit` |
| `n4m_sparse_pls_da_fit` | `n4m_estimators_sparse_pls_da_fit` |
| `n4m_pp_snv_create` / `_transform` | `n4m_transform_snv_create` / `_transform` |
| `n4m_pp_msc_transform` | `n4m_transform_msc_transform` |
| `n4m_pp_epo_fit` | `n4m_domain_adaptation_epo_fit` |
| `n4m_pp_flex_pca_transform` | `n4m_decomposition_flexible_pca_transform` |
| `n4m_aug_gaussian_noise_apply` | `n4m_augmentation_gaussian_noise_apply` |
| `n4m_cars_select` | `n4m_feature_selection_cars_select` |
| `n4m_wvc_threshold_select` | `n4m_feature_selection_wvc_threshold_select` |
| `n4m_filter_correlation_fit` | `n4m_feature_selection_correlation_fit` |
| `n4m_split_kennard_stone_split` | `n4m_model_selection_kennard_stone_split` |
| `n4m_sweep_run` | `n4m_model_selection_sweep_run` |
| `n4m_aom_global_result_get_best_score` | `n4m_model_selection_aom_pls_result_get_best_score` |
| `n4m_aom_ridge_blender_fit` | `n4m_ensemble_aom_ridge_blender_fit` |
| `n4m_transfer_metrics_compute` | `n4m_domain_adaptation_transfer_metrics_compute` |
| `n4m_util_hotelling_t2` | `n4m_outlier_detection_hotelling_t2` |
| `n4m_metric_rmse` | `n4m_metrics_regression_metrics_rmse` |
| `n4m_moments_subset_compute` | `n4m_lowlevel_moments_subset_compute` |

For the complete table, read `proposals/namespace/_rename_map.tsv`. To migrate
a binding mechanically, look each old symbol up in the `old_symbol` column and
substitute the `new_symbol` value.

## Headers

The flat header set was replaced by an umbrella + 12 role headers + second-level
subheaders. **Stop including** the deleted headers (no shim exists):
`pls.h`, `preprocessing.h`, `models.h`, `selection.h`, `splitters.h`,
`filters.h`, `diagnostics.h`, `transfer.h`, `utilities.h`, `aom_pop.h`,
`context.h`.

```c
/* before (ABI 1.x) */
#include <n4m/pls.h>
#include <n4m/preprocessing.h>

/* after (ABI 2.0): umbrella pulls in every role header */
#include <n4m/n4m.h>
/* …or include just the role(s) you need */
#include <n4m/estimators.h>
#include <n4m/transform/scatter.h>
```

Role headers: `transform.h`, `augmentation.h`, `estimators.h`,
`feature_selection.h`, `model_selection.h`, `domain_adaptation.h`,
`outlier_detection.h`, `ensemble.h`, `compose.h`, `metrics.h`,
`decomposition.h`, `lowlevel.h`. Subheaders live under
`transform/*.h`, `estimators/*.h`, `augmentation/*.h`.

## Python imports

The flat `n4m.python` / `n4m.sklearn` surface is gone. Public classes now live
in role subpackages that mirror the namespace; `n4m.__init__` exposes only
metadata/helpers and the subpackages. Use the public class names (not
`Native*`).

```python
# before (ABI 1.x)
from n4m.sklearn import SNV, Ridge, CARS, KennardStoneSplitter

# after (ABI 2.0)
from n4m.estimators.regression.regularized import Ridge, RidgePLS
from n4m.estimators.regression.latent import PLS, PCR
from n4m.transform.scatter import SNV, MSC
from n4m.transform.smoothing import SavitzkyGolay
from n4m.feature_selection.wrapper import CARS
from n4m.model_selection.splitters import KennardStone
from n4m.domain_adaptation.orthogonalization import EPO
from n4m.augmentation.noise import GaussianAdditiveNoise
from n4m.ensemble import AOMRidgeBlender
from n4m.compose.aom_superblock import AOMRidgePLSSuperblock
from n4m.decomposition import FlexiblePCA
```

Python layout (role → subpackages):

```text
n4m/
  transform/{scatter,baseline,smoothing,wavelet,signal_conversion,resampling,scaling,alignment,orthogonalization,specialized}
  augmentation/{noise,drift,wavelength,spectral,scattering,instrument,splines,mixup}
  estimators/regression/{latent,regularized,robust,kernel,sparse,tensor,online,local,glm}
  estimators/{classification,multiblock,survival}
  feature_selection/{wrapper,filter,interval,ranking}
  model_selection/{splitters,aom_search,aom_campaign,sweep}
  domain_adaptation/{standardization,invariant,metrics,orthogonalization}
  outlier_detection/
  ensemble/
  compose/aom_superblock/
  metrics/{scoring,diagnostics}
  decomposition/
  lowlevel/moments.py
```

The slim `pls4all` package keeps its name (subset contract) but every
underlying C call now uses the ABI-2 symbols.

## R / MATLAB / JS

- **R** uses a shorter singular role prefix for the public wrappers:
  `n4m_regression_ridge()`, `n4m_transform_snv()`, `n4m_feature_select_cars()`,
  `n4m_model_selection_kennard_stone()`. The internal `.Call` names use the
  ABI-2 C symbols.
- **MATLAB/Octave** MEX entry points keep their filenames; the C calls inside
  use the ABI-2 names (e.g. `n4m_pls_fit_simple → n4m_estimators_pls_fit`).
- **JS/WASM** `EXPORTED_FUNCTIONS` is snapshot-driven; direct `ccall`/`cwrap`
  call sites use the ABI-2 names. Regenerate `dist`.

## Detecting the version at runtime

```c
if (n4m_check_abi_compatibility(2, 0) != N4M_OK) {
    /* header/runtime skew — built against a different ABI major */
}
```
