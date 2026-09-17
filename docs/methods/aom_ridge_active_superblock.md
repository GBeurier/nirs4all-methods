# `ridge_active_superblock` — n4m.compose.aom_superblock.ridge_active_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.ridge_active_superblock` · _Catalog id_: `aom_pop.ridge_active_superblock`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.compose.aom_superblock import aom_ridge_active_superblock`

**Signature:** [`aom_ridge_active_superblock(X, y, *, operators = None, alpha: float | None = None, alphas: Sequence[float] = (0.0001, 0.01, 1.0, 100.0), cv: int = 5, fold_ids = None, active_top_m: int = 20, active_diversity_threshold: float = 0.98, active_score_method: str = 'norm', active_max_per_family: int | None = None, keep_identity: bool = True, block_scaling: str = 'rms', center_x: bool = True, center_y: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L7576)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `operators` | `—` | `None` |
| `alpha` | `float \| None` | `None` |
| `alphas` | `Sequence[float]` | `(0.0001, 0.01, 1.0, 100.0)` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `active_top_m` | `int` | `20` |
| `active_diversity_threshold` | `float` | `0.98` |
| `active_score_method` | `str` | `'norm'` |
| `active_max_per_family` | `int \| None` | `None` |
| `keep_identity` | `bool` | `True` |
| `block_scaling` | `str` | `'rms'` |
| `center_x` | `bool` | `True` |
| `center_y` | `bool` | `True` |

## Explanations

### Bibliographic source

No canonical paper defines this active-superblock variant. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Form centered (optionally RMS-scaled) blocks $Z_b$. With centered response $Y_c$, each candidate has signature $s_b=c_bZ_b^\top Y_c$ and default activity score $\lVert s_b\rVert_F^2$ (KTA and blend scores are optional). The selector keeps top-$M$ candidates while pruning highly correlated signatures, enforcing family caps/identity, then fits Ridge on the selected concatenated superblock.

### Appropriate uses

Sparse or constrained superblock workflows that need a linear Ridge prediction surface and branch diagnostics.

### Limits and validation

Activity, signature-correlation pruning, family caps, and Ridge penalty are refit inside each fold. This is not a substitute for external validation and cannot fold non-linear or stateful branches into raw-space coefficients.

### Implementation

`n4m.compose.aom_superblock.aom_ridge_active_superblock` and `AOMRidgeActiveSuperblock`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

## Catalog note

Python-backed donor-style AOM Ridge active-superblock constrained to strict-linear single-operator AOM views. It screens operators fold-locally during alpha CV using response signatures computed from native aom_preprocess outputs, refits the active subset on the full calibration set, and folds final superblock coefficients back to original-input input_coefficients plus intercept. It intentionally excludes donor branch_global, MKL/kernel, row-reference-dependent preprocessing and nonlinear AOM Ridge modes; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU active-superblock grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_active_superblock_timing.py`


_See also_: [methods index](index.md).