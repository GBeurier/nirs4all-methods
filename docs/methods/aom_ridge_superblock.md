# `ridge_superblock` — n4m.compose.aom_superblock.ridge_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.ridge_superblock` · _Catalog id_: `aom_pop.ridge_superblock`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.compose.aom_superblock import aom_ridge_superblock`

**Signature:** [`aom_ridge_superblock(X, y, *, operators = None, alpha: float | None = None, alphas: Sequence[float] = (0.0001, 0.01, 1.0, 100.0), cv: int = 5, fold_ids = None, block_scaling: str = 'rms', center_x: bool = True, center_y: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L7233)

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
| `block_scaling` | `str` | `'rms'` |
| `center_x` | `bool` | `True` |
| `center_y` | `bool` | `True` |

## Explanations

### Bibliographic source

No canonical paper defines this ABI-2 superblock wrapper. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Create $B$ branch outputs $Z_b\in\mathbb{R}^{n\times p}$, concatenate $Z=[Z_1|\cdots|Z_B]\in\mathbb{R}^{n\times Bp}$, center columns, optionally RMS-scale each block, then fit Ridge on the resulting superblock. Its penalty is selected and refit within the declared CV protocol.

### Appropriate uses

Linear calibration from multiple explicitly declared preprocessing branches.

### Limits and validation

Column scale and block width affect Ridge regularization; centering, RMS scaling and penalty selection must be fold-local. The method is restricted to the declared strict-linear branch family for raw-space replay.

### Implementation

`n4m.compose.aom_superblock.aom_ridge_superblock` and `AOMRidgeSuperblock`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

## Catalog note

Python-backed donor-style AOM Ridge superblock constrained to strict-linear single-operator AOM views. It builds operator outputs through native aom_preprocess, selects/fits Ridge alphas fold-locally through the native Ridge binding, applies train-fold centering and optional block RMS scaling to validation folds, and folds final superblock coefficients back to original-input input_coefficients plus intercept. It intentionally excludes donor branch_global, MKL/kernel, row-reference-dependent preprocessing and nonlinear AOM Ridge modes; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU superblock grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py`


_See also_: [methods index](index.md).