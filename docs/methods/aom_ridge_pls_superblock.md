# `aom_ridge_pls_superblock` — n4m.compose.aom_superblock.aom_ridge_pls_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.aom_ridge_pls_superblock` · _Catalog id_: `aom_pop.aom_ridge_pls_superblock`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.compose.aom_superblock import aom_ridge_pls_superblock`

**Signature:** [`aom_ridge_pls_superblock(X, y, *, operators = None, n_components: int = 2, pls_components: Sequence[int] | None = None, ridge_lambda: float | None = None, ridge_lambdas: Sequence[float] = (0.0, 0.1, 1.0, 10.0), cv: int = 5, fold_ids = None, block_scaling: str = 'rms', center_x: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8023)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `operators` | `—` | `None` |
| `n_components` | `int` | `2` |
| `pls_components` | `Sequence[int] \| None` | `None` |
| `ridge_lambda` | `float \| None` | `None` |
| `ridge_lambdas` | `Sequence[float]` | `(0.0, 0.1, 1.0, 10.0)` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `block_scaling` | `str` | `'rms'` |
| `center_x` | `bool` | `True` |

## Explanations

### Bibliographic source

No canonical publication defines this combined superblock head. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Build $Z=[Z_1|\cdots|Z_B]\in\mathbb{R}^{n\times Bp}$ from strict operator outputs, center columns, and optionally RMS-scale each block. Ridge-PLS is then fitted on $Z$ while CV evaluates the declared component-count and $\lambda$ grids; the winning preprocessing and head are refit on all training rows.

### Appropriate uses

Correlated multi-branch spectral representations where a linear, coefficient-exporting head is required.

### Limits and validation

The additional Ridge penalty and number of components must be selected fold-locally without using a test set. Only the declared strict-linear branches support raw-space replay; block scaling changes the penalty geometry.

### Implementation

`n4m.compose.aom_superblock.aom_ridge_pls_superblock` and `AOMRidgePLSSuperblock`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

## Catalog note

Python-backed donor-style AOM Ridge-PLS superblock constrained to strict-linear single-operator AOM views. It builds concatenated native aom_preprocess operator outputs, selects the PLS component count and Ridge-PLS lambda by train CV over the superblock, fits through the native ridge_pls binding, and folds final superblock coefficients back to original-input input_coefficients plus intercept. It intentionally excludes row-reference-dependent preprocessing, nonlinear lifts, MKL/kernel modes and dataset/source routing; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU Ridge-PLS superblock grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py`


_See also_: [methods index](index.md).