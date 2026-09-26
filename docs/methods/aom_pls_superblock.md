# `aom_pls_superblock` — n4m.compose.aom_superblock.aom_pls_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.aom_pls_superblock` · _Catalog id_: `aom_pop.aom_pls_superblock`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.compose.aom_superblock import aom_pls_superblock`

**Signature:** [`aom_pls_superblock(X, y, *, operators = None, n_components: int = 2, pls_components: Sequence[int] | None = None, cv: int = 5, fold_ids = None, block_scaling: str = 'rms', center_x: bool = True, center_y: bool = True, cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L7838)

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
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `block_scaling` | `str` | `'rms'` |
| `center_x` | `bool` | `True` |
| `center_y` | `bool` | `True` |
| `cuda_pls_parallel_folds` | `bool \| None` | `None` |
| `cuda_pls_min_device_features` | `int \| None` | `None` |
| `cuda_pls_many_batched` | `bool \| None` | `None` |

## Explanations

### Bibliographic source

No publication specifies this ABI-2 superblock wrapper; it composes the AOM-PLS family. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

For $B$ declared operators, form outputs $Z_b\in\mathbb{R}^{n\times p}$ and concatenate $Z=[Z_1|\cdots|Z_B]\in\mathbb{R}^{n\times Bp}$. Columns are centered and each block can be RMS-scaled before PLS is fitted on $Z$; the component-count grid is evaluated fold-locally and the selected head is refit on the full training rows.

### Appropriate uses

Multi-block spectroscopy where distinct preprocessing branches are intentionally retained as model inputs.

### Limits and validation

Block construction changes the feature geometry and can overweight blocks with more columns. Centering, RMS scaling, and component selection must be fold-local; raw-space fold-back is only valid for the declared strict-linear branches.

### Implementation

`n4m.compose.aom_superblock.aom_pls_superblock` and `AOMPLSSuperblock`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

## Catalog note

Python-backed donor-style AOM-PLS superblock constrained to strict-linear single-operator AOM views. It builds concatenated native aom_preprocess operator outputs, selects the PLS component count by train CV over the superblock, fits through the native PLS binding, and folds final superblock coefficients back to original-input input_coefficients plus intercept. It intentionally excludes row-reference-dependent preprocessing, nonlinear lifts and dataset/source routing; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU PLS superblock grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_pls_superblock_timing.py`


_See also_: [methods index](index.md).