# `ridge_mkl_superblock` — n4m.compose.aom_superblock.ridge_mkl_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.ridge_mkl_superblock` · _Catalog id_: `aom_pop.ridge_mkl_superblock`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.compose.aom_superblock import aom_ridge_mkl_superblock`

**Signature:** [`aom_ridge_mkl_superblock(X, y, *, operators = None, alpha: float | None = None, alphas: Sequence[float] = (0.0001, 0.01, 1.0, 100.0), cv: int = 5, fold_ids = None, mkl_top_k: int = 6, block_scaling: str = 'none', center_x: bool = True, center_y: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L7414)

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
| `mkl_top_k` | `int` | `6` |
| `block_scaling` | `str` | `'none'` |
| `center_x` | `bool` | `True` |
| `center_y` | `bool` | `True` |

## Explanations

### Bibliographic source

No canonical publication defines this product-specific MKL-light surface. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

For each centered block $Z_b$, compute $K_b=Z_bZ_b^\top$ and its alignment $A_b=\langle K_b,Y_cY_c^\top\rangle_F/(\lVert K_b\rVert_F\lVert Y_cY_c^\top\rVert_F)$. The top-$k$ blocks receive simplex weights $w_b=\max(A_b,0)/\sum_b\max(A_b,0)$; multiplying each retained block by $\sqrt{w_b}$ before Ridge yields the final weighted superblock. If every retained alignment is nonpositive, the implementation uses uniform weights $1/k$ instead.

### Appropriate uses

Comparing a small collection of complementary strict spectral representations with a controlled linear head.

### Limits and validation

The weighting is not equivalent to full multiple-kernel learning. The uniform fallback makes nonpositive alignments non-discriminative. Top-$k$, weights, and Ridge penalty are selected fold-locally and refit on all rows; stateful/nonlinear branches break simple coefficient replay.

### Implementation

`n4m.compose.aom_superblock.aom_ridge_mkl_superblock` and `AOMRidgeMKLSuperblock`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

## Catalog note

Native AOM Ridge MKL-light superblock over strict-linear single-operator AOM views (core fit_aom_superblock). It learns non-negative train-only kernel-target-alignment weights over the top operator blocks inside every alpha-CV fold, refits the weights on all rows, fits Ridge on the equivalent weighted superblock, and folds the final coefficients back to original-input input_coefficients plus intercept (affine N4MM state). It excludes donor branch_global, row-reference-dependent preprocessing, nonlinear kernels and nonlinear AOM Ridge modes.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py`


_See also_: [methods index](index.md).