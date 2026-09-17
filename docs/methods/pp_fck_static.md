# `pp_fck_static` — Static fractional convolutional-kernel bank

_Group_: **Feature extraction** · _C ABI_: `n4m_transform_fck_static_*`

## Description

Static fractional convolutional kernel bank transformer.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `kernel_size` | `int` | `—` |
| `alphas` | `Sequence[float] \| None` | `None` |
| `sigmas` | `Sequence[float] \| None` | `None` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_fck_static_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/specialized.h#L26) · [`n4m_transform_fck_static_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/specialized.h#L31) · [`n4m_transform_fck_static_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/specialized.h#L35) · [`n4m_transform_fck_static_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/specialized.h#L32). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.specialized import FCKStaticTransformer
```

Source signature: [`FCKStaticTransformer(kernel_size: int, alphas: Sequence[float] | None = None, sigmas: Sequence[float] | None = None, *, filter_orders: Sequence[float] | None = None, filter_scales: Sequence[float] | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/feature_extraction.py#L170).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper uniquely defines this n4m operator. It is an implementation-specific bank of fractional convolutional kernels; its generated kernel equation and fixed $\sigma=3$ are defined by `fck_kernel.h` and `fck_static.c`.

### Mathematical principle

At construction, one length-`kernel_size` kernel is generated for every Cartesian pair of fractional order `alpha` and scale. Each row is convolved with all kernels using nearest-edge extension, and the bands are concatenated into `n_kernels * n_features` outputs.

### Appropriate uses

Fixed multiscale, fractional-order feature expansion before a linear or sparse model.

### Limits and validation

Output dimensionality grows multiplicatively with the kernel bank. Edge clamping and the fixed kernel parameterization may not suit every wavelength grid; it is not a learned convolutional model.

### Implementation

`n4m.transform.specialized.FCKStaticTransformer` calls `n4m_transform_fck_static_*`; kernel construction and convolution are in `preprocessing/specialized/fck_static.c`.

The ABI-2 implementation is the `n4m_transform_fck_static_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/specialized/fck_static.c; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/common/fck_kernel.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)