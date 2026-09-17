# `split_split_splitter` — SPlit sequential data-twinning split

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_data_twinning_*`

## Description

SPlit data-twinning splitter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `test_size` | `float` | `0.25` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_data_twinning_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L132) · [`n4m_model_selection_data_twinning_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L134) · [`n4m_model_selection_data_twinning_split`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L135). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import DataTwinning
```

Source signature: [`DataTwinning(test_size: float = 0.25, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L447).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Joseph & Vakayil (2021/2022), *SPlit: An Optimal Method for Data Splitting*, Technometrics 64, 166–176, DOI 10.1080/00401706.2021.1921037 (https://doi.org/10.1080/00401706.2021.1921037). The shipped routine is the paper-inspired sequential nearest-neighbor algorithm documented in source.

### Mathematical principle

Drop constant columns and z-score the rest. Let $r=\lfloor1/test\_size\rfloor$ and choose a seeded start. Repeatedly take the r nearest active samples to the current point, place the closest representative in the smaller twin (test), deactivate that neighborhood, and choose the next current point from the remaining set near the farthest deactivated member. Training is the ascending complement.

### Appropriate uses

Producing model-independent train/test twins intended to have similar multivariate distributions.

### Limits and validation

The class name is `SPlitSplitter`/role alias `DataTwinning`, but the native algorithm implements the SPlit-style sequential routine, not the later full Twinning package. The reciprocal conversion means arbitrary `test_size` values are quantized through integer r; seed changes the start.

### Implementation

Python role API `n4m.model_selection.splitters.DataTwinning`; binding class `SPlitSplitter`; ABI 2 family `n4m_model_selection_data_twinning_{create,split,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_data_twinning_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/split_splitter.h; https://doi.org/10.1080/00401706.2021.1921037


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)