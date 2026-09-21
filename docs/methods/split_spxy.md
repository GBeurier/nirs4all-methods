# `split_spxy` — SPXY joint X–Y maximin split

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_spxy_*`

## Description

SPXY (Sample set Partitioning based on X and Y) train/test split.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `test_size` | `float` | `0.25` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_spxy_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L46) · [`n4m_model_selection_spxy_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L48) · [`n4m_model_selection_spxy_split`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L49). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import SPXY
```

Source signature: [`SPXY(test_size: float = 0.25)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L116).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Galvão et al. (2005), *A method for calibration and validation subset partitioning*, Talanta 67, 736–740, DOI 10.1016/j.talanta.2005.03.025 (https://doi.org/10.1016/j.talanta.2005.03.025).

### Mathematical principle

Compute Euclidean pairwise matrices $D_X$ and $D_Y$, normalize each by its maximum, then form $D=D_X/\max D_X+D_Y/\max D_Y$. Apply the Kennard–Stone farthest-pair then maximin sequence to D. Y may have one or several columns.

### Appropriate uses

Calibration/validation partitioning that covers both spectra and reference-value space.

### Limits and validation

It uses all y values to design the split, so it is unsuitable for a final blind test and can be optimistic. Euclidean geometry remains scale-sensitive within multicolumn X or Y, and full distances are quadratic. In the current Python binding, univariate y must be passed explicitly as shape `(n_samples, 1)`; a 1-D array is promoted to one row and fails the X/Y row check.

### Implementation

Python role API `n4m.model_selection.splitters.SPXY`; binding class `SPXYSplitter`; ABI 2 family `n4m_model_selection_spxy_{create,split,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_spxy_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/spxy.h; https://doi.org/10.1016/j.talanta.2005.03.025


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)