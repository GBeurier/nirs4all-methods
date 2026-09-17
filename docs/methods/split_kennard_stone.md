# `split_kennard_stone` — Kennard–Stone maximin calibration split

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_kennard_stone_*`

## Description

Kennard-Stone train/test split.

<details>
<summary>Full binding docstring</summary>

```text
Kennard-Stone train/test split.

Picks the most diverse samples for the training set, in descending order
of pairwise Euclidean distance.
```
</details>

## Parameters

| Name | Type | Default |
|------|------|---------|
| `test_size` | `float` | `0.25` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_kennard_stone_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L37) · [`n4m_model_selection_kennard_stone_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L39) · [`n4m_model_selection_kennard_stone_split`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L40). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import KennardStone
```

Source signature: [`KennardStone(test_size: float = 0.25)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L82).

**R (source-verified):** [`kennard_stone_split(X, test_size = 0.25, zero_based = FALSE)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/preprocessing.R).

```r
library(n4m)
result <- kennard_stone_split(X)
```

**MATLAB / Octave (source-verified):** [`kennard_stone_split(X, varargin)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/kennard_stone_split.m).

```matlab
addpath('bindings/matlab')
result = n4m.kennard_stone_split(X);
```

## Explanations

### Bibliographic source

Kennard & Stone (1969), *Computer Aided Design of Experiments*, Technometrics 11, 137–148, DOI 10.1080/00401706.1969.10490666 (https://doi.org/10.1080/00401706.1969.10490666).

### Mathematical principle

Compute all Euclidean distances in X, initialize the training set with the globally farthest pair, then repeatedly add the unselected sample maximizing its minimum distance to the selected set. For n samples, `test_size` gives $n_{test}=\lceil n\,test\_size\rceil$ and the remaining maximin points are training.

### Appropriate uses

Constructing a calibration set that covers X-space while reserving the complement for testing.

### Limits and validation

It is deterministic, quadratic in memory/time for distances, sensitive to feature scaling, and ignores y. The test complement is not an IID random sample.

### Implementation

Python role API `n4m.model_selection.splitters.KennardStone`; binding class `KennardStoneSplitter`; ABI 2 family `n4m_model_selection_kennard_stone_{create,split,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_kennard_stone_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/kennard_stone.h; https://doi.org/10.1080/00401706.1969.10490666


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)