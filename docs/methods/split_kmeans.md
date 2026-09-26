# `split_kmeans` — K-means++ representative-sample split

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_kmeans_*`

## Description

K-means++ diversity splitter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `test_size` | `float` | `0.25` |
| `seed` | `int` | `0` |
| `max_iter` | `int` | `100` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_kmeans_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L124) · [`n4m_model_selection_kmeans_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L127) · [`n4m_model_selection_kmeans_split`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L128). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import KMeans
```

Source signature: [`KMeans(test_size: float = 0.25, seed: int = 0, max_iter: int = 100)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L268).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*, SODA, 1027–1035 (https://dl.acm.org/doi/10.5555/1283383.1283494). The surrounding representative split is implementation-specific.

### Mathematical principle

Set k to the requested training count, initialize k centroids with seeded k-means++, and run Lloyd iterations up to `max_iter`. Select the nearest observed sample to each centroid, deduplicate those indices, and use the sorted complement as test.

### Appropriate uses

Selecting observed spectra near cluster centers as a representative training subset.

### Limits and validation

Deduplication can yield fewer training samples than requested when centroids choose the same row. Results depend on scaling and seed; the native PCG64 implementation is not scikit-learn KMeans.

### Implementation

Python role API `n4m.model_selection.splitters.KMeans`; binding class `KMeansSplitter`; ABI 2 family `n4m_model_selection_kmeans_{create,split,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_kmeans_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/kmeans.h; https://dl.acm.org/doi/10.5555/1283383.1283494


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)