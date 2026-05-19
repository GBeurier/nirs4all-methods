# `composite_filter` — CompositeFilter

_Group_: **Filter** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/composite_filter.md`](../algorithms/composite_filter.md)

## Description

Aggregates the keep-masks of any number of sub-filters via either ANY or ALL semantics. The composite does NOT replay sub-filter algorithms — it holds references to the sub-filter handles, asks each one for its mask, and combines the masks.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- nirs4all 0.8.x — `nirs4all.operators.filters.CompositeFilter` in `base.py`
- `parity/python_generator/src/c4a_parity_filters_ref/composite.py`

### Mathematical principle

`composite_filter` follows the standard filter operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

Composite is purely a boolean aggregator over already-computed sub-filter masks. The output is exact (mask equality with the frozen reference) since no float arithmetic is involved.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
c4a_*
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
