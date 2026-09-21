# `pp_kubelka_munk` — Kubelka–Munk remission transform

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_kubelka_munk_*`

## Description

KM = (1 - R)^2 / (2 R), with R guarded by epsilon.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `is_percent` | `bool` | `False` |
| `epsilon` | `float` | `1e-10` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_kubelka_munk_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L54) · [`n4m_transform_kubelka_munk_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L56) · [`n4m_transform_kubelka_munk_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L58). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.signal_conversion import KubelkaMunk
```

Source signature: [`KubelkaMunk(is_percent: bool = False, epsilon: float = 1e-10)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L519).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Kubelka & Munk (1931), *Ein Beitrag zur Optik der Farbanstriche*, Zeitschrift für Technische Physik 12, 593–601; English translation: https://doi.org/10.1002/col.5080010404.

### Mathematical principle

Fractional reflectance $R$ is converted element-wise to the infinite-layer remission function $F(R)=(1-R)^2/(2R)$. Percent input is divided by 100 first and the denominator is guarded by `epsilon`.

### Appropriate uses

Linearizing diffuse-reflectance measurements of optically thick scattering samples under the Kubelka–Munk assumptions.

### Limits and validation

The two-flux, infinite-thickness, homogeneous-sample assumptions often fail. Values near zero explode, and specular reflection or finite thickness invalidates the physical interpretation.

### Implementation

`n4m.transform.signal_conversion.KubelkaMunk` wraps `n4m_transform_kubelka_munk_*`; the guarded formula is in `signal_conversion/kubelka_munk.c`.

The ABI-2 implementation is the `n4m_transform_kubelka_munk_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1002/col.5080010404; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/kubelka_munk.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)