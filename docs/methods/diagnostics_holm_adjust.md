# `holm_adjust` — n4m.metrics.multiple_testing.holm_adjust

_Namespace_: **`n4m.metrics.multiple_testing`** · _Fully-qualified_: `n4m.metrics.multiple_testing.holm_adjust` · _Catalog id_: `diagnostics.holm_adjust`

## API surface

**C ABI (ABI 2):** [`n4m_preflight_holm_adjust`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L278). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*, Scandinavian Journal of Statistics 6(2), 65–70, https://www.jstor.org/stable/4615733.

### Mathematical principle

For one family of $m$ p-values sorted ascending as $p_{(1)}\le\dots\le p_{(m)}$, the adjusted value of the hypothesis of rank $i$ is $\min(1,\max_{j\le i}(m-j+1)p_{(j)})$. The sort is stable, so tied p-values keep their input order, as in R `p.adjust(method = "holm")`.

### Appropriate uses

Controlling the family-wise error rate when several Preflight factor-association tests are reported together, without assuming independence between the tests.

### Limits and validation

Holm controls the family-wise error rate, not the false discovery rate, and is conservative for many correlated tests. The caller defines the family. Inputs must lie in $[0,1]$ and $m\ge 1$.

### Implementation

The ABI-2 symbol `n4m_preflight_holm_adjust` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `preflightHolmAdjust`. R `p.adjust` reference values are pinned in `cpp/tests/test_preflight.cpp`.

### Sources and provenance

https://www.jstor.org/stable/4615733; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_preflight.cpp

## Catalog note

Holm step-down adjusted p-values for one hypothesis family, matching R p.adjust(method = "holm") with a stable ascending order.


_See also_: [methods index](index.md).