# `derivate` — Derivate

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/derivate.md`](../algorithms/derivate.md)

## Description

`derivate` is a chemometrics4all preprocessing method exposed through the C ABI and the generated language bindings.

### Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `order` | — | Derivative order $k \geq 1$. |
| `delta` | — | Wavelength spacing $\delta \neq 0$. |

A helper `c4a_pp_derivate_output_cols(order, input_cols)` returns the output
column count for the caller's pre-sizing of the output buffer. It returns 0
for degenerate cases (`order >= input_cols` or `input_cols <= 0`).

## Explanations

### Bibliographic source

Savitzky, A., Golay, M. J. E. (1964). "Smoothing and Differentiation of Data
by Simplified Least Squares Procedures." *Analytical Chemistry* 36 (8),
1627-1639 — for the broader family. The current implementation is the bare
finite-difference variant; Savitzky-Golay arrives in Phase 4.

### Mathematical principle

Finite-difference derivative of order $k$ along axis 1 (the wavelength
axis), normalised by the wavelength spacing $\delta$:

$$
X' = \frac{\mathrm{np.diff}(X,\; n=k,\; \mathrm{axis}=1)}{\delta^k}.
$$

`np.diff` applies first differences along the chosen axis; `n=k` repeats
the operation $k$ times, shortening the column count by one each pass.
Output shape:

$$
X' \in \mathbb{R}^{n \times (p - k)}.
$$

For $k = 1$, $X'_{i,j} = (X_{i, j+1} - X_{i, j}) / \delta$.
For $k = 2$, $X''_{i,j} = (X_{i, j+2} - 2 X_{i, j+1} + X_{i, j}) / \delta^2$.

Implementation: a single scratch buffer of length $p$ holds the row across
the $k$ passes. Each pass scans left-to-right and overwrites entries in
place — `scratch[j] := scratch[j+1] - scratch[j]`. After $k$ passes the
first $p - k$ slots hold the $k$-th differences, and we divide each by the
precomputed scalar $\delta^k$.

### Implementation

Mathematically stateless, but exposes the stateful contract
(`_create / _fit / _transform / _destroy`) to fit uniformly with MSC,
EMSC and Baseline. `_fit` records the input column count and marks the
state fitted; calling `_transform` before `_fit` returns
`C4A_ERR_NOT_FITTED`.

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
