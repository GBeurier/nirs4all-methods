# Derivate (Finite-Difference Derivative)

## Algorithm

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

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `order` | — | Derivative order $k \geq 1$. |
| `delta` | — | Wavelength spacing $\delta \neq 0$. |

A helper `c4a_pp_derivate_output_cols(order, input_cols)` returns the output
column count for the caller's pre-sizing of the output buffer. It returns 0
for degenerate cases (`order >= input_cols` or `input_cols <= 0`).

## Lifecycle

Mathematically stateless, but exposes the stateful contract
(`_create / _fit / _transform / _destroy`) to fit uniformly with MSC,
EMSC and Baseline. `_fit` records the input column count and marks the
state fitted; calling `_transform` before `_fit` returns
`C4A_ERR_NOT_FITTED`.

## Algorithm-vs-nirs4all note

The c4a Derivate operator implements `np.diff(X, n=order, axis=1) /
delta**order` per the Phase 3 contract. The nirs4all class
`nirs4all.operators.transforms.scalers.Derivate` (as of 0.8.x) uses
`np.gradient(X, delta, axis=0)` — a different operator that preserves the
column count and differs from the c4a definition at the boundaries. Phase 3
locks in the `np.diff(..., axis=1)` variant since it is the convention used
in the rest of the chemometrics literature for spectroscopic derivatives.

## Numerical contract

* `_fit` requires `cols > order`. Returns `C4A_ERR_INVALID_ARGUMENT` otherwise.
* `_transform` returns `C4A_ERR_SHAPE_MISMATCH` if the output column count
  is not exactly `cols - order`.
* Tolerance against `np.diff` + scalar division: 1e-12 absolute / 1e-13
  relative — well within the byte-equality region for typical NIRS data.

## Reference

Savitzky, A., Golay, M. J. E. (1964). "Smoothing and Differentiation of Data
by Simplified Least Squares Procedures." *Analytical Chemistry* 36 (8),
1627-1639 — for the broader family. The current implementation is the bare
finite-difference variant; Savitzky-Golay arrives in Phase 4.
