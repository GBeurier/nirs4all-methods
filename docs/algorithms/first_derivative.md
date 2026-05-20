# FirstDerivative

## Algorithm

Numerical first derivative along the wavelength axis with second-order
accuracy at the edges, mirroring `np.gradient(X, delta, axis=1,
edge_order)`:

$$
\hat{X}_{i,j} = \begin{cases}
\dfrac{X_{i, j+1} - X_{i, j-1}}{2 \delta} & 1 \leq j \leq p - 2 \\[4pt]
\dfrac{-3 X_{i, 0} + 4 X_{i, 1} - X_{i, 2}}{2 \delta} & j = 0,\;\text{edge\_order} = 2 \\[4pt]
\dfrac{X_{i, 1} - X_{i, 0}}{\delta} & j = 0,\;\text{edge\_order} = 1 \\
\end{cases}
$$

(and symmetrically at the right edge).  Output shape matches input shape
(shape-preserving).

## Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `delta`      | $\neq 0$ | Sample spacing.   |
| `edge_order` | $\in \{1, 2\}$ | Boundary accuracy. |

## Numerical contract

- `_create` returns `N4M_ERR_INVALID_ARGUMENT` for $\delta = 0$ or an
  edge order outside $\{1, 2\}$.
- `_transform` requires `cols >= 3` for `edge_order = 2` and `cols >= 2`
  for `edge_order = 1`.
- Tolerance against `np.gradient`: $10^{-12}$ absolute / $10^{-13}$
  relative — pure arithmetic over the input samples.

## Reference

Documentation: [`numpy.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).
