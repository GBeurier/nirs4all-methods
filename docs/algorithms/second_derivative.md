# SecondDerivative

## Algorithm

Numerical second derivative along the wavelength axis as the composition
of two `np.gradient` passes:

$$
\hat{X}^{(2)} = \nabla\big(\nabla(X)\big) \quad \text{along axis=1, with } \delta, \text{edge\_order}.
$$

This is the operator nirs4all uses by default for the SecondDerivative
preprocessing.  Shape-preserving; the second pass reuses the FirstDerivative
edge formula on the already-differentiated row.

## Parameters

Same as [FirstDerivative](first_derivative.md): `delta` $\neq 0$ and
`edge_order` $\in \{1, 2\}$.

## Numerical contract

- `_transform` requires `cols >= 3` for `edge_order = 2` (since two
  passes still need a 3-element interior).
- Tolerance: $10^{-12}$ absolute / $10^{-13}$ relative.

## Reference

Documentation: [`numpy.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).
