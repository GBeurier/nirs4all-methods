# Simple Scale

## Algorithm

Column-wise (axis=0) min-max scaling to the $[0, 1]$ range. For each column $j$:

$$
\mu_j = \min_i x_{i,j}, \qquad
M_j = \max_i x_{i,j}, \qquad
x'_{i,j} = \frac{x_{i,j} - \mu_j}{M_j - \mu_j}
$$

This is the special case of [Normalize](normalize.md) with `feature_range = (0, 1)` written as a dedicated operator for performance and API symmetry.

## Parameters

None — the transform is fully data-driven.

## Numerical contract

* Per-column min/max found via a single left-to-right scan.
* Final scaling uses `(x - min) / (max - min)` — a single subtraction and a single division per element, matching nirs4all.

## Reference

Implementation matches `nirs4all.operators.transforms.scalers.SimpleScale`.
