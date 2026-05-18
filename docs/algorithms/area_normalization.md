# Area Normalization

## Algorithm

Per-spectrum normalisation by a row-area metric. For each row $x_i \in \mathbb{R}^m$:

$$
A_i = \text{area}(x_i), \qquad
x'_{i,j} = \frac{x_{i,j}}{A_i}
$$

where the area metric is one of:

| Method | Formula | Notes |
| --- | --- | --- |
| `sum` | $A_i = \sum_j x_{i,j}$ | Signed total intensity. |
| `abs_sum` | $A_i = \sum_j |x_{i,j}|$ | Absolute intensity (robust to mean-centered data). |
| `trapz` | $A_i = \frac{1}{2} \sum_{j=0}^{m-2} (x_{i,j} + x_{i,j+1})$ | Trapezoidal integration with unit spacing. |

If $|A_i| < 10^{-10}$, the divisor is replaced by $1.0$ to avoid amplifying noise on near-zero rows.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `method` | `sum` | One of `{sum, abs_sum, trapz}`. |

## Numerical contract

* `sum` / `abs_sum` accumulate left-to-right with the same rounding order as `numpy.add.reduce`.
* `trapz` matches `numpy.trapz` / `scipy.integrate.trapezoid` with default $\Delta x = 1$, computed as $\frac{1}{2} \sum (x[:-1] + x[1:])$ — the pair-sum form is preserved bit-for-bit.
* Final scaling uses `x / area`, not `x * (1/area)`.

## Reference

Implementation matches `nirs4all.operators.transforms.nirs.AreaNormalization`. The `trapz` integrand follows the implementation used by `numpy.trapz` (which is bit-identical to `scipy.integrate.trapezoid` for unit spacing).
