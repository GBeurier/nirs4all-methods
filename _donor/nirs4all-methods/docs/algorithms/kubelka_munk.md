# Kubelka-Munk

## Algorithm

Diffuse-reflectance transformation. For each element of the input matrix:

$$
R_{i,j} = \min\!\bigl(\max(\tilde{x}_{i,j},\, \varepsilon),\, 1 - \varepsilon\bigr), \qquad
F(R_{i,j}) = \frac{(1 - R_{i,j})^2}{2 R_{i,j}}, \qquad
\tilde{x}_{i,j} = \begin{cases} x_{i,j} / 100 & \text{if `is\_percent`} \\ x_{i,j} & \text{otherwise} \end{cases}
$$

The clamp keeps $R \in [\varepsilon,\, 1 - \varepsilon]$ to avoid division by zero (when $R \to 0$) and to keep the function well-defined for reflectance-like inputs that may slightly overshoot 1.

This is theoretically more appropriate for scattering media (powders) than simple $-\log_{10}(R)$, though in NIR the benefit is dataset-dependent.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `is_percent` | `false` | When set, divide input by 100 before the clamp. |
| `epsilon` | `1e-10` | Symmetric clamp target. Must satisfy `0 < epsilon < 0.5` so the clamp interval is non-empty. |

## Numerical contract

* `is_percent` divides by `100.0` element-wise.
* The clamp is two-sided symmetric: `R = clip(x, epsilon, 1 - epsilon)`. `1 - epsilon` is computed once (not per element) and compared via the `if v > upper, v = upper` branch.
* The squared term is `(1 - R) * (1 - R)` — a single multiply of the subtraction result, matching numpy's `np.square` ufunc.
* The division `(1 - R)^2 / (2 R)` is performed as a true element-wise divide; the denominator `2.0 * R` is computed inline to mirror nirs4all's operator-precedence evaluation of the literal `(1.0 - R)**2 / (2.0 * R)` expression.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

## Reference

Implementation matches `nirs4all.operators.transforms.signal_conversion.KubelkaMunk` (0.8.11).

For background see Kubelka, P. & Munk, F. (1931), "An article on optics of paint layers", Z. Tech. Phys., 12, 593-601.
