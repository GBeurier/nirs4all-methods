# Robust Standard Normal Variate (Robust SNV / RNV)

## Algorithm

A median-based robust variant of SNV. For each spectrum $x_i$:

$$
\mathrm{med}_i = \operatorname{median}_j x_{i,j}, \qquad
\mathrm{MAD}_i = \operatorname{median}_j |x_{i,j} - \mathrm{med}_i|
$$

$$
x'_{i,j} = \frac{x_{i,j} - \mathrm{med}_i}{k \cdot \mathrm{MAD}_i}
$$

where $k = 1.4826$ is the consistency constant that makes $k \cdot \text{MAD}$ an unbiased estimator of the standard deviation under Gaussian assumptions.

When `with_center=False`, the MAD is computed on $|x_{i,j}|$ rather than on centered residuals — this matches nirs4all's `if/if` semantics (each block reads from the current state of `X`). If $k \cdot \text{MAD}_i = 0$, the divisor is replaced by $1.0$.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `with_center` | `true` | Subtract the row median. |
| `with_scale` | `true` | Divide by `k * MAD`. |
| `k` | `1.4826` | MAD-to-std consistency constant. |

## Numerical contract

* Median computed via in-place median-of-three quickselect with Lomuto partitioning. For even sample counts the two middle order statistics are averaged via $(a + b) / 2$, matching `numpy.median`.
* The final scaling is evaluated as $(x - \text{med}) / (k \cdot \text{MAD})$ — a single division per element, matching NumPy's broadcast.

## Reference

Implementation matches `nirs4all.operators.transforms.scalers.RobustStandardNormalVariate`. The MAD consistency constant $k = 1/\Phi^{-1}(3/4) \approx 1.4826$ comes from Hampel (1974), "The Influence Curve and its Role in Robust Estimation."
