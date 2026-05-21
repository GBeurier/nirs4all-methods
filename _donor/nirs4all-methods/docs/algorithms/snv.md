# Standard Normal Variate (SNV)

## Algorithm

Per-sample (row-wise) centering and scaling. For each spectrum $x_i \in \mathbb{R}^m$:

$$
\bar{x}_i = \frac{1}{m} \sum_{j=1}^{m} x_{i,j}, \qquad
s_i = \sqrt{\frac{1}{m - \text{ddof}} \sum_{j=1}^{m} (x_{i,j} - \bar{x}_i)^2}
$$

$$
x'_{i,j} = \frac{x_{i,j} - \bar{x}_i}{s_i}
$$

When the row standard deviation falls below $10^{-15}$ (a flat or quasi-flat spectrum) the divisor is replaced by $1.0$ to avoid amplifying numerical noise — this mirrors `np.std`-then-`std[std == 0] = 1.0` in nirs4all.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `with_mean` | `true` | Subtract the row mean. |
| `with_std`  | `true` | Divide by the row standard deviation. |
| `ddof`      | `0`    | Delta degrees of freedom for std (`0` = population, `1` = sample). |

## Numerical contract

* Bit-exact against `numpy.mean(X, axis=1, keepdims=True)` followed by `numpy.std(X, axis=1, ddof=ddof, keepdims=True)`.
* Both `mean = sum / N` and `(X - mean) / std` are evaluated as true element-wise divisions (not by reciprocal-multiplication) to match NumPy's rounding sequence.

## Reference

Barnes, R. J., Dhanoa, M. S., Lister, S. J. (1989). "Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra." *Applied Spectroscopy* 43 (5), 772-777. Implementation matches `nirs4all.operators.transforms.scalers.StandardNormalVariate`.
