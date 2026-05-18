# Local Standard Normal Variate (LSNV)

## Algorithm

A sliding-window variant of SNV that normalises each sample with statistics computed in a local neighbourhood along the wavelength axis. With odd window size $w = 2h + 1$, for each spectrum $x_i$ and each feature index $j$:

$$
\mu_{i,j} = \frac{1}{w} \sum_{k = j - h}^{j + h} \tilde{x}_{i,k}, \qquad
\sigma_{i,j} = \sqrt{\max\!\left(\frac{1}{w} \sum_{k = j - h}^{j + h} \tilde{x}_{i,k}^2 - \mu_{i,j}^2, \; 0\right)}
$$

$$
x'_{i,j} = \frac{x_{i,j} - \mu_{i,j}}{\sigma_{i,j}}
$$

where $\tilde{x}_i$ is the row padded by $h$ samples on each side. Boundary padding follows scipy's convention: **reflect** (mirror without repeating the edge, iterating when the pad exceeds $m - 1$), **edge** (repeat the boundary sample), or **constant** (a user-supplied value).

Variance is clamped at zero to absorb the cancellation that arises when $\mu^2 \approx \overline{x^2}$ on smooth spectra. If $\sigma_{i,j} = 0$ exactly, the divisor is replaced by $1.0$.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `window` | `11` | Odd integer $\geq 3$ — the sliding window size. |
| `pad_mode` | `reflect` | One of `{reflect, edge, constant}`. |
| `constant_value` | `0.0` | Pad fill value when `pad_mode = constant`. |

## Numerical contract

* Bit-exact against `np.cumsum(Xp, axis=1) / w` for both first and second moment moving averages.
* The reference reflect padding matches `np.pad(..., mode='reflect')` including the iterative reflection used when pad width exceeds $m - 1$.

## Reference

Implementation matches `nirs4all.operators.transforms.scalers.LocalStandardNormalVariate`. Sliding-window mean/variance via cumulative sums is the conventional $O(m)$ formulation used in spectral preprocessing toolkits.
