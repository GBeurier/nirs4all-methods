# WaveletFeatures

## Algorithm

Multi-level wavelet decomposition followed by per-band statistical
summarisation.  For each input row a multi-level DWT produces
$K + 1$ coefficient bands $\{c_A^{(K)}, c_D^{(K)}, c_D^{(K-1)}, \ldots, c_D^{(1)}\}$
(where $K$ is the effective level, clamped to
`pywt.dwt_max_level(n, family)`), and each band is summarised by four
statistics:

$$
\text{mean}(c) = \frac{1}{|c|} \sum_i c_i, \qquad
\text{std}(c) = \sqrt{\tfrac{1}{|c|} \sum_i (c_i - \text{mean}(c))^2}
$$
$$
\text{energy}(c) = \sum_i c_i^2, \qquad
\text{entropy}(c) = -\sum_i p_i \log p_i, \quad p_i = \frac{c_i^2}{\text{energy}(c)}
$$

(`std` uses population variance `ddof=0`; zero-energy bands return
entropy 0 by convention.)

The output row has length $4 (K + 1)$ laid out as
$[\text{approx stats}, \text{detail}_K \text{ stats}, \ldots,
\text{detail}_1 \text{ stats}]$.  `_output_cols(input_cols)` returns
this value.

## Parameters

| Parameter   | Default | Meaning                                         |
| ----------- | ------- | ----------------------------------------------- |
| `family`    | (db4)   | Wavelet family                                  |
| `mode`      | (per.)  | Boundary mode                                   |
| `max_level` | 5       | Cap on decomposition depth                     |

## Lifecycle

Stateless: `_create / _transform / _destroy` plus `_output_cols`.

## Numerical contract

- The output column count depends on the input column count (because
  `max_level` is clamped to `dwt_max_level(cols, family)`); callers
  must size the output buffer with `_output_cols`.
- Parity tolerance against the frozen NumPy reference: `1e-12`
  absolute / `1e-13` relative (closed-form arithmetic on a small set of
  summary statistics).

## Reference

`nirs4all.operators.transforms.nirs.WaveletFeatures` (Mallat 1989,
"A theory for multiresolution signal decomposition").
chemometrics4all reduces the nirs4all surface to the four canonical
statistics — top-K coefficient retention is omitted in v1 to keep the
parity surface compact.
