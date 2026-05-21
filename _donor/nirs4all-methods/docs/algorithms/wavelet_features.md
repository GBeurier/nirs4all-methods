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
\text{energy}(c) = \sum_i c_i^2
$$

(`std` uses population variance `ddof=0`.)

The entropy statistic is selectable:

- `entropy="energy"`: Shannon entropy of the squared-coefficient
  distribution, `p_i = c_i^2 / sum(c^2)`. Zero-energy bands return 0.
- `entropy="histogram"`: ten equal-width histogram bins, normalized as a
  probability distribution before `-sum(p log p)`. This matches
  `nirs4all.WaveletFeatures` when the boundary mode is also aligned to
  PyWavelets' default `symmetric`.

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
| `entropy`   | energy  | `energy` or `histogram` entropy definition      |

## Lifecycle

Stateless: `_create / _transform / _destroy` plus `_output_cols`.

## Numerical contract

- The output column count depends on the input column count (because
  `max_level` is clamped to `dwt_max_level(cols, family)`); callers
  must size the output buffer with `_output_cols`.
- Parity tolerance against the internal parity fixture: `1e-12`
  absolute / `1e-13` relative (closed-form arithmetic on a small set of
  summary statistics).

## Reference

`PyWavelets.wavedec(...)+band stats` is the canonical external reference used
for exact parity snapshots. The dashboard benchmark currently uses
`haar + symmetric + histogram` so the n4m, PyWavelets, and nirs4all rows compare
the same feature contract.

`nirs4all.operators.transforms.nirs.WaveletFeatures` is also timed as an
external comparator. Its default surface adds top-K retained coefficients per
band; the benchmark calls it with `n_coeffs_per_level=0` so its output has the
same feature count as n4m. The benchmark selects the same
histogram entropy and symmetric boundary contract, so this row carries an
external comparator parity gate.

nirs4all-methods reduces the nirs4all surface to the four canonical statistics
in v1; top-K coefficient retention is omitted to keep the ABI and parity
surface compact.
