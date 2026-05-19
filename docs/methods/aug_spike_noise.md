# `aug_spike_noise` — SpikeNoise

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_spike_noise.md`](../algorithms/aug_spike_noise.md)

## Description

Adds a small number of spikes at random wavelength positions per sample.

Algorithm (RNG draw order is locked — the C side and the Python reference consume the same PCG64 stream):

1. **Per-row spike count** (`n_samples` draws):
   `n_spikes[i] = n_min + floor(u_i * (n_max - n_min + 1))` where `u_i ~ Uniform[0, 1)`.
2. **Candidate indices** (`n_samples * n_max` draws):
   `indices[i, k] = floor(u_ik * n_features)`.
3. **Candidate amplitudes** (`n_samples * n_max` draws):
   `amps[i, k] = amp_min + (amp_max - amp_min) * u_ik`.
4. **Apply**: take `indices[i, :n_spikes[i]]`, deduplicate via sort+unique (mirrors `np.unique`), then add the first `len(uniq)` amplitudes at those positions.

Mirrors `nirs4all.operators.augmentation.spectral.SpikeNoise`. The integer / uniform draws are unrolled to `floor(u * range)` so the C side does not have to re-implement NumPy's Lemire rejection for bounded integers.

### Parameters

- `n_spikes_min, n_spikes_max: int32_t` (`0 <= min <= max`) — inclusive range for the per-row spike count.
- `amplitude_min, amplitude_max: double` (`min <= max`) — half-open range for spike amplitudes.

## Explanations

### Bibliographic source

- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/spike_noise.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_spike_noise` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

Implementation details are defined by the public C ABI and the generated binding wrappers.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
c4a_*
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
