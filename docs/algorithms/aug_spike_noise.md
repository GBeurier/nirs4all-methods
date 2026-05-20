# SpikeNoise — `n4m_aug_spike_noise_*`

Adds a small number of spikes at random wavelength positions per sample.

Algorithm (RNG draw order is locked — the C side and the Python reference consume the same PCG64 stream):

1. **Per-row spike count** (`n_samples` bounded-integer draws):
   `n_spikes[i] = rng.integers(n_min, n_max + 1)`.
2. **Candidate indices** (`n_samples * n_max` draws):
   `indices[i, k] = rng.integers(0, n_features)`.
3. **Candidate amplitudes** (`n_samples * n_max` draws):
   `amps[i, k] = amp_min + (amp_max - amp_min) * u_ik`.
4. **Apply**: take `indices[i, :n_spikes[i]]`, deduplicate via sort+unique (mirrors `np.unique`), then add the first `len(uniq)` amplitudes at those positions.

Mirrors `nirs4all.operators.augmentation.spectral.SpikeNoise`. Bounded integer draws use the same buffered NumPy PCG64 path as `Generator.integers` for the tested ranges; amplitudes use the standard uniform double stream.

## Parameters
- `n_spikes_min, n_spikes_max: int32_t` (`0 <= min <= max`) — inclusive range for the per-row spike count.
- `amplitude_min, amplitude_max: double` (`min <= max`) — half-open range for spike amplitudes.

## ABI
```c
n4m_status_t n4m_aug_spike_noise_create(
    n4m_aug_spike_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_spikes_min, int32_t n_spikes_max,
    double  amplitude_min, double amplitude_max);
void         n4m_aug_spike_noise_destroy(n4m_aug_spike_noise_handle_t* h);
n4m_status_t n4m_aug_spike_noise_apply(
    const n4m_aug_spike_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Reference
- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/spike_noise.py`.
- Parity tolerance: 1e-15 abs.
