# SpikeNoise — `c4a_aug_spike_noise_*`

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

## Parameters
- `n_spikes_min, n_spikes_max: int32_t` (`0 <= min <= max`) — inclusive range for the per-row spike count.
- `amplitude_min, amplitude_max: double` (`min <= max`) — half-open range for spike amplitudes.

## ABI
```c
c4a_status_t c4a_aug_spike_noise_create(
    c4a_aug_spike_noise_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_spikes_min, int32_t n_spikes_max,
    double  amplitude_min, double amplitude_max);
void         c4a_aug_spike_noise_destroy(c4a_aug_spike_noise_handle_t* h);
c4a_status_t c4a_aug_spike_noise_apply(
    const c4a_aug_spike_noise_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Reference
- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/spike_noise.py`.
- Parity tolerance: 1e-15 abs.
