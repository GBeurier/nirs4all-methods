# PathLengthAugmenter — `n4m_aug_path_length_*`

Applies a per-sample multiplicative path-length factor (with a lower clip to prevent sign inversion).

Algorithm:

1. Draw `n_samples` standard normals in one bulk call.
2. `L[i] = 1.0 + path_length_std * normal[i]`.
3. `L[i] = max(L[i], min_path_length)`.
4. `out[i, j] = L[i] * X[i, j]`.

Mirrors `nirs4all.operators.augmentation.synthesis.PathLengthAugmenter` with `variation_scope="sample"` (default).

## Parameters
- `path_length_std: double` (`>= 0.0`) — standard deviation of the per-sample factor (centered at 1.0).
- `min_path_length: double` — lower clip applied to `L` after the normal draw.

## ABI
```c
n4m_status_t n4m_aug_path_length_create(
    n4m_aug_path_length_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double path_length_std,
    double min_path_length);
void         n4m_aug_path_length_destroy(n4m_aug_path_length_handle_t* h);
n4m_status_t n4m_aug_path_length_apply(
    const n4m_aug_path_length_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Reference
- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/path_length.py`.
- Parity tolerance: 1e-15 abs.
