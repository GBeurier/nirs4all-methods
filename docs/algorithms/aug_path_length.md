# PathLengthAugmenter — `c4a_aug_path_length_*`

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
c4a_status_t c4a_aug_path_length_create(
    c4a_aug_path_length_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double path_length_std,
    double min_path_length);
void         c4a_aug_path_length_destroy(c4a_aug_path_length_handle_t* h);
c4a_status_t c4a_aug_path_length_apply(
    const c4a_aug_path_length_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Reference
- Internal parity fixture: `parity/python_generator/src/c4a_parity_augmenters_ref/path_length.py`.
- Parity tolerance: 1e-15 abs.
