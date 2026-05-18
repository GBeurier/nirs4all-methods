# SNIP — Statistics-sensitive Non-linear Iterative Peak-clipping

Ryan 1988 / Morháč 1997 baseline. Pure-arithmetic algorithm (no linear algebra). For each row `y` of length `n`:

1. **LLS transform**: `v[i] = log(log(sqrt(|y[i]| + 1) + 1) + 1)`.
2. **Peak clipping** at growing half-windows: for each `w = 1, 2, .., max_half_window`, for `i in [w, n - w)`:
   `v[i] = min(v[i], (v[i - w] + v[i + w]) / 2)`.
3. **Inverse LLS**: `z = (exp(exp(v) - 1) - 1)^2 - 1`.
4. Output: `out = y - z`.

## Parameters
- `max_half_window: int` (default 20) — maximum half-window for the iterative clip.

## ABI
```c
c4a_status_t c4a_pp_snip_create(c4a_pp_snip_handle_t** out,
                                 int32_t max_half_window);
void         c4a_pp_snip_destroy(c4a_pp_snip_handle_t* h);
c4a_status_t c4a_pp_snip_transform(const c4a_pp_snip_handle_t* h,
                                    c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical contract
- Pure arithmetic; the clipping step is applied IN-PLACE so each window pass references just-updated neighbours (matches Morháč 1997 / pybaselines).
- One scratch buffer per row (length `n` doubles).
- Parity tolerance vs frozen NumPy reference: `1e-12 abs / 1e-13 rel`.

## Reference
- Ryan, C. et al. (1988). "SNIP: A statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications." Nuclear Instruments and Methods B34, 396-402.
- Morháč, M. et al. (1997). "Background elimination methods for multidimensional coincidence γ-ray spectra." Nuclear Instruments and Methods A 401(1), 113-132.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/snip.py`.
