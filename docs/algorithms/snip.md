# SNIP — Statistics-sensitive Non-linear Iterative Peak-clipping

Ryan 1988 / Morháč 1997 baseline correction, implemented on the same public
contract as `pybaselines.smooth.snip` with `filter_order=2`.

For each row `y` of length `n`:

1. Clamp `max_half_window` to the largest valid half-window for the row.
2. Linearly extrapolate `max_half_window` samples at both edges.
3. For each `w = 1, 2, ..., max_half_window`, compute
   `filters[i] = (baseline[i - w] + baseline[i + w]) / 2` for the whole valid
   padded slice, then update `baseline[i] = min(baseline[i], filters[i])`.
4. Output `out = y - baseline[unpadded]`.

## Parameters
- `max_half_window: int` (default 20) — maximum half-window for the iterative
  clip.

## ABI
```c
n4m_status_t n4m_pp_snip_create(n4m_pp_snip_handle_t** out,
                                 int32_t max_half_window);
void         n4m_pp_snip_destroy(n4m_pp_snip_handle_t* h);
n4m_status_t n4m_pp_snip_transform(const n4m_pp_snip_handle_t* h,
                                    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Numerical Contract
- Pure arithmetic, row-wise.
- Edge handling and per-window update order match `pybaselines.smooth.snip`.
- External benchmark gate: `pybaselines.Baseline().snip`.

## Reference
- Ryan, C. et al. (1988). "SNIP: A statistics-sensitive background treatment
  for the quantitative analysis of PIXE spectra in geoscience applications."
  Nuclear Instruments and Methods B34, 396-402.
- Morháč, M. et al. (1997). "Background elimination methods for
  multidimensional coincidence γ-ray spectra." Nuclear Instruments and Methods
  A 401(1), 113-132.
- Pybaselines SNIP reference: `pybaselines.smooth.snip`.
