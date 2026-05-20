# RollingBall — Morphological baseline

Kneen & Annegarn 1996 rolling-ball baseline. For each row `y` of length `n`:

1. **Min filter** of half-window `W`: `e[i] = min over [max(0, i - W), min(n - 1, i + W)] of y`.
2. **Max filter** of half-window `W` on the result: `z[i] = max over [max(0, i - W), min(n - 1, i + W)] of e`.
3. **Optional moving-average smoothing** of `z` over a `(2 * S + 1)` clipped centred window (skipped when `S = 0`).
4. Output: `out = y - z`.

Edges use centred windows clipped to `[0, n - 1]` (no padding / reflection).

## Parameters
- `half_window: int` (default 20).
- `smooth_half_window: int` (default 0) — disables smoothing when 0.

## ABI
```c
c4a_status_t c4a_pp_rolling_ball_create(c4a_pp_rolling_ball_handle_t** out,
                                         int32_t half_window,
                                         int32_t smooth_half_window);
void         c4a_pp_rolling_ball_destroy(c4a_pp_rolling_ball_handle_t* h);
c4a_status_t c4a_pp_rolling_ball_transform(const c4a_pp_rolling_ball_handle_t* h,
                                            c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical contract
- Two scratch buffers per row (length `n` doubles). No linear algebra.
- Parity tolerance vs internal parity fixture: `1e-12 abs / 1e-13 rel`.

## Reference
- Kneen, M. A. & Annegarn, H. J. (1996). "Algorithm for fitting XRF, SEM and PIXE X-ray spectra backgrounds." Nuclear Instruments and Methods B109/110, 209-213.
- Internal parity fixture: `parity/python_generator/src/c4a_parity_pybaselines_ref/rolling_ball.py`.
