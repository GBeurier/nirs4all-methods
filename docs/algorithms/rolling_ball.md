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
n4m_status_t n4m_pp_rolling_ball_create(n4m_pp_rolling_ball_handle_t** out,
                                         int32_t half_window,
                                         int32_t smooth_half_window);
void         n4m_pp_rolling_ball_destroy(n4m_pp_rolling_ball_handle_t* h);
n4m_status_t n4m_pp_rolling_ball_transform(const n4m_pp_rolling_ball_handle_t* h,
                                            n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Numerical contract
- Two scratch buffers per row (length `n` doubles). No linear algebra.
- Parity tolerance vs internal parity fixture: `1e-12 abs / 1e-13 rel`.

## Reference
- Kneen, M. A. & Annegarn, H. J. (1996). "Algorithm for fitting XRF, SEM and PIXE X-ray spectra backgrounds." Nuclear Instruments and Methods B109/110, 209-213.
- Internal parity fixture: `parity/python_generator/src/n4m_parity_pybaselines_ref/rolling_ball.py`.
