/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/smoothing.h — transform.smoothing methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_SMOOTHING_H
#define N4M_TRANSFORM_SMOOTHING_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Derivate (finite-difference derivative along axis=1) --------- */
/*
 * Finite-difference derivative of order `order` (>= 1) along the wavelength
 * axis (axis=1). The output has shape (rows, cols - order); each derivative
 * step shortens the column count by one. Division by `delta**order` yields
 * the physical-units derivative when `delta` is the wavelength sample
 * spacing.
 *
 *   out = np.diff(X, n=order, axis=1) / delta ** order
 *
 * `_fit` memoises the input column count for shape validation at transform
 * time — it carries no statistical learned state, but it does record `cols`
 * so that `_transform` can reject inputs whose width disagrees with what
 * was seen at fit time. Calling `_transform` before `_fit` returns
 * N4M_ERR_NOT_FITTED; calling it with a column count different from the
 * fitted value returns N4M_ERR_SHAPE_MISMATCH. This lets bindings treat
 * Derivate uniformly with the other stateful preprocessings without a
 * special case.
 *
 * Use `n4m_transform_derivative_output_cols(order, input_cols)` to compute the
 * output column count expected by `_transform`. The helper returns 0 when
 * `order >= input_cols`.
 */
typedef struct n4m_pp_derivate_handle_t n4m_pp_derivate_handle_t;
N4M_API n4m_status_t n4m_transform_derivative_create(n4m_pp_derivate_handle_t** out,
                                             int32_t order, double delta);
N4M_API void         n4m_transform_derivative_destroy(n4m_pp_derivate_handle_t* handle);
N4M_API n4m_status_t n4m_transform_derivative_fit(n4m_pp_derivate_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_derivative_transform(
    const n4m_pp_derivate_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API int64_t      n4m_transform_derivative_output_cols(int32_t order,
                                                  int64_t input_cols);

/* ---------- SavitzkyGolay -------------------------------------------------
 *
 * Parameters:
 *   window_length : odd integer >= 1
 *   polyorder     : 0 <= polyorder < window_length
 *   deriv         : derivative order >= 0
 *   delta         : sample spacing (non-zero)
 *   mode          : boundary handling, see n4m_pp_savgol_mode_t below
 *   cval          : fill value used when mode == N4M_PP_SAVGOL_CONSTANT
 *
 * The five modes mirror `scipy.signal.savgol_filter`:
 *   - MIRROR   : reflection without repeating the edge (default).
 *   - CONSTANT : pad with `cval`.
 *   - NEAREST  : replicate the edge sample.
 *   - WRAP     : cyclic.
 *   - INTERP   : fit `polyorder` polynomials to the first and last
 *                window_length samples, evaluate at the half-window
 *                positions on each side.
 */
typedef struct n4m_pp_savgol_handle_t n4m_pp_savgol_handle_t;
typedef enum n4m_pp_savgol_mode_t {
    N4M_PP_SAVGOL_MIRROR   = 0,
    N4M_PP_SAVGOL_CONSTANT = 1,
    N4M_PP_SAVGOL_NEAREST  = 2,
    N4M_PP_SAVGOL_WRAP     = 3,
    N4M_PP_SAVGOL_INTERP   = 4
} n4m_pp_savgol_mode_t;
N4M_API n4m_status_t n4m_transform_savitzky_golay_create(n4m_pp_savgol_handle_t** out,
                                           int32_t window_length,
                                           int32_t polyorder,
                                           int32_t deriv,
                                           double delta,
                                           n4m_pp_savgol_mode_t mode,
                                           double cval);
N4M_API void         n4m_transform_savitzky_golay_destroy(n4m_pp_savgol_handle_t* handle);
N4M_API n4m_status_t n4m_transform_savitzky_golay_transform(
    const n4m_pp_savgol_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- FirstDerivative -----------------------------------------------
 *
 * `np.gradient(X, delta, axis=1, edge_order)` with edge_order in {1, 2}.
 * Output shape matches input shape.  `delta` must be non-zero.
 */
typedef struct n4m_pp_first_derivative_handle_t
    n4m_pp_first_derivative_handle_t;
N4M_API n4m_status_t n4m_transform_first_derivative_create(
    n4m_pp_first_derivative_handle_t** out, double delta, int32_t edge_order);
N4M_API void         n4m_transform_first_derivative_destroy(
    n4m_pp_first_derivative_handle_t* handle);
N4M_API n4m_status_t n4m_transform_first_derivative_transform(
    const n4m_pp_first_derivative_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- SecondDerivative ----------------------------------------------
 *
 * Two passes of `np.gradient(X, delta, axis=1, edge_order)`.  Shape-preserving.
 */
typedef struct n4m_pp_second_derivative_handle_t
    n4m_pp_second_derivative_handle_t;
N4M_API n4m_status_t n4m_transform_second_derivative_create(
    n4m_pp_second_derivative_handle_t** out, double delta, int32_t edge_order);
N4M_API void         n4m_transform_second_derivative_destroy(
    n4m_pp_second_derivative_handle_t* handle);
N4M_API n4m_status_t n4m_transform_second_derivative_transform(
    const n4m_pp_second_derivative_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- NorrisWilliams -------------------------------------------------
 *
 * `segment` must be odd and >= 1 (1 disables smoothing).  `gap` must be >= 1.
 * `derivative_order` must be 1 or 2.  `delta` must be non-zero.  Shape-preserving.
 */
typedef struct n4m_pp_norris_williams_handle_t
    n4m_pp_norris_williams_handle_t;
N4M_API n4m_status_t n4m_transform_norris_williams_create(
    n4m_pp_norris_williams_handle_t** out,
    int32_t segment, int32_t gap, int32_t derivative_order, double delta);
N4M_API void         n4m_transform_norris_williams_destroy(
    n4m_pp_norris_williams_handle_t* handle);
N4M_API n4m_status_t n4m_transform_norris_williams_transform(
    const n4m_pp_norris_williams_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- Gaussian -------------------------------------------------------
 *
 * `scipy.ndimage.gaussian_filter1d(X, sigma, order, axis=1, mode, cval,
 * truncate)`.  Kernel radius is `int(truncate * sigma + 0.5)`.  Sigma must be
 * positive, order >= 0, truncate >= 0.
 *
 * Boundary modes mirror SciPy.ndimage:
 *   - REFLECT  : edge-repeating reflection (default).
 *   - CONSTANT : pad with `cval`.
 *   - NEAREST  : replicate the edge sample.
 *   - MIRROR   : non-edge-repeating reflection.
 *   - WRAP     : cyclic.
 */
typedef struct n4m_pp_gaussian_handle_t n4m_pp_gaussian_handle_t;
typedef enum n4m_pp_gaussian_mode_t {
    N4M_PP_GAUSSIAN_REFLECT  = 0,
    N4M_PP_GAUSSIAN_CONSTANT = 1,
    N4M_PP_GAUSSIAN_NEAREST  = 2,
    N4M_PP_GAUSSIAN_MIRROR   = 3,
    N4M_PP_GAUSSIAN_WRAP     = 4
} n4m_pp_gaussian_mode_t;
N4M_API n4m_status_t n4m_transform_gaussian_create(
    n4m_pp_gaussian_handle_t** out,
    double sigma, int32_t order,
    n4m_pp_gaussian_mode_t mode,
    double cval, double truncate);
N4M_API void         n4m_transform_gaussian_destroy(n4m_pp_gaussian_handle_t* handle);
N4M_API n4m_status_t n4m_transform_gaussian_transform(
    const n4m_pp_gaussian_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_SMOOTHING_H */
