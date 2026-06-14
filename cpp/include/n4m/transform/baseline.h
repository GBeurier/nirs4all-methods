/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/baseline.h — transform.baseline methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_BASELINE_H
#define N4M_TRANSFORM_BASELINE_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Detrend (polynomial baseline subtraction) ------------------- */
typedef struct n4m_pp_detrend_handle_t n4m_pp_detrend_handle_t;
/* `polyorder` >= 0. Default in pybaselines.polynomial: 1 (linear detrend). */
N4M_API n4m_status_t n4m_transform_detrend_create(n4m_pp_detrend_handle_t** out,
                                            int32_t polyorder);
N4M_API void         n4m_transform_detrend_destroy(n4m_pp_detrend_handle_t* handle);
N4M_API n4m_status_t n4m_transform_detrend_transform(
    const n4m_pp_detrend_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- AsLS (Eilers & Boelens 2005) -------------------------------- */
typedef struct n4m_pp_asls_handle_t n4m_pp_asls_handle_t;
/* `lam` > 0  (smoothing penalty; default 1e6).
 * `p`        (asymmetry, 0 < p < 1; default 1e-2).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 weight change). */
N4M_API n4m_status_t n4m_transform_asls_create(n4m_pp_asls_handle_t** out,
                                         double lam, double p,
                                         int32_t max_iter, double tol);
N4M_API void         n4m_transform_asls_destroy(n4m_pp_asls_handle_t* handle);
N4M_API n4m_status_t n4m_transform_asls_transform(const n4m_pp_asls_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- AirPLS (Zhang 2010) ----------------------------------------- */
typedef struct n4m_pp_airpls_handle_t n4m_pp_airpls_handle_t;
/* `lam`      > 0 (default 1e6).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, |sum(neg residuals)| / |y|_1). */
N4M_API n4m_status_t n4m_transform_airpls_create(n4m_pp_airpls_handle_t** out,
                                           double lam,
                                           int32_t max_iter, double tol);
N4M_API void         n4m_transform_airpls_destroy(n4m_pp_airpls_handle_t* handle);
N4M_API n4m_status_t n4m_transform_airpls_transform(
    const n4m_pp_airpls_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- ArPLS (Baek 2015) ------------------------------------------- */
typedef struct n4m_pp_arpls_handle_t n4m_pp_arpls_handle_t;
/* `lam`      > 0 (default 1e5).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 weight change). */
N4M_API n4m_status_t n4m_transform_arpls_create(n4m_pp_arpls_handle_t** out,
                                          double lam,
                                          int32_t max_iter, double tol);
N4M_API void         n4m_transform_arpls_destroy(n4m_pp_arpls_handle_t* handle);
N4M_API n4m_status_t n4m_transform_arpls_transform(const n4m_pp_arpls_handle_t* handle,
                                             n4m_matrix_view_t X,
                                             n4m_matrix_view_t out);

/* ---------- ModPoly (Lieber & Mahadevan-Jansen 2003) -------------------- */
typedef struct n4m_pp_modpoly_handle_t n4m_pp_modpoly_handle_t;
/* `polyorder` >= 0 (default 2).
 * `max_iter`  >= 0 (default 250).
 * `tol`       >= 0 (default 1e-3, relative L2 of the baseline change). */
N4M_API n4m_status_t n4m_transform_modpoly_create(n4m_pp_modpoly_handle_t** out,
                                            int32_t polyorder,
                                            int32_t max_iter, double tol);
N4M_API void         n4m_transform_modpoly_destroy(n4m_pp_modpoly_handle_t* handle);
N4M_API n4m_status_t n4m_transform_modpoly_transform(
    const n4m_pp_modpoly_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- IModPoly (Gan, Ruan, Mo 2006) ------------------------------ */
typedef struct n4m_pp_imodpoly_handle_t n4m_pp_imodpoly_handle_t;
/* `polyorder` >= 0 (default 2).
 * `max_iter`  >= 0 (default 250).
 * `tol`       >= 0 (default 1e-3, relative change in residual stdev). */
N4M_API n4m_status_t n4m_transform_imodpoly_create(n4m_pp_imodpoly_handle_t** out,
                                             int32_t polyorder,
                                             int32_t max_iter, double tol);
N4M_API void         n4m_transform_imodpoly_destroy(n4m_pp_imodpoly_handle_t* handle);
N4M_API n4m_status_t n4m_transform_imodpoly_transform(
    const n4m_pp_imodpoly_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- SNIP (Ryan 1988, Morháč 1997) ------------------------------ */
typedef struct n4m_pp_snip_handle_t n4m_pp_snip_handle_t;
/* `max_half_window` >= 1 (default 20). The algorithm matches
 * pybaselines.smooth.snip's raw-data filter_order=2 contract: linear edge
 * extrapolation, then simultaneous local-min clipping for each window width
 * w in [1, max_half_window]. */
N4M_API n4m_status_t n4m_transform_snip_create(n4m_pp_snip_handle_t** out,
                                         int32_t max_half_window);
N4M_API void         n4m_transform_snip_destroy(n4m_pp_snip_handle_t* handle);
N4M_API n4m_status_t n4m_transform_snip_transform(const n4m_pp_snip_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- RollingBall (Kneen & Annegarn 1996) ------------------------ */
typedef struct n4m_pp_rolling_ball_handle_t n4m_pp_rolling_ball_handle_t;
/* `half_window`        >= 1 (default 20). Window radius for the min-then-max
 *                            morphological filter.
 * `smooth_half_window` >= 0 (default 0). Optional moving-average smoothing of
 *                            the final baseline. 0 disables smoothing. */
N4M_API n4m_status_t n4m_transform_rolling_ball_create(
    n4m_pp_rolling_ball_handle_t** out,
    int32_t half_window, int32_t smooth_half_window);
N4M_API void         n4m_transform_rolling_ball_destroy(
    n4m_pp_rolling_ball_handle_t* handle);
N4M_API n4m_status_t n4m_transform_rolling_ball_transform(
    const n4m_pp_rolling_ball_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- IAsLS (He 2014) -------------------------------------------- */
typedef struct n4m_pp_iasls_handle_t n4m_pp_iasls_handle_t;
/* `lam`        > 0 (default 1e6).
 * `p`          (asymmetry, 0 < p < 1; default 1e-2).
 * `lam_1`      > 0 (default 1e-4, first-derivative residual penalty).
 * `polyorder`  >= 0 (default 2, prefit polynomial degree; pybaselines uses 2).
 * `diff_order` == 2 in this ABI revision.
 * `max_iter`   >= 0 (default 50).
 * `tol`        >= 0 (default 1e-3, relative L2 weight change). */
N4M_API n4m_status_t n4m_transform_iasls_create(n4m_pp_iasls_handle_t** out,
                                          double lam, double p,
                                          int32_t polyorder,
                                          int32_t max_iter, double tol);
N4M_API n4m_status_t n4m_transform_iasls_create_ex(n4m_pp_iasls_handle_t** out,
                                             double lam, double p,
                                             double lam_1,
                                             int32_t polyorder,
                                             int32_t diff_order,
                                             int32_t max_iter, double tol);
N4M_API void         n4m_transform_iasls_destroy(n4m_pp_iasls_handle_t* handle);
N4M_API n4m_status_t n4m_transform_iasls_transform(const n4m_pp_iasls_handle_t* handle,
                                             n4m_matrix_view_t X,
                                             n4m_matrix_view_t out);

/* ---------- BEADS — Ning & Selesnick 2014 ------------------------------ */
typedef struct n4m_pp_beads_handle_t n4m_pp_beads_handle_t;
/* `lam_0`    > 0 (default 1e2, sparsity weight on baseline residual).
 * `lam_1`    > 0 (default 0.5, banded 1st-difference smoothness weight).
 * `lam_2`    > 0 (default 0.5, banded 2nd-difference smoothness weight).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 baseline change).
 *
 * Fixed BEADS options match pybaselines defaults for the public n4m surface:
 * freq_cutoff=0.005, asymmetry=6, filter_type=1, cost_function=2,
 * eps_0=eps_1=1e-6, fit_parabola=true. */
N4M_API n4m_status_t n4m_transform_beads_create(n4m_pp_beads_handle_t** out,
                                          double lam_0, double lam_1,
                                          double lam_2,
                                          int32_t max_iter, double tol);
N4M_API void         n4m_transform_beads_destroy(n4m_pp_beads_handle_t* handle);
N4M_API n4m_status_t n4m_transform_beads_transform(const n4m_pp_beads_handle_t* handle,
                                             n4m_matrix_view_t X,
                                             n4m_matrix_view_t out);

typedef struct n4m_pp_saps_handle_t n4m_pp_saps_handle_t;
N4M_API n4m_status_t n4m_transform_saps_create(
    n4m_pp_saps_handle_t** out, int32_t n_components, double score_weight,
    int fit_intercept, double ridge);
N4M_API void n4m_transform_saps_destroy(n4m_pp_saps_handle_t* handle);
N4M_API n4m_status_t n4m_transform_saps_fit(
    n4m_pp_saps_handle_t* handle, n4m_matrix_view_t source,
    n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_transform_saps_transform(
    const n4m_pp_saps_handle_t* handle, n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_saps_is_fitted(
    const n4m_pp_saps_handle_t* handle, int* out_fitted);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_BASELINE_H */
