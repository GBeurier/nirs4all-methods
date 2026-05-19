// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 19 NIRS regression metrics
// (c4a_metric_*) and the multivariate outlier statistics
// (c4a_util_hotelling_t2, c4a_util_q_residuals). All wrappers are stateless
// — there is no handle lifecycle.
//
// The 8 metric wrappers translate (y_true, y_pred, n) to the internal
// implementations in cpp/src/core/utilities/nirs_metrics.{c,h}. The 2 utility
// wrappers validate the matrix view and the output array length, then
// dispatch to cpp/src/core/utilities/{hotelling_t2,q_residuals}.{c,h}.

#include <cstddef>
#include <cstdint>
#include <exception>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/utilities/hotelling_t2.h"
#include "core/utilities/nirs_metrics.h"
#include "core/utilities/q_residuals.h"

namespace {

c4a_status_t require_rowmajor_f64(const c4a_matrix_view_t& v,
                                  const double*& out_ptr,
                                  std::int64_t& out_rows,
                                  std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) {
        return s;
    }
    if (v.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

template <c4a_status_t (*Fn)(const double*, const double*, std::int64_t, double*)>
c4a_status_t dispatch_metric(const double* y_true,
                              const double* y_pred,
                              std::int64_t n,
                              double* out) noexcept {
    if (y_true == nullptr || y_pred == nullptr || out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n <= 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        return Fn(y_true, y_pred, n, out);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // namespace

extern "C" {

C4A_API c4a_status_t c4a_metric_rmse(const double* y_true, const double* y_pred,
                                      int64_t n, double* out) {
    return dispatch_metric<c4a_metric_rmse_impl>(y_true, y_pred, n, out);
}

C4A_API c4a_status_t c4a_metric_mae(const double* y_true, const double* y_pred,
                                     int64_t n, double* out) {
    return dispatch_metric<c4a_metric_mae_impl>(y_true, y_pred, n, out);
}

C4A_API c4a_status_t c4a_metric_bias(const double* y_true, const double* y_pred,
                                      int64_t n, double* out) {
    return dispatch_metric<c4a_metric_bias_impl>(y_true, y_pred, n, out);
}

C4A_API c4a_status_t c4a_metric_sep(const double* y_true, const double* y_pred,
                                     int64_t n, double* out) {
    return dispatch_metric<c4a_metric_sep_impl>(y_true, y_pred, n, out);
}

C4A_API c4a_status_t c4a_metric_rpd(const double* y_true, const double* y_pred,
                                     int64_t n, double* out) {
    return dispatch_metric<c4a_metric_rpd_impl>(y_true, y_pred, n, out);
}

C4A_API c4a_status_t c4a_metric_rpiq(const double* y_true, const double* y_pred,
                                      int64_t n, double* out) {
    return dispatch_metric<c4a_metric_rpiq_impl>(y_true, y_pred, n, out);
}

C4A_API c4a_status_t c4a_metric_r2(const double* y_true, const double* y_pred,
                                    int64_t n, double* out) {
    return dispatch_metric<c4a_metric_r2_impl>(y_true, y_pred, n, out);
}

C4A_API c4a_status_t c4a_metric_nrmse(const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out) {
    return dispatch_metric<c4a_metric_nrmse_impl>(y_true, y_pred, n, out);
}

// ---------------------------------------------------------------------------
// Hotelling T²
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_util_hotelling_t2(c4a_matrix_view_t X,
                                            int32_t n_components,
                                            double alpha,
                                            double* t2_per_sample,
                                            int64_t n_samples,
                                            double* ucl_out) {
    if (t2_per_sample == nullptr || ucl_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        if (n_samples != xr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_util_hotelling_t2_impl(xp, xr, xc, n_components, alpha,
                                           t2_per_sample, ucl_out);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Q-residuals
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_util_q_residuals(c4a_matrix_view_t X,
                                           int32_t n_components,
                                           double alpha,
                                           double* q_per_sample,
                                           int64_t n_samples,
                                           double* ucl_out) {
    if (q_per_sample == nullptr || ucl_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        if (n_samples != xr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_util_q_residuals_impl(xp, xr, xc, n_components, alpha,
                                          q_per_sample, ucl_out);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
