// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 19 NIRS regression metrics
// (n4m_metric_*) and the multivariate outlier statistics
// (n4m_util_hotelling_t2, n4m_util_q_residuals). All wrappers are stateless
// — there is no handle lifecycle.
//
// The 8 metric wrappers translate (y_true, y_pred, n) to the internal
// implementations in cpp/src/core/utilities/nirs_metrics.{c,h}. The 2 utility
// wrappers validate the matrix view and the output array length, then
// dispatch to cpp/src/core/utilities/{hotelling_t2,q_residuals}.{c,h}.

#include <cstddef>
#include <cstdint>
#include <exception>

#include "n4m/n4m.h"

#include "core/common/matrix_view.hpp"
#include "core/utilities/hotelling_t2.h"
#include "core/utilities/nirs_metrics.h"
#include "core/utilities/q_residuals.h"

namespace {

n4m_status_t require_rowmajor_f64(const n4m_matrix_view_t& v,
                                  const double*& out_ptr,
                                  std::int64_t& out_rows,
                                  std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) {
        return s;
    }
    if (v.dtype != N4M_DTYPE_F64) {
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return N4M_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

template <n4m_status_t (*Fn)(const double*, const double*, std::int64_t, double*)>
n4m_status_t dispatch_metric(const double* y_true,
                              const double* y_pred,
                              std::int64_t n,
                              double* out) noexcept {
    if (y_true == nullptr || y_pred == nullptr || out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n <= 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        return Fn(y_true, y_pred, n, out);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_metric_rmse(const double* y_true, const double* y_pred,
                                      int64_t n, double* out) {
    return dispatch_metric<n4m_metric_rmse_impl>(y_true, y_pred, n, out);
}

N4M_API n4m_status_t n4m_metric_mae(const double* y_true, const double* y_pred,
                                     int64_t n, double* out) {
    return dispatch_metric<n4m_metric_mae_impl>(y_true, y_pred, n, out);
}

N4M_API n4m_status_t n4m_metric_bias(const double* y_true, const double* y_pred,
                                      int64_t n, double* out) {
    return dispatch_metric<n4m_metric_bias_impl>(y_true, y_pred, n, out);
}

N4M_API n4m_status_t n4m_metric_sep(const double* y_true, const double* y_pred,
                                     int64_t n, double* out) {
    return dispatch_metric<n4m_metric_sep_impl>(y_true, y_pred, n, out);
}

N4M_API n4m_status_t n4m_metric_rpd(const double* y_true, const double* y_pred,
                                     int64_t n, double* out) {
    return dispatch_metric<n4m_metric_rpd_impl>(y_true, y_pred, n, out);
}

N4M_API n4m_status_t n4m_metric_rpiq(const double* y_true, const double* y_pred,
                                      int64_t n, double* out) {
    return dispatch_metric<n4m_metric_rpiq_impl>(y_true, y_pred, n, out);
}

N4M_API n4m_status_t n4m_metric_r2(const double* y_true, const double* y_pred,
                                    int64_t n, double* out) {
    return dispatch_metric<n4m_metric_r2_impl>(y_true, y_pred, n, out);
}

N4M_API n4m_status_t n4m_metric_nrmse(const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out) {
    return dispatch_metric<n4m_metric_nrmse_impl>(y_true, y_pred, n, out);
}

// ---------------------------------------------------------------------------
// Hotelling T²
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_util_hotelling_t2(n4m_matrix_view_t X,
                                            int32_t n_components,
                                            double alpha,
                                            double* t2_per_sample,
                                            int64_t n_samples,
                                            double* ucl_out) {
    if (t2_per_sample == nullptr || ucl_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        if (n_samples != xr) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return n4m_util_hotelling_t2_impl(xp, xr, xc, n_components, alpha,
                                           t2_per_sample, ucl_out);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Q-residuals
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_util_q_residuals(n4m_matrix_view_t X,
                                           int32_t n_components,
                                           double alpha,
                                           double* q_per_sample,
                                           int64_t n_samples,
                                           double* ucl_out) {
    if (q_per_sample == nullptr || ucl_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        if (n_samples != xr) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return n4m_util_q_residuals_impl(xp, xr, xc, n_components, alpha,
                                          q_per_sample, ucl_out);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
