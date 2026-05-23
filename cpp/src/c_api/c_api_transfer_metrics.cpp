// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrapper for the Phase 20 TransferMetricsComputer utility. A
// single entry point computes a 9-element struct of transfer-alignment
// metrics between a source and a target dataset; everything else is private.
//
// Universal rules (mirrored from c_api_preprocessing.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The matrix views must be row-major contiguous F64 (the internal engine
//     under cpp/src/core/utilities/transfer_metrics.{c,h} only knows that
//     layout).
//   - `out` is a caller-allocated `n4m_transfer_metrics_t`; it is overwritten
//     unconditionally on success.

#include <cstddef>
#include <cstdint>
#include <exception>

#include "n4m/n4m.h"

#include "core/common/matrix_view.hpp"
#include "core/utilities/transfer_metrics.h"

namespace {

// Re-use the same view validator as the preprocessing wrappers. Kept TU-local
// because the preprocessing TU's helper is `static`-equivalent (in an
// anonymous namespace).
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

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_transfer_metrics_compute(
    n4m_matrix_view_t X_source,
    n4m_matrix_view_t X_target,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    n4m_transfer_metrics_t* out) {

    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xs = nullptr;
        const double* xt = nullptr;
        std::int64_t  rs = 0, cs = 0, rt = 0, ct = 0;
        n4m_status_t s = require_rowmajor_f64(X_source, xs, rs, cs);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64(X_target, xt, rt, ct);
        if (s != N4M_OK) return s;

        n4m_transfer_metrics_result_t r{};
        s = n4m_transfer_metrics_compute_impl(
            xs, rs, cs,
            xt, rt, ct,
            n_components, k_neighbors, seed,
            &r);
        if (s != N4M_OK) {
            return s;
        }
        out->centroid_distance    = r.centroid_distance;
        out->cka_similarity       = r.cka_similarity;
        out->grassmann_distance   = r.grassmann_distance;
        out->rv_coefficient       = r.rv_coefficient;
        out->procrustes_disparity = r.procrustes_disparity;
        out->trustworthiness      = r.trustworthiness;
        out->spread_distance      = r.spread_distance;
        out->evr_source           = r.evr_source;
        out->evr_target           = r.evr_target;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
