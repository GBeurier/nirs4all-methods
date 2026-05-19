// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrapper for the Phase 19 SignalTypeDetector ABI entry-point
// (c4a_signal_detect). Stateless function: no handle lifecycle. The body
// validates the matrix view, then dispatches to the core implementation
// under cpp/src/core/utilities/signal_type_detector.{c,h}.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/utilities/signal_type_detector.h"

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous, and
// return the underlying double pointer + dimensions. Same convention as the
// Phase 2+ wrappers.
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

}  // namespace

extern "C" {

C4A_API c4a_status_t c4a_signal_detect(c4a_matrix_view_t X,
                                        const double* wavelengths_optional,
                                        int64_t wl_length,
                                        double confidence_threshold,
                                        c4a_signal_type_t* type_out,
                                        double* confidence_out,
                                        char reason_buf[256]) {
    if (type_out == nullptr || confidence_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *type_out       = C4A_SIGNAL_UNKNOWN;
    *confidence_out = 0.0;
    if (reason_buf != nullptr) {
        reason_buf[0] = '\0';
    }
    if (wavelengths_optional != nullptr && wl_length < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        // Allow zero-sized matrices through: the core handles "empty data".
        if (s != C4A_OK && !(X.rows == 0 || X.cols == 0)) {
            return s;
        }
        int32_t kind = 0;
        const c4a_status_t st = c4a_signal_detect_impl(
            xp, xr, xc,
            wavelengths_optional, wl_length,
            confidence_threshold,
            &kind, confidence_out, reason_buf);
        if (st != C4A_OK) return st;
        *type_out = static_cast<c4a_signal_type_t>(kind);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
