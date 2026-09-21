// SPDX-License-Identifier: CECILL-2.1
#include <algorithm>
#include <limits>
#include <new>

#include "core/aom_calibration.hpp"
#include "n4m/n4m.h"
extern "C" {
N4M_API n4m_status_t n4m_model_selection_aom_calibration_fit(n4m_context_t* ctx,
                                                             const n4m_matrix_view_t* X,
                                                             const n4m_matrix_view_t* Y,
                                                             const n4m_operator_bank_t* bank,
                                                             const int32_t* offsets,
                                                             int32_t chains,
                                                             const int32_t* branches,
                                                             int32_t nb,
                                                             const int32_t* folds,
                                                             int32_t nf,
                                                             int32_t head,
                                                             int32_t fast,
                                                             int32_t rank,
                                                             const double* alphas,
                                                             int32_t np,
                                                             double* coefficients,
                                                             int64_t nc,
                                                             double* state,
                                                             int64_t ns,
                                                             double* scores,
                                                             int64_t nscore,
                                                             int32_t* selected) {
    if (!ctx || !X || !Y || !bank || !offsets || !branches || !folds || !coefficients || !state
        || !scores || !selected || (head != 0 && !alphas))
        return N4M_ERR_NULL_POINTER;
    if (chains < 1 || nb < 1 || np < 1 || X->cols < 1 || fast < 0 || fast > 1)
        return N4M_ERR_INVALID_ARGUMENT;
    if (X->cols > (std::numeric_limits<int64_t>::max() - 2) / 2)
        return N4M_ERR_INVALID_ARGUMENT;
    int64_t cells = np;
    if (!fast) {
        if (static_cast<int64_t>(chains) * nb > std::numeric_limits<int64_t>::max() / np)
            return N4M_ERR_INVALID_ARGUMENT;
        cells = static_cast<int64_t>(chains) * nb * np;
    }
    if (nc < X->cols || ns < 2 * X->cols + 2 || nscore < cells)
        return N4M_ERR_SHAPE_MISMATCH;
    try {
        n4m::core::AomCalibrationResult result;
        auto status = n4m::core::fit_aom_calibration(*ctx,
                                                     *X,
                                                     *Y,
                                                     *bank,
                                                     offsets,
                                                     chains,
                                                     branches,
                                                     nb,
                                                     folds,
                                                     nf,
                                                     head,
                                                     fast != 0,
                                                     rank,
                                                     alphas,
                                                     np,
                                                     result);
        if (status != N4M_OK) {
            ctx->set_error(
                "AOM calibration failed; check folds, branch/operator parameters and numerical rank");
            return status;
        }
        std::copy(result.coefficients.begin(), result.coefficients.end(), coefficients);
        std::copy(result.state.begin(), result.state.end(), state);
        std::copy(result.scores.begin(), result.scores.end(), scores);
        selected[0] = result.branch;
        selected[1] = result.chain;
        selected[2] = result.parameter;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
N4M_API n4m_status_t n4m_model_selection_aom_calibration_predict(const n4m_matrix_view_t* X,
                                                                 int32_t branch,
                                                                 const double* coefficients,
                                                                 int64_t nc,
                                                                 const double* state,
                                                                 int64_t ns,
                                                                 n4m_matrix_view_t* out) {
    if (!X || !out || !coefficients || !state)
        return N4M_ERR_NULL_POINTER;
    if (X->cols < 1 || X->cols > (std::numeric_limits<int64_t>::max() - 2) / 2 || nc < X->cols
        || ns < 2 * X->cols + 2)
        return N4M_ERR_SHAPE_MISMATCH;
    return n4m::core::predict_aom_calibration(*X, branch, coefficients, state, *out);
}
}
