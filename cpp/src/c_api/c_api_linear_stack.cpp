// SPDX-License-Identifier: CECILL-2.1
#include "core/linear_stack.hpp"
#include "n4m/n4m.h"
extern "C" N4M_API n4m_status_t
n4m_ensemble_linear_stack_compress(const n4m_matrix_view_t* base,
                                   const n4m_matrix_view_t* bias,
                                   const n4m_matrix_view_t* weights,
                                   const n4m_matrix_view_t* meta_bias,
                                   n4m_matrix_view_t* coefficients,
                                   n4m_matrix_view_t* intercept) {
    if (!base || !bias || !weights || !meta_bias || !coefficients || !intercept)
        return N4M_ERR_NULL_POINTER;
    try {
        return n4m::core::compress_linear_stack(
            *base, *bias, *weights, *meta_bias, *coefficients, *intercept);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
