/* SPDX-License-Identifier: CECILL-2.1 */
#include "random_x_op.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/rng_pcg64.h"

struct n4m_aug_random_x_op_state_t {
    int32_t op_kind;
    double  operator_range_min;
    double  operator_range_max;
};

n4m_aug_random_x_op_state_t* n4m_aug_random_x_op_state_new(
    int32_t op_kind, double operator_range_min, double operator_range_max) {
    if (op_kind < 0 || op_kind > 2) return NULL;
    if (operator_range_min > operator_range_max) return NULL;
    n4m_aug_random_x_op_state_t* s =
        (n4m_aug_random_x_op_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->op_kind            = op_kind;
    s->operator_range_min = operator_range_min;
    s->operator_range_max = operator_range_max;
    return s;
}

void n4m_aug_random_x_op_state_free(n4m_aug_random_x_op_state_t* state) {
    free(state);
}

/* float32 max ≈ 3.4028234663852886e+38. */
#define N4M_FLOAT32_MAX 3.4028234663852886e+38

n4m_status_t n4m_aug_random_x_op_state_apply(
    const n4m_aug_random_x_op_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL || rng_void == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return N4M_OK;
    }
    n4m_rng_pcg64* rng = (n4m_rng_pcg64*)rng_void;
    const double lo = state->operator_range_min;
    const double interval = state->operator_range_max - lo;
    const int64_t total = rows * cols;
    for (int64_t k = 0; k < total; ++k) {
        const double u = n4m_pcg64_engine_next_double(rng);
        const double v = lo + u * interval;
        double r;
        switch (state->op_kind) {
            case N4M_AUG_RANDOM_X_OP_MUL: r = X[k] * v; break;
            case N4M_AUG_RANDOM_X_OP_ADD: r = X[k] + v; break;
            case N4M_AUG_RANDOM_X_OP_SUB: r = X[k] - v; break;
            default:                        r = X[k];      break;
        }
        if (r >  N4M_FLOAT32_MAX) r =  N4M_FLOAT32_MAX;
        if (r < -N4M_FLOAT32_MAX) r = -N4M_FLOAT32_MAX;
        out[k] = r;
    }
    return N4M_OK;
}
