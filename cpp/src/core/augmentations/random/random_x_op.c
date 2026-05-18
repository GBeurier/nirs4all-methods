/* SPDX-License-Identifier: CECILL-2.1 */
#include "random_x_op.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/rng_pcg64.h"

struct c4a_aug_random_x_op_state_t {
    int32_t op_kind;
    double  operator_range_min;
    double  operator_range_max;
};

c4a_aug_random_x_op_state_t* c4a_aug_random_x_op_state_new(
    int32_t op_kind, double operator_range_min, double operator_range_max) {
    if (op_kind < 0 || op_kind > 2) return NULL;
    if (operator_range_min > operator_range_max) return NULL;
    c4a_aug_random_x_op_state_t* s =
        (c4a_aug_random_x_op_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->op_kind            = op_kind;
    s->operator_range_min = operator_range_min;
    s->operator_range_max = operator_range_max;
    return s;
}

void c4a_aug_random_x_op_state_free(c4a_aug_random_x_op_state_t* state) {
    free(state);
}

/* float32 max ≈ 3.4028234663852886e+38. */
#define C4A_FLOAT32_MAX 3.4028234663852886e+38

c4a_status_t c4a_aug_random_x_op_state_apply(
    const c4a_aug_random_x_op_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL || rng_void == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return C4A_OK;
    }
    c4a_rng_pcg64* rng = (c4a_rng_pcg64*)rng_void;
    const double lo = state->operator_range_min;
    const double interval = state->operator_range_max - lo;
    const int64_t total = rows * cols;
    for (int64_t k = 0; k < total; ++k) {
        const double u = c4a_pcg64_engine_next_double(rng);
        const double v = lo + u * interval;
        double r;
        switch (state->op_kind) {
            case C4A_AUG_RANDOM_X_OP_MUL: r = X[k] * v; break;
            case C4A_AUG_RANDOM_X_OP_ADD: r = X[k] + v; break;
            case C4A_AUG_RANDOM_X_OP_SUB: r = X[k] - v; break;
            default:                        r = X[k];      break;
        }
        if (r >  C4A_FLOAT32_MAX) r =  C4A_FLOAT32_MAX;
        if (r < -C4A_FLOAT32_MAX) r = -C4A_FLOAT32_MAX;
        out[k] = r;
    }
    return C4A_OK;
}
