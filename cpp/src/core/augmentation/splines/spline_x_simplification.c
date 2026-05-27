/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_x_simplification.h"

#include <stdlib.h>
#include <string.h>

#include "spline_simplify_common.h"

struct n4m_aug_spline_x_simplify_state_t {
    int32_t spline_points;
    int     uniform;
};

n4m_aug_spline_x_simplify_state_t* n4m_aug_spline_x_simplify_state_new(
    int32_t spline_points, int uniform) {
    n4m_aug_spline_x_simplify_state_t* s =
        (n4m_aug_spline_x_simplify_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->spline_points = spline_points;
    s->uniform       = uniform ? 1 : 0;
    return s;
}

void n4m_aug_spline_x_simplify_state_free(
    n4m_aug_spline_x_simplify_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_spline_x_simplify_state_apply(
    const n4m_aug_spline_x_simplify_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL) return N4M_ERR_NULL_POINTER;
    /* Spline_X_Simplification does NOT np.unique the uniform linspace indices. */
    return n4m_spline_simplify_apply_impl(rng_void, X, rows, cols,
                                          state->spline_points, state->uniform,
                                          /*unique_in_uniform=*/0, out);
}
