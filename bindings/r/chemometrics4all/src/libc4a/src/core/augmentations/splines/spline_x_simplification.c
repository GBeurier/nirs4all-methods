/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_x_simplification.h"

#include <stdlib.h>
#include <string.h>

struct c4a_aug_spline_x_simplify_state_t {
    int32_t spline_points;
    int     uniform;
};

c4a_aug_spline_x_simplify_state_t* c4a_aug_spline_x_simplify_state_new(
    int32_t spline_points, int uniform) {
    c4a_aug_spline_x_simplify_state_t* s =
        (c4a_aug_spline_x_simplify_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->spline_points = spline_points;
    s->uniform       = uniform ? 1 : 0;
    return s;
}

void c4a_aug_spline_x_simplify_state_free(
    c4a_aug_spline_x_simplify_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_spline_x_simplify_state_apply(
    const c4a_aug_spline_x_simplify_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    (void)state;
    (void)rng_void;
    (void)X;
    (void)rows;
    (void)cols;
    (void)out;
    /* v2-deferred — see DEFERRALS.md. */
    return C4A_ERR_NOT_IMPLEMENTED;
}
