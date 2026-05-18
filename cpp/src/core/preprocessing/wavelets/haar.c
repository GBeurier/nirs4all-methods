/* SPDX-License-Identifier: CECILL-2.1 */

#include "haar.h"

#include <stdlib.h>

#include "core/common/wavelet_kernels.h"
#include "wavelet.h"

struct c4a_pp_haar_state_t {
    c4a_pp_wavelet_state_t* inner;
};

c4a_pp_haar_state_t* c4a_pp_haar_state_new(void) {
    c4a_pp_haar_state_t* s = (c4a_pp_haar_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->inner = c4a_pp_wavelet_state_new(C4A_WAVELET_HAAR,
                                          C4A_WAVELET_MODE_PERIODIZATION);
    if (s->inner == NULL) {
        free(s);
        return NULL;
    }
    return s;
}

void c4a_pp_haar_state_free(c4a_pp_haar_state_t* state) {
    if (state == NULL) return;
    c4a_pp_wavelet_state_free(state->inner);
    free(state);
}

c4a_status_t c4a_pp_haar_state_apply(
    const c4a_pp_haar_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out) {
    if (state == NULL) return C4A_ERR_NULL_POINTER;
    return c4a_pp_wavelet_state_apply(state->inner, X, rows, cols,
                                       out_cols, out);
}
