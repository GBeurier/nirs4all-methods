/* SPDX-License-Identifier: CECILL-2.1 */

#include "haar.h"

#include <stdlib.h>

#include "core/common/wavelet_kernels.h"

struct c4a_pp_haar_state_t {
    int unused;
};

c4a_pp_haar_state_t* c4a_pp_haar_state_new(void) {
    c4a_pp_haar_state_t* s = (c4a_pp_haar_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->unused = 0;
    return s;
}

void c4a_pp_haar_state_free(c4a_pp_haar_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_haar_state_apply(
    const c4a_pp_haar_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0) return C4A_OK;
    const int64_t m = c4a_wavelet_dwt_output_length(
        cols, C4A_WAVELET_HAAR, C4A_WAVELET_MODE_PERIODIZATION);
    if (out_cols != 2 * m) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    const double s = 0.7071067811865476;
    const int64_t pairs = cols / 2;
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        double* o = out + (size_t)i * (size_t)out_cols;
        double* cA = o;
        double* cD = o + m;
        for (int64_t k = 0; k < pairs; ++k) {
            const int64_t j = 2 * k;
            const double a = row[j];
            const double b = row[j + 1];
            cA[k] = (a + b) * s;
            cD[k] = (a - b) * s;
        }
        if ((cols & 1) != 0) {
            const double a = row[cols - 1];
            cA[pairs] = (a + a) * s;
            cD[pairs] = 0.0;
        }
    }
    return C4A_OK;
}
