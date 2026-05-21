/* SPDX-License-Identifier: CECILL-2.1 */

#include "haar.h"

#include <stdlib.h>

#include "core/common/wavelet_kernels.h"

struct n4m_pp_haar_state_t {
    int unused;
};

n4m_pp_haar_state_t* n4m_pp_haar_state_new(void) {
    n4m_pp_haar_state_t* s = (n4m_pp_haar_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->unused = 0;
    return s;
}

void n4m_pp_haar_state_free(n4m_pp_haar_state_t* state) {
    free(state);
}

n4m_status_t n4m_pp_haar_state_apply(
    const n4m_pp_haar_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0) return N4M_OK;
    const int64_t m = n4m_wavelet_dwt_output_length(
        cols, N4M_WAVELET_HAAR, N4M_WAVELET_MODE_PERIODIZATION);
    if (out_cols != 2 * m) {
        return N4M_ERR_SHAPE_MISMATCH;
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
    return N4M_OK;
}
