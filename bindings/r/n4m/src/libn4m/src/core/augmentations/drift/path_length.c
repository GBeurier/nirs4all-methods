/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * PathLengthAugmenter — reference implementation.
 *
 * Draws one Gaussian per row, scales by `path_length_std`, adds the unit
 * mean, then clips below at `min_path_length` and broadcasts across the
 * row.
 *
 * `L[i] = max(1.0 + path_length_std * N[i], min_path_length)`
 * `out[i, j] = L[i] * X[i, j]`.
 */

#include "path_length.h"

#include <stdlib.h>

struct n4m_aug_path_length_state_t {
    n4m_rng_pcg64* rng;   /* borrowed; not owned. */
    double path_length_std;
    double min_path_length;
};

n4m_aug_path_length_state_t* n4m_aug_path_length_state_new(
    n4m_rng_pcg64* rng,
    double path_length_std,
    double min_path_length) {
    n4m_aug_path_length_state_t* s =
        (n4m_aug_path_length_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng             = rng;
    s->path_length_std = path_length_std;
    s->min_path_length = min_path_length;
    return s;
}

void n4m_aug_path_length_state_free(n4m_aug_path_length_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_path_length_state_apply(
    n4m_aug_path_length_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return N4M_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return N4M_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0)               return N4M_OK;

    double* normals = (double*)malloc((size_t)rows * sizeof(double));
    if (normals == NULL) return N4M_ERR_OUT_OF_MEMORY;
    n4m_pcg64_engine_standard_normal_fill(state->rng, normals, (size_t)rows);

    const double std_ = state->path_length_std;
    const double min_ = state->min_path_length;
    for (int64_t i = 0; i < rows; ++i) {
        double L = 1.0 + std_ * normals[(size_t)i];
        if (L < min_) L = min_;
        const size_t off = (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            out[off + (size_t)j] = L * X[off + (size_t)j];
        }
    }

    free(normals);
    return N4M_OK;
}
