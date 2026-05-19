/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Gaussian filter reference implementation.  Matches
 * `scipy.ndimage.gaussian_filter1d` byte-for-byte on the kernel construction
 * (Hermite recursion) and applies the resulting weights via the same
 * correlate-1d index map (`out[i] = sum_k weights[k] * input[i + k - lw]`).
 *
 * Kernel build (per `_gaussian_kernel1d` in scipy 1.11.4):
 *   lw      = int(truncate * sigma + 0.5)
 *   x[i]    = i - lw           for i in [0, 2*lw + 1)
 *   phi[i]  = exp(-x[i]^2 / (2 * sigma^2)) / sum(phi)
 *   if order == 0:
 *       kernel_natural = phi
 *   else:
 *       q       = [1, 0, ..., 0]                # length order + 1
 *       sigma2  = sigma * sigma
 *       for _ in range(order):
 *           q_new[i]     += (i + 1) * q[i + 1]     # D matrix
 *           q_new[i + 1] += -q[i] / sigma2         # P matrix
 *       poly[i] = sum_j q[j] * x[i]^j
 *       kernel_natural = poly * phi
 *   kernel = reverse(kernel_natural)             # for cross-correlation
 */

#include "gaussian.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/padding.h"

struct c4a_pp_gaussian_state_t {
    double                 sigma;
    int32_t                order;
    c4a_pp_gaussian_mode_t mode;
    double                 cval;
    double                 truncate;
    int32_t                lw;        /* kernel radius */
    double*                kernel;    /* length 2 * lw + 1, ready for correlation */
};

c4a_pp_gaussian_state_t* c4a_pp_gaussian_state_new(
    double sigma, int32_t order,
    c4a_pp_gaussian_mode_t mode,
    double cval, double truncate) {
    if (!(sigma > 0.0)) {
        return NULL;
    }
    if (order < 0) {
        return NULL;
    }
    if (!(truncate >= 0.0)) {
        return NULL;
    }
    c4a_pp_gaussian_state_t* s =
        (c4a_pp_gaussian_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->sigma    = sigma;
    s->order    = order;
    s->mode     = mode;
    s->cval     = cval;
    s->truncate = truncate;
    /* `int(truncate * sigma + 0.5)` matches scipy's lw computation
     * exactly (Python int() truncates toward zero, but since the operand
     * is >= 0 this is equivalent to floor()). */
    const int32_t lw = (int32_t)(truncate * sigma + 0.5);
    s->lw = lw;
    const int32_t length = 2 * lw + 1;
    s->kernel = (double*)malloc((size_t)length * sizeof(double));
    if (s->kernel == NULL) {
        free(s);
        return NULL;
    }
    /* Build natural kernel into the destination, then reverse in-place. */
    const double sigma2 = sigma * sigma;
    double* phi = s->kernel;
    /* phi[i] = exp(-0.5 / sigma^2 * x[i]^2). */
    double phi_sum = 0.0;
    for (int32_t i = 0; i < length; ++i) {
        const double xi = (double)(i - lw);
        phi[i] = exp(-0.5 / sigma2 * xi * xi);
        phi_sum += phi[i];
    }
    /* Normalize phi. */
    for (int32_t i = 0; i < length; ++i) {
        phi[i] /= phi_sum;
    }
    if (order > 0) {
        /* Hermite-polynomial recursion to build q (length order + 1). */
        const int32_t qn = order + 1;
        double* q     = (double*)malloc((size_t)qn * sizeof(double));
        double* q_new = (double*)malloc((size_t)qn * sizeof(double));
        if (q == NULL || q_new == NULL) {
            free(q); free(q_new);
            free(s->kernel);
            free(s);
            return NULL;
        }
        for (int32_t i = 0; i < qn; ++i) q[i] = 0.0;
        q[0] = 1.0;
        for (int32_t step = 0; step < order; ++step) {
            for (int32_t i = 0; i < qn; ++i) q_new[i] = 0.0;
            /* D matrix: q_new[i] += (i + 1) * q[i + 1] for i in [0, order). */
            for (int32_t i = 0; i < order; ++i) {
                q_new[i] += (double)(i + 1) * q[i + 1];
            }
            /* P matrix: q_new[i + 1] += -q[i] / sigma2 for i in [0, order). */
            for (int32_t i = 0; i < order; ++i) {
                q_new[i + 1] += -q[i] / sigma2;
            }
            /* Copy back. */
            for (int32_t i = 0; i < qn; ++i) q[i] = q_new[i];
        }
        /* Evaluate polynomial poly[i] = sum_j q[j] * x[i]^j via Horner. */
        for (int32_t i = 0; i < length; ++i) {
            const double xi = (double)(i - lw);
            double val = q[qn - 1];
            for (int32_t j = qn - 2; j >= 0; --j) {
                val = val * xi + q[j];
            }
            phi[i] *= val;
        }
        free(q);
        free(q_new);
    }
    /* Reverse the kernel in place (scipy passes weights[::-1] to correlate1d). */
    for (int32_t i = 0, j = length - 1; i < j; ++i, --j) {
        const double tmp = phi[i];
        phi[i] = phi[j];
        phi[j] = tmp;
    }
    return s;
}

void c4a_pp_gaussian_state_free(c4a_pp_gaussian_state_t* state) {
    if (state == NULL) return;
    free(state->kernel);
    free(state);
}

c4a_status_t c4a_pp_gaussian_state_apply(
    const c4a_pp_gaussian_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }
    const int32_t lw = state->lw;
    const int32_t length = 2 * lw + 1;
    const double* kernel = state->kernel;
    const int in_place = (X == out);

    c4a_pad_mode_t pad = C4A_PAD_REFLECT;
    switch (state->mode) {
        case C4A_PP_GAUSSIAN_REFLECT:  pad = C4A_PAD_REFLECT;  break;
        case C4A_PP_GAUSSIAN_CONSTANT: pad = C4A_PAD_CONSTANT; break;
        case C4A_PP_GAUSSIAN_NEAREST:  pad = C4A_PAD_NEAREST;  break;
        case C4A_PP_GAUSSIAN_MIRROR:   pad = C4A_PAD_MIRROR;   break;
        case C4A_PP_GAUSSIAN_WRAP:     pad = C4A_PAD_WRAP;     break;
        default:                       return C4A_ERR_INVALID_ARGUMENT;
    }

    double* scratch = NULL;
    if (in_place) {
        scratch = (double*)malloc((size_t)cols * sizeof(double));
    }
    if (in_place && scratch == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const double* row_in  = X + (size_t)r * (size_t)cols;
        double*       row_out = out + (size_t)r * (size_t)cols;
        double*       dst     = in_place ? scratch : row_out;

        const int64_t left_end = ((int64_t)lw < cols) ? (int64_t)lw : cols;
        const int64_t interior_begin = (int64_t)lw;
        const int64_t interior_end = (cols - (int64_t)lw > interior_begin)
            ? cols - (int64_t)lw
            : interior_begin;

        for (int64_t j = 0; j < left_end; ++j) {
            double acc = 0.0;
            for (int32_t k = 0; k < length; ++k) {
                const int64_t src_idx = j + (int64_t)k - (int64_t)lw;
                double v;
                if (src_idx >= 0 && src_idx < cols) {
                    v = row_in[src_idx];
                } else if (pad == C4A_PAD_CONSTANT) {
                    v = state->cval;
                } else {
                    const int64_t resolved =
                        c4a_pad_resolve_index(src_idx, cols, pad);
                    v = row_in[resolved];
                }
                acc += kernel[k] * v;
            }
            dst[j] = acc;
        }

        if (interior_end > interior_begin) {
            if (length == 9) {
                for (int64_t j = interior_begin; j < interior_end; ++j) {
                    const double* x = row_in + (size_t)(j - (int64_t)lw);
                    double acc = 0.0;
                    acc += kernel[0] * x[0];
                    acc += kernel[1] * x[1];
                    acc += kernel[2] * x[2];
                    acc += kernel[3] * x[3];
                    acc += kernel[4] * x[4];
                    acc += kernel[5] * x[5];
                    acc += kernel[6] * x[6];
                    acc += kernel[7] * x[7];
                    acc += kernel[8] * x[8];
                    dst[j] = acc;
                }
            } else {
                for (int64_t j = interior_begin; j < interior_end; ++j) {
                    const double* x = row_in + (size_t)(j - (int64_t)lw);
                    double acc = 0.0;
                    for (int32_t k = 0; k < length; ++k) {
                        acc += kernel[k] * x[k];
                    }
                    dst[j] = acc;
                }
            }
        }

        const int64_t right_begin =
            (interior_end > left_end) ? interior_end : left_end;
        for (int64_t j = right_begin; j < cols; ++j) {
            double acc = 0.0;
            for (int32_t k = 0; k < length; ++k) {
                const int64_t src_idx = j + (int64_t)k - (int64_t)lw;
                double v;
                if (src_idx >= 0 && src_idx < cols) {
                    v = row_in[src_idx];
                } else if (pad == C4A_PAD_CONSTANT) {
                    v = state->cval;
                } else {
                    const int64_t resolved =
                        c4a_pad_resolve_index(src_idx, cols, pad);
                    v = row_in[resolved];
                }
                acc += kernel[k] * v;
            }
            dst[j] = acc;
        }

        if (in_place) {
            memcpy(row_out, scratch, (size_t)cols * sizeof(double));
        }
    }
    free(scratch);
    return C4A_OK;
}
