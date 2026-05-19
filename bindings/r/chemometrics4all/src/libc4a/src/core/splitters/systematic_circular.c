/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SystematicCircularSplitter — sort by y, rotate by offset, take every
 * n-th sample for training.
 *
 * The C engine and the Python reference fixture use the same PCG64
 * uniform draw for the rotation offset so the integer indices agree.
 */

#include "systematic_circular.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

c4a_split_syscirc_state_t* c4a_split_syscirc_state_new(double test_size,
                                                        uint64_t seed) {
    c4a_split_syscirc_state_t* s =
        (c4a_split_syscirc_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    s->seed      = seed;
    return s;
}

void c4a_split_syscirc_state_free(c4a_split_syscirc_state_t* state) {
    free(state);
}

/* Stable ascending argsort by y. NumPy's argsort with kind='stable' uses
 * timsort which preserves sample-index order on ties. Plain insertion sort
 * on a small comparison key matches that on integer ties. */
static const double* g_y_ref_syscirc;
static int cmp_idx_by_y_stable(const void* a, const void* b) {
    const int64_t ia = *(const int64_t*)a;
    const int64_t ib = *(const int64_t*)b;
    if (g_y_ref_syscirc[ia] < g_y_ref_syscirc[ib]) return -1;
    if (g_y_ref_syscirc[ia] > g_y_ref_syscirc[ib]) return 1;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static int cmp_int64_asc(const void* a, const void* b) {
    const int64_t ia = *(const int64_t*)a;
    const int64_t ib = *(const int64_t*)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

c4a_status_t c4a_split_syscirc_apply(const c4a_split_syscirc_state_t* state,
                                      const double* y, int64_t rows,
                                      c4a_split_result_t* out) {
    if (state == NULL || y == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2) return C4A_ERR_INVALID_ARGUMENT;
    int64_t n_train = 0, n_test = 0;
    c4a_status_t st = c4a_split_compute_train_test_count(rows, state->test_size,
                                                          &n_train, &n_test);
    if (st != C4A_OK) return st;

    /* 1. argsort y ascending (stable). */
    int64_t* ordered = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    int64_t* rotated = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (ordered == NULL || rotated == NULL) {
        free(ordered); free(rotated);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) ordered[i] = i;
    g_y_ref_syscirc = y;
    qsort(ordered, (size_t)rows, sizeof(int64_t), cmp_idx_by_y_stable);

    /* 2. Random offset in [0, rows). */
    c4a_rng_pcg64 rng;
    c4a_pcg64_engine_seed(&rng, state->seed);
    const double u = c4a_split_rng_uniform01(&rng);
    int64_t offset = (int64_t)floor(u * (double)rows);
    if (offset < 0) offset = 0;
    if (offset >= rows) offset = rows - 1;

    /* 3. np.roll(ordered, offset): rotated[i] = ordered[(i - offset) mod rows]. */
    for (int64_t i = 0; i < rows; ++i) {
        int64_t src = i - offset;
        src %= rows;
        if (src < 0) src += rows;
        rotated[i] = ordered[src];
    }

    /* 4. step = rows / n_train; indices = [round(step * k) for k in range(n_train)]. */
    const double step = (double)rows / (double)n_train;
    int64_t* picked_positions = (int64_t*)malloc((size_t)n_train * sizeof(int64_t));
    if (picked_positions == NULL) {
        free(ordered); free(rotated);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    char* picked_mask = (char*)calloc((size_t)rows, sizeof(char));
    if (picked_mask == NULL) {
        free(ordered); free(rotated); free(picked_positions);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    /* Python's round() uses banker's rounding for halves; C's rint with
     * FE_TONEAREST matches. We use rint here. */
    for (int64_t k = 0; k < n_train; ++k) {
        int64_t p = (int64_t)rint(step * (double)k);
        if (p < 0) p = 0;
        if (p >= rows) p = rows - 1;
        picked_positions[k] = p;
        picked_mask[p] = 1;
    }

    /* 5. Output: index_train = rotated[picked_positions] (kept in the
     *    order positions were picked, matching Python's rotated_idx[indices]);
     *    index_test = np.delete(rotated, picked_positions) (rotation order,
     *    with picked positions removed). */
    int64_t actual_train = 0;
    for (int64_t p = 0; p < rows; ++p) if (picked_mask[p]) ++actual_train;
    int64_t actual_test = rows - actual_train;
    st = c4a_split_result_alloc(out, actual_train, actual_test);
    if (st != C4A_OK) {
        free(ordered); free(rotated); free(picked_positions); free(picked_mask);
        return st;
    }
    /* Emit train in pick order, dedup'd against duplicate positions
     * (rint collisions on small datasets). */
    int64_t ti = 0;
    for (int64_t k = 0; k < n_train; ++k) {
        const int64_t pos = picked_positions[k];
        if (picked_mask[pos]) {
            out->train_idx[ti++] = rotated[pos];
            picked_mask[pos] = 0;
        }
    }
    /* Re-mark picked positions for the test scan. */
    for (int64_t k = 0; k < n_train; ++k) picked_mask[picked_positions[k]] = 1;
    int64_t te = 0;
    for (int64_t p = 0; p < rows; ++p) {
        if (!picked_mask[p]) out->test_idx[te++] = rotated[p];
    }
    out->n_train = ti;
    out->n_test  = te;
    (void)cmp_int64_asc;

    free(ordered); free(rotated); free(picked_positions); free(picked_mask);
    return C4A_OK;
}
