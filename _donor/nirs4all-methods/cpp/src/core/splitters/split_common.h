/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal helpers shared by the Phase 11 sample-partitioning splitters.
 *
 * The public C ABI lives in n4m.h (§15). The wrappers in
 * c_api/c_api_splitters.cpp delegate to the per-operator engines under
 * cpp/src/core/splitters/. This header centralises the result-buffer
 * allocation strategy and the distance-matrix primitives.
 *
 * Pure C, no dependencies. INTERNAL: never installed.
 */
#ifndef N4M_CORE_SPLITTERS_COMMON_H
#define N4M_CORE_SPLITTERS_COMMON_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Result buffer (canonical definition lives in §17 of n4m.h; we include
 * n4m.h above and reuse that type here).
 * ------------------------------------------------------------------------- */

/* Allocate train/test index arrays sized exactly n_train and n_test in a
 * single backing buffer; assigns out->train_idx, out->test_idx, out->n_train,
 * out->n_test, out->_owner. Returns N4M_OK or N4M_ERR_OUT_OF_MEMORY.
 *
 * On failure all out->* fields are cleared. On success the caller fills the
 * indices and ownership passes to the caller / public destroy function. */
n4m_status_t n4m_split_result_alloc(n4m_split_result_t* out,
                                    int64_t n_train, int64_t n_test);

/* Reset a result to its empty state without freeing memory. Used when an
 * engine wants to clear an already-allocated output before re-populating it. */
void n4m_split_result_clear(n4m_split_result_t* r);

/* Free the backing buffer of a result. NULL-safe. After the call the result
 * struct is in its empty state. */
void n4m_split_result_free(n4m_split_result_t* r);

/* ---------------------------------------------------------------------------
 * Validation helpers.
 *
 * Compute n_train from a test_size fraction with the same convention as
 * sklearn's `_validate_shuffle_split`:
 *   - 0 < test_size < 1
 *   - n_test = ceil(test_size * n_samples)
 *   - n_train = n_samples - n_test
 * Returns N4M_OK on success (and writes n_train / n_test), or
 * N4M_ERR_INVALID_ARGUMENT on out-of-range test_size / too-small n_samples.
 * ------------------------------------------------------------------------- */
n4m_status_t n4m_split_compute_train_test_count(int64_t n_samples,
                                                double test_size,
                                                int64_t* out_n_train,
                                                int64_t* out_n_test);

/* ---------------------------------------------------------------------------
 * Pairwise Euclidean distance matrix.
 *
 * Computes D[i, j] = ||X[i, :] - X[j, :]||_2 for i, j in [0, rows). The
 * output buffer must be of length rows*rows. Memory layout: row-major,
 * D[i*rows + j].
 *
 * Returns N4M_OK or N4M_ERR_NULL_POINTER / N4M_ERR_INVALID_ARGUMENT.
 * ------------------------------------------------------------------------- */
n4m_status_t n4m_split_euclidean_dist(const double* X, int64_t rows,
                                       int64_t cols, double* D);

/* Compute Euclidean distances for a [rows x cols] X relative to a single
 * reference point r of length cols, writing into out[rows]. */
n4m_status_t n4m_split_euclidean_dist_to_point(const double* X,
                                                int64_t rows, int64_t cols,
                                                const double* r, double* out);

/* ---------------------------------------------------------------------------
 * Max-min distance subset selection.
 *
 * Implements the Kennard-Stone / SPXY core: from the n x n distance matrix
 * D, pick the two farthest points, then iteratively pick the next point that
 * maximises the minimum distance to the already-selected set.
 *
 * Writes train_size indices to train_idx[0..train_size) and the remaining
 * n - train_size to test_idx[0..n - train_size). Both buffers must be
 * pre-allocated by the caller.
 *
 * Returns N4M_OK or N4M_ERR_INVALID_ARGUMENT for train_size < 2 or > n.
 * ------------------------------------------------------------------------- */
n4m_status_t n4m_split_max_min_selection(const double* D, int64_t n,
                                          int64_t train_size,
                                          int64_t* train_idx,
                                          int64_t* test_idx);

/* In-place normalise an n x n distance matrix so its max value becomes 1
 * (safe no-op when the max is already 0). */
void n4m_split_normalise_distance_matrix(double* D, int64_t n);

/* Uniform double in [0, 1) drawn from a PCG64 engine. Matches NumPy's
 * `next_double` callback `(u64 >> 11) * 2^-53`. */
double n4m_split_rng_uniform01(n4m_rng_pcg64* rng);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_SPLITTERS_COMMON_H */
