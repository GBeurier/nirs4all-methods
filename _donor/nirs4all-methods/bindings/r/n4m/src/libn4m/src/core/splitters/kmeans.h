/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the KMeans-based splitter. Public ABI lives in
 * n4m.h §15. Uses a vendored k-means++ initialisation + Lloyd iterations.
 *
 * Reference: nirs4all.operators.splitters.KMeansSplitter (which delegates
 * to sklearn.cluster.KMeans). The C engine ships its own k-means++
 * implementation seeded by PCG64 so the reference fixture (which uses the
 * same algorithm via NumPy default_rng) reproduces byte-exact integer
 * indices.
 *
 * Algorithm (matches the Python reference shipped with the parity fixture):
 *   1. k = n_train (one cluster per training sample).
 *   2. Initialise k centroids via k-means++ using PCG64-drawn uniforms.
 *   3. Lloyd iterations until centroids stabilise or max_iter reached.
 *   4. For each centroid, pick the sample with the minimum Euclidean
 *      distance (NumPy argmin convention: first occurrence wins on ties).
 *   5. Deduplicate (np.unique) — train set is the unique sorted set of the
 *      k argmins. Test set is the complement in ascending order.
 */
#ifndef N4M_CORE_SPLITTERS_KMEANS_H
#define N4M_CORE_SPLITTERS_KMEANS_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_split_kmeans_state_t {
    double   test_size;
    uint64_t seed;
    int32_t  max_iter;
} n4m_split_kmeans_state_t;

n4m_split_kmeans_state_t* n4m_split_kmeans_state_new(double test_size,
                                                      uint64_t seed,
                                                      int32_t max_iter);
void n4m_split_kmeans_state_free(n4m_split_kmeans_state_t* state);

n4m_status_t n4m_split_kmeans_apply(const n4m_split_kmeans_state_t* state,
                                     const double* X, int64_t rows, int64_t cols,
                                     n4m_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_SPLITTERS_KMEANS_H */
