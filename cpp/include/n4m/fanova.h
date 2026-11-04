/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#ifndef N4M_FANOVA_H
#define N4M_FANOVA_H
#include "n4m/n4m.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_fanova_fit_state_t {
    N4M_FANOVA_FITTED = 0,
    N4M_FANOVA_CONSTANT_RESPONSE = 1,
    N4M_FANOVA_NO_INFORMATIVE_TREES = 2,
    N4M_FANOVA_NUMERICAL_REFUSAL = 3
} n4m_fanova_fit_state_t;

typedef struct n4m_fanova_options_v1_t {
    uint64_t struct_size;
    int32_t num_trees; /* 1..128, default 32 */
    int32_t max_depth; /* 1..12, default 8 */
    int32_t min_samples_leaf; /* 1..8192, default 2 */
    int32_t max_features; /* 0 uses all features; otherwise 1..p */
    uint64_t seed; /* std::mt19937_64 seed, default 0 */
    int32_t max_order; /* exactly 1 or 2, default 2 */
    int32_t reserved; /* must be 0 */
    int64_t max_marginal_evaluations; /* 1..10000000, default 1000000 */
} n4m_fanova_options_v1_t;

typedef struct n4m_fanova_summary_v1_t {
    uint64_t struct_size; /* caller sets sizeof(this struct) */
    int32_t fit_state; /* n4m_fanova_fit_state_t */
    int32_t reserved;
    int64_t observations;
    int64_t features;
    int64_t trees;
    int64_t informative_trees;
    int64_t marginal_evaluations;
    double mean_tree_variance; /* population variance under the product measure */
    double remainder_fraction; /* mean remaining fraction after requested orders */
    double train_rmse;
    double validation_rmse; /* NaN without validation */
    double validation_r2; /* NaN without validation or for constant validation y */
} n4m_fanova_summary_v1_t;

N4M_API n4m_status_t n4m_fanova_options_init_v1(n4m_fanova_options_v1_t* options);
N4M_API int n4m_fanova_backend_available_v1(void);

/* Fit a fresh regression forest, then integrate its per-tree conditional means
 * exactly over the tree partitions under an INDEPENDENT UNIFORM PRODUCT measure.
 * X is finite F64 n*p, y finite F64 n*1, bounds finite F64 p*2. types has p entries:
 * 0 is continuous with strictly increasing bounds; K=1..128 is categorical with
 * exact integer codes 0..K-1 and bounds exactly [0,K-1]. All observed and optional
 * validation X values must belong to these domains. Nonnegative element strides
 * are supported. X_validation and y_validation must both be null or both supplied.
 * Domain choice, independent-variable assumption, matching and response direction
 * belong to the host. This is surrogate variance attribution, never a causal effect
 * or a generalization guarantee; train fit is descriptive and holdout is optional.
 *
 * Resource caps: 2<=n<=8192, 1<=p<=32, n*p<=262144; validation n<=8192.
 * Exact integration also respects max_marginal_evaluations; budget exhaustion is
 * N4M_ERR_UNSUPPORTED. main_effects has EXACT size p*2, each row fraction, tree SD.
 * pair_effects has EXACT size p*p*2, symmetric with NaN diagonal; all pairs are NaN
 * for max_order=1. Fractions normalize EACH informative tree by its total variance
 * before averaging. SD is sample variability across those trees, NOT an SE or CI;
 * SD is NaN with one informative tree. Remainder includes all unreported orders.
 * Constant trees are excluded and counted; mean_tree_variance includes all trees.
 *
 * N4M_OK includes statistical refusal: constant y, no informative trees or numerical
 * refusal yield NaN effect buffers and explicit fit_state. Technical errors leave
 * ALL outputs unchanged. Output buffers and summary must not overlap each other or
 * inputs. Backend-off builds retain symbols and return N4M_ERR_BACKEND_UNAVAILABLE
 * for version-compatible nonnull calls. No alternative engine or allocation crosses
 * the ABI. summary.struct_size must be set before every call. */
N4M_API n4m_status_t n4m_fanova_fit_v1(
    const n4m_matrix_view_t* X, const n4m_matrix_view_t* y,
    const n4m_matrix_view_t* bounds, const uint32_t* types,
    const n4m_matrix_view_t* X_validation, const n4m_matrix_view_t* y_validation,
    const n4m_fanova_options_v1_t* options,
    double* main_effects, int64_t main_effects_size,
    double* pair_effects, int64_t pair_effects_size,
    n4m_fanova_summary_v1_t* summary);
#ifdef __cplusplus
}
#endif
#endif
