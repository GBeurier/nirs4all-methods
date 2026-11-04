/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#ifndef N4M_MIXED_EFFECTS_H
#define N4M_MIXED_EFFECTS_H
#include "n4m/n4m.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_mixed_effects_fit_state_t {
    N4M_MIXED_EFFECTS_FITTED = 0,
    N4M_MIXED_EFFECTS_FITTED_BOUNDARY = 1,
    N4M_MIXED_EFFECTS_RANK_DEFICIENT = 2,
    N4M_MIXED_EFFECTS_UNIDENTIFIABLE = 3,
    N4M_MIXED_EFFECTS_NONCONVERGED = 4,
    N4M_MIXED_EFFECTS_NUMERICAL_REFUSAL = 5
} n4m_mixed_effects_fit_state_t;

typedef enum n4m_mixed_effects_reason_t {
    N4M_MIXED_EFFECTS_REASON_NONE = 0,
    N4M_MIXED_EFFECTS_REASON_FIXED_RANK = 1,
    N4M_MIXED_EFFECTS_REASON_NO_RESIDUAL_DF = 2,
    N4M_MIXED_EFFECTS_REASON_CONSTANT_RESPONSE = 3,
    N4M_MIXED_EFFECTS_REASON_INSUFFICIENT_GROUP_LEVELS = 4,
    N4M_MIXED_EFFECTS_REASON_CONFOUNDED_VARIANCES = 5,
    N4M_MIXED_EFFECTS_REASON_ZERO_RESIDUAL = 6,
    N4M_MIXED_EFFECTS_REASON_OPTIMIZER_BUDGET = 7,
    N4M_MIXED_EFFECTS_REASON_OPTIMIZER_ROUNDOFF = 8,
    N4M_MIXED_EFFECTS_REASON_OPTIMIZER_FAILURE = 9,
    N4M_MIXED_EFFECTS_REASON_NONFINITE_RESULT = 10,
    N4M_MIXED_EFFECTS_REASON_ENGINE_REJECTED = 11
} n4m_mixed_effects_reason_t;

typedef struct n4m_mixed_effects_options_v1_t {
    uint64_t struct_size;
    int32_t reml;              /* exactly 0 (ML) or 1 (REML); default 1 */
    int32_t reserved;          /* must be 0 */
    int64_t max_evaluations;   /* 1..100000, default 1000 */
    double tolerance;          /* relative objective tolerance; default 1e-8 */
    double rank_tolerance;     /* relative singular-value cutoff; default 1e-10 */
    double boundary_tolerance; /* random/residual variance ratio cutoff; default 1e-8 */
} n4m_mixed_effects_options_v1_t;

typedef struct n4m_mixed_effects_summary_v1_t {
    uint64_t struct_size; /* caller must set to sizeof(this struct) before fit */
    int32_t fit_state;    /* n4m_mixed_effects_fit_state_t */
    int32_t reason;       /* n4m_mixed_effects_reason_t */
    int32_t converged;    /* genuine optimizer termination, not an adequacy certificate */
    int32_t singular;
    int32_t boundary;
    int32_t optimizer_status; /* 0 not run, 1 converged, 2 budget, 3 roundoff, 4 failed */
    int64_t observations;
    int64_t fixed_effects;
    int64_t group_terms;
    int64_t design_rank;
    int64_t total_group_levels;
    int64_t evaluations;
    double objective;        /* ML deviance or REML criterion; NaN if unavailable */
    double log_likelihood;   /* -objective/2; RESTRICTED log likelihood for REML */
    double residual_df;      /* n-rank(X), also reported for ML */
    double condition_number; /* unscaled X 2-norm condition; Inf if rank deficient */
} n4m_mixed_effects_summary_v1_t;

N4M_API n4m_status_t n4m_mixed_effects_options_init_v1(n4m_mixed_effects_options_v1_t* options);
N4M_API int n4m_mixed_effects_backend_available_v1(void);

/* Gaussian y = X beta + sum_j Z_j b_j + epsilon, independent zero-mean Gaussian
 * random intercepts and IID Gaussian residual. X is finite F64 n*p, y is finite
 * F64 n*1; groups is I64 n*k with nonnegative IDs, interpreted independently in
 * each column. Include any fixed intercept explicitly in X; no intercept,
 * formula parsing, contrasts, centering or normalization are added. Crossed and
 * nested factors use separate columns; nested IDs must encode their full parent
 * context. Equal IDs across different columns have no implied relationship.
 *
 * Resource caps: n<=8192, p<=64, 1<=k<=8, n*p<=262144, sum of distinct group
 * levels<=1024, n*sum(levels)<=4194304. Each random term needs >=2 levels.
 * Confounded residual/random covariance directions after removing fixed effects
 * are refused. The caller owns matching, metric scale and dependence policy;
 * independent folds or causal effects are never inferred.
 *
 * Buffers have EXACT sizes p, p*p, p, k+1; covariance is row-major. Coefficients
 * and covariance axes preserve X column order. Variances are residual first,
 * then one per groups column in original order. Buffers and summary must not
 * overlap each other. All views support nonnegative element strides.
 *
 * N4M_OK reports an atomic statistical outcome. Only FITTED/FITTED_BOUNDARY
 * expose estimates; other fit_state values withhold ALL estimate buffers as NaN
 * and provide refusal diagnostics. Boundary estimates are explicitly singular/
 * boundary flagged. SE is conditional plug-in sqrt(diag(cov_beta)); no p-values,
 * confidence intervals, variance-parameter uncertainty or causal claims.
 *
 * Technical errors (bad views, nonfinite inputs, options, resource cap, backend
 * absent, Rust panic or C++ exception) leave ALL outputs unchanged. Backend-off
 * builds preserve these symbols and return N4M_ERR_BACKEND_UNAVAILABLE for a
 * version-compatible nonnull call, without an alternative engine. No allocation
 * crosses the ABI. Set summary.struct_size before every call. */
N4M_API n4m_status_t n4m_mixed_effects_fit_v1(const n4m_matrix_view_t* X,
                                              const n4m_matrix_view_t* y,
                                              const n4m_matrix_view_t* groups,
                                              const n4m_mixed_effects_options_v1_t* options,
                                              double* beta,
                                              int64_t beta_size,
                                              double* covariance,
                                              int64_t covariance_size,
                                              double* standard_errors,
                                              int64_t standard_errors_size,
                                              double* variances,
                                              int64_t variances_size,
                                              n4m_mixed_effects_summary_v1_t* summary);
#ifdef __cplusplus
}
#endif
#endif
