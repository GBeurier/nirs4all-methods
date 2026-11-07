/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#ifndef N4M_RANKING_H
#define N4M_RANKING_H
#include "n4m/n4m.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Each row is ONE comparable task/dataset/split block; columns are arbitrary
 * pipeline/component candidates. Hosts own identity, matching, and dependence
 * policy. No cross-row or unmatched pair is ever manufactured. */
typedef enum n4m_ranking_method_t {
    N4M_RANKING_AVERAGE_RANK = 0,
    N4M_RANKING_ELO = 1,
    N4M_RANKING_BRADLEY_TERRY = 2,
    N4M_RANKING_TRUESKILL_1V1 = 3,
    N4M_RANKING_WENG_LIN_BRADLEY_TERRY = 4
} n4m_ranking_method_t;

typedef struct n4m_ranking_options_v1_t {
    uint64_t struct_size;
    int32_t method;
    int32_t higher_is_better;
    double tie_tolerance; /* absolute metric units; nonnegative */
    double initial_mean;
    double k_factor;
    double elo_scale;
    double prior_sigma;
    double beta;
    double dynamics;
    double draw_margin;    /* latent performance units, strictly positive */
    double regularization; /* BT L2 penalty, strictly positive */
    double convergence_tolerance;
    int64_t max_iterations;
} n4m_ranking_options_v1_t;

N4M_API n4m_status_t n4m_ranking_options_init_v1(n4m_ranking_options_v1_t* options);

/* F64 strided input. present is a REQUIRED row-major blocks*candidates mask
 * with exactly 0/1 entries; masked input values may be nonfinite. candidates
 * is caller-owned, row-major candidates*10: score, posterior_sd (NaN unless
 * Bayesian), mean_rank, observations, matches, connected_component (0-based), wins, ties, losses,
 * global_rank. Global rank is NaN for disconnected graphs or candidates with no matches. For
 * AVERAGE_RANK it is also NaN unless every block is complete (complete_blocks == blocks): mean
 * within-block ranks of incomplete blocks lie on different scales, so they stay descriptive scores.
 * Score ties
 * in global rank use exact equality, with averaged ordinal ranks. A candidate with no observations
 * has NaN score/SD/rank. Bayesian SD is an algorithm posterior approximation, NOT a frequentist
 * standard error. summary has 9 doubles: blocks, complete_blocks, pair_count, components, Friedman
 * tie-corrected statistic, df, converged, iterations, chi-square approximate Friedman p-value.
 * Friedman is computed on complete blocks only, NaN for <2 blocks/candidates
 * or all ties. The asymptotic p-value assumes independent blocks; the host must assess
 * that assumption and adequate sample size.
 * Online methods process rows then column pairs in supplied order, once.
 * BT uses all pair outcomes with ties as half wins and L2 regularization.
 * Invalid input/unsupported method fails without writing either output. */
N4M_API n4m_status_t n4m_ranking_compute_v1(const n4m_matrix_view_t* scores,
                                            const uint8_t* present,
                                            const n4m_ranking_options_v1_t* options,
                                            double* candidates,
                                            int64_t candidates_size,
                                            double* summary,
                                            int64_t summary_size);

/* Paired relative gain (baseline-candidate)/baseline; BOTH losses must be
 * finite and strictly positive on present rows. present is required 0/1.
 * variances, if nonnull, are finite strictly positive sampling variances of
 * THESE RELATIVE EFFECTS (including covariance), never raw loss variances.
 * Output 8 doubles: n, mean_gain, sample_sd, DL_pooled_gain, DL_SE, tau_squared,
 * Q, I_squared. Without variances or with n<2, last five outputs are NaN.
 * No invented uncertainty from folds; caller must supply independent study
 * effects/variances for valid random-effects inference. */
N4M_API n4m_status_t n4m_ranking_relative_gain_v1(const double* baseline,
                                                  const double* candidate,
                                                  const uint8_t* present,
                                                  const double* variances,
                                                  int64_t n,
                                                  double* output,
                                                  int64_t output_size);
#ifdef __cplusplus
}
#endif
#endif
