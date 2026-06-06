// SPDX-License-Identifier: CECILL-2.1
//
// Internal tests for moment-sweep accelerators that are hidden from the public
// C ABI. This TU is linked into n4m_internal_tests via the static archive.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "core/common/context.hpp"
#include "core/config.hpp"
#include "core/moments.hpp"
#include "core/sweep.hpp"

namespace {

int g_sweep_failures = 0;

#define SWEEP_CHECK(cond)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            ++g_sweep_failures;                                              \
        }                                                                    \
    } while (0)

constexpr std::int64_t kN = 8;
constexpr std::int64_t kP = 3;
constexpr std::int32_t kCv = 4;
constexpr double kTol = 1e-8;

const double kX[static_cast<std::size_t>(kN * kP)] = {
    0.5, -1.0,  2.0,
    1.5,  0.2, -0.5,
   -0.3,  2.2,  1.1,
    2.0, -0.7,  0.3,
    0.9,  1.4, -1.2,
   -1.1,  0.5,  1.7,
    1.2, -1.5,  0.8,
   -0.8,  1.0, -0.4,
};

const double kY[static_cast<std::size_t>(kN)] = {
    1.2, 0.4, -0.7, 2.1, 0.1, -1.4, 1.8, -0.2,
};

const double kDegenerateX[static_cast<std::size_t>(kN * kP)] = {
    0.0,  1.0,  3.0,
    1.0,  2.0,  5.0,
    2.0,  3.0,  7.0,
    3.0,  4.0,  9.0,
    4.0,  5.0, 11.0,
    5.0,  6.0, 13.0,
    6.0,  7.0, 15.0,
    7.0,  8.0, 17.0,
};

const double kDegenerateY[static_cast<std::size_t>(kN)] = {
    -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
};

const std::int32_t kFolds[static_cast<std::size_t>(kN)] = {
    0, 0, 1, 1, 2, 2, 3, 3,
};

n4m_matrix_view_t make_view(const double* data,
                            std::int64_t rows,
                            std::int64_t cols) {
    n4m_matrix_view_t v{};
    v.data = const_cast<double*>(data);
    v.rows = rows;
    v.cols = cols;
    v.row_stride = cols;
    v.col_stride = 1;
    v.dtype = N4M_DTYPE_F64;
    return v;
}

void build_heldout_stats(::n4m::core::Context& ctx,
                         const n4m_matrix_view_t& X,
                         const n4m_matrix_view_t& Y,
                         std::vector<::n4m::core::MomentStats>& heldout) {
    heldout.assign(static_cast<std::size_t>(kCv), ::n4m::core::MomentStats{});
    for (std::int32_t fold = 0; fold < kCv; ++fold) {
        std::vector<std::int64_t> rows;
        for (std::int64_t row = 0; row < kN; ++row) {
            if (kFolds[static_cast<std::size_t>(row)] == fold) {
                rows.push_back(row);
            }
        }
        SWEEP_CHECK(::n4m::core::compute_moments_subset(
                        ctx, X, Y, rows.data(),
                        static_cast<std::int64_t>(rows.size()),
                        heldout[static_cast<std::size_t>(fold)]) == N4M_OK);
    }
}

void test_pls1_moment_scores_match_materialized_cv(bool scale_x) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.scale_x = scale_x ? 1 : 0;

    n4m_matrix_view_t X = make_view(kX, kN, kP);
    n4m_matrix_view_t Y = make_view(kY, kN, 1);
    const std::int32_t comps[2] = {1, 2};

    ::n4m::core::SweepResult materialized;
    SWEEP_CHECK(::n4m::core::run_moment_sweep(
                    ctx, cfg, X, Y, kCv, kFolds, kN,
                    nullptr, 0, comps, 2,
                    ::n4m::core::kSweepHeadPls, materialized) == N4M_OK);

    ::n4m::core::MomentStats all_stats;
    SWEEP_CHECK(::n4m::core::compute_moments(ctx, X, Y, all_stats) == N4M_OK);
    std::vector<::n4m::core::MomentStats> heldout;
    build_heldout_stats(ctx, X, Y, heldout);

    ::n4m::core::SweepResult scored;
    SWEEP_CHECK(::n4m::core::score_pls1_moment_sweep(
                    ctx, cfg, all_stats, heldout, comps, 2, scored) == N4M_OK);

    SWEEP_CHECK(scored.n_candidates == materialized.n_candidates);
    for (std::size_t i = 0; i < scored.candidate_scores.size(); ++i) {
        const double diff =
            std::fabs(scored.candidate_scores[i] -
                      materialized.candidate_scores[i]);
        SWEEP_CHECK(diff <= kTol);
    }
    SWEEP_CHECK(scored.selected_candidate_id == materialized.selected_candidate_id);
    SWEEP_CHECK(scored.selected_head_id == materialized.selected_head_id);
    SWEEP_CHECK(std::fabs(scored.selected_param -
                          materialized.selected_param) <= kTol);
    SWEEP_CHECK(std::fabs(scored.selected_cv_rmse -
                          materialized.selected_cv_rmse) <= kTol);
}

void test_pls1_moment_batch_scores_match_single_chain() {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.aom_score_only = 1;

    n4m_matrix_view_t X = make_view(kX, kN, kP);
    n4m_matrix_view_t Y = make_view(kY, kN, 1);
    const std::int32_t comps[2] = {1, 2};

    ::n4m::core::MomentStats all_stats;
    SWEEP_CHECK(::n4m::core::compute_moments(ctx, X, Y, all_stats) == N4M_OK);
    std::vector<::n4m::core::MomentStats> heldout;
    build_heldout_stats(ctx, X, Y, heldout);

    ::n4m::core::SweepResult single;
    SWEEP_CHECK(::n4m::core::score_pls1_moment_sweep(
                    ctx, cfg, all_stats, heldout, comps, 2, single) == N4M_OK);

    std::vector<::n4m::core::MomentStats> all_many = {all_stats, all_stats};
    std::vector<std::vector<::n4m::core::MomentStats>> heldout_many = {
        heldout,
        heldout,
    };
    std::vector<::n4m::core::SweepResult> batched;
    SWEEP_CHECK(::n4m::core::score_pls1_moment_sweeps_score_only(
                    ctx, cfg, all_many, heldout_many, comps, 2,
                    batched) == N4M_OK);

    SWEEP_CHECK(batched.size() == 2U);
    for (std::size_t row_ix = 0; row_ix < batched.size(); ++row_ix) {
        const auto& row = batched[row_ix];
        SWEEP_CHECK(row.score_only == 1);
        SWEEP_CHECK(row.n_pls_moment_cv_fits == kCv);
        if (row_ix == 0U) {
            SWEEP_CHECK(row.n_pls_moment_score_batch_calls == 1);
            SWEEP_CHECK(row.n_pls_moment_score_batch_jobs == 2 * kCv);
        } else {
            SWEEP_CHECK(row.n_pls_moment_score_batch_calls == 0);
            SWEEP_CHECK(row.n_pls_moment_score_batch_jobs == 0);
        }
        SWEEP_CHECK(row.n_candidates == single.n_candidates);
        SWEEP_CHECK(row.candidate_scores.size() == single.candidate_scores.size());
        for (std::size_t i = 0; i < row.candidate_scores.size(); ++i) {
            SWEEP_CHECK(std::fabs(row.candidate_scores[i] -
                                  single.candidate_scores[i]) <= kTol);
        }
        SWEEP_CHECK(row.selected_candidate_id == single.selected_candidate_id);
        SWEEP_CHECK(row.selected_head_id == single.selected_head_id);
        SWEEP_CHECK(std::fabs(row.selected_param -
                              single.selected_param) <= kTol);
        SWEEP_CHECK(std::fabs(row.selected_cv_rmse -
                              single.selected_cv_rmse) <= kTol);
    }
}

void test_pls1_moment_batch_fallback_reuses_recovered_prefixes() {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.aom_score_only = 1;

    n4m_matrix_view_t X = make_view(kX, kN, kP);
    n4m_matrix_view_t Y = make_view(kY, kN, 1);
    n4m_matrix_view_t X_degenerate = make_view(kDegenerateX, kN, kP);
    n4m_matrix_view_t Y_degenerate = make_view(kDegenerateY, kN, 1);
    const std::int32_t comps[2] = {1, 2};

    ::n4m::core::MomentStats all_stats;
    SWEEP_CHECK(::n4m::core::compute_moments(ctx, X, Y, all_stats) == N4M_OK);
    std::vector<::n4m::core::MomentStats> heldout;
    build_heldout_stats(ctx, X, Y, heldout);

    ::n4m::core::MomentStats degenerate_all_stats;
    SWEEP_CHECK(::n4m::core::compute_moments(
                    ctx, X_degenerate, Y_degenerate,
                    degenerate_all_stats) == N4M_OK);
    std::vector<::n4m::core::MomentStats> degenerate_heldout;
    build_heldout_stats(ctx, X_degenerate, Y_degenerate,
                        degenerate_heldout);

    std::vector<::n4m::core::MomentStats> all_many = {
        all_stats,
        degenerate_all_stats,
    };
    std::vector<std::vector<::n4m::core::MomentStats>> heldout_many = {
        heldout,
        degenerate_heldout,
    };

    std::vector<::n4m::core::SweepResult> batched;
    SWEEP_CHECK(::n4m::core::score_pls1_moment_sweeps_score_only(
                    ctx, cfg, all_many, heldout_many, comps, 2,
                    batched) == N4M_OK);

    SWEEP_CHECK(batched.size() == 2U);
    const auto& healthy = batched[0];
    SWEEP_CHECK(healthy.n_pls_moment_score_batch_calls == 0);
    SWEEP_CHECK(healthy.n_pls_moment_cv_fits == kCv);
    SWEEP_CHECK(healthy.n_pls_moment_host_cv_fits == kCv);
    SWEEP_CHECK(healthy.n_candidates == 2);
    SWEEP_CHECK(std::isfinite(healthy.candidate_scores[3]));
    SWEEP_CHECK(std::isfinite(healthy.candidate_scores[7]));

    const auto& degenerate = batched[1];
    SWEEP_CHECK(degenerate.n_pls_moment_score_batch_calls == 0);
    SWEEP_CHECK(degenerate.n_pls_moment_cv_fits == kCv + 1);
    SWEEP_CHECK(degenerate.n_pls_moment_host_cv_fits == kCv + 1);
    SWEEP_CHECK(degenerate.n_candidates == 2);
    SWEEP_CHECK(std::isfinite(degenerate.candidate_scores[3]));
    SWEEP_CHECK(std::isinf(degenerate.candidate_scores[7]));
}

void test_pls1_moment_single_fallback_recovers_prefix() {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;

    n4m_matrix_view_t X = make_view(kDegenerateX, kN, kP);
    n4m_matrix_view_t Y = make_view(kDegenerateY, kN, 1);
    const std::int32_t comps[2] = {1, 2};

    ::n4m::core::MomentStats all_stats;
    SWEEP_CHECK(::n4m::core::compute_moments(ctx, X, Y, all_stats) == N4M_OK);
    std::vector<::n4m::core::MomentStats> heldout;
    build_heldout_stats(ctx, X, Y, heldout);

    ::n4m::core::SweepResult scored;
    SWEEP_CHECK(::n4m::core::score_pls1_moment_sweep(
                    ctx, cfg, all_stats, heldout, comps, 2, scored) == N4M_OK);

    SWEEP_CHECK(scored.n_candidates == 2);
    SWEEP_CHECK(scored.n_pls_moment_cv_fits == kCv + 1);
    SWEEP_CHECK(scored.n_pls_moment_host_cv_fits == kCv + 1);
    SWEEP_CHECK(scored.n_pls_materialized_cv_fits == 0);
    SWEEP_CHECK(std::isfinite(scored.candidate_scores[3]));
    SWEEP_CHECK(std::isinf(scored.candidate_scores[7]));
    SWEEP_CHECK(scored.selected_candidate_id == 0);
    SWEEP_CHECK(scored.selected_head_id == 1);
    SWEEP_CHECK(std::fabs(scored.selected_param - 1.0) <= kTol);
}

void test_ridge_moment_batch_scores_match_single_chain() {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.aom_score_only = 1;

    n4m_matrix_view_t X = make_view(kX, kN, kP);
    n4m_matrix_view_t Y = make_view(kY, kN, 1);
    const double lambdas[2] = {0.01, 0.1};

    ::n4m::core::MomentStats all_stats;
    SWEEP_CHECK(::n4m::core::compute_moments(ctx, X, Y, all_stats) == N4M_OK);
    std::vector<::n4m::core::MomentStats> heldout;
    build_heldout_stats(ctx, X, Y, heldout);

    ::n4m::core::SweepResult single;
    SWEEP_CHECK(::n4m::core::score_ridge_moment_sweep(
                    ctx, cfg, all_stats, heldout, lambdas, 2,
                    single) == N4M_OK);

    std::vector<::n4m::core::MomentStats> all_many = {all_stats, all_stats};
    std::vector<std::vector<::n4m::core::MomentStats>> heldout_many = {
        heldout,
        heldout,
    };
    std::vector<::n4m::core::SweepResult> batched;
    SWEEP_CHECK(::n4m::core::score_ridge_moment_sweeps_score_only(
                    ctx, cfg, all_many, heldout_many, lambdas, 2,
                    batched) == N4M_OK);

    SWEEP_CHECK(batched.size() == 2U);
    for (std::size_t row_ix = 0; row_ix < batched.size(); ++row_ix) {
        const auto& row = batched[row_ix];
        SWEEP_CHECK(row.score_only == 1);
        SWEEP_CHECK(row.n_ridge_moment_cv_fits == 2 * kCv);
        if (row_ix == 0U) {
            SWEEP_CHECK(row.n_ridge_moment_score_batch_calls == 1);
            SWEEP_CHECK(row.n_ridge_moment_score_batch_jobs == 2 * 2 * kCv);
        } else {
            SWEEP_CHECK(row.n_ridge_moment_score_batch_calls == 0);
            SWEEP_CHECK(row.n_ridge_moment_score_batch_jobs == 0);
        }
        SWEEP_CHECK(row.n_candidates == single.n_candidates);
        SWEEP_CHECK(row.candidate_scores.size() == single.candidate_scores.size());
        for (std::size_t i = 0; i < row.candidate_scores.size(); ++i) {
            SWEEP_CHECK(std::fabs(row.candidate_scores[i] -
                                  single.candidate_scores[i]) <= kTol);
        }
        SWEEP_CHECK(row.selected_candidate_id == single.selected_candidate_id);
        SWEEP_CHECK(row.selected_head_id == single.selected_head_id);
        SWEEP_CHECK(std::fabs(row.selected_param -
                              single.selected_param) <= kTol);
        SWEEP_CHECK(std::fabs(row.selected_cv_rmse -
                              single.selected_cv_rmse) <= kTol);
    }
}

void test_pls1_gcv_moment_batch_scores_match_single_chain() {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.aom_score_only = 1;

    n4m_matrix_view_t X = make_view(kX, kN, kP);
    n4m_matrix_view_t Y = make_view(kY, kN, 1);
    const std::int32_t comps[2] = {1, 2};

    ::n4m::core::MomentStats all_stats;
    SWEEP_CHECK(::n4m::core::compute_moments(ctx, X, Y, all_stats) == N4M_OK);

    ::n4m::core::SweepResult single;
    SWEEP_CHECK(::n4m::core::score_pls1_gcv_moment_screen(
                    ctx, cfg, all_stats, comps, 2, single) == N4M_OK);

    std::vector<::n4m::core::MomentStats> all_many = {all_stats, all_stats};
    std::vector<::n4m::core::SweepResult> batched;
    SWEEP_CHECK(::n4m::core::score_pls1_gcv_moment_screens_score_only(
                    ctx, cfg, all_many, comps, 2, batched) == N4M_OK);

    SWEEP_CHECK(batched.size() == 2U);
    for (std::size_t row_ix = 0; row_ix < batched.size(); ++row_ix) {
        const auto& row = batched[row_ix];
        SWEEP_CHECK(row.score_only == 1);
        SWEEP_CHECK(row.n_pls_gcv_proxy_fits == 1);
        if (row_ix == 0U) {
            SWEEP_CHECK(row.n_pls_gcv_proxy_batch_calls == 1);
            SWEEP_CHECK(row.n_pls_gcv_proxy_batch_jobs == 2);
        } else {
            SWEEP_CHECK(row.n_pls_gcv_proxy_batch_calls == 0);
            SWEEP_CHECK(row.n_pls_gcv_proxy_batch_jobs == 0);
        }
        SWEEP_CHECK(row.n_candidates == single.n_candidates);
        SWEEP_CHECK(row.candidate_scores.size() == single.candidate_scores.size());
        for (std::size_t i = 0; i < row.candidate_scores.size(); ++i) {
            SWEEP_CHECK(std::fabs(row.candidate_scores[i] -
                                  single.candidate_scores[i]) <= kTol);
        }
        SWEEP_CHECK(row.selected_candidate_id == single.selected_candidate_id);
        SWEEP_CHECK(row.selected_head_id == single.selected_head_id);
        SWEEP_CHECK(std::fabs(row.selected_param -
                              single.selected_param) <= kTol);
        SWEEP_CHECK(std::fabs(row.selected_cv_rmse -
                              single.selected_cv_rmse) <= kTol);
    }
}

}  // namespace

int run_internal_sweep_tests() {
    g_sweep_failures = 0;
    test_pls1_moment_scores_match_materialized_cv(false);
    test_pls1_moment_scores_match_materialized_cv(true);
    test_pls1_moment_batch_scores_match_single_chain();
    test_pls1_moment_batch_fallback_reuses_recovered_prefixes();
    test_pls1_moment_single_fallback_recovers_prefix();
    test_ridge_moment_batch_scores_match_single_chain();
    test_pls1_gcv_moment_batch_scores_match_single_chain();
    if (g_sweep_failures == 0) {
        std::printf("[internal] sweep Ridge/PLS1 moment scoring ... ok\n");
    } else {
        std::printf("[internal] sweep Ridge/PLS1 moment scoring ... %d failure(s)\n",
                    g_sweep_failures);
    }
    return g_sweep_failures;
}
