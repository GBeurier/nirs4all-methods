// SPDX-License-Identifier: CECILL-2.1
//
// Tier C: R-exact MCUVE parity (internal-core test).
//
// Proves that the additive R-exact UVE path (cfg.rng_kind == N4M_RNG_MT_R)
// reproduces R `plsVarSel::mcuve_pls`'s selected feature set EXACTLY
// (Jaccard 1.0). The fixture is regenerated bit-exactly from R's own rnorm
// stream via n4m_rng_mt_r_norm (set.seed(1)), so the test is fully
// self-contained — no CSV.
//
// R ground truth (set.seed(1); 40x12; y = 2*X[,3] - 1.5*X[,8] + 0.1*rnorm;
// then set.seed(11); mcuve_pls(N=3, ratio=0.75, ncomp=5)) selects, 0-based,
// {2, 3, 4, 7, 8}. Verified live against R 4.3.3 and the proven Python
// reference on 2026-05-30.
//
// This was ported from a stale doctest-based file (the repo has no doctest
// framework). It exercises INTERNAL core symbols (select_by_uve etc.) that are
// hidden in libn4m, so it links the static archive via n4m_internal_tests and
// uses the local CHECK macro defined in test_internal_rng_choice.cpp's main
// translation unit's style (a small self-contained failure counter).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/common/context.hpp"
#include "core/common/rng_mt_r.h"
#include "core/config.hpp"
#include "core/uve_selection.hpp"
#include "core/validation.hpp"

int run_uve_r_exact_tests();

namespace {

using ::n4m::core::Config;
using ::n4m::core::Context;
using ::n4m::core::select_by_uve;
using ::n4m::core::UveSelectionResult;
using ::n4m::core::ValidationFold;
using ::n4m::core::ValidationPlan;

int g_uve_failures = 0;

#define UVE_CHECK(cond)                                                      \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
            ++g_uve_failures;                                                \
        }                                                                    \
    } while (0)

// Regenerate R's exact dataset:
//   set.seed(1); X <- matrix(rnorm(n*p), n, p)   (column-major fill)
//                y <- 2*X[,3] - 1.5*X[,8] + 0.1*rnorm(n)
// Returns X as a row-major (n x p) buffer and y as length n.
void regen_fixture(std::size_t n, std::size_t p,
                   std::vector<double>& X, std::vector<double>& y) {
    n4m_rng_mt_r rng;
    n4m_rng_mt_r_set_seed(&rng, 1U);
    std::vector<double> col_major(n * p, 0.0);
    for (std::size_t k = 0; k < n * p; ++k) col_major[k] = n4m_rng_mt_r_norm(&rng);
    X.assign(n * p, 0.0);
    for (std::size_t row = 0; row < n; ++row)
        for (std::size_t col = 0; col < p; ++col)
            X[row * p + col] = col_major[col * n + row];  // R matrix(): col*n + row
    std::vector<double> ynoise(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) ynoise[i] = n4m_rng_mt_r_norm(&rng);
    y.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = 2.0 * X[i * p + 2] - 1.5 * X[i * p + 7] + 0.1 * ynoise[i];  // cols 3 & 8 (1-based)
}

n4m_matrix_view_t make_view(const std::vector<double>& data,
                            std::int64_t rows, std::int64_t cols) {
    n4m_matrix_view_t view{};
    view.data = const_cast<double*>(data.data());
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

// Build N Monte-Carlo folds, each using the first `train_count` rows as train
// and the remainder as the held-out validation set. The current ValidationFold
// stores train + test index vectors (no fold_id); ValidationPlan stores only
// n_samples + folds.
ValidationPlan mc_plan(std::int64_t n_samples, std::int64_t train_count,
                       std::int32_t n_folds) {
    ValidationPlan plan;
    plan.n_samples = n_samples;
    for (std::int32_t f = 0; f < n_folds; ++f) {
        ValidationFold fold;
        for (std::int64_t i = 0; i < train_count; ++i) fold.train_indices.push_back(i);
        for (std::int64_t i = train_count; i < n_samples; ++i) fold.test_indices.push_back(i);
        plan.folds.push_back(fold);
    }
    return plan;
}

void test_fixture_regen_is_deterministic() {
    std::vector<double> X1, y1, X2, y2;
    regen_fixture(40, 12, X1, y1);
    regen_fixture(40, 12, X2, y2);
    UVE_CHECK(X1.size() == 40U * 12U);
    UVE_CHECK(y1.size() == 40U);
    bool x_eq = true, y_eq = true;
    for (std::size_t k = 0; k < X1.size(); ++k) if (X1[k] != X2[k]) x_eq = false;
    for (std::size_t k = 0; k < y1.size(); ++k) if (y1[k] != y2[k]) y_eq = false;
    UVE_CHECK(x_eq);
    UVE_CHECK(y_eq);
}

void test_select_by_uve_matches_r() {
    const std::size_t n = 40, p = 12;
    std::vector<double> X, y;
    regen_fixture(n, p, X, y);

    n4m_matrix_view_t x_view = make_view(X, static_cast<std::int64_t>(n),
                                         static_cast<std::int64_t>(p));
    n4m_matrix_view_t y_view = make_view(y, static_cast<std::int64_t>(n), 1);

    Config cfg;                       // defaults: PLS_REGRESSION / NIPALS, centered, scaled
    cfg.rng_kind = N4M_RNG_MT_R;      // opt into the R-exact MCUVE path
    cfg.n_components = 5;             // R mcuve_pls ncomp

    // R: N = 3 Monte-Carlo iterations, ratio = 0.75 -> K = floor(40*0.75) = 30.
    ValidationPlan plan = mc_plan(static_cast<std::int64_t>(n), 30, 3);

    Context ctx;
    UveSelectionResult result;
    const n4m_status_t status =
        select_by_uve(ctx, cfg, x_view, y_view, plan,
                      /*noise_features=*/static_cast<std::int32_t>(p),
                      /*noise_seed=*/11ULL, result);
    UVE_CHECK(status == N4M_OK);

    const std::vector<std::int64_t> expected = {2, 3, 4, 7, 8};
    UVE_CHECK(result.selected_indices == expected);
    UVE_CHECK(result.n_features == static_cast<std::int32_t>(p));
    UVE_CHECK(result.n_repeats == 3);
}

}  // namespace

// Entry point invoked from the n4m_internal_tests main (test_internal_rng_choice.cpp).
// Returns the number of failures so the caller can fold it into its exit code.
int run_uve_r_exact_tests() {
    test_fixture_regen_is_deterministic();
    test_select_by_uve_matches_r();
    if (g_uve_failures == 0) {
        std::printf("=== uve_r_exact: MCUVE R-exact selection parity OK ===\n");
    } else {
        std::printf("=== uve_r_exact: %d failure(s) ===\n", g_uve_failures);
    }
    return g_uve_failures;
}
