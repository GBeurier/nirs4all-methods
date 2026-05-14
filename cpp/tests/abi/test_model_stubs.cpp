// SPDX-License-Identifier: CeCILL-2.1
//
// Phase 1 model tests. The file name is kept stable so the CMake test list
// from Phase 0 does not need churn.

#include "pls4all/p4a.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fixtures/kernel_fixtures.hpp"
#include "fixtures/orthogonal_scores_fixtures.hpp"
#include "fixtures/phase1_fixtures.hpp"
#include "fixtures/pcr_fixtures.hpp"
#include "fixtures/power_fixtures.hpp"
#include "fixtures/randomized_svd_fixtures.hpp"
#include "fixtures/simpls_fixtures.hpp"
#include "fixtures/svd_fixtures.hpp"
#include "fixtures/wide_kernel_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-9;
constexpr double kRelTol = 1e-9;
constexpr std::uint64_t kRandomizedSvdFixtureSeed = 123456789ULL;

#if defined(__GNUC__) || defined(__clang__)
#  define P4A_TEST_NOINLINE __attribute__((noinline))
#else
#  define P4A_TEST_NOINLINE
#endif

struct Handles {
    p4a_context_t* ctx{nullptr};
    p4a_config_t* cfg{nullptr};
    p4a_model_t* model{nullptr};

    ~Handles() {
        p4a_model_destroy(model);
        p4a_config_destroy(cfg);
        p4a_context_destroy(ctx);
    }
};

P4A_TEST_NOINLINE void flip_last_byte(std::vector<unsigned char>& bytes) {
    if (bytes.empty()) {
        return;
    }
    const std::size_t last = bytes.size() - 1U;
    bytes[last] = static_cast<unsigned char>(bytes[last] ^ 0x01U);
}

[[nodiscard]] std::vector<double> copy_values(const p4a_array_t* arr) {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    p4a_array_shape(arr, &rows, &cols);
    p4a_matrix_view_t view{};
    p4a_array_view(arr, &view);
    const auto* ptr = static_cast<const double*>(view.data);
    return std::vector<double>(ptr, ptr + rows * cols);
}

void check_close_values(int& failures,
                        const char* label,
                        const double* actual,
                        const ::pls4all::test::fixtures::MatrixRef& expected) {
    for (std::size_t i = 0; i < expected.size; ++i) {
        const double a = expected.sign_invariant ? std::fabs(actual[i]) : actual[i];
        const double b = expected.sign_invariant ? std::fabs(expected.values[i]) : expected.values[i];
        const double diff = std::fabs(a - b);
        const double scale = std::max(std::max(std::fabs(a), std::fabs(b)), 1.0);
        if (diff > kAbsTol && diff > kRelTol * scale) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu]: actual %.17g expected %.17g diff %.3g\n",
                         label,
                         static_cast<unsigned long>(i),
                         actual[i],
                         expected.values[i],
                         diff);
            return;
        }
    }
}

void check_array(int& failures,
                 p4a_context_t* ctx,
                 p4a_model_t* model,
                 p4a_model_array_t which,
                 const char* label,
                 const ::pls4all::test::fixtures::MatrixRef& expected) {
    p4a_array_t* arr = nullptr;
    CHECK_EQ(p4a_model_get_array(ctx, model, which, &arr), P4A_OK);
    CHECK_NE(arr, nullptr);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(p4a_array_shape(arr, &rows, &cols), P4A_OK);
    CHECK_EQ(rows, expected.rows);
    CHECK_EQ(cols, expected.cols);
    p4a_dtype_t dtype = P4A_DTYPE_UNKNOWN;
    CHECK_EQ(p4a_array_dtype(arr, &dtype), P4A_OK);
    CHECK_EQ(dtype, P4A_DTYPE_F64);
    p4a_matrix_view_t view{};
    CHECK_EQ(p4a_array_view(arr, &view), P4A_OK);
    const auto* values = static_cast<const double*>(view.data);
    check_close_values(failures, label, values, expected);
    p4a_array_free(arr);
}

void fit_fixture(int& failures,
                 const ::pls4all::test::fixtures::Phase1Fixture& fixture,
                 Handles& h,
                 bool store_scores = false,
                 p4a_solver_t solver = P4A_SOLVER_NIPALS) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, solver), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_simpls_fixture(int& failures,
                        const ::pls4all::test::fixtures::SimplsFixture& fixture,
                        Handles& h,
                        bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_SIMPLS), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_svd_fixture(int& failures,
                     const ::pls4all::test::fixtures::SvdFixture& fixture,
                     Handles& h,
                     bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_SVD), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_kernel_fixture(int& failures,
                        const ::pls4all::test::fixtures::KernelFixture& fixture,
                        Handles& h,
                        bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_KERNEL_ALGORITHM), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_pcr_fixture(int& failures,
                     const ::pls4all::test::fixtures::PcrFixture& fixture,
                     Handles& h,
                     bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_algorithm(h.cfg, P4A_ALGO_PCR), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_SVD), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_wide_kernel_fixture(
    int& failures,
    const ::pls4all::test::fixtures::WideKernelFixture& fixture,
    Handles& h,
    bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_WIDE_KERNEL), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_orthogonal_scores_fixture(
    int& failures,
    const ::pls4all::test::fixtures::OrthogonalScoresFixture& fixture,
    Handles& h,
    bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_ORTHOGONAL_SCORES), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_power_fixture(int& failures,
                       const ::pls4all::test::fixtures::PowerFixture& fixture,
                       Handles& h,
                       bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_POWER), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_randomized_svd_fixture(
    int& failures,
    const ::pls4all::test::fixtures::RandomizedSvdFixture& fixture,
    Handles& h,
    bool store_scores = false) {
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_context_set_seed(h.ctx, kRandomizedSvdFixtureSeed), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_RANDOMIZED_SVD), P4A_OK);
    if (store_scores) {
        CHECK_EQ(p4a_config_set_store_scores(h.cfg, 1), P4A_OK);
    }

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_OK);
    CHECK_NE(h.model, nullptr);
}

}  // namespace

#undef P4A_TEST_NOINLINE

TEST(model_phase1, fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kPhase1Fixtures) {
        Handles h;
        fit_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::int64_t rows = 0;
        std::int64_t cols = 0;
        CHECK_EQ(p4a_array_shape(pred, &rows, &cols), P4A_OK);
        CHECK_EQ(rows, fixture.predict_train.rows);
        CHECK_EQ(cols, fixture.predict_train.cols);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        std::vector<double> caller_pred(fixture.predict_train.size, 0.0);
        p4a_matrix_view_t out{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&out,
                                               caller_pred.data(),
                                               fixture.predict_train.rows,
                                               fixture.predict_train.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        CHECK_EQ(p4a_model_predict(h.ctx, h.model, &X, &out), P4A_OK);
        check_close_values(failures, fixture.id, caller_pred.data(), fixture.predict_train);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, simpls_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kSimplsFixtures) {
        Handles h;
        fit_simpls_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "simpls coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "simpls intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "simpls x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "simpls x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "simpls y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "simpls y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "simpls weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "simpls loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "simpls y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "simpls rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, svd_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kSvdFixtures) {
        Handles h;
        fit_svd_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "svd coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "svd intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "svd x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "svd x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "svd y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "svd y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "svd weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "svd loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "svd y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "svd rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, power_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kPowerFixtures) {
        Handles h;
        fit_power_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "power coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "power intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "power x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "power x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "power y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "power y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "power weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "power loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "power y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "power rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, randomized_svd_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kRandomizedSvdFixtures) {
        Handles h;
        fit_randomized_svd_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "randomized svd coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "randomized svd intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "randomized svd x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "randomized svd x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "randomized svd y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "randomized svd y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "randomized svd weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "randomized svd loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "randomized svd y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "randomized svd rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, kernel_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kKernelFixtures) {
        Handles h;
        fit_kernel_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "kernel coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "kernel intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "kernel x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "kernel x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "kernel y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "kernel y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "kernel weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "kernel loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "kernel y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "kernel rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, wide_kernel_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kWideKernelFixtures) {
        Handles h;
        fit_wide_kernel_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "wide kernel coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "wide kernel intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "wide kernel x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "wide kernel x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "wide kernel y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "wide kernel y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "wide kernel weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "wide kernel loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "wide kernel y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "wide kernel rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, orthogonal_scores_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kOrthogonalScoresFixtures) {
        Handles h;
        fit_orthogonal_scores_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "orthogonal scores coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "orthogonal scores intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "orthogonal scores x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "orthogonal scores x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "orthogonal scores y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "orthogonal scores y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "orthogonal scores weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "orthogonal scores loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "orthogonal scores y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "orthogonal scores rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, pcr_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::pls4all::test::fixtures::kPcrFixtures) {
        Handles h;
        fit_pcr_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(p4a_model_get_n_components(h.model, &n_components), P4A_OK);
        CHECK_EQ(p4a_model_get_n_features(h.model, &n_features), P4A_OK);
        CHECK_EQ(p4a_model_get_n_targets(h.model, &n_targets), P4A_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        p4a_array_t* pred = nullptr;
        CHECK_EQ(p4a_model_predict_alloc(h.ctx, h.model, &X, &pred), P4A_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        p4a_array_free(pred);

        check_array(failures, h.ctx, h.model, P4A_MODEL_COEFFICIENTS,
                    "pcr coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, P4A_MODEL_INTERCEPT,
                    "pcr intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_MEAN,
                    "pcr x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_X_SCALE,
                    "pcr x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_MEAN,
                    "pcr y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_SCALE,
                    "pcr y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, P4A_MODEL_WEIGHTS_W,
                    "pcr weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, P4A_MODEL_LOADINGS_P,
                    "pcr loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, P4A_MODEL_Y_LOADINGS_Q,
                    "pcr y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, P4A_MODEL_ROTATIONS_R,
                    "pcr rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, transform_and_stored_scores_agree) {
    const auto& fixture = ::pls4all::test::fixtures::kPhase1Fixtures[0];
    Handles h;
    fit_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    CHECK_EQ(transform_values.size(), stored_values.size());
    for (std::size_t i = 0; i < transform_values.size(); ++i) {
        const double diff = std::fabs(transform_values[i] - stored_values[i]);
        if (diff > kAbsTol) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL transform/scores[%lu] diff %.3g\n",
                         static_cast<unsigned long>(i),
                         diff);
            break;
        }
    }
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, simpls_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kSimplsFixtures[0];
    Handles h;
    fit_simpls_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "simpls transform", transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "simpls stored scores", stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, svd_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kSvdFixtures[0];
    Handles h;
    fit_svd_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "svd transform", transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "svd stored scores", stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, power_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kPowerFixtures[0];
    Handles h;
    fit_power_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "power transform", transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "power stored scores", stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, randomized_svd_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kRandomizedSvdFixtures[0];
    Handles h;
    fit_randomized_svd_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "randomized svd transform",
                       transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "randomized svd stored scores",
                       stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, kernel_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kKernelFixtures[0];
    Handles h;
    fit_kernel_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "kernel transform", transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "kernel stored scores", stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, wide_kernel_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kWideKernelFixtures[0];
    Handles h;
    fit_wide_kernel_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "wide kernel transform",
                       transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "wide kernel stored scores",
                       stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, orthogonal_scores_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kOrthogonalScoresFixtures[0];
    Handles h;
    fit_orthogonal_scores_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "orthogonal scores transform",
                       transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "orthogonal scores stored scores",
                       stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, pcr_transform_matches_reference_scores) {
    const auto& fixture = ::pls4all::test::fixtures::kPcrFixtures[0];
    Handles h;
    fit_pcr_fixture(failures, fixture, h, true);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    p4a_array_t* transformed = nullptr;
    CHECK_EQ(p4a_model_transform_alloc(h.ctx, h.model, &X, &transformed), P4A_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "pcr transform", transform_values.data(), fixture.scores_t);

    p4a_array_t* stored_scores = nullptr;
    CHECK_EQ(p4a_model_get_array(h.ctx, h.model, P4A_MODEL_SCORES_T, &stored_scores), P4A_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "pcr stored scores", stored_values.data(), fixture.scores_t);
    p4a_array_free(stored_scores);
    p4a_array_free(transformed);
}

TEST(model_phase1, serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kPhase1Fixtures[2];
    Handles h;
    fit_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written), P4A_OK);
    CHECK_EQ(written, size);

    std::uint32_t fmt = 0;
    std::uint32_t abi_major = 0;
    std::uint32_t abi_minor = 0;
    std::uint32_t abi_patch = 0;
    CHECK_EQ(p4a_serialization_inspect(buffer.data(), buffer.size(),
                                       &fmt, &abi_major, &abi_minor, &abi_patch),
             P4A_OK);
    CHECK_EQ(fmt, static_cast<std::uint32_t>(P4A_SERIALIZATION_FORMAT_VERSION));
    CHECK_EQ(abi_major, static_cast<std::uint32_t>(P4A_ABI_VERSION_MAJOR));
    CHECK_EQ(abi_minor, static_cast<std::uint32_t>(P4A_ABI_VERSION_MINOR));
    CHECK_EQ(abi_patch, static_cast<std::uint32_t>(P4A_ABI_VERSION_PATCH));

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported), P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_predict", pred_values.data(), fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);

    std::vector<unsigned char> corrupted = buffer;
    flip_last_byte(corrupted);
    p4a_model_t* corrupt_model = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, corrupted.data(), corrupted.size(), &corrupt_model),
             P4A_ERR_CORRUPT_BUFFER);
    CHECK_EQ(corrupt_model, nullptr);

    std::vector<unsigned char> short_buffer(size - 1U, 0U);
    CHECK_EQ(p4a_model_export_to_buffer(h.model, short_buffer.data(), short_buffer.size(), &written),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(written, 0U);
}

TEST(model_phase1, simpls_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kSimplsFixtures[1];
    Handles h;
    fit_simpls_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written), P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported), P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_simpls_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, svd_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kSvdFixtures[1];
    Handles h;
    fit_svd_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_svd_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, power_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kPowerFixtures[1];
    Handles h;
    fit_power_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_power_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, randomized_svd_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kRandomizedSvdFixtures[1];
    Handles h;
    fit_randomized_svd_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_randomized_svd_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, kernel_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kKernelFixtures[1];
    Handles h;
    fit_kernel_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_kernel_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, wide_kernel_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kWideKernelFixtures[1];
    Handles h;
    fit_wide_kernel_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_wide_kernel_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, orthogonal_scores_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kOrthogonalScoresFixtures[1];
    Handles h;
    fit_orthogonal_scores_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_orthogonal_scores_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, pcr_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::pls4all::test::fixtures::kPcrFixtures[1];
    Handles h;
    fit_pcr_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(p4a_model_export_size(h.model, &size), P4A_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(p4a_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             P4A_OK);
    CHECK_EQ(written, size);

    p4a_model_t* imported = nullptr;
    CHECK_EQ(p4a_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             P4A_OK);
    CHECK_NE(imported, nullptr);

    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    p4a_array_t* pred = nullptr;
    CHECK_EQ(p4a_model_predict_alloc(h.ctx, imported, &X, &pred), P4A_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_pcr_predict", pred_values.data(),
                       fixture.predict_train);
    p4a_array_free(pred);
    p4a_model_destroy(imported);
}

TEST(model_phase1, validation_errors_are_deterministic) {
    const auto& fixture = ::pls4all::test::fixtures::kPhase1Fixtures[0];
    Handles h;
    CHECK_EQ(p4a_context_create(&h.ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&h.cfg), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(h.cfg, fixture.n_components), P4A_OK);
    CHECK_EQ(p4a_config_set_algorithm(h.cfg, P4A_ALGO_PLS_CANONICAL), P4A_OK);

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_ERR_UNSUPPORTED);
    CHECK_STR_CONTAINS(p4a_context_last_error(h.ctx), "KERNEL_ALGORITHM");

    CHECK_EQ(p4a_config_set_algorithm(h.cfg, P4A_ALGO_PLS_REGRESSION), P4A_OK);
    CHECK_EQ(p4a_config_set_solver(h.cfg, P4A_SOLVER_NIPALS), P4A_OK);
    p4a_matrix_view_t bad_y = Y;
    bad_y.rows = Y.rows - 1;
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &bad_y, &h.model), P4A_ERR_SHAPE_MISMATCH);

    std::vector<double> x_with_nan(fixture.X.values, fixture.X.values + fixture.X.size);
    x_with_nan[0] = std::nan("");
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           x_with_nan.data(),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_ERR_INVALID_ARGUMENT);

    std::vector<std::int32_t> x_i32(fixture.X.size, 1);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           x_i32.data(),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_I32),
             P4A_OK);
    CHECK_EQ(p4a_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), P4A_ERR_DTYPE_MISMATCH);
}

TEST(model_phase1, null_pointer_contracts_are_live) {
    p4a_context_t* ctx = nullptr;
    p4a_config_t* cfg = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    CHECK_EQ(p4a_config_create(&cfg), P4A_OK);

    p4a_model_t* model = reinterpret_cast<p4a_model_t*>(0x1);
    CHECK_EQ(p4a_model_fit(ctx, cfg, nullptr, nullptr, &model), P4A_ERR_NULL_POINTER);
    CHECK_EQ(model, nullptr);

    p4a_array_t* arr = reinterpret_cast<p4a_array_t*>(0x1);
    CHECK_EQ(p4a_model_predict_alloc(ctx, nullptr, nullptr, &arr), P4A_ERR_NULL_POINTER);
    CHECK_EQ(arr, nullptr);
    CHECK_EQ(p4a_model_get_array(ctx, nullptr, P4A_MODEL_COEFFICIENTS, &arr),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(arr, nullptr);

    p4a_dtype_t dtype = P4A_DTYPE_UNKNOWN;
    CHECK_EQ(p4a_array_dtype(nullptr, &dtype), P4A_ERR_NULL_POINTER);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(p4a_array_shape(nullptr, &rows, &cols), P4A_ERR_NULL_POINTER);
    p4a_matrix_view_t view{};
    CHECK_EQ(p4a_array_view(nullptr, &view), P4A_ERR_NULL_POINTER);
    p4a_array_free(nullptr);
    p4a_model_destroy(nullptr);

    p4a_config_destroy(cfg);
    p4a_context_destroy(ctx);
}
