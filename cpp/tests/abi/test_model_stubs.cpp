// SPDX-License-Identifier: CECILL-2.1
//
// Phase 1 model tests. The file name is kept stable so the CMake test list
// from Phase 0 does not need churn.

#include "pls4all/p4a.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fixtures/canonical_fixtures.hpp"
#include "fixtures/canonical_svd_fixtures.hpp"
#include "fixtures/kernel_fixtures.hpp"
#include "fixtures/opls_da_fixtures.hpp"
#include "fixtures/opls_fixtures.hpp"
#include "fixtures/orthogonal_scores_fixtures.hpp"
#include "fixtures/phase1_fixtures.hpp"
#include "fixtures/pcr_fixtures.hpp"
#include "fixtures/pls_da_fixtures.hpp"
#include "fixtures/pls_svd_fixtures.hpp"
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
#  define N4M_TEST_NOINLINE __attribute__((noinline))
#else
#  define N4M_TEST_NOINLINE
#endif

struct Handles {
    n4m_context_t* ctx{nullptr};
    n4m_config_t* cfg{nullptr};
    n4m_model_t* model{nullptr};

    ~Handles() {
        n4m_model_destroy(model);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
};

N4M_TEST_NOINLINE void flip_last_byte(std::vector<unsigned char>& bytes) {
    if (bytes.empty()) {
        return;
    }
    const std::size_t last = bytes.size() - 1U;
    bytes[last] = static_cast<unsigned char>(bytes[last] ^ 0x01U);
}

[[nodiscard]] std::vector<double> copy_values(const n4m_array_t* arr) {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    n4m_array_shape(arr, &rows, &cols);
    n4m_matrix_view_t view{};
    n4m_array_view(arr, &view);
    const auto* ptr = static_cast<const double*>(view.data);
    return std::vector<double>(ptr, ptr + rows * cols);
}

void check_close_values(int& failures,
                        const char* label,
                        const double* actual,
                        const ::n4m::test::fixtures::MatrixRef& expected) {
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
                 n4m_context_t* ctx,
                 n4m_model_t* model,
                 n4m_model_array_t which,
                 const char* label,
                 const ::n4m::test::fixtures::MatrixRef& expected) {
    n4m_array_t* arr = nullptr;
    CHECK_EQ(n4m_model_get_array(ctx, model, which, &arr), N4M_OK);
    CHECK_NE(arr, nullptr);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(n4m_array_shape(arr, &rows, &cols), N4M_OK);
    CHECK_EQ(rows, expected.rows);
    CHECK_EQ(cols, expected.cols);
    n4m_dtype_t dtype = N4M_DTYPE_UNKNOWN;
    CHECK_EQ(n4m_array_dtype(arr, &dtype), N4M_OK);
    CHECK_EQ(dtype, N4M_DTYPE_F64);
    n4m_matrix_view_t view{};
    CHECK_EQ(n4m_array_view(arr, &view), N4M_OK);
    const auto* values = static_cast<const double*>(view.data);
    check_close_values(failures, label, values, expected);
    n4m_array_free(arr);
}

void fit_fixture(int& failures,
                 const ::n4m::test::fixtures::Phase1Fixture& fixture,
                 Handles& h,
                 bool store_scores = false,
                 n4m_solver_t solver = N4M_SOLVER_NIPALS) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, solver), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_simpls_fixture(int& failures,
                        const ::n4m::test::fixtures::SimplsFixture& fixture,
                        Handles& h,
                        bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_SIMPLS), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_svd_fixture(int& failures,
                     const ::n4m::test::fixtures::SvdFixture& fixture,
                     Handles& h,
                     bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_SVD), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_pls_svd_fixture(int& failures,
                         const ::n4m::test::fixtures::PlsSvdFixture& fixture,
                         Handles& h,
                         bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_PLS_SVD), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_SVD), N4M_OK);
    CHECK_EQ(n4m_config_set_deflation(h.cfg, N4M_DEFLATION_CANONICAL), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_kernel_fixture(int& failures,
                        const ::n4m::test::fixtures::KernelFixture& fixture,
                        Handles& h,
                        bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_KERNEL_ALGORITHM), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_pcr_fixture(int& failures,
                     const ::n4m::test::fixtures::PcrFixture& fixture,
                     Handles& h,
                     bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_PCR), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_SVD), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_wide_kernel_fixture(
    int& failures,
    const ::n4m::test::fixtures::WideKernelFixture& fixture,
    Handles& h,
    bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_WIDE_KERNEL), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_orthogonal_scores_fixture(
    int& failures,
    const ::n4m::test::fixtures::OrthogonalScoresFixture& fixture,
    Handles& h,
    bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_ORTHOGONAL_SCORES), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_power_fixture(int& failures,
                       const ::n4m::test::fixtures::PowerFixture& fixture,
                       Handles& h,
                       bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_POWER), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_randomized_svd_fixture(
    int& failures,
    const ::n4m::test::fixtures::RandomizedSvdFixture& fixture,
    Handles& h,
    bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_context_set_seed(h.ctx, kRandomizedSvdFixtureSeed), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_RANDOMIZED_SVD), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

template <typename Fixture>
void fit_canonical_fixture(int& failures,
                           const Fixture& fixture,
                           Handles& h,
                           n4m_solver_t solver,
                           bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_PLS_CANONICAL), N4M_OK);
    CHECK_EQ(n4m_config_set_deflation(h.cfg, N4M_DEFLATION_CANONICAL), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, solver), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_pls_da_fixture(int& failures,
                        const ::n4m::test::fixtures::PlsDaFixture& fixture,
                        Handles& h,
                        bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_PLS_DA), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_NIPALS), N4M_OK);
    CHECK_EQ(n4m_config_set_deflation(h.cfg, N4M_DEFLATION_REGRESSION), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_opls_fixture(int& failures,
                      const ::n4m::test::fixtures::OplsFixture& fixture,
                      Handles& h,
                      bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_OPLS), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_NIPALS), N4M_OK);
    CHECK_EQ(n4m_config_set_deflation(h.cfg, N4M_DEFLATION_ORTHOGONAL), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

void fit_opls_da_fixture(int& failures,
                         const ::n4m::test::fixtures::OplsDaFixture& fixture,
                         Handles& h,
                         bool store_scores = false) {
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_OPLS_DA), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_NIPALS), N4M_OK);
    CHECK_EQ(n4m_config_set_deflation(h.cfg, N4M_DEFLATION_ORTHOGONAL), N4M_OK);
    if (store_scores) {
        CHECK_EQ(n4m_config_set_store_scores(h.cfg, 1), N4M_OK);
    }

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_OK);
    CHECK_NE(h.model, nullptr);
}

}  // namespace

#undef N4M_TEST_NOINLINE

TEST(model_phase1, fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kPhase1Fixtures) {
        Handles h;
        fit_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::int64_t rows = 0;
        std::int64_t cols = 0;
        CHECK_EQ(n4m_array_shape(pred, &rows, &cols), N4M_OK);
        CHECK_EQ(rows, fixture.predict_train.rows);
        CHECK_EQ(cols, fixture.predict_train.cols);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        std::vector<double> caller_pred(fixture.predict_train.size, 0.0);
        n4m_matrix_view_t out{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&out,
                                               caller_pred.data(),
                                               fixture.predict_train.rows,
                                               fixture.predict_train.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        CHECK_EQ(n4m_model_predict(h.ctx, h.model, &X, &out), N4M_OK);
        check_close_values(failures, fixture.id, caller_pred.data(), fixture.predict_train);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, simpls_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kSimplsFixtures) {
        Handles h;
        fit_simpls_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "simpls coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "simpls intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "simpls x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "simpls x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "simpls y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "simpls y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "simpls weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "simpls loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "simpls y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "simpls rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, svd_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kSvdFixtures) {
        Handles h;
        fit_svd_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "svd coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "svd intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "svd x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "svd x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "svd y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "svd y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "svd weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "svd loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "svd y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "svd rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, pls_svd_fixture_parity_for_cross_covariance_scores) {
    for (const auto& fixture : ::n4m::test::fixtures::kPlsSvdFixtures) {
        Handles h;
        fit_pls_svd_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "pls-svd coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "pls-svd intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "pls-svd x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "pls-svd x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "pls-svd y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "pls-svd y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "pls-svd weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "pls-svd loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "pls-svd y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "pls-svd rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, power_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kPowerFixtures) {
        Handles h;
        fit_power_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "power coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "power intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "power x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "power x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "power y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "power y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "power weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "power loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "power y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "power rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, randomized_svd_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kRandomizedSvdFixtures) {
        Handles h;
        fit_randomized_svd_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "randomized svd coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "randomized svd intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "randomized svd x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "randomized svd x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "randomized svd y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "randomized svd y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "randomized svd weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "randomized svd loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "randomized svd y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "randomized svd rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, canonical_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kCanonicalFixtures) {
        Handles h;
        fit_canonical_fixture(failures, fixture, h, N4M_SOLVER_NIPALS);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "canonical coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "canonical intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "canonical x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "canonical x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "canonical y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "canonical y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "canonical weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "canonical loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "canonical y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "canonical rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, canonical_svd_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kCanonicalSvdFixtures) {
        Handles h;
        fit_canonical_fixture(failures, fixture, h, N4M_SOLVER_SVD);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "canonical svd coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "canonical svd intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "canonical svd x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "canonical svd x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "canonical svd y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "canonical svd y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "canonical svd weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "canonical svd loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "canonical svd y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "canonical svd rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, pls_da_fixture_parity_for_dummy_targets) {
    for (const auto& fixture : ::n4m::test::fixtures::kPlsDaFixtures) {
        Handles h;
        fit_pls_da_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "pls-da coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "pls-da intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "pls-da x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "pls-da x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "pls-da y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "pls-da y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "pls-da weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "pls-da loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "pls-da y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "pls-da rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, opls_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kOplsFixtures) {
        Handles h;
        fit_opls_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "opls coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "opls intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "opls x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "opls x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "opls y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "opls y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "opls weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "opls loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "opls y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "opls rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, opls_da_fixture_parity_for_binary_and_multiclass_dummy_targets) {
    for (const auto& fixture : ::n4m::test::fixtures::kOplsDaFixtures) {
        Handles h;
        fit_opls_da_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "opls-da coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "opls-da intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "opls-da x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "opls-da x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "opls-da y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "opls-da y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "opls-da weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "opls-da loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "opls-da y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "opls-da rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, kernel_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kKernelFixtures) {
        Handles h;
        fit_kernel_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "kernel coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "kernel intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "kernel x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "kernel x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "kernel y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "kernel y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "kernel weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "kernel loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "kernel y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "kernel rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, wide_kernel_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kWideKernelFixtures) {
        Handles h;
        fit_wide_kernel_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "wide kernel coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "wide kernel intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "wide kernel x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "wide kernel x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "wide kernel y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "wide kernel y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "wide kernel weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "wide kernel loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "wide kernel y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "wide kernel rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, orthogonal_scores_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kOrthogonalScoresFixtures) {
        Handles h;
        fit_orthogonal_scores_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "orthogonal scores coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "orthogonal scores intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "orthogonal scores x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "orthogonal scores x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "orthogonal scores y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "orthogonal scores y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "orthogonal scores weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "orthogonal scores loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "orthogonal scores y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "orthogonal scores rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, pcr_fixture_parity_for_pls1_and_pls2) {
    for (const auto& fixture : ::n4m::test::fixtures::kPcrFixtures) {
        Handles h;
        fit_pcr_fixture(failures, fixture, h);

        std::int32_t n_components = 0;
        std::int32_t n_features = 0;
        std::int32_t n_targets = 0;
        CHECK_EQ(n4m_model_get_n_components(h.model, &n_components), N4M_OK);
        CHECK_EQ(n4m_model_get_n_features(h.model, &n_features), N4M_OK);
        CHECK_EQ(n4m_model_get_n_targets(h.model, &n_targets), N4M_OK);
        CHECK_EQ(n_components, fixture.n_components);
        CHECK_EQ(n_features, static_cast<std::int32_t>(fixture.X.cols));
        CHECK_EQ(n_targets, static_cast<std::int32_t>(fixture.Y.cols));

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_array_t* pred = nullptr;
        CHECK_EQ(n4m_model_predict_alloc(h.ctx, h.model, &X, &pred), N4M_OK);
        CHECK_NE(pred, nullptr);
        std::vector<double> pred_values = copy_values(pred);
        check_close_values(failures, fixture.id, pred_values.data(), fixture.predict_train);
        n4m_array_free(pred);

        check_array(failures, h.ctx, h.model, N4M_MODEL_COEFFICIENTS,
                    "pcr coefficients", fixture.coefficients);
        check_array(failures, h.ctx, h.model, N4M_MODEL_INTERCEPT,
                    "pcr intercept", fixture.intercept);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_MEAN,
                    "pcr x_mean", fixture.x_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_X_SCALE,
                    "pcr x_scale", fixture.x_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_MEAN,
                    "pcr y_mean", fixture.y_mean);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_SCALE,
                    "pcr y_scale", fixture.y_scale);
        check_array(failures, h.ctx, h.model, N4M_MODEL_WEIGHTS_W,
                    "pcr weights_w", fixture.weights_w);
        check_array(failures, h.ctx, h.model, N4M_MODEL_LOADINGS_P,
                    "pcr loadings_p", fixture.loadings_p);
        check_array(failures, h.ctx, h.model, N4M_MODEL_Y_LOADINGS_Q,
                    "pcr y_loadings_q", fixture.y_loadings_q);
        check_array(failures, h.ctx, h.model, N4M_MODEL_ROTATIONS_R,
                    "pcr rotations_r", fixture.rotations_r);
    }
}

TEST(model_phase1, transform_and_stored_scores_agree) {
    const auto& fixture = ::n4m::test::fixtures::kPhase1Fixtures[0];
    Handles h;
    fit_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
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
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, simpls_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kSimplsFixtures[0];
    Handles h;
    fit_simpls_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "simpls transform", transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "simpls stored scores", stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, svd_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kSvdFixtures[0];
    Handles h;
    fit_svd_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "svd transform", transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "svd stored scores", stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, pls_svd_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kPlsSvdFixtures[1];
    Handles h;
    fit_pls_svd_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "pls-svd transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "pls-svd stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, power_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kPowerFixtures[0];
    Handles h;
    fit_power_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "power transform", transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "power stored scores", stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, randomized_svd_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kRandomizedSvdFixtures[0];
    Handles h;
    fit_randomized_svd_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "randomized svd transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "randomized svd stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, canonical_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kCanonicalFixtures[0];
    Handles h;
    fit_canonical_fixture(failures, fixture, h, N4M_SOLVER_NIPALS, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "canonical transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "canonical stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, canonical_svd_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kCanonicalSvdFixtures[0];
    Handles h;
    fit_canonical_fixture(failures, fixture, h, N4M_SOLVER_SVD, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "canonical svd transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "canonical svd stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, pls_da_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kPlsDaFixtures[0];
    Handles h;
    fit_pls_da_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "pls-da transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "pls-da stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, opls_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kOplsFixtures[2];
    Handles h;
    fit_opls_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "opls transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "opls stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, opls_da_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kOplsDaFixtures[1];
    Handles h;
    fit_opls_da_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "opls-da transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "opls-da stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, kernel_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kKernelFixtures[0];
    Handles h;
    fit_kernel_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "kernel transform", transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "kernel stored scores", stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, wide_kernel_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kWideKernelFixtures[0];
    Handles h;
    fit_wide_kernel_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "wide kernel transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "wide kernel stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, orthogonal_scores_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kOrthogonalScoresFixtures[0];
    Handles h;
    fit_orthogonal_scores_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "orthogonal scores transform",
                       transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "orthogonal scores stored scores",
                       stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, pcr_transform_matches_reference_scores) {
    const auto& fixture = ::n4m::test::fixtures::kPcrFixtures[0];
    Handles h;
    fit_pcr_fixture(failures, fixture, h, true);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    n4m_array_t* transformed = nullptr;
    CHECK_EQ(n4m_model_transform_alloc(h.ctx, h.model, &X, &transformed), N4M_OK);
    CHECK_NE(transformed, nullptr);
    std::vector<double> transform_values = copy_values(transformed);
    check_close_values(failures, "pcr transform", transform_values.data(), fixture.scores_t);

    n4m_array_t* stored_scores = nullptr;
    CHECK_EQ(n4m_model_get_array(h.ctx, h.model, N4M_MODEL_SCORES_T, &stored_scores), N4M_OK);
    CHECK_NE(stored_scores, nullptr);
    std::vector<double> stored_values = copy_values(stored_scores);
    check_close_values(failures, "pcr stored scores", stored_values.data(), fixture.scores_t);
    n4m_array_free(stored_scores);
    n4m_array_free(transformed);
}

TEST(model_phase1, serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kPhase1Fixtures[2];
    Handles h;
    fit_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written), N4M_OK);
    CHECK_EQ(written, size);

    std::uint32_t fmt = 0;
    std::uint32_t abi_major = 0;
    std::uint32_t abi_minor = 0;
    std::uint32_t abi_patch = 0;
    CHECK_EQ(n4m_serialization_inspect(buffer.data(), buffer.size(),
                                       &fmt, &abi_major, &abi_minor, &abi_patch),
             N4M_OK);
    CHECK_EQ(fmt, static_cast<std::uint32_t>(N4M_SERIALIZATION_FORMAT_VERSION));
    CHECK_EQ(abi_major, static_cast<std::uint32_t>(N4M_ABI_VERSION_MAJOR));
    CHECK_EQ(abi_minor, static_cast<std::uint32_t>(N4M_ABI_VERSION_MINOR));
    CHECK_EQ(abi_patch, static_cast<std::uint32_t>(N4M_ABI_VERSION_PATCH));

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported), N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_predict", pred_values.data(), fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);

    std::vector<unsigned char> corrupted = buffer;
    flip_last_byte(corrupted);
    n4m_model_t* corrupt_model = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, corrupted.data(), corrupted.size(), &corrupt_model),
             N4M_ERR_CORRUPT_BUFFER);
    CHECK_EQ(corrupt_model, nullptr);

    std::vector<unsigned char> short_buffer(size - 1U, 0U);
    CHECK_EQ(n4m_model_export_to_buffer(h.model, short_buffer.data(), short_buffer.size(), &written),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(written, 0U);
}

TEST(model_phase1, simpls_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kSimplsFixtures[1];
    Handles h;
    fit_simpls_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written), N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported), N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_simpls_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, svd_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kSvdFixtures[1];
    Handles h;
    fit_svd_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_svd_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, pls_svd_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kPlsSvdFixtures[1];
    Handles h;
    fit_pls_svd_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_pls_svd_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, power_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kPowerFixtures[1];
    Handles h;
    fit_power_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_power_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, randomized_svd_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kRandomizedSvdFixtures[1];
    Handles h;
    fit_randomized_svd_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_randomized_svd_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, canonical_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kCanonicalFixtures[1];
    Handles h;
    fit_canonical_fixture(failures, fixture, h, N4M_SOLVER_NIPALS);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_canonical_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, canonical_svd_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kCanonicalSvdFixtures[1];
    Handles h;
    fit_canonical_fixture(failures, fixture, h, N4M_SOLVER_SVD);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_canonical_svd_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, pls_da_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kPlsDaFixtures[1];
    Handles h;
    fit_pls_da_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_pls_da_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, opls_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kOplsFixtures[2];
    Handles h;
    fit_opls_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_opls_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, opls_da_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kOplsDaFixtures[1];
    Handles h;
    fit_opls_da_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_opls_da_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, kernel_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kKernelFixtures[1];
    Handles h;
    fit_kernel_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_kernel_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, wide_kernel_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kWideKernelFixtures[1];
    Handles h;
    fit_wide_kernel_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_wide_kernel_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, orthogonal_scores_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kOrthogonalScoresFixtures[1];
    Handles h;
    fit_orthogonal_scores_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_orthogonal_scores_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, pcr_serialization_roundtrip_preserves_predictions) {
    const auto& fixture = ::n4m::test::fixtures::kPcrFixtures[1];
    Handles h;
    fit_pcr_fixture(failures, fixture, h);

    std::size_t size = 0;
    CHECK_EQ(n4m_model_export_size(h.model, &size), N4M_OK);
    CHECK(size > 64U);

    std::vector<unsigned char> buffer(size, 0U);
    std::size_t written = 0;
    CHECK_EQ(n4m_model_export_to_buffer(h.model, buffer.data(), buffer.size(), &written),
             N4M_OK);
    CHECK_EQ(written, size);

    n4m_model_t* imported = nullptr;
    CHECK_EQ(n4m_model_import_from_buffer(h.ctx, buffer.data(), buffer.size(), &imported),
             N4M_OK);
    CHECK_NE(imported, nullptr);

    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    n4m_array_t* pred = nullptr;
    CHECK_EQ(n4m_model_predict_alloc(h.ctx, imported, &X, &pred), N4M_OK);
    std::vector<double> pred_values = copy_values(pred);
    check_close_values(failures, "imported_pcr_predict", pred_values.data(),
                       fixture.predict_train);
    n4m_array_free(pred);
    n4m_model_destroy(imported);
}

TEST(model_phase1, validation_errors_are_deterministic) {
    const auto& fixture = ::n4m::test::fixtures::kPhase1Fixtures[0];
    Handles h;
    CHECK_EQ(n4m_context_create(&h.ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&h.cfg), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(h.cfg, fixture.n_components), N4M_OK);
    // Pick an algorithm that the public ABI does not yet dispatch through
    // n4m_model_fit. N4M_ALGO_MB_PLS (multi-block) is implemented as an
    // internal kernel but is not yet exposed through the public model
    // fit/predict surface.
    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_MB_PLS), N4M_OK);

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_ERR_UNSUPPORTED);
    CHECK_STR_CONTAINS(n4m_context_last_error(h.ctx), "PLS_DA");

    CHECK_EQ(n4m_config_set_algorithm(h.cfg, N4M_ALGO_PLS_REGRESSION), N4M_OK);
    CHECK_EQ(n4m_config_set_solver(h.cfg, N4M_SOLVER_NIPALS), N4M_OK);
    n4m_matrix_view_t bad_y = Y;
    bad_y.rows = Y.rows - 1;
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &bad_y, &h.model), N4M_ERR_SHAPE_MISMATCH);

    std::vector<double> x_with_nan(fixture.X.values, fixture.X.values + fixture.X.size);
    x_with_nan[0] = std::nan("");
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           x_with_nan.data(),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_ERR_INVALID_ARGUMENT);

    std::vector<std::int32_t> x_i32(fixture.X.size, 1);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           x_i32.data(),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_I32),
             N4M_OK);
    CHECK_EQ(n4m_model_fit(h.ctx, h.cfg, &X, &Y, &h.model), N4M_ERR_DTYPE_MISMATCH);
}

TEST(model_phase1, null_pointer_contracts_are_live) {
    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_config_create(&cfg), N4M_OK);

    n4m_model_t* model = reinterpret_cast<n4m_model_t*>(0x1);
    CHECK_EQ(n4m_model_fit(ctx, cfg, nullptr, nullptr, &model), N4M_ERR_NULL_POINTER);
    CHECK_EQ(model, nullptr);

    n4m_array_t* arr = reinterpret_cast<n4m_array_t*>(0x1);
    CHECK_EQ(n4m_model_predict_alloc(ctx, nullptr, nullptr, &arr), N4M_ERR_NULL_POINTER);
    CHECK_EQ(arr, nullptr);
    CHECK_EQ(n4m_model_get_array(ctx, nullptr, N4M_MODEL_COEFFICIENTS, &arr),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(arr, nullptr);

    n4m_dtype_t dtype = N4M_DTYPE_UNKNOWN;
    CHECK_EQ(n4m_array_dtype(nullptr, &dtype), N4M_ERR_NULL_POINTER);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(n4m_array_shape(nullptr, &rows, &cols), N4M_ERR_NULL_POINTER);
    n4m_matrix_view_t view{};
    CHECK_EQ(n4m_array_view(nullptr, &view), N4M_ERR_NULL_POINTER);
    n4m_array_free(nullptr);
    n4m_model_destroy(nullptr);

    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}
