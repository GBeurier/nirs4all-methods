// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 9 (partial) feature-selection /
// dimensionality-reduction operators (FlexiblePCA, FlexibleSVD).
//
// CARS and MCUVE require an internal PLS callback and are deferred to a
// later phase. Only the SVD-based operators are exercised here.
//
// Each operator contributes:
//   * a smoke test exercising create / fit / transform / destroy /
//     output_cols / is_fitted on a small inline matrix to validate the
//     public ABI surface and lifecycle semantics;
//   * a parity test loading a JSON fixture (fit on a training matrix,
//     transform on a test matrix, validate per-case 1e-10 abs / 1e-11 rel
//     tolerance against the frozen NumPy reference);
//   * a refit-replaces-state test asserting that a second `_fit` on the
//     same handle overrides the prior fit (sklearn-style refit semantics).
//
// Total contributions: 2 smoke + 2 parity + 2 refit = 6 new tests.
//
// Since main.cpp is frozen, the new register function is chained from
// `register_preprocessing_stateful_tests` in test_preprocessing_stateful.cpp.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef N4M_PARITY_FIXTURE_DIR
#  error "N4M_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

using ParityFixture = ::n4m_testing::Fixture;
using ParityCase    = ::n4m_testing::Case;

ParityFixture load_fixture(const std::string& filename) {
    return ::n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/true);
}

using ::n4m_testing::params_get_double;

void assert_close(const std::vector<double>& got,
                  const std::vector<double>& want,
                  const std::string& tag,
                  double abs_tol = 1e-10,
                  double rel_tol = 1e-11) {
    ::n4m_testing::assert_close(got, want, tag, abs_tol, rel_tol);
}

n4m_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st =
        n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

void test_flex_pca_smoke() {
    // 3 samples in 4 features. Choose data with one dominant component so
    // we can sanity-check that count_mode picks a 1-d projection that
    // captures most of the variance.
    double Xfit[12] = {
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0,
        3.0, 6.0, 9.0, 12.0,
    };
    n4m_pp_flex_pca_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_create(&h, /*n_components=*/2.0) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);

    int fitted = 1;
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 0);

    /* output_cols before fit -> NOT_FITTED */
    int64_t kcols = -1;
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_output_cols(h, &kcols)
                     == N4M_ERR_NOT_FITTED);

    n4m_matrix_view_t Xfv = make_rowmajor_view(Xfit, 3, 4);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_fit(h, Xfv) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_pp_flex_pca_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 1);

    /* With 3 samples and 4 features, k_full = min(3, 4) = 3. We requested
     * 2 components, so we should get 2. */
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_output_cols(h, &kcols) == N4M_OK);
    N4M_TEST_REQUIRE(kcols == 2);

    /* transform the fit matrix back through the operator. */
    double Y[6] = {0};
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 3, 2);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_transform(h, Xfv, Yv) == N4M_OK);
    for (int k = 0; k < 6; ++k) {
        N4M_TEST_REQUIRE(std::isfinite(Y[k]));
    }
    /* The data is a perfect rank-1 multiple — the second component should
     * have essentially zero magnitude. Allow a small floating-point
     * cushion since we're not testing exact bit-equality here. */
    double max_pc2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double v = std::fabs(Y[i * 2 + 1]);
        if (v > max_pc2) max_pc2 = v;
    }
    N4M_TEST_REQUIRE(max_pc2 < 1e-10);

    /* shape-mismatch transform: ask for k=3 -> SHAPE_MISMATCH */
    double Ybad[9] = {0};
    n4m_matrix_view_t Ybadv = make_rowmajor_view(Ybad, 3, 3);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_transform(h, Xfv, Ybadv)
                     == N4M_ERR_SHAPE_MISMATCH);

    /* not-fitted transform after a fresh handle */
    n4m_pp_flex_pca_handle_t* h2 = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_create(&h2, 1.0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_transform(h2, Xfv, Yv)
                     == N4M_ERR_NOT_FITTED);
    n4m_pp_flex_pca_destroy(h2);

    /* invalid constructor (n_components <= 0) */
    n4m_pp_flex_pca_handle_t* h_bad = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_create(&h_bad, 0.0)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(h_bad == nullptr);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_create(&h_bad, -1.0)
                     == N4M_ERR_INVALID_ARGUMENT);

    n4m_pp_flex_pca_destroy(h);
    n4m_pp_flex_pca_destroy(nullptr);  // null-safe
}

void test_flex_svd_smoke() {
    double Xfit[12] = {
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 0.0, 5.0,
        3.0, 5.0, 2.0, 1.0,
    };
    n4m_pp_flex_svd_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_create(&h, /*n_components=*/2.0) == N4M_OK);

    int fitted = 1;
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 0);

    int64_t kcols = -1;
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_output_cols(h, &kcols)
                     == N4M_ERR_NOT_FITTED);

    n4m_matrix_view_t Xfv = make_rowmajor_view(Xfit, 3, 4);
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_fit(h, Xfv) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_pp_flex_svd_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 1);

    N4M_TEST_REQUIRE(n4m_pp_flex_svd_output_cols(h, &kcols) == N4M_OK);
    N4M_TEST_REQUIRE(kcols == 2);

    double Y[6] = {0};
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 3, 2);
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_transform(h, Xfv, Yv) == N4M_OK);
    for (int k = 0; k < 6; ++k) {
        N4M_TEST_REQUIRE(std::isfinite(Y[k]));
    }
    /* shape-mismatch */
    double Ybad[9] = {0};
    n4m_matrix_view_t Ybadv = make_rowmajor_view(Ybad, 3, 3);
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_transform(h, Xfv, Ybadv)
                     == N4M_ERR_SHAPE_MISMATCH);

    /* not-fitted */
    n4m_pp_flex_svd_handle_t* h2 = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_create(&h2, 1.0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_transform(h2, Xfv, Yv)
                     == N4M_ERR_NOT_FITTED);
    n4m_pp_flex_svd_destroy(h2);

    /* invalid constructor */
    n4m_pp_flex_svd_handle_t* h_bad = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_create(&h_bad, 0.0)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(h_bad == nullptr);

    n4m_pp_flex_svd_destroy(h);
    n4m_pp_flex_svd_destroy(nullptr);
}

// ---------------------------------------------------------------------------
// Parity tests
// ---------------------------------------------------------------------------

void verify_flex_pca_parity() {
    ParityFixture fx = load_fixture("flexible_pca_v1.json");
    for (const auto& c : fx.cases) {
        const double n_components =
            params_get_double(c.params_json, "n_components", 5.0);
        n4m_pp_flex_pca_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_flex_pca_create(&h, n_components) == N4M_OK);

        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_flex_pca_fit(h, Xfv) == N4M_OK);

        int64_t k_learned = -1;
        N4M_TEST_REQUIRE(n4m_pp_flex_pca_output_cols(h, &k_learned) == N4M_OK);
        N4M_TEST_REQUIRE(k_learned == c.output_cols);

        std::vector<double> in = fx.input;
        const std::int64_t out_size = c.output_rows * c.output_cols;
        std::vector<double> out(static_cast<std::size_t>(out_size), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_flex_pca_transform(h, Xv, Yv) == N4M_OK);
        /* Tolerance widened to 1e-10 abs / 1e-11 rel — the gap between
         * the n4m Jacobi SVD and NumPy's LAPACK gesdd is a few ULPs on
         * mid-to-low singular values, well within this budget. */
        assert_close(out, c.expected_output, "flex_pca/" + c.name);
        n4m_pp_flex_pca_destroy(h);
    }
}

void verify_flex_svd_parity() {
    ParityFixture fx = load_fixture("flexible_svd_v1.json");
    for (const auto& c : fx.cases) {
        const double n_components =
            params_get_double(c.params_json, "n_components", 5.0);
        n4m_pp_flex_svd_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_flex_svd_create(&h, n_components) == N4M_OK);

        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_flex_svd_fit(h, Xfv) == N4M_OK);

        int64_t k_learned = -1;
        N4M_TEST_REQUIRE(n4m_pp_flex_svd_output_cols(h, &k_learned) == N4M_OK);
        N4M_TEST_REQUIRE(k_learned == c.output_cols);

        std::vector<double> in = fx.input;
        const std::int64_t out_size = c.output_rows * c.output_cols;
        std::vector<double> out(static_cast<std::size_t>(out_size), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_flex_svd_transform(h, Xv, Yv) == N4M_OK);
        assert_close(out, c.expected_output, "flex_svd/" + c.name);
        n4m_pp_flex_svd_destroy(h);
    }
}

// ---------------------------------------------------------------------------
// Refit-replaces-state tests
// ---------------------------------------------------------------------------

void test_flex_pca_refit_replaces_state() {
    n4m_pp_flex_pca_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_create(&h, 1.0) == N4M_OK);

    /* X1 vs X2 produce very different principal axes; a third matrix X3
     * projected through each fit should give different outputs. */
    double X1[12] = {1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0};
    double X2[12] = {0.0, 0.0, 0.0, 1.0,
                     0.0, 0.0, 1.0, 0.0,
                     1.0, 0.0, 0.0, 0.0};
    double X3[8]  = {1.0, 0.0, 0.5, 0.25,
                     0.0, 0.5, 0.0, 0.75};
    double outA[2], outB[2];

    n4m_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    n4m_matrix_view_init_rowmajor(&vX1, X1, 3, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX2, X2, 3, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX3, X3, 2, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutA, outA, 2, 1, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutB, outB, 2, 1, N4M_DTYPE_F64);

    N4M_TEST_REQUIRE(n4m_pp_flex_pca_fit(h, vX1) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_transform(h, vX3, voutA) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_fit(h, vX2) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_pca_transform(h, vX3, voutB) == N4M_OK);

    bool any_diff = false;
    for (int i = 0; i < 2; ++i) {
        if (std::fabs(outA[i] - outB[i]) > 1e-9) { any_diff = true; break; }
    }
    N4M_TEST_REQUIRE(any_diff);
    n4m_pp_flex_pca_destroy(h);
}

void test_flex_svd_refit_replaces_state() {
    n4m_pp_flex_svd_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_create(&h, 1.0) == N4M_OK);

    double X1[12] = {1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0};
    double X2[12] = {0.0, 0.0, 0.0, 1.0,
                     0.0, 0.0, 1.0, 0.0,
                     1.0, 0.0, 0.0, 0.0};
    double X3[8]  = {1.0, 0.0, 0.5, 0.25,
                     0.0, 0.5, 0.0, 0.75};
    double outA[2], outB[2];

    n4m_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    n4m_matrix_view_init_rowmajor(&vX1, X1, 3, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX2, X2, 3, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX3, X3, 2, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutA, outA, 2, 1, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutB, outB, 2, 1, N4M_DTYPE_F64);

    N4M_TEST_REQUIRE(n4m_pp_flex_svd_fit(h, vX1) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_transform(h, vX3, voutA) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_fit(h, vX2) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_flex_svd_transform(h, vX3, voutB) == N4M_OK);

    bool any_diff = false;
    for (int i = 0; i < 2; ++i) {
        if (std::fabs(outA[i] - outB[i]) > 1e-9) { any_diff = true; break; }
    }
    N4M_TEST_REQUIRE(any_diff);
    n4m_pp_flex_svd_destroy(h);
}

}  // namespace

void register_preprocessing_feature_selection_tests(n4m_testing::Runner& r);
void register_preprocessing_feature_selection_tests(n4m_testing::Runner& r) {
    r.run("pp_flex_pca_smoke",  test_flex_pca_smoke);
    r.run("pp_flex_pca_parity", verify_flex_pca_parity);
    r.run("pp_flex_pca_refit",  test_flex_pca_refit_replaces_state);
    r.run("pp_flex_svd_smoke",  test_flex_svd_smoke);
    r.run("pp_flex_svd_parity", verify_flex_svd_parity);
    r.run("pp_flex_svd_refit",  test_flex_svd_refit_replaces_state);
}
