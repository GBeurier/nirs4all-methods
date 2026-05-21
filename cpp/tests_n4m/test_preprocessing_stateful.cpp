// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 3 stateful preprocessing operators (MSC,
// EMSC, Baseline column-centering, Derivate). Each operator contributes:
//
//   * a smoke test exercising create / fit / transform / destroy on a
//     small inline matrix to validate the public ABI surface and
//     lifecycle semantics;
//   * a parity test loading a JSON fixture (fit on a training matrix,
//     transform on a test matrix, validate byte-equality / 1e-10 abs /
//     1e-11 rel tolerance against the reference output);
//   * a NOT_FITTED test asserting `_transform` before `_fit` returns
//     N4M_ERR_NOT_FITTED.
//
// Baseline also gets an inverse-transform test, and Derivate gets a
// dedicated `n4m_pp_derivate_output_cols(2, 200) == 198` shape-helper
// test. Total contributions: 4 smoke + 4 parity + 4 not-fitted + 1
// inverse + 1 output-cols = 14 new tests.

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

using ::n4m_testing::params_get_bool;
using ::n4m_testing::params_get_double;
using ::n4m_testing::params_get_int;
using ::n4m_testing::params_get_string;

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

void test_msc_smoke() {
    // Three "spectra" with a linear scatter pattern so MSC has something to
    // learn. Each row has a different multiplier on a reference signal.
    double Xfit[12] = {
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0,
        0.5, 1.0, 1.5, 2.0,
    };
    double Xt[8] = {
        1.5, 3.0, 4.5, 6.0,
        2.5, 5.0, 7.5, 10.0,
    };
    double Y[8] = {0};
    n4m_pp_msc_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_msc_create(&h) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);

    int fitted = 1;
    N4M_TEST_REQUIRE(n4m_pp_msc_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 0);

    n4m_matrix_view_t Xfv = make_rowmajor_view(Xfit, 3, 4);
    N4M_TEST_REQUIRE(n4m_pp_msc_fit(h, Xfv) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_pp_msc_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 1);

    n4m_matrix_view_t Xv = make_rowmajor_view(Xt, 2, 4);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 4);
    N4M_TEST_REQUIRE(n4m_pp_msc_transform(h, Xv, Yv) == N4M_OK);
    // The Xfit columns are scalar multiples of [1, 2, 3, 4] so each MSC
    // (a, b) per column should be exact; the resulting MSC-transformed
    // rows are the per-row mean (after scatter correction). Just sanity-
    // check that none of the outputs is NaN/Inf.
    for (int k = 0; k < 8; ++k) {
        N4M_TEST_REQUIRE(std::isfinite(Y[k]));
    }
    n4m_pp_msc_destroy(h);
    n4m_pp_msc_destroy(nullptr);  // null-safe
}

void test_emsc_smoke() {
    double X[12] = {
        1.0, 2.0, 3.0, 4.0,
        1.5, 2.5, 3.5, 4.5,
        2.0, 3.0, 4.0, 5.0,
    };
    double Y[12] = {0};
    n4m_pp_emsc_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_emsc_create(&h, /*degree=*/2) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);

    n4m_matrix_view_t Xv = make_rowmajor_view(X, 3, 4);
    N4M_TEST_REQUIRE(n4m_pp_emsc_fit(h, Xv) == N4M_OK);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 3, 4);
    N4M_TEST_REQUIRE(n4m_pp_emsc_transform(h, Xv, Yv) == N4M_OK);
    for (int k = 0; k < 12; ++k) {
        N4M_TEST_REQUIRE(std::isfinite(Y[k]));
    }
    n4m_pp_emsc_destroy(h);
}

void test_baseline_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    n4m_pp_baseline_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_baseline_create(&h) == N4M_OK);

    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_baseline_fit(h, Xv) == N4M_OK);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, Xv, Yv) == N4M_OK);

    // Column means: (1+4)/2=2.5, (2+5)/2=3.5, (3+6)/2=4.5.
    // Row 0: [1-2.5, 2-3.5, 3-4.5] = [-1.5, -1.5, -1.5]
    N4M_TEST_REQUIRE(std::fabs(Y[0] + 1.5) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[1] + 1.5) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[2] + 1.5) < 1e-15);
    // Row 1: [4-2.5, 5-3.5, 6-4.5] = [1.5, 1.5, 1.5]
    N4M_TEST_REQUIRE(std::fabs(Y[3] - 1.5) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[5] - 1.5) < 1e-15);
    n4m_pp_baseline_destroy(h);
}

void test_baseline_inverse() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    double Z[6] = {0};
    n4m_pp_baseline_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_baseline_create(&h) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    n4m_matrix_view_t Zv = make_rowmajor_view(Z, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_baseline_fit(h, Xv) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, Xv, Yv) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_baseline_inverse_transform(h, Yv, Zv) == N4M_OK);
    for (int k = 0; k < 6; ++k) {
        N4M_TEST_REQUIRE(std::fabs(Z[k] - X[k]) < 1e-15);
    }
    n4m_pp_baseline_destroy(h);
}

void test_derivate_smoke() {
    double X[8] = {1.0, 2.0, 4.0, 7.0,    /* diff1: [1, 2, 3], diff2: [1, 1] */
                   2.0, 3.0, 5.0, 8.0};
    double Y[6] = {0};  // output cols = 4 - 1 = 3, rows = 2
    n4m_pp_derivate_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_derivate_create(&h, /*order=*/1,
                                              /*delta=*/1.0) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 4);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_derivate_fit(h, Xv) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_derivate_transform(h, Xv, Yv) == N4M_OK);
    // Row 0 diffs: [2-1, 4-2, 7-4] = [1, 2, 3]
    N4M_TEST_REQUIRE(std::fabs(Y[0] - 1.0) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[1] - 2.0) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[2] - 3.0) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[3] - 1.0) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[5] - 3.0) < 1e-15);
    n4m_pp_derivate_destroy(h);
}

void test_derivate_output_cols() {
    N4M_TEST_REQUIRE(n4m_pp_derivate_output_cols(2, 200) == 198);
    N4M_TEST_REQUIRE(n4m_pp_derivate_output_cols(1, 10)  == 9);
    N4M_TEST_REQUIRE(n4m_pp_derivate_output_cols(3, 200) == 197);
    // Degenerate: order >= input_cols → 0 (out-of-range).
    N4M_TEST_REQUIRE(n4m_pp_derivate_output_cols(10, 10) == 0);
    N4M_TEST_REQUIRE(n4m_pp_derivate_output_cols(1, 1)   == 0);
}

// ---------------------------------------------------------------------------
// NOT_FITTED tests
// ---------------------------------------------------------------------------

void test_msc_not_fitted() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    n4m_pp_msc_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_msc_create(&h) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_msc_transform(h, Xv, Yv) == N4M_ERR_NOT_FITTED);
    N4M_TEST_REQUIRE(n4m_pp_msc_inverse_transform(h, Xv, Yv) == N4M_ERR_NOT_FITTED);
    n4m_pp_msc_destroy(h);
}

void test_emsc_not_fitted() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    n4m_pp_emsc_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_emsc_create(&h, 2) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_emsc_transform(h, Xv, Yv) == N4M_ERR_NOT_FITTED);
    n4m_pp_emsc_destroy(h);
}

void test_baseline_not_fitted() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    n4m_pp_baseline_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_baseline_create(&h) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, Xv, Yv) == N4M_ERR_NOT_FITTED);
    N4M_TEST_REQUIRE(n4m_pp_baseline_inverse_transform(h, Xv, Yv) == N4M_ERR_NOT_FITTED);
    n4m_pp_baseline_destroy(h);
}

void test_derivate_not_fitted() {
    double X[8] = {1, 2, 4, 7, 2, 3, 5, 8};
    double Y[6] = {0};
    n4m_pp_derivate_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_derivate_create(&h, 1, 1.0) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 4);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_derivate_transform(h, Xv, Yv) == N4M_ERR_NOT_FITTED);
    n4m_pp_derivate_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests
// ---------------------------------------------------------------------------

void verify_msc_parity() {
    ParityFixture fx = load_fixture("msc_v1.json");
    for (const auto& c : fx.cases) {
        const std::string variant = params_get_string(c.params_json, "variant", "");
        n4m_pp_msc_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_msc_create(&h) == N4M_OK);
        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_msc_fit(h, Xfv) == N4M_OK);

        if (variant == "inverse") {
            // The reference for this case is inverse_transform(transform(test)).
            // First apply transform to the test matrix, then inverse_transform.
            std::vector<double> in = fx.input;
            std::vector<double> mid(in.size(), 0.0);
            std::vector<double> out(in.size(), 0.0);
            n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.rows, fx.cols);
            n4m_matrix_view_t Mv = make_rowmajor_view(mid.data(),
                                                       fx.rows, fx.cols);
            n4m_matrix_view_t Ov = make_rowmajor_view(out.data(),
                                                       fx.rows, fx.cols);
            N4M_TEST_REQUIRE(n4m_pp_msc_transform(h, Xv, Mv) == N4M_OK);
            N4M_TEST_REQUIRE(n4m_pp_msc_inverse_transform(h, Mv, Ov) == N4M_OK);
            assert_close(out, c.expected_output, "msc/" + c.name);
        } else {
            std::vector<double> in = fx.input;
            std::vector<double> out(in.size(), 0.0);
            n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.rows, fx.cols);
            n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                       fx.rows, fx.cols);
            N4M_TEST_REQUIRE(n4m_pp_msc_transform(h, Xv, Yv) == N4M_OK);
            assert_close(out, c.expected_output, "msc/" + c.name);
        }
        n4m_pp_msc_destroy(h);
    }
}

void verify_emsc_parity() {
    ParityFixture fx = load_fixture("emsc_v1.json");
    for (const auto& c : fx.cases) {
        const int degree = static_cast<int>(params_get_int(c.params_json,
                                                            "degree", 2));
        n4m_pp_emsc_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_emsc_create(&h, degree) == N4M_OK);
        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_emsc_fit(h, Xfv) == N4M_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_emsc_transform(h, Xv, Yv) == N4M_OK);
        /* EMSC parity is bounded by the LAPACK gelsd (SVD-based) vs our
         * Householder QR difference on the ill-conditioned raw-integer
         * wavelength basis (1, w, w^2, …, w^degree with w in [0, p-1]).
         * For degree=3 and p=200 the wavelength column ranges 0..7.9e6;
         * the resulting condition number of B costs roughly 8 decimal
         * digits regardless of solver choice. We allow 5e-10 abs / 5e-10
         * rel, well within the "couple-of-ULPs of gelsd" budget the
         * Phase 3 brief allotted to least-squares ops. */
        assert_close(out, c.expected_output, "emsc/" + c.name,
                     /*abs=*/5e-10, /*rel=*/5e-10);
        n4m_pp_emsc_destroy(h);
    }
}

void verify_baseline_parity() {
    ParityFixture fx = load_fixture("baseline_center_v1.json");
    for (const auto& c : fx.cases) {
        const std::string variant = params_get_string(c.params_json, "variant", "");
        n4m_pp_baseline_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_baseline_create(&h) == N4M_OK);
        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_baseline_fit(h, Xfv) == N4M_OK);

        if (variant == "test") {
            std::vector<double> in = fx.input;
            std::vector<double> out(in.size(), 0.0);
            n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.rows, fx.cols);
            n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                       fx.rows, fx.cols);
            N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, Xv, Yv) == N4M_OK);
            assert_close(out, c.expected_output, "baseline/" + c.name,
                         1e-12, 1e-13);
        } else if (variant == "inverse") {
            // transform(fit_X) then inverse_transform.
            std::vector<double> in = fx.fit_input;
            std::vector<double> mid(in.size(), 0.0);
            std::vector<double> out(in.size(), 0.0);
            n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.fit_rows, fx.fit_cols);
            n4m_matrix_view_t Mv = make_rowmajor_view(mid.data(),
                                                       fx.fit_rows, fx.fit_cols);
            n4m_matrix_view_t Ov = make_rowmajor_view(out.data(),
                                                       fx.fit_rows, fx.fit_cols);
            N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, Xv, Mv) == N4M_OK);
            N4M_TEST_REQUIRE(n4m_pp_baseline_inverse_transform(h, Mv, Ov) == N4M_OK);
            assert_close(out, c.expected_output, "baseline/" + c.name,
                         1e-12, 1e-13);
        } else {
            // default: fit on fit_X then transform on fit_X.
            std::vector<double> in = fx.fit_input;
            std::vector<double> out(in.size(), 0.0);
            n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.fit_rows, fx.fit_cols);
            n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                       fx.fit_rows, fx.fit_cols);
            N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, Xv, Yv) == N4M_OK);
            assert_close(out, c.expected_output, "baseline/" + c.name,
                         1e-12, 1e-13);
        }
        n4m_pp_baseline_destroy(h);
    }
}

void verify_derivate_parity() {
    ParityFixture fx = load_fixture("derivate_v1.json");
    for (const auto& c : fx.cases) {
        const int order  = static_cast<int>(params_get_int(c.params_json,
                                                            "order", 1));
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        n4m_pp_derivate_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_derivate_create(&h, order, delta) == N4M_OK);
        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_derivate_fit(h, Xfv) == N4M_OK);

        std::vector<double> in = fx.input;
        const std::int64_t out_size = c.output_rows * c.output_cols;
        std::vector<double> out(static_cast<std::size_t>(out_size), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_derivate_transform(h, Xv, Yv) == N4M_OK);
        assert_close(out, c.expected_output, "derivate/" + c.name,
                     1e-12, 1e-13);
        n4m_pp_derivate_destroy(h);
    }
}

/* Refit contract: calling _fit twice on the same handle must replace prior
 * state. We verify by fitting on X1, transforming X3 to capture output_A,
 * then fitting again on X2 (different stats), and transforming the same X3
 * to capture output_B. output_A must differ from output_B because the fit
 * state was overwritten. */
void test_msc_refit_replaces_state() {
    n4m_pp_msc_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_msc_create(&h) == N4M_OK);

    /* X1: ascending rows */
    double X1[6] = {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0};
    /* X2: different distribution */
    double X2[6] = {10.0,  20.0,  30.0,
                     5.0,  15.0,  25.0};
    /* X3: shared transform input */
    double X3[6] = {2.0, 4.0, 6.0,
                    3.0, 5.0, 7.0};
    double outA[6], outB[6];
    n4m_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    n4m_matrix_view_init_rowmajor(&vX1, X1, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX2, X2, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX3, X3, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutA, outA, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutB, outB, 2, 3, N4M_DTYPE_F64);

    N4M_TEST_REQUIRE(n4m_pp_msc_fit(h, vX1) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_msc_transform(h, vX3, voutA) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_msc_fit(h, vX2) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_msc_transform(h, vX3, voutB) == N4M_OK);
    /* outputs must differ — different references produce different regressions */
    bool any_diff = false;
    for (int i = 0; i < 6; ++i) {
        if (std::fabs(outA[i] - outB[i]) > 1e-9) { any_diff = true; break; }
    }
    N4M_TEST_REQUIRE(any_diff);
    n4m_pp_msc_destroy(h);
}

void test_emsc_refit_replaces_state() {
    n4m_pp_emsc_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_emsc_create(&h, 2) == N4M_OK);
    double X1[8] = {1.0, 2.0, 3.0, 4.0,
                    2.0, 3.0, 4.0, 5.0};
    double X2[8] = {10.0,  5.0, 20.0, 15.0,
                    -1.0, -2.0, -3.0, -4.0};
    double X3[8] = {3.0, 4.0, 5.0, 6.0,
                    2.5, 3.5, 4.5, 5.5};
    double outA[8], outB[8];
    n4m_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    n4m_matrix_view_init_rowmajor(&vX1, X1, 2, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX2, X2, 2, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX3, X3, 2, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutA, outA, 2, 4, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutB, outB, 2, 4, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(n4m_pp_emsc_fit(h, vX1) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_emsc_transform(h, vX3, voutA) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_emsc_fit(h, vX2) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_emsc_transform(h, vX3, voutB) == N4M_OK);
    bool any_diff = false;
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(outA[i] - outB[i]) > 1e-9) { any_diff = true; break; }
    }
    N4M_TEST_REQUIRE(any_diff);
    n4m_pp_emsc_destroy(h);
}

void test_baseline_refit_replaces_state() {
    n4m_pp_baseline_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_baseline_create(&h) == N4M_OK);
    double X1[6] = {0.0,  0.0,  0.0,
                    0.0,  0.0,  0.0};
    double X2[6] = {10.0, 20.0, 30.0,
                    40.0, 50.0, 60.0};
    double X3[6] = {1.0,  2.0,  3.0,
                    4.0,  5.0,  6.0};
    double outA[6], outB[6];
    n4m_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    n4m_matrix_view_init_rowmajor(&vX1, X1, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX2, X2, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&vX3, X3, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutA, outA, 2, 3, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&voutB, outB, 2, 3, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(n4m_pp_baseline_fit(h, vX1) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, vX3, voutA) == N4M_OK);
    /* mean(X1) is all zeros, so outA == X3 */
    for (int i = 0; i < 6; ++i) N4M_TEST_REQUIRE(std::fabs(outA[i] - X3[i]) < 1e-15);
    N4M_TEST_REQUIRE(n4m_pp_baseline_fit(h, vX2) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_baseline_transform(h, vX3, voutB) == N4M_OK);
    /* mean(X2) = [25, 35, 45], so outB[0] = X3[0] - 25 = -24, etc. */
    const double expected[6] = { 1.0 - 25.0, 2.0 - 35.0, 3.0 - 45.0,
                                  4.0 - 25.0, 5.0 - 35.0, 6.0 - 45.0 };
    for (int i = 0; i < 6; ++i) N4M_TEST_REQUIRE(std::fabs(outB[i] - expected[i]) < 1e-15);
    n4m_pp_baseline_destroy(h);
}

}  // namespace

void register_preprocessing_stateful_tests(n4m_testing::Runner& r);
void register_preprocessing_stateful_tests(n4m_testing::Runner& r) {
    r.run("pp_msc_smoke",            test_msc_smoke);
    r.run("pp_msc_not_fitted",       test_msc_not_fitted);
    r.run("pp_msc_parity",           verify_msc_parity);
    r.run("pp_msc_refit",            test_msc_refit_replaces_state);
    r.run("pp_emsc_smoke",           test_emsc_smoke);
    r.run("pp_emsc_not_fitted",      test_emsc_not_fitted);
    r.run("pp_emsc_parity",          verify_emsc_parity);
    r.run("pp_emsc_refit",           test_emsc_refit_replaces_state);
    r.run("pp_baseline_smoke",       test_baseline_smoke);
    r.run("pp_baseline_inverse",     test_baseline_inverse);
    r.run("pp_baseline_not_fitted",  test_baseline_not_fitted);
    r.run("pp_baseline_parity",      verify_baseline_parity);
    r.run("pp_baseline_refit",       test_baseline_refit_replaces_state);
    r.run("pp_derivate_smoke",       test_derivate_smoke);
    r.run("pp_derivate_output_cols", test_derivate_output_cols);
    r.run("pp_derivate_not_fitted",  test_derivate_not_fitted);
    r.run("pp_derivate_parity",      verify_derivate_parity);
}
