// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 6 wavelets preprocessing operators
// (Wavelet, Haar, WaveletDenoise, WaveletFeatures, WaveletPCA,
//  WaveletSVD).
//
// Each operator contributes:
//   - a smoke test exercising create / transform / destroy (and
//     fit / transform / is_fitted / output_cols for the stateful pair),
//   - a parity test loading a JSON fixture and verifying the output
//     matches the frozen NumPy reference to within the per-operator
//     tolerance described in the Phase 6 brief.
//
// Since main.cpp is frozen, the new register function is chained from
// `register_preprocessing_feature_selection_tests` in
// test_preprocessing_feature_selection.cpp (FlexiblePCA/SVD is the
// closest existing test surface).
//
// Total contributions: 6 smoke + 6 parity = 12 new tests.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef N4M_PARITY_FIXTURE_DIR
#  error "N4M_PARITY_FIXTURE_DIR must be defined"
#endif

// Phase 6 ABI declarations live in n4m.h §23 (consumed via the
// `#include "n4m/n4m.h"` above).

namespace {

using ParityFixture = ::n4m_testing::Fixture;
using ParityCase    = ::n4m_testing::Case;
using ::n4m_testing::params_get_double;
using ::n4m_testing::params_get_int;
using ::n4m_testing::params_get_string;

ParityFixture load_fixture(const std::string& filename,
                            bool require_per_case_shape) {
    return ::n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/" + filename,
        require_per_case_shape);
}

n4m_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st =
        n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

n4m_pp_wavelet_family_t family_from_string(const std::string& s) {
    if (s == "haar")  return N4M_PP_WAVELET_HAAR;
    if (s == "db4")   return N4M_PP_WAVELET_DB4;
    if (s == "sym4")  return N4M_PP_WAVELET_SYM4;
    if (s == "coif1") return N4M_PP_WAVELET_COIF1;
    throw std::runtime_error("unknown wavelet family: " + s);
}

n4m_pp_wavelet_boundary_t mode_from_string(const std::string& s) {
    if (s == "periodization") return N4M_PP_WAVELET_BOUNDARY_PERIODIZATION;
    if (s == "symmetric")     return N4M_PP_WAVELET_BOUNDARY_SYMMETRIC;
    if (s == "zero")          return N4M_PP_WAVELET_BOUNDARY_ZERO;
    throw std::runtime_error("unknown boundary mode: " + s);
}

n4m_pp_wavelet_threshold_t threshold_from_string(const std::string& s) {
    if (s == "soft") return N4M_PP_WAVELET_THRESHOLD_SOFT;
    if (s == "hard") return N4M_PP_WAVELET_THRESHOLD_HARD;
    throw std::runtime_error("unknown threshold mode: " + s);
}

n4m_pp_wavelet_noise_t noise_from_string(const std::string& s) {
    if (s == "median") return N4M_PP_WAVELET_NOISE_MEDIAN;
    if (s == "std")    return N4M_PP_WAVELET_NOISE_STD;
    throw std::runtime_error("unknown noise estimator: " + s);
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

void test_wavelet_smoke() {
    double X[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    n4m_pp_wavelet_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_create(
        &h, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION) == N4M_OK);
    int64_t oc = -1;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_output_cols(h, 8, &oc) == N4M_OK);
    N4M_TEST_REQUIRE(oc == 8);  // 2 * ceil(8/2) = 8 (cA len + cD len)

    double Y[16] = {0};
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 8);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 8);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 16; ++i) N4M_TEST_REQUIRE(std::isfinite(Y[i]));

    /* Wrong output cols -> SHAPE_MISMATCH */
    double Ybad[8] = {0};
    n4m_matrix_view_t YBv = make_rowmajor_view(Ybad, 2, 4);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_transform(h, Xv, YBv) == N4M_ERR_SHAPE_MISMATCH);

    /* NULL out-pointer -> NULL_POINTER */
    N4M_TEST_REQUIRE(n4m_pp_wavelet_create(
        nullptr, N4M_PP_WAVELET_HAAR,
        N4M_PP_WAVELET_BOUNDARY_PERIODIZATION) == N4M_ERR_NULL_POINTER);

    n4m_pp_wavelet_destroy(h);
    n4m_pp_wavelet_destroy(nullptr);  // null-safe
}

void test_haar_smoke() {
    double X[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    n4m_pp_haar_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_haar_create(&h) == N4M_OK);

    int64_t oc = -1;
    N4M_TEST_REQUIRE(n4m_pp_haar_output_cols(8, &oc) == N4M_OK);
    N4M_TEST_REQUIRE(oc == 8);

    double Y[8] = {0};
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 8);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 8);
    N4M_TEST_REQUIRE(n4m_pp_haar_transform(h, Xv, Yv) == N4M_OK);
    /* cA[0..3] = (1+2)/sqrt(2), (3+4)/sqrt(2), ... */
    const double s2 = 1.41421356237309515;
    N4M_TEST_REQUIRE(std::fabs(Y[0] - 3.0 / s2) < 1e-12);
    N4M_TEST_REQUIRE(std::fabs(Y[1] - 7.0 / s2) < 1e-12);
    /* cD all -1/sqrt(2). */
    N4M_TEST_REQUIRE(std::fabs(Y[4] - (-1.0 / s2)) < 1e-12);
    n4m_pp_haar_destroy(h);
}

void test_wavelet_denoise_smoke() {
    /* 1 row of length 32, smooth signal + noise.  We just need to
     * verify the API works end-to-end. */
    double X[32];
    for (int i = 0; i < 32; ++i) X[i] = std::sin(0.3 * i) + ((i % 3) - 1) * 0.05;
    n4m_pp_wavelet_denoise_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_denoise_create(
        &h, N4M_PP_WAVELET_DB4, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*level=*/3, N4M_PP_WAVELET_THRESHOLD_SOFT,
        N4M_PP_WAVELET_NOISE_MEDIAN) == N4M_OK);
    double Y[32] = {0};
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 32);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 32);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_denoise_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 32; ++i) N4M_TEST_REQUIRE(std::isfinite(Y[i]));

    /* negative level -> INVALID_ARGUMENT */
    n4m_pp_wavelet_denoise_handle_t* hbad = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_denoise_create(
        &hbad, N4M_PP_WAVELET_DB4, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION, -1,
        N4M_PP_WAVELET_THRESHOLD_SOFT,
        N4M_PP_WAVELET_NOISE_MEDIAN) == N4M_ERR_INVALID_ARGUMENT);

    n4m_pp_wavelet_denoise_destroy(h);
}

void test_wavelet_features_smoke() {
    double X[32];
    for (int i = 0; i < 32; ++i) X[i] = std::cos(0.2 * i) + 0.1 * i;
    n4m_pp_wavelet_features_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_features_create(
        &h, N4M_PP_WAVELET_DB4, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*max_level=*/2) == N4M_OK);
    int64_t oc = -1;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_features_output_cols(h, 32, &oc) == N4M_OK);
    N4M_TEST_REQUIRE(oc == 4 * 3);  // (max_level=2 → 3 bands) * 4 stats

    std::vector<double> Y(static_cast<std::size_t>(oc), 0.0);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 32);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y.data(), 1, oc);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_features_transform(h, Xv, Yv) == N4M_OK);
    for (auto v : Y) N4M_TEST_REQUIRE(std::isfinite(v));
    n4m_pp_wavelet_features_destroy(h);
}

void test_wavelet_pca_smoke() {
    /* 4 samples in 8 features. */
    double X[32];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            X[i * 8 + j] = (i + 1) * (j + 1.0) + 0.01 * (i * j);
        }
    }
    n4m_pp_wavelet_pca_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_create(
        &h, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*max_level=*/1, /*n_components=*/2.0) == N4M_OK);

    int fitted = 1;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 0);

    int64_t kcols = -1;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_output_cols(h, &kcols) == N4M_ERR_NOT_FITTED);

    n4m_matrix_view_t Xv = make_rowmajor_view(X, 4, 8);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_fit(h, Xv) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 1);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_output_cols(h, &kcols) == N4M_OK);
    N4M_TEST_REQUIRE(kcols == 2);

    double Y[8] = {0};
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 4, 2);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 8; ++i) N4M_TEST_REQUIRE(std::isfinite(Y[i]));

    /* Invalid n_components <= 0 */
    n4m_pp_wavelet_pca_handle_t* hbad = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_create(
        &hbad, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION,
        1, 0.0) == N4M_ERR_INVALID_ARGUMENT);

    n4m_pp_wavelet_pca_destroy(h);
}

void test_wavelet_svd_smoke() {
    double X[32];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            X[i * 8 + j] = std::sin((i + 1) * 0.5 + j * 0.2);
        }
    }
    n4m_pp_wavelet_svd_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_create(
        &h, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*max_level=*/1, /*n_components=*/2.0) == N4M_OK);

    int fitted = 1;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 0);

    n4m_matrix_view_t Xv = make_rowmajor_view(X, 4, 8);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_fit(h, Xv) == N4M_OK);

    int64_t kcols = -1;
    N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_output_cols(h, &kcols) == N4M_OK);
    N4M_TEST_REQUIRE(kcols == 2);

    double Y[8] = {0};
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 4, 2);
    N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 8; ++i) N4M_TEST_REQUIRE(std::isfinite(Y[i]));

    n4m_pp_wavelet_svd_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests
// ---------------------------------------------------------------------------

void verify_wavelet_parity() {
    ParityFixture fx = load_fixture("wavelet_v1.json",
                                     /*require_per_case_shape=*/true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "haar");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");

        n4m_pp_wavelet_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_create(
            &h, family_from_string(fam), mode_from_string(md)) == N4M_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_wavelet_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "wavelet/" + c.name, 1e-10, 1e-11);
        n4m_pp_wavelet_destroy(h);
    }
}

void verify_haar_parity() {
    ParityFixture fx = load_fixture("haar_v1.json", true);
    for (const auto& c : fx.cases) {
        n4m_pp_haar_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_haar_create(&h) == N4M_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_haar_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "haar/" + c.name, 1e-10, 1e-11);
        n4m_pp_haar_destroy(h);
    }
}

void verify_wavelet_denoise_parity() {
    ParityFixture fx = load_fixture("wavelet_denoise_v1.json", true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "db4");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");
        const std::int64_t lvl = params_get_int(c.params_json, "level", 3);
        const std::string tm  = params_get_string(c.params_json, "threshold_mode", "soft");
        const std::string ne  = params_get_string(c.params_json, "noise_estimator", "median");

        n4m_pp_wavelet_denoise_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_denoise_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(lvl),
            threshold_from_string(tm),
            noise_from_string(ne)) == N4M_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_wavelet_denoise_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "wavelet_denoise/" + c.name, 1e-9, 1e-10);
        n4m_pp_wavelet_denoise_destroy(h);
    }
}

void verify_wavelet_features_parity() {
    ParityFixture fx = load_fixture("wavelet_features_v1.json", true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "db4");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");
        const std::int64_t mx = params_get_int(c.params_json, "max_level", 3);

        n4m_pp_wavelet_features_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_features_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(mx)) == N4M_OK);

        int64_t expected_oc = -1;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_features_output_cols(h, fx.cols, &expected_oc) == N4M_OK);
        N4M_TEST_REQUIRE(expected_oc == c.output_cols);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_wavelet_features_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "wavelet_features/" + c.name, 1e-12, 1e-13);
        n4m_pp_wavelet_features_destroy(h);
    }
}

void verify_wavelet_pca_parity() {
    ParityFixture fx = load_fixture("wavelet_pca_v1.json", true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "db4");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");
        const std::int64_t mx = params_get_int(c.params_json, "max_level", 3);
        const double nc = params_get_double(c.params_json, "n_components", 3.0);

        n4m_pp_wavelet_pca_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(mx), nc) == N4M_OK);

        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_fit(h, Xfv) == N4M_OK);

        int64_t k_learned = -1;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_output_cols(h, &k_learned) == N4M_OK);
        N4M_TEST_REQUIRE(k_learned == c.output_cols);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_wavelet_pca_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "wavelet_pca/" + c.name, 1e-10, 1e-11);
        n4m_pp_wavelet_pca_destroy(h);
    }
}

void verify_wavelet_svd_parity() {
    ParityFixture fx = load_fixture("wavelet_svd_v1.json", true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "db4");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");
        const std::int64_t mx = params_get_int(c.params_json, "max_level", 3);
        const double nc = params_get_double(c.params_json, "n_components", 3.0);

        n4m_pp_wavelet_svd_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(mx), nc) == N4M_OK);

        std::vector<double> fit_in = fx.fit_input;
        n4m_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_fit(h, Xfv) == N4M_OK);

        int64_t k_learned = -1;
        N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_output_cols(h, &k_learned) == N4M_OK);
        N4M_TEST_REQUIRE(k_learned == c.output_cols);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        N4M_TEST_REQUIRE(n4m_pp_wavelet_svd_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "wavelet_svd/" + c.name, 1e-10, 1e-11);
        n4m_pp_wavelet_svd_destroy(h);
    }
}

}  // namespace

void register_preprocessing_wavelets_tests(n4m_testing::Runner& r);
void register_preprocessing_wavelets_tests(n4m_testing::Runner& r) {
    r.run("pp_wavelet_smoke",            test_wavelet_smoke);
    r.run("pp_wavelet_parity",           verify_wavelet_parity);
    r.run("pp_haar_smoke",               test_haar_smoke);
    r.run("pp_haar_parity",              verify_haar_parity);
    r.run("pp_wavelet_denoise_smoke",    test_wavelet_denoise_smoke);
    r.run("pp_wavelet_denoise_parity",   verify_wavelet_denoise_parity);
    r.run("pp_wavelet_features_smoke",   test_wavelet_features_smoke);
    r.run("pp_wavelet_features_parity",  verify_wavelet_features_parity);
    r.run("pp_wavelet_pca_smoke",        test_wavelet_pca_smoke);
    r.run("pp_wavelet_pca_parity",       verify_wavelet_pca_parity);
    r.run("pp_wavelet_svd_smoke",        test_wavelet_svd_smoke);
    r.run("pp_wavelet_svd_parity",       verify_wavelet_svd_parity);
}
