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

#include "chemometrics4all/c4a.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined"
#endif

// Phase 6 ABI declarations live in c4a.h §23 (consumed via the
// `#include "chemometrics4all/c4a.h"` above).

namespace {

using ParityFixture = ::c4a_testing::Fixture;
using ParityCase    = ::c4a_testing::Case;
using ::c4a_testing::params_get_double;
using ::c4a_testing::params_get_int;
using ::c4a_testing::params_get_string;

ParityFixture load_fixture(const std::string& filename,
                            bool require_per_case_shape) {
    return ::c4a_testing::load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename,
        require_per_case_shape);
}

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

c4a_pp_wavelet_family_t family_from_string(const std::string& s) {
    if (s == "haar")  return C4A_PP_WAVELET_HAAR;
    if (s == "db4")   return C4A_PP_WAVELET_DB4;
    if (s == "sym4")  return C4A_PP_WAVELET_SYM4;
    if (s == "coif1") return C4A_PP_WAVELET_COIF1;
    throw std::runtime_error("unknown wavelet family: " + s);
}

c4a_pp_wavelet_boundary_t mode_from_string(const std::string& s) {
    if (s == "periodization") return C4A_PP_WAVELET_BOUNDARY_PERIODIZATION;
    if (s == "symmetric")     return C4A_PP_WAVELET_BOUNDARY_SYMMETRIC;
    if (s == "zero")          return C4A_PP_WAVELET_BOUNDARY_ZERO;
    throw std::runtime_error("unknown boundary mode: " + s);
}

c4a_pp_wavelet_threshold_t threshold_from_string(const std::string& s) {
    if (s == "soft") return C4A_PP_WAVELET_THRESHOLD_SOFT;
    if (s == "hard") return C4A_PP_WAVELET_THRESHOLD_HARD;
    throw std::runtime_error("unknown threshold mode: " + s);
}

c4a_pp_wavelet_noise_t noise_from_string(const std::string& s) {
    if (s == "median") return C4A_PP_WAVELET_NOISE_MEDIAN;
    if (s == "std")    return C4A_PP_WAVELET_NOISE_STD;
    throw std::runtime_error("unknown noise estimator: " + s);
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

void test_wavelet_smoke() {
    double X[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    c4a_pp_wavelet_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_create(
        &h, C4A_PP_WAVELET_HAAR, C4A_PP_WAVELET_BOUNDARY_PERIODIZATION) == C4A_OK);
    int64_t oc = -1;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_output_cols(h, 8, &oc) == C4A_OK);
    C4A_TEST_REQUIRE(oc == 8);  // 2 * ceil(8/2) = 8 (cA len + cD len)

    double Y[16] = {0};
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 8);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 8);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_transform(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 16; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));

    /* Wrong output cols -> SHAPE_MISMATCH */
    double Ybad[8] = {0};
    c4a_matrix_view_t YBv = make_rowmajor_view(Ybad, 2, 4);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_transform(h, Xv, YBv) == C4A_ERR_SHAPE_MISMATCH);

    /* NULL out-pointer -> NULL_POINTER */
    C4A_TEST_REQUIRE(c4a_pp_wavelet_create(
        nullptr, C4A_PP_WAVELET_HAAR,
        C4A_PP_WAVELET_BOUNDARY_PERIODIZATION) == C4A_ERR_NULL_POINTER);

    c4a_pp_wavelet_destroy(h);
    c4a_pp_wavelet_destroy(nullptr);  // null-safe
}

void test_haar_smoke() {
    double X[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    c4a_pp_haar_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_haar_create(&h) == C4A_OK);

    int64_t oc = -1;
    C4A_TEST_REQUIRE(c4a_pp_haar_output_cols(8, &oc) == C4A_OK);
    C4A_TEST_REQUIRE(oc == 8);

    double Y[8] = {0};
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 8);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 8);
    C4A_TEST_REQUIRE(c4a_pp_haar_transform(h, Xv, Yv) == C4A_OK);
    /* cA[0..3] = (1+2)/sqrt(2), (3+4)/sqrt(2), ... */
    const double s2 = 1.41421356237309515;
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 3.0 / s2) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[1] - 7.0 / s2) < 1e-12);
    /* cD all -1/sqrt(2). */
    C4A_TEST_REQUIRE(std::fabs(Y[4] - (-1.0 / s2)) < 1e-12);
    c4a_pp_haar_destroy(h);
}

void test_wavelet_denoise_smoke() {
    /* 1 row of length 32, smooth signal + noise.  We just need to
     * verify the API works end-to-end. */
    double X[32];
    for (int i = 0; i < 32; ++i) X[i] = std::sin(0.3 * i) + ((i % 3) - 1) * 0.05;
    c4a_pp_wavelet_denoise_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_denoise_create(
        &h, C4A_PP_WAVELET_DB4, C4A_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*level=*/3, C4A_PP_WAVELET_THRESHOLD_SOFT,
        C4A_PP_WAVELET_NOISE_MEDIAN) == C4A_OK);
    double Y[32] = {0};
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 32);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 32);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_denoise_transform(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 32; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));

    /* negative level -> INVALID_ARGUMENT */
    c4a_pp_wavelet_denoise_handle_t* hbad = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_denoise_create(
        &hbad, C4A_PP_WAVELET_DB4, C4A_PP_WAVELET_BOUNDARY_PERIODIZATION, -1,
        C4A_PP_WAVELET_THRESHOLD_SOFT,
        C4A_PP_WAVELET_NOISE_MEDIAN) == C4A_ERR_INVALID_ARGUMENT);

    c4a_pp_wavelet_denoise_destroy(h);
}

void test_wavelet_features_smoke() {
    double X[32];
    for (int i = 0; i < 32; ++i) X[i] = std::cos(0.2 * i) + 0.1 * i;
    c4a_pp_wavelet_features_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_features_create(
        &h, C4A_PP_WAVELET_DB4, C4A_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*max_level=*/2) == C4A_OK);
    int64_t oc = -1;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_features_output_cols(h, 32, &oc) == C4A_OK);
    C4A_TEST_REQUIRE(oc == 4 * 3);  // (max_level=2 → 3 bands) * 4 stats

    std::vector<double> Y(static_cast<std::size_t>(oc), 0.0);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 32);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y.data(), 1, oc);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_features_transform(h, Xv, Yv) == C4A_OK);
    for (auto v : Y) C4A_TEST_REQUIRE(std::isfinite(v));
    c4a_pp_wavelet_features_destroy(h);
}

void test_wavelet_pca_smoke() {
    /* 4 samples in 8 features. */
    double X[32];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            X[i * 8 + j] = (i + 1) * (j + 1.0) + 0.01 * (i * j);
        }
    }
    c4a_pp_wavelet_pca_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_create(
        &h, C4A_PP_WAVELET_HAAR, C4A_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*max_level=*/1, /*n_components=*/2.0) == C4A_OK);

    int fitted = 1;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 0);

    int64_t kcols = -1;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_output_cols(h, &kcols) == C4A_ERR_NOT_FITTED);

    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 8);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_fit(h, Xv) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 1);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_output_cols(h, &kcols) == C4A_OK);
    C4A_TEST_REQUIRE(kcols == 2);

    double Y[8] = {0};
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 2);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_transform(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 8; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));

    /* Invalid n_components <= 0 */
    c4a_pp_wavelet_pca_handle_t* hbad = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_create(
        &hbad, C4A_PP_WAVELET_HAAR, C4A_PP_WAVELET_BOUNDARY_PERIODIZATION,
        1, 0.0) == C4A_ERR_INVALID_ARGUMENT);

    c4a_pp_wavelet_pca_destroy(h);
}

void test_wavelet_svd_smoke() {
    double X[32];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            X[i * 8 + j] = std::sin((i + 1) * 0.5 + j * 0.2);
        }
    }
    c4a_pp_wavelet_svd_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_create(
        &h, C4A_PP_WAVELET_HAAR, C4A_PP_WAVELET_BOUNDARY_PERIODIZATION,
        /*max_level=*/1, /*n_components=*/2.0) == C4A_OK);

    int fitted = 1;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 0);

    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 8);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_fit(h, Xv) == C4A_OK);

    int64_t kcols = -1;
    C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_output_cols(h, &kcols) == C4A_OK);
    C4A_TEST_REQUIRE(kcols == 2);

    double Y[8] = {0};
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 2);
    C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_transform(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 8; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));

    c4a_pp_wavelet_svd_destroy(h);
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

        c4a_pp_wavelet_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_create(
            &h, family_from_string(fam), mode_from_string(md)) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_wavelet_transform(h, Xv, Yv) == C4A_OK);
        ::c4a_testing::assert_close(out, c.expected_output,
                                     "wavelet/" + c.name, 1e-10, 1e-11);
        c4a_pp_wavelet_destroy(h);
    }
}

void verify_haar_parity() {
    ParityFixture fx = load_fixture("haar_v1.json", true);
    for (const auto& c : fx.cases) {
        c4a_pp_haar_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_haar_create(&h) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_haar_transform(h, Xv, Yv) == C4A_OK);
        ::c4a_testing::assert_close(out, c.expected_output,
                                     "haar/" + c.name, 1e-10, 1e-11);
        c4a_pp_haar_destroy(h);
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

        c4a_pp_wavelet_denoise_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_denoise_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(lvl),
            threshold_from_string(tm),
            noise_from_string(ne)) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_wavelet_denoise_transform(h, Xv, Yv) == C4A_OK);
        ::c4a_testing::assert_close(out, c.expected_output,
                                     "wavelet_denoise/" + c.name, 1e-9, 1e-10);
        c4a_pp_wavelet_denoise_destroy(h);
    }
}

void verify_wavelet_features_parity() {
    ParityFixture fx = load_fixture("wavelet_features_v1.json", true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "db4");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");
        const std::int64_t mx = params_get_int(c.params_json, "max_level", 3);

        c4a_pp_wavelet_features_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_features_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(mx)) == C4A_OK);

        int64_t expected_oc = -1;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_features_output_cols(h, fx.cols, &expected_oc) == C4A_OK);
        C4A_TEST_REQUIRE(expected_oc == c.output_cols);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_wavelet_features_transform(h, Xv, Yv) == C4A_OK);
        ::c4a_testing::assert_close(out, c.expected_output,
                                     "wavelet_features/" + c.name, 1e-12, 1e-13);
        c4a_pp_wavelet_features_destroy(h);
    }
}

void verify_wavelet_pca_parity() {
    ParityFixture fx = load_fixture("wavelet_pca_v1.json", true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "db4");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");
        const std::int64_t mx = params_get_int(c.params_json, "max_level", 3);
        const double nc = params_get_double(c.params_json, "n_components", 3.0);

        c4a_pp_wavelet_pca_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(mx), nc) == C4A_OK);

        std::vector<double> fit_in = fx.fit_input;
        c4a_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_fit(h, Xfv) == C4A_OK);

        int64_t k_learned = -1;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_output_cols(h, &k_learned) == C4A_OK);
        C4A_TEST_REQUIRE(k_learned == c.output_cols);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_wavelet_pca_transform(h, Xv, Yv) == C4A_OK);
        ::c4a_testing::assert_close(out, c.expected_output,
                                     "wavelet_pca/" + c.name, 1e-10, 1e-11);
        c4a_pp_wavelet_pca_destroy(h);
    }
}

void verify_wavelet_svd_parity() {
    ParityFixture fx = load_fixture("wavelet_svd_v1.json", true);
    for (const auto& c : fx.cases) {
        const std::string fam = params_get_string(c.params_json, "family", "db4");
        const std::string md  = params_get_string(c.params_json, "mode", "periodization");
        const std::int64_t mx = params_get_int(c.params_json, "max_level", 3);
        const double nc = params_get_double(c.params_json, "n_components", 3.0);

        c4a_pp_wavelet_svd_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_create(
            &h, family_from_string(fam), mode_from_string(md),
            static_cast<int32_t>(mx), nc) == C4A_OK);

        std::vector<double> fit_in = fx.fit_input;
        c4a_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_fit(h, Xfv) == C4A_OK);

        int64_t k_learned = -1;
        C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_output_cols(h, &k_learned) == C4A_OK);
        C4A_TEST_REQUIRE(k_learned == c.output_cols);

        std::vector<double> in = fx.input;
        std::vector<double> out(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_wavelet_svd_transform(h, Xv, Yv) == C4A_OK);
        ::c4a_testing::assert_close(out, c.expected_output,
                                     "wavelet_svd/" + c.name, 1e-10, 1e-11);
        c4a_pp_wavelet_svd_destroy(h);
    }
}

}  // namespace

void register_preprocessing_wavelets_tests(c4a_testing::Runner& r);
void register_preprocessing_wavelets_tests(c4a_testing::Runner& r) {
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
