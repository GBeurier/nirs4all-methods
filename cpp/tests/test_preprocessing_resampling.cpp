// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 10 resampling, cropping, and target
// discretization operators:
//   - Resampler              (stateful)
//   - CropTransformer        (stateless)
//   - ResampleTransformer    (stateless)
//   - IntegerKBinsDiscretizer (stateful, int32 output)
//   - RangeDiscretizer       (stateless, int32 output)
//
// Each operator contributes:
//   * a smoke test exercising the public ABI lifecycle on a small inline
//     matrix;
//   * a parity test loading a JSON fixture and comparing against the
//     reference output;
//   * a NOT_FITTED test (stateful ops only).
//
// Total contributions: 5 smoke + 5 parity + 2 not-fitted = 12 tests.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "chemometrics4all/c4a.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

using ParityFixture = ::c4a_testing::Fixture;
using ParityCase    = ::c4a_testing::Case;

ParityFixture load_fixture(const std::string& filename) {
    return ::c4a_testing::load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/true);
}

using ::c4a_testing::params_get_bool;
using ::c4a_testing::params_get_double;
using ::c4a_testing::params_get_int;
using ::c4a_testing::params_get_string;

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

c4a_matrix_view_t make_rowmajor_view_i32(std::int32_t* data, std::int64_t rows,
                                          std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_I32);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

void assert_close(const std::vector<double>& got,
                  const std::vector<double>& want,
                  const std::string& tag,
                  double abs_tol = 1e-12,
                  double rel_tol = 1e-13) {
    ::c4a_testing::assert_close(got, want, tag, abs_tol, rel_tol);
}

void assert_int_equal(const std::vector<std::int32_t>& got,
                       const std::vector<double>& want,
                       const std::string& tag) {
    if (got.size() != want.size()) {
        throw std::runtime_error(tag + " size mismatch");
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const std::int32_t g = got[i];
        const std::int32_t w = static_cast<std::int32_t>(want[i]);
        if (g != w) {
            throw std::runtime_error(
                tag + " mismatch at i=" + std::to_string(i) +
                " got=" + std::to_string(g) +
                " want=" + std::to_string(w));
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

void test_resampler_smoke() {
    // Source axis: 5 evenly spaced wavelengths. Target axis: 3 evenly spaced.
    const double src_wl[5] = {1000.0, 1100.0, 1200.0, 1300.0, 1400.0};
    const double tgt_wl[3] = {1050.0, 1200.0, 1350.0};
    double X[10] = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        2.0, 4.0, 6.0, 8.0, 10.0,
    };
    double Y[6] = {0};
    c4a_pp_resampler_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_resampler_create(&h, tgt_wl, 3,
                                              /*method=linear*/0,
                                              0.0, 0.0, /*use_crop=*/0,
                                              0.0, /*bounds_error=*/0,
                                              /*extrapolate=*/0) == C4A_OK);
    int fitted = 1;
    C4A_TEST_REQUIRE(c4a_pp_resampler_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 0);
    C4A_TEST_REQUIRE(c4a_pp_resampler_fit(h, src_wl, 5) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_resampler_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 1);
    C4A_TEST_REQUIRE(c4a_pp_resampler_output_cols(h) == 3);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 5);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_resampler_transform(h, Xv, Yv) == C4A_OK);
    // Linear interp at exact src points: tgt=1200 falls on src[2] -> Y == X[2]
    // Y[0..2] = [X(1050), X(1200), X(1350)] = [(1+2)/2=1.5, 3, (4+5)/2=4.5]
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 1.5) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[1] - 3.0) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[2] - 4.5) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 3.0) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[4] - 6.0) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[5] - 9.0) < 1e-12);
    c4a_pp_resampler_destroy(h);
    c4a_pp_resampler_destroy(nullptr);
}

void test_crop_smoke() {
    double X[12] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    };
    double Y[6] = {0};
    c4a_pp_crop_handle_t* h = nullptr;
    // crop [start=1, end=4) -> cols 1, 2, 3
    C4A_TEST_REQUIRE(c4a_pp_crop_create(&h, 1, 4) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_crop_output_cols(h, 6) == 3);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 6);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_crop_transform(h, Xv, Yv) == C4A_OK);
    C4A_TEST_REQUIRE(Y[0] == 2.0 && Y[1] == 3.0 && Y[2] == 4.0);
    C4A_TEST_REQUIRE(Y[3] == 8.0 && Y[4] == 9.0 && Y[5] == 10.0);
    c4a_pp_crop_destroy(h);

    // end == -1 (to end of row)
    C4A_TEST_REQUIRE(c4a_pp_crop_create(&h, 4, -1) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_crop_output_cols(h, 6) == 2);
    double Z[4] = {0};
    c4a_matrix_view_t Zv = make_rowmajor_view(Z, 2, 2);
    C4A_TEST_REQUIRE(c4a_pp_crop_transform(h, Xv, Zv) == C4A_OK);
    C4A_TEST_REQUIRE(Z[0] == 5.0 && Z[1] == 6.0 && Z[2] == 11.0 && Z[3] == 12.0);
    c4a_pp_crop_destroy(h);
}

void test_resample_smoke() {
    // Resample 5 -> 3 with linear interp.
    double X[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double Y[3] = {0};
    c4a_pp_resample_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_resample_create(&h, 3) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_resample_output_cols(h, 5) == 3);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 5);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 3);
    C4A_TEST_REQUIRE(c4a_pp_resample_transform(h, Xv, Yv) == C4A_OK);
    // src axis = [0, 0.25, 0.5, 0.75, 1.0], dst axis = [0, 0.5, 1.0]
    // interp(0.0) = X[0] = 1, interp(0.5) = X[2] = 3, interp(1.0) = X[4] = 5
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 1.0) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[1] - 3.0) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[2] - 5.0) < 1e-12);
    c4a_pp_resample_destroy(h);

    // num_samples == -1 (identity)
    C4A_TEST_REQUIRE(c4a_pp_resample_create(&h, -1) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_resample_output_cols(h, 7) == 7);
    c4a_pp_resample_destroy(h);
}

void test_kbins_disc_smoke() {
    // Uniform discretization: 3 bins, 1 column.
    double Xfit[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double Xt[5]    = {1.0, 4.0, 5.0, 7.0, 10.0};
    std::int32_t Y[5] = {-1, -1, -1, -1, -1};
    c4a_pp_kbins_disc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_kbins_disc_create(&h, /*n_bins=*/3,
                                                /*strategy=uniform*/0) == C4A_OK);
    int fitted = 1;
    C4A_TEST_REQUIRE(c4a_pp_kbins_disc_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 0);
    c4a_matrix_view_t Xfv = make_rowmajor_view(Xfit, 10, 1);
    C4A_TEST_REQUIRE(c4a_pp_kbins_disc_fit(h, Xfv) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_kbins_disc_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 1);
    c4a_matrix_view_t Xv = make_rowmajor_view(Xt, 5, 1);
    c4a_matrix_view_t Yv = make_rowmajor_view_i32(Y, 5, 1);
    C4A_TEST_REQUIRE(c4a_pp_kbins_disc_transform(h, Xv, Yv) == C4A_OK);
    // Uniform edges on [1, 10]: linspace(1, 10, 4) = [1, 4, 7, 10]
    // Inner edges = [4, 7]
    // searchsorted([4,7], val, 'right'):
    //   1 -> 0, 4 -> 1, 5 -> 1, 7 -> 2, 10 -> 2
    C4A_TEST_REQUIRE(Y[0] == 0);
    C4A_TEST_REQUIRE(Y[1] == 1);
    C4A_TEST_REQUIRE(Y[2] == 1);
    C4A_TEST_REQUIRE(Y[3] == 2);
    C4A_TEST_REQUIRE(Y[4] == 2);
    c4a_pp_kbins_disc_destroy(h);
}

void test_range_disc_smoke() {
    // Bins = [1, 2, 3]
    // X = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    // digitize: [0, 1, 1, 2, 2, 3, 3]
    const double bins[3] = {1.0, 2.0, 3.0};
    double X[7] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    std::int32_t Y[7] = {-1, -1, -1, -1, -1, -1, -1};
    c4a_pp_range_disc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_range_disc_create(&h, bins, 3) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 7, 1);
    c4a_matrix_view_t Yv = make_rowmajor_view_i32(Y, 7, 1);
    C4A_TEST_REQUIRE(c4a_pp_range_disc_transform(h, Xv, Yv) == C4A_OK);
    const std::int32_t expected[7] = {0, 1, 1, 2, 2, 3, 3};
    for (int i = 0; i < 7; ++i) {
        C4A_TEST_REQUIRE(Y[i] == expected[i]);
    }
    c4a_pp_range_disc_destroy(h);
}

// ---------------------------------------------------------------------------
// NOT_FITTED tests
// ---------------------------------------------------------------------------

void test_resampler_not_fitted() {
    const double tgt_wl[3] = {1050.0, 1200.0, 1350.0};
    double X[10] = {1.0, 2.0, 3.0, 4.0, 5.0,
                    2.0, 4.0, 6.0, 8.0, 10.0};
    double Y[6] = {0};
    c4a_pp_resampler_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_resampler_create(&h, tgt_wl, 3, 0,
                                              0.0, 0.0, 0,
                                              0.0, 0, 0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 5);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_resampler_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    c4a_pp_resampler_destroy(h);
}

void test_kbins_disc_not_fitted() {
    double X[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::int32_t Y[5] = {0, 0, 0, 0, 0};
    c4a_pp_kbins_disc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_kbins_disc_create(&h, 3, 0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 5, 1);
    c4a_matrix_view_t Yv = make_rowmajor_view_i32(Y, 5, 1);
    C4A_TEST_REQUIRE(c4a_pp_kbins_disc_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    c4a_pp_kbins_disc_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests (one per operator)
// ---------------------------------------------------------------------------

void verify_resampler_parity() {
    ParityFixture fx = load_fixture("resampler_v1.json");
    for (const auto& c : fx.cases) {
        const int method = static_cast<int>(params_get_int(c.params_json,
                                                            "method", 0));
        const int n_target = static_cast<int>(params_get_int(c.params_json,
                                                              "n_target", 0));
        const int use_crop = static_cast<int>(params_get_int(c.params_json,
                                                              "use_crop", 0));
        const double crop_min = params_get_double(c.params_json, "crop_min", 0.0);
        const double crop_max = params_get_double(c.params_json, "crop_max", 0.0);
        const double fill_value = params_get_double(c.params_json,
                                                     "fill_value", 0.0);
        const int bounds_error = static_cast<int>(params_get_int(c.params_json,
                                                                  "bounds_error",
                                                                  0));
        const int extrapolate = static_cast<int>(params_get_int(c.params_json,
                                                                  "extrapolate",
                                                                  0));

        // Each fixture case carries its own per-case target wavelengths and
        // source wavelengths in the params block. We decode them from the
        // params_json string using a small helper here because the shared
        // fixture parser doesn't decode hex arrays nested in params.
        // For simplicity we parse them from explicit min/step/count fields.
        const double src_min  = params_get_double(c.params_json, "src_min", 0.0);
        const double src_step = params_get_double(c.params_json, "src_step", 1.0);
        const int    src_n    = static_cast<int>(params_get_int(c.params_json,
                                                                 "src_n", 0));
        const double tgt_min  = params_get_double(c.params_json, "tgt_min", 0.0);
        const double tgt_step = params_get_double(c.params_json, "tgt_step", 1.0);
        const int    tgt_n    = static_cast<int>(params_get_int(c.params_json,
                                                                 "tgt_n", 0));
        std::vector<double> src_wl(static_cast<std::size_t>(src_n));
        for (std::size_t i = 0; i < static_cast<std::size_t>(src_n); ++i) {
            src_wl[i] = src_min + src_step * static_cast<double>(i);
        }
        std::vector<double> tgt_wl(static_cast<std::size_t>(tgt_n));
        for (std::size_t i = 0; i < static_cast<std::size_t>(tgt_n); ++i) {
            tgt_wl[i] = tgt_min + tgt_step * static_cast<double>(i);
        }
        (void)n_target;

        c4a_pp_resampler_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_resampler_create(&h,
            tgt_wl.empty() ? nullptr : tgt_wl.data(),
            tgt_wl.empty() ? -1 : static_cast<int64_t>(tgt_wl.size()),
            method, crop_min, crop_max, use_crop,
            fill_value, bounds_error, extrapolate) == C4A_OK);
        C4A_TEST_REQUIRE(c4a_pp_resampler_fit(h, src_wl.data(),
                                              static_cast<int64_t>(src_wl.size()))
                          == C4A_OK);

        const std::int64_t expected_out_cols = c4a_pp_resampler_output_cols(h);
        C4A_TEST_REQUIRE(expected_out_cols == c.output_cols);
        std::vector<double> in = fx.input;
        std::vector<double> outv(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(outv.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_resampler_transform(h, Xv, Yv) == C4A_OK);
        // Linear/nearest interpolation: tighter tolerance.
        // Cubic spline: slightly looser tolerance (per the Phase 10 brief).
        const double abs_tol = (method == 2) ? 1e-10 : 1e-12;
        const double rel_tol = (method == 2) ? 1e-11 : 1e-13;
        assert_close(outv, c.expected_output, "resampler/" + c.name,
                     abs_tol, rel_tol);
        c4a_pp_resampler_destroy(h);
    }
}

void verify_crop_parity() {
    ParityFixture fx = load_fixture("crop_v1.json");
    for (const auto& c : fx.cases) {
        const std::int64_t start =
            params_get_int(c.params_json, "start", 0);
        const std::int64_t end =
            params_get_int(c.params_json, "end", -1);
        c4a_pp_crop_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_crop_create(&h, start, end) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<double> outv(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(outv.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_crop_transform(h, Xv, Yv) == C4A_OK);
        // Crop is pure memcpy — byte exact.
        assert_close(outv, c.expected_output, "crop/" + c.name,
                     0.0, 0.0);
        c4a_pp_crop_destroy(h);
    }
}

void verify_resample_parity() {
    ParityFixture fx = load_fixture("resample_v1.json");
    for (const auto& c : fx.cases) {
        const std::int64_t num_samples =
            params_get_int(c.params_json, "num_samples", -1);
        c4a_pp_resample_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_resample_create(&h, num_samples) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<double> outv(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(outv.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_resample_transform(h, Xv, Yv) == C4A_OK);
        assert_close(outv, c.expected_output, "resample/" + c.name,
                     1e-12, 1e-13);
        c4a_pp_resample_destroy(h);
    }
}

void verify_kbins_disc_parity() {
    ParityFixture fx = load_fixture("kbins_disc_v1.json");
    for (const auto& c : fx.cases) {
        const int n_bins   = static_cast<int>(params_get_int(c.params_json,
                                                              "n_bins", 5));
        const int strategy = static_cast<int>(params_get_int(c.params_json,
                                                              "strategy", 0));
        c4a_pp_kbins_disc_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_kbins_disc_create(&h, n_bins, strategy)
                          == C4A_OK);
        std::vector<double> fit_in = fx.fit_input;
        c4a_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        C4A_TEST_REQUIRE(c4a_pp_kbins_disc_fit(h, Xfv) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<std::int32_t> outv(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view_i32(outv.data(),
                                                       c.output_rows,
                                                       c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_kbins_disc_transform(h, Xv, Yv) == C4A_OK);
        // Integer parity (no tolerance).
        assert_int_equal(outv, c.expected_output, "kbins_disc/" + c.name);
        c4a_pp_kbins_disc_destroy(h);
    }
}

void verify_range_disc_parity() {
    ParityFixture fx = load_fixture("range_disc_v1.json");
    for (const auto& c : fx.cases) {
        // Edges are encoded as a comma-separated string in params for
        // simplicity (zero-dependency JSON parser doesn't decode numeric
        // arrays in nested objects). Decode manually.
        const std::string edges_str = params_get_string(c.params_json,
                                                         "edges_csv", "");
        std::vector<double> edges;
        {
            std::size_t pos = 0;
            while (pos < edges_str.size()) {
                char* tail = nullptr;
                const double v = std::strtod(edges_str.c_str() + pos, &tail);
                if (tail == edges_str.c_str() + pos) break;
                edges.push_back(v);
                pos = static_cast<std::size_t>(tail - edges_str.c_str());
                while (pos < edges_str.size() && edges_str[pos] == ',') ++pos;
            }
        }
        c4a_pp_range_disc_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_range_disc_create(&h,
            edges.empty() ? nullptr : edges.data(),
            static_cast<int64_t>(edges.size())) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<std::int32_t> outv(
            static_cast<std::size_t>(c.output_rows * c.output_cols), 0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view_i32(outv.data(),
                                                       c.output_rows,
                                                       c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_range_disc_transform(h, Xv, Yv) == C4A_OK);
        assert_int_equal(outv, c.expected_output, "range_disc/" + c.name);
        c4a_pp_range_disc_destroy(h);
    }
}

}  // namespace

void register_preprocessing_resampling_tests(c4a_testing::Runner& r);
void register_preprocessing_resampling_tests(c4a_testing::Runner& r) {
    r.run("pp_resampler_smoke",            test_resampler_smoke);
    r.run("pp_resampler_not_fitted",       test_resampler_not_fitted);
    r.run("pp_resampler_parity",           verify_resampler_parity);
    r.run("pp_crop_smoke",                 test_crop_smoke);
    r.run("pp_crop_parity",                verify_crop_parity);
    r.run("pp_resample_smoke",             test_resample_smoke);
    r.run("pp_resample_parity",            verify_resample_parity);
    r.run("pp_kbins_disc_smoke",           test_kbins_disc_smoke);
    r.run("pp_kbins_disc_not_fitted",      test_kbins_disc_not_fitted);
    r.run("pp_kbins_disc_parity",          verify_kbins_disc_parity);
    r.run("pp_range_disc_smoke",           test_range_disc_smoke);
    r.run("pp_range_disc_parity",          verify_range_disc_parity);
}
