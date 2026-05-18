// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 2 stateless preprocessing operators (SNV,
// LocalSNV, RobustSNV, AreaNormalization, Normalize, SimpleScale,
// LogTransform). For each operator the test loads a JSON fixture produced
// by parity/python_generator/scripts/generate_phase2_fixtures.py, decodes
// the input/output matrices from big-endian hex doubles, applies the C
// engine through its public ABI, and asserts byte-equality (or 1e-12 abs /
// 1e-13 rel tolerance) against the reference output.
//
// Each operator contributes two tests:
//
//   * a smoke test exercising create/transform/destroy on a small inline
//     matrix to validate the public ABI surface and lifecycle semantics;
//   * a parity test sweeping every case in the fixture and asserting the
//     output matches the reference within tolerance.

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
        /*require_per_case_output_shape=*/false);
}

using ::c4a_testing::params_get_bool;
using ::c4a_testing::params_get_double;
using ::c4a_testing::params_get_int;
using ::c4a_testing::params_get_string;

void assert_close(const std::vector<double>& got,
                  const std::vector<double>& want,
                  const std::string& tag) {
    ::c4a_testing::assert_close(got, want, tag, /*abs_tol=*/1e-12,
                                 /*rel_tol=*/1e-13);
}

// ---------------------------------------------------------------------------
// Smoke tests — exercise the create/transform/destroy lifecycle on a small
// hand-crafted matrix for each operator.
// ---------------------------------------------------------------------------

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

void test_snv_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_snv_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_snv_create(&h, /*with_mean=*/1, /*with_std=*/1,
                                         /*ddof=*/0) == C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_snv_transform(h, Xv, Yv) == C4A_OK);
    // Row 0: mean = 2, std = sqrt(2/3); expected SNV ≈ [-1.224..., 0, 1.224...]
    C4A_TEST_REQUIRE(std::fabs(Y[1] - 0.0) < 1e-12);
    C4A_TEST_REQUIRE(Y[0] < 0.0 && Y[2] > 0.0);
    c4a_pp_snv_destroy(h);
    c4a_pp_snv_destroy(nullptr);  // null-safe
}

void test_lsnv_smoke() {
    double X[10] = {1.0, 1.0, 2.0, 5.0, 3.0, 2.0, 4.0, 6.0, 5.0, 3.0};
    double Y[10] = {0};
    c4a_pp_lsnv_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_lsnv_create(&h, /*window=*/3, /*pad_mode=*/0,
                                          /*constant_value=*/0.0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    C4A_TEST_REQUIRE(c4a_pp_lsnv_transform(h, Xv, Yv) == C4A_OK);
    c4a_pp_lsnv_destroy(h);
}

void test_rnv_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_rnv_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_rnv_create(&h, /*center=*/1, /*scale=*/1,
                                        /*k=*/1.4826) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_rnv_transform(h, Xv, Yv) == C4A_OK);
    c4a_pp_rnv_destroy(h);
}

void test_area_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_area_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_area_create(&h, /*sum=*/0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_area_transform(h, Xv, Yv) == C4A_OK);
    // Row 0 area = 6 → [1/6, 2/6, 3/6]; row 1 area = 15 → [4/15, 5/15, 6/15].
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 1.0 / 6.0) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[5] - 6.0 / 15.0) < 1e-15);
    c4a_pp_area_destroy(h);
}

void test_normalize_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_normalize_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_normalize_create(&h, /*min=*/-1.0, /*max=*/1.0)
                     == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_normalize_transform(h, Xv, Yv) == C4A_OK);
    c4a_pp_normalize_destroy(h);
}

void test_simple_scale_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_simple_scale_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_simple_scale_create(&h) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_simple_scale_transform(h, Xv, Yv) == C4A_OK);
    // For axis=0 min-max: each column scales [min, max] → [0, 1].
    // Col 0: min=1, max=4 → Y[0]=(1-1)/3=0, Y[3]=(4-1)/3=1.
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 0.0) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 1.0) < 1e-15);
    c4a_pp_simple_scale_destroy(h);
}

void test_log_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_log_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_log_create(&h, /*base=*/0.0, /*offset=*/0.0,
                                        /*auto_offset=*/1, /*min_value=*/1e-8)
                     == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_log_transform(h, Xv, Yv) == C4A_OK);
    C4A_TEST_REQUIRE(std::fabs(Y[0] - std::log(1.0)) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[5] - std::log(6.0)) < 1e-15);
    c4a_pp_log_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests — load fixture, dispatch each case to the matching engine.
// ---------------------------------------------------------------------------

void verify_snv_parity() {
    ParityFixture fx = load_fixture("snv_v1.json");
    for (const auto& c : fx.cases) {
        const int with_mean = params_get_bool(c.params_json, "with_mean", true) ? 1 : 0;
        const int with_std  = params_get_bool(c.params_json, "with_std",  true) ? 1 : 0;
        const int ddof      = static_cast<int>(params_get_int(c.params_json, "ddof", 0));

        c4a_pp_snv_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_snv_create(&h, with_mean, with_std, ddof) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_snv_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "snv/" + c.name);
        c4a_pp_snv_destroy(h);
    }
}

void verify_lsnv_parity() {
    ParityFixture fx = load_fixture("lsnv_v1.json");
    for (const auto& c : fx.cases) {
        const int window = static_cast<int>(params_get_int(c.params_json, "window", 11));
        const std::string pad = params_get_string(c.params_json, "pad_mode", "reflect");
        const double cv = params_get_double(c.params_json, "constant_value", 0.0);
        int pad_mode = 0;
        if      (pad == "reflect")  pad_mode = 0;
        else if (pad == "edge")     pad_mode = 1;
        else if (pad == "constant") pad_mode = 2;
        else throw std::runtime_error("lsnv: unknown pad_mode " + pad);

        c4a_pp_lsnv_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_lsnv_create(&h, window, pad_mode, cv) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_lsnv_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "lsnv/" + c.name);
        c4a_pp_lsnv_destroy(h);
    }
}

void verify_rnv_parity() {
    ParityFixture fx = load_fixture("rnv_v1.json");
    for (const auto& c : fx.cases) {
        const int with_center = params_get_bool(c.params_json, "with_center", true) ? 1 : 0;
        const int with_scale  = params_get_bool(c.params_json, "with_scale",  true) ? 1 : 0;
        const double k        = params_get_double(c.params_json, "k", 1.4826);

        c4a_pp_rnv_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_rnv_create(&h, with_center, with_scale, k) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_rnv_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "rnv/" + c.name);
        c4a_pp_rnv_destroy(h);
    }
}

void verify_area_parity() {
    ParityFixture fx = load_fixture("area_norm_v1.json");
    for (const auto& c : fx.cases) {
        const std::string m = params_get_string(c.params_json, "method", "sum");
        int method = 0;
        if      (m == "sum")     method = 0;
        else if (m == "abs_sum") method = 1;
        else if (m == "trapz")   method = 2;
        else throw std::runtime_error("area: unknown method " + m);

        c4a_pp_area_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_area_create(&h, method) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_area_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "area/" + c.name);
        c4a_pp_area_destroy(h);
    }
}

void verify_normalize_parity() {
    ParityFixture fx = load_fixture("normalize_v1.json");
    for (const auto& c : fx.cases) {
        const double fmin = params_get_double(c.params_json, "feature_min", -1.0);
        const double fmax = params_get_double(c.params_json, "feature_max",  1.0);

        c4a_pp_normalize_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_normalize_create(&h, fmin, fmax) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_normalize_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "normalize/" + c.name);
        c4a_pp_normalize_destroy(h);
    }
}

void verify_simple_scale_parity() {
    ParityFixture fx = load_fixture("simple_scale_v1.json");
    for (const auto& c : fx.cases) {
        c4a_pp_simple_scale_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_simple_scale_create(&h) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_simple_scale_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "simple_scale/" + c.name);
        c4a_pp_simple_scale_destroy(h);
    }
}

void verify_log_parity() {
    ParityFixture fx = load_fixture("log_transform_v1.json");
    for (const auto& c : fx.cases) {
        const double base       = params_get_double(c.params_json, "base", 0.0);
        const double offset     = params_get_double(c.params_json, "offset", 0.0);
        const int auto_offset   = params_get_bool(c.params_json, "auto_offset", true) ? 1 : 0;
        const double min_value  = params_get_double(c.params_json, "min_value", 1e-8);

        c4a_pp_log_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_log_create(&h, base, offset, auto_offset, min_value)
                         == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_log_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "log/" + c.name);
        c4a_pp_log_destroy(h);
    }
}

}  // namespace

void register_preprocessing_stateless_tests(c4a_testing::Runner& r);
void register_preprocessing_stateless_tests(c4a_testing::Runner& r) {
    r.run("pp_snv_smoke",                  test_snv_smoke);
    r.run("pp_snv_parity",                 verify_snv_parity);
    r.run("pp_lsnv_smoke",                 test_lsnv_smoke);
    r.run("pp_lsnv_parity",                verify_lsnv_parity);
    r.run("pp_rnv_smoke",                  test_rnv_smoke);
    r.run("pp_rnv_parity",                 verify_rnv_parity);
    r.run("pp_area_smoke",                 test_area_smoke);
    r.run("pp_area_parity",                verify_area_parity);
    r.run("pp_normalize_smoke",            test_normalize_smoke);
    r.run("pp_normalize_parity",           verify_normalize_parity);
    r.run("pp_simple_scale_smoke",         test_simple_scale_smoke);
    r.run("pp_simple_scale_parity",        verify_simple_scale_parity);
    r.run("pp_log_smoke",                  test_log_smoke);
    r.run("pp_log_parity",                 verify_log_parity);
}
