// SPDX-License-Identifier: CECILL-2.1
//
// Phase 19 — SignalTypeDetector smoke + parity tests.
//
// Smoke:   a tiny reflectance-shaped matrix detects as reflectance.
// Parity:  load `signal_type_detect_v1.json`, replay each case, compare the
//          enum, confidence (bit-for-bit hex), and the human-readable reason
//          string against the Python reference.

#include <cmath>
#include <cstddef>
#include <cstdint>
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

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Smoke tests.
// ---------------------------------------------------------------------------

void test_signal_detect_smoke_reflectance() {
    // Values in [0.1, 0.8] — typical reflectance.
    double X[3 * 4] = {
        0.45, 0.50, 0.55, 0.48,
        0.42, 0.51, 0.53, 0.49,
        0.47, 0.49, 0.52, 0.46,
    };
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 3, 4);
    c4a_signal_type_t type_out = C4A_SIGNAL_UNKNOWN;
    double confidence = -1.0;
    char reason_buf[256] = {0};
    const c4a_status_t st = c4a_signal_detect(
        Xv, nullptr, 0, /*threshold=*/0.3, &type_out, &confidence, reason_buf);
    C4A_TEST_REQUIRE(st == C4A_OK);
    C4A_TEST_REQUIRE(type_out == C4A_SIGNAL_REFLECTANCE);
    C4A_TEST_REQUIRE(confidence > 0.0 && confidence <= 1.0);
    C4A_TEST_REQUIRE(reason_buf[0] != '\0');
}

void test_signal_detect_smoke_empty() {
    // 0x0 input — empty data path.
    c4a_matrix_view_t Xv{};
    Xv.data       = nullptr;
    Xv.rows       = 0;
    Xv.cols       = 0;
    Xv.row_stride = 0;
    Xv.col_stride = 1;
    Xv.dtype      = C4A_DTYPE_F64;
    c4a_signal_type_t type_out = C4A_SIGNAL_ABSORBANCE;
    double confidence = -1.0;
    char reason_buf[256] = {0};
    const c4a_status_t st = c4a_signal_detect(
        Xv, nullptr, 0, /*threshold=*/0.7, &type_out, &confidence, reason_buf);
    C4A_TEST_REQUIRE(st == C4A_OK);
    C4A_TEST_REQUIRE(type_out == C4A_SIGNAL_UNKNOWN);
    C4A_TEST_REQUIRE(confidence == 0.0);
}

void test_signal_detect_smoke_nullptr_out() {
    double X[2] = {0.4, 0.6};
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 2);
    double conf = 0.0;
    char reason[256] = {0};
    const c4a_status_t st1 = c4a_signal_detect(
        Xv, nullptr, 0, 0.5, nullptr, &conf, reason);
    C4A_TEST_REQUIRE(st1 == C4A_ERR_NULL_POINTER);
    c4a_signal_type_t t = C4A_SIGNAL_UNKNOWN;
    const c4a_status_t st2 = c4a_signal_detect(
        Xv, nullptr, 0, 0.5, &t, nullptr, reason);
    C4A_TEST_REQUIRE(st2 == C4A_ERR_NULL_POINTER);
}

// ---------------------------------------------------------------------------
// Parity test against the fixture.
// ---------------------------------------------------------------------------

// Light JSON helpers — the existing fixture_parser was built for the matrix-
// based fixture schema and doesn't expose direct case-body slicing for the
// non-matrix per-case fields here. We slurp the file, locate the case body,
// and decode the values inline.
struct SignalCase {
    std::string         name;
    std::int64_t        rows = 0;
    std::int64_t        cols = 0;
    std::vector<double> input;
    std::int64_t        wl_length = 0;
    std::vector<double> wavelengths;   // empty if wl_length == 0
    double              confidence_threshold = 0.0;
    int32_t             expected_type = 0;
    double              expected_confidence = 0.0;
    std::string         expected_reason;
};

std::vector<SignalCase> load_signal_cases(const std::string& filename) {
    const std::string body = ::c4a_testing::slurp(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename);
    auto [cases_lo, cases_hi] = ::c4a_testing::find_cases_array(body);
    auto spans = ::c4a_testing::list_case_object_spans(body, cases_lo,
                                                        cases_hi);
    std::vector<SignalCase> out;
    for (const auto& [lo, hi] : spans) {
        SignalCase c;
        // name
        std::size_t np = ::c4a_testing::find_value_for_key(body, "name", lo, hi);
        c.name = ::c4a_testing::parse_string(body, np);
        // rows, cols
        c.rows = ::c4a_testing::parse_int64(body,
            ::c4a_testing::find_value_for_key(body, "rows", lo, hi));
        c.cols = ::c4a_testing::parse_int64(body,
            ::c4a_testing::find_value_for_key(body, "cols", lo, hi));
        // input_hex
        std::size_t ip = ::c4a_testing::find_value_for_key(body, "input_hex",
                                                            lo, hi);
        ::c4a_testing::parse_hex_double_array(body, ip, c.input);
        // confidence_threshold, expected_type
        c.confidence_threshold = std::strtod(
            body.c_str() + ::c4a_testing::find_value_for_key(
                body, "confidence_threshold", lo, hi), nullptr);
        c.expected_type = static_cast<int32_t>(
            ::c4a_testing::parse_int64(body,
                ::c4a_testing::find_value_for_key(body, "expected_type",
                                                   lo, hi)));
        // expected_confidence_hex
        std::size_t hp = ::c4a_testing::find_value_for_key(
            body, "expected_confidence_hex", lo, hi);
        c.expected_confidence = ::c4a_testing::parse_hex_double(body, hp);
        // expected_reason
        std::size_t rp = ::c4a_testing::find_value_for_key(
            body, "expected_reason", lo, hi);
        c.expected_reason = ::c4a_testing::parse_string(body, rp);
        // wl_length, optional wavelengths_hex
        c.wl_length = ::c4a_testing::parse_int64(body,
            ::c4a_testing::find_value_for_key(body, "wl_length", lo, hi));
        if (c.wl_length > 0) {
            std::size_t wp = ::c4a_testing::find_value_for_key(
                body, "wavelengths_hex", lo, hi);
            ::c4a_testing::parse_hex_double_array(body, wp, c.wavelengths);
        }
        out.push_back(std::move(c));
    }
    return out;
}

void verify_signal_detect_parity() {
    auto cases = load_signal_cases("signal_type_detect_v1.json");
    C4A_TEST_REQUIRE(!cases.empty());
    for (const auto& c : cases) {
        std::vector<double> X = c.input;
        c4a_matrix_view_t Xv = make_rowmajor_view(X.data(), c.rows, c.cols);
        c4a_signal_type_t type_out = C4A_SIGNAL_UNKNOWN;
        double confidence = 0.0;
        char reason_buf[256] = {0};
        const double*  wl_ptr = (c.wl_length > 0) ? c.wavelengths.data() : nullptr;
        const c4a_status_t st = c4a_signal_detect(
            Xv, wl_ptr, c.wl_length, c.confidence_threshold,
            &type_out, &confidence, reason_buf);
        C4A_TEST_REQUIRE(st == C4A_OK);

        // Enum: exact match.
        if (static_cast<int32_t>(type_out) != c.expected_type) {
            throw std::runtime_error(
                "signal_detect/" + c.name + " enum mismatch: got=" +
                std::to_string(static_cast<int32_t>(type_out)) +
                " want=" + std::to_string(c.expected_type));
        }
        // Confidence: tight tolerance (deterministic heuristic).
        const double dconf = std::fabs(confidence - c.expected_confidence);
        if (dconf > 1e-12) {
            throw std::runtime_error(
                "signal_detect/" + c.name + " confidence mismatch: got=" +
                std::to_string(confidence) +
                " want=" + std::to_string(c.expected_confidence) +
                " |diff|=" + std::to_string(dconf));
        }
        // Reason: exact byte-for-byte match.
        if (std::strcmp(reason_buf, c.expected_reason.c_str()) != 0) {
            throw std::runtime_error(
                "signal_detect/" + c.name + " reason mismatch:\n  got:  '" +
                std::string(reason_buf) + "'\n  want: '" +
                c.expected_reason + "'");
        }
    }
}

}  // namespace

void register_signal_type_tests(c4a_testing::Runner& r);
void register_signal_type_tests(c4a_testing::Runner& r) {
    r.run("signal_detect_smoke_reflectance", test_signal_detect_smoke_reflectance);
    r.run("signal_detect_smoke_empty",       test_signal_detect_smoke_empty);
    r.run("signal_detect_smoke_nullptr_out", test_signal_detect_smoke_nullptr_out);
    r.run("signal_detect_parity",            verify_signal_detect_parity);
}
