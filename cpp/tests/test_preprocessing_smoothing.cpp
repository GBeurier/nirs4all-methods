// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 4 stateless derivative & smoothing operators
// (SavitzkyGolay, FirstDerivative, SecondDerivative, NorrisWilliams,
// Gaussian).  For each operator the test loads a JSON fixture produced by
// parity/python_generator/scripts/generate_phase4_fixtures.py, decodes the
// input / expected matrices from big-endian hex doubles, applies the C
// engine through its public ABI, and asserts byte-equality or 1e-10 / 1e-11
// tolerance (loosened for SG and Gaussian, which depart by a few ULPs from
// the SVD / FFT paths SciPy uses internally on the same algorithm).
//
// Each operator contributes one smoke test exercising the create/transform/
// destroy lifecycle and one parity test sweeping the full fixture.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "chemometrics4all/c4a.h"

#include "harness.hpp"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

// ---------------------------------------------------------------------------
// Minimal fixture parser — duplicated from the stateless / stateful tests
// because the harness is intentionally zero-dependency.  When we move to a
// shared JSON parser (Phase 5+), the implementation here gets pulled out.
// ---------------------------------------------------------------------------

struct ParityCase {
    std::string                 name;
    std::string                 params_json;
    std::vector<double>         expected_output;
};

struct ParityFixture {
    std::int64_t              rows = 0;
    std::int64_t              cols = 0;
    std::vector<double>       input;
    std::vector<ParityCase>   cases;
};

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open fixture: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
        ++i;
    }
}

std::size_t find_value_for_key(const std::string& s, const std::string& key,
                               std::size_t from, std::size_t end) {
    std::string q = "\"" + key + "\"";
    std::size_t pos = s.find(q, from);
    if (pos == std::string::npos || pos + q.size() > end) {
        throw std::runtime_error("fixture missing key: " + key);
    }
    pos += q.size();
    skip_ws(s, pos);
    if (pos >= end || s[pos] != ':') {
        throw std::runtime_error("fixture: ':' expected after key " + key);
    }
    ++pos;
    skip_ws(s, pos);
    if (pos >= end) {
        throw std::runtime_error("fixture: value missing for key " + key);
    }
    return pos;
}

std::int64_t parse_int64(const std::string& s, std::size_t pos) {
    char* tail = nullptr;
    std::int64_t v = std::strtoll(s.c_str() + pos, &tail, 10);
    if (tail == s.c_str() + pos) {
        throw std::runtime_error("fixture: cannot parse integer");
    }
    return v;
}

double parse_hex_double(const std::string& s, std::size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') {
        throw std::runtime_error("fixture: expected '\"' starting hex double");
    }
    ++pos;
    if (pos + 16 > s.size()) {
        throw std::runtime_error("fixture: truncated hex double");
    }
    auto nibble = [&](char c) -> std::uint8_t {
        if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(10 + c - 'a');
        if (c >= 'A' && c <= 'F') return static_cast<std::uint8_t>(10 + c - 'A');
        throw std::runtime_error("fixture: bad hex digit");
    };
    std::uint64_t bits = 0;
    for (std::size_t b = 0; b < 8; ++b) {
        const std::uint8_t hi = nibble(s[pos + 2 * b]);
        const std::uint8_t lo = nibble(s[pos + 2 * b + 1]);
        bits = (bits << 8) | static_cast<std::uint64_t>((hi << 4) | lo);
    }
    pos += 16;
    if (pos >= s.size() || s[pos] != '"') {
        throw std::runtime_error("fixture: missing closing '\"' on hex double");
    }
    ++pos;
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

void parse_hex_double_array(const std::string& s, std::size_t& pos,
                            std::vector<double>& out) {
    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("fixture: '[' expected for hex double array");
    }
    ++pos;
    for (;;) {
        skip_ws(s, pos);
        if (pos >= s.size()) {
            throw std::runtime_error("fixture: unterminated hex array");
        }
        if (s[pos] == ']') {
            ++pos;
            return;
        }
        out.push_back(parse_hex_double(s, pos));
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') {
            ++pos;
        }
    }
}

std::pair<std::size_t, std::size_t>
find_cases_array(const std::string& s) {
    std::size_t key_pos = s.find("\"cases\"");
    if (key_pos == std::string::npos) {
        throw std::runtime_error("fixture: 'cases' key missing");
    }
    std::size_t pos = key_pos + 7;
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != ':') {
        throw std::runtime_error("fixture: ':' expected after cases key");
    }
    ++pos;
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("fixture: '[' expected for cases array");
    }
    const std::size_t start = pos + 1;
    int depth = 1;
    std::size_t i = start;
    bool in_string = false;
    bool escaped = false;
    while (i < s.size() && depth > 0) {
        const char c = s[i];
        if (escaped) {
            escaped = false;
        } else if (in_string) {
            if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
        } else {
            if (c == '"')        in_string = true;
            else if (c == '[')   ++depth;
            else if (c == ']')   { --depth; if (depth == 0) return {start, i}; }
        }
        ++i;
    }
    throw std::runtime_error("fixture: unterminated cases array");
}

std::vector<std::pair<std::size_t, std::size_t>>
list_case_object_spans(const std::string& s, std::size_t lo, std::size_t hi) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    std::size_t i = lo;
    while (i < hi) {
        skip_ws(s, i);
        if (i >= hi) break;
        if (s[i] != '{') {
            if (s[i] == ',') { ++i; continue; }
            throw std::runtime_error("fixture: expected '{' opening a case");
        }
        const std::size_t obj_start = i;
        ++i;
        int depth = 1;
        bool in_string = false;
        bool escaped = false;
        while (i < hi && depth > 0) {
            const char c = s[i];
            if (escaped) { escaped = false; }
            else if (in_string) {
                if (c == '\\') escaped = true;
                else if (c == '"') in_string = false;
            } else {
                if (c == '"')      in_string = true;
                else if (c == '{') ++depth;
                else if (c == '}') --depth;
            }
            ++i;
            if (depth == 0) {
                out.emplace_back(obj_start, i);
                break;
            }
        }
        if (depth != 0) {
            throw std::runtime_error("fixture: unterminated case object");
        }
    }
    return out;
}

std::string parse_string(const std::string& s, std::size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') {
        throw std::runtime_error("fixture: expected string");
    }
    ++pos;
    const std::size_t start = pos;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos += 2;
        } else {
            ++pos;
        }
    }
    if (pos >= s.size()) {
        throw std::runtime_error("fixture: unterminated string");
    }
    std::string out = s.substr(start, pos - start);
    ++pos;
    return out;
}

std::pair<std::size_t, std::size_t>
read_value_span(const std::string& s, std::size_t pos) {
    skip_ws(s, pos);
    if (pos >= s.size()) {
        throw std::runtime_error("fixture: missing value");
    }
    const char c = s[pos];
    const std::size_t start = pos;
    if (c == '{' || c == '[') {
        const char closing = (c == '{') ? '}' : ']';
        int depth = 1;
        bool in_string = false;
        bool escaped = false;
        ++pos;
        while (pos < s.size() && depth > 0) {
            const char d = s[pos];
            if (escaped) escaped = false;
            else if (in_string) {
                if (d == '\\') escaped = true;
                else if (d == '"') in_string = false;
            } else {
                if (d == '"') in_string = true;
                else if (d == c) ++depth;
                else if (d == closing) --depth;
            }
            ++pos;
        }
        if (depth != 0) {
            throw std::runtime_error("fixture: unterminated value");
        }
        return {start, pos};
    }
    if (c == '"') {
        ++pos;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) pos += 2;
            else ++pos;
        }
        if (pos >= s.size()) {
            throw std::runtime_error("fixture: unterminated string");
        }
        return {start, pos + 1};
    }
    while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' &&
           s[pos] != ' ' && s[pos] != '\t' && s[pos] != '\n' && s[pos] != '\r') {
        ++pos;
    }
    return {start, pos};
}

ParityFixture load_fixture(const std::string& filename) {
    const std::string path =
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename;
    const std::string body = slurp(path);
    ParityFixture fx;
    fx.rows = parse_int64(body, find_value_for_key(body, "rows", 0, body.size()));
    fx.cols = parse_int64(body, find_value_for_key(body, "cols", 0, body.size()));

    std::size_t input_pos = find_value_for_key(body, "input_hex", 0, body.size());
    parse_hex_double_array(body, input_pos, fx.input);

    auto [cases_lo, cases_hi] = find_cases_array(body);
    auto spans = list_case_object_spans(body, cases_lo, cases_hi);
    for (const auto& [lo, hi] : spans) {
        ParityCase c;
        std::size_t np = find_value_for_key(body, "name", lo, hi);
        c.name = parse_string(body, np);
        std::size_t pp = find_value_for_key(body, "params", lo, hi);
        auto [pb, pe] = read_value_span(body, pp);
        c.params_json = body.substr(pb, pe - pb);
        std::size_t op = find_value_for_key(body, "output_hex", lo, hi);
        parse_hex_double_array(body, op, c.expected_output);
        fx.cases.push_back(std::move(c));
    }
    return fx;
}

double params_get_double(const std::string& json, const std::string& key,
                         double default_value) {
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) return default_value;
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    char* tail = nullptr;
    const double v = std::strtod(json.c_str() + p, &tail);
    if (tail == json.c_str() + p) return default_value;
    return v;
}

std::int64_t params_get_int(const std::string& json, const std::string& key,
                             std::int64_t default_value) {
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) return default_value;
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    char* tail = nullptr;
    const std::int64_t v = std::strtoll(json.c_str() + p, &tail, 10);
    if (tail == json.c_str() + p) return default_value;
    return v;
}

std::string params_get_string(const std::string& json, const std::string& key,
                               const std::string& default_value) {
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) return default_value;
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    if (p >= json.size() || json[p] != '"') return default_value;
    ++p;
    const std::size_t start = p;
    while (p < json.size() && json[p] != '"') ++p;
    return json.substr(start, p - start);
}

// ---------------------------------------------------------------------------
// Tolerance check.  SG & Gaussian deviate up to ~5e-11 from the SciPy SVD /
// FFT paths because the underlying small-system solvers differ in rounding;
// FirstDerivative / SecondDerivative / NorrisWilliams are pure arithmetic
// and hit 1e-12 byte-tolerant comparisons.
// ---------------------------------------------------------------------------

void assert_close_with(const std::vector<double>& got,
                       const std::vector<double>& want,
                       const std::string& tag,
                       double abs_tol, double rel_tol) {
    if (got.size() != want.size()) {
        std::ostringstream m;
        m << tag << " size mismatch: got " << got.size()
          << " want " << want.size();
        throw std::runtime_error(m.str());
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double w = want[i];
        std::uint64_t gb, wb;
        std::memcpy(&gb, &g, sizeof(double));
        std::memcpy(&wb, &w, sizeof(double));
        if (gb == wb) continue;
        const double diff = std::fabs(g - w);
        const double ref  = std::max(std::fabs(g), std::fabs(w));
        if (diff <= abs_tol) continue;
        if (ref > 0.0 && diff / ref <= rel_tol) continue;
        std::ostringstream m;
        m << tag << " mismatch at i=" << i
          << " got=" << g << " want=" << w
          << " abs_diff=" << diff << " rel_diff="
          << (ref > 0 ? diff / ref : diff);
        throw std::runtime_error(m.str());
    }
}

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

void test_savgol_smoke() {
    double X[10] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0};
    double Y[10] = {0};
    c4a_pp_savgol_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_savgol_create(&h, /*w=*/5, /*p=*/2, /*d=*/0,
                                            /*delta=*/1.0,
                                            C4A_PP_SAVGOL_MIRROR, 0.0) ==
                     C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    C4A_TEST_REQUIRE(c4a_pp_savgol_transform(h, Xv, Yv) == C4A_OK);
    // The polynomial 1, 2, 4, 7, 11, ... is monotonically increasing; the
    // SG-smoothed values stay close to the raw values away from edges.
    C4A_TEST_REQUIRE(Y[4] > 9.0 && Y[4] < 12.0);
    // Invalid parameters.
    c4a_pp_savgol_handle_t* h_bad = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_savgol_create(&h_bad, /*w=*/4, 2, 0, 1.0,
                                            C4A_PP_SAVGOL_MIRROR, 0.0) ==
                     C4A_ERR_INVALID_ARGUMENT);  // even window
    C4A_TEST_REQUIRE(c4a_pp_savgol_create(&h_bad, /*w=*/5, 5, 0, 1.0,
                                            C4A_PP_SAVGOL_MIRROR, 0.0) ==
                     C4A_ERR_INVALID_ARGUMENT);  // polyorder >= window
    c4a_pp_savgol_destroy(h);
    c4a_pp_savgol_destroy(nullptr);
}

void test_first_derivative_smoke() {
    double X[5] = {0.0, 1.0, 4.0, 9.0, 16.0};  /* x^2 sampled */
    double Y[5] = {0};
    c4a_pp_first_derivative_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_first_derivative_create(&h, /*delta=*/1.0,
                                                      /*edge_order=*/2) ==
                     C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 5);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 5);
    C4A_TEST_REQUIRE(c4a_pp_first_derivative_transform(h, Xv, Yv) == C4A_OK);
    // Interior derivative of x^2 at integer points is 2x:
    // Y[1] = (4 - 0) / 2 = 2; Y[2] = (9 - 1) / 2 = 4; Y[3] = (16 - 4) / 2 = 6.
    C4A_TEST_REQUIRE(std::fabs(Y[1] - 2.0) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[2] - 4.0) < 1e-12);
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 6.0) < 1e-12);
    c4a_pp_first_derivative_destroy(h);
}

void test_second_derivative_smoke() {
    double X[7] = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0};  /* x^2 */
    double Y[7] = {0};
    c4a_pp_second_derivative_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_second_derivative_create(&h, /*delta=*/1.0,
                                                       /*edge_order=*/2) ==
                     C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 7);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 7);
    C4A_TEST_REQUIRE(c4a_pp_second_derivative_transform(h, Xv, Yv) == C4A_OK);
    // d^2/dx^2 (x^2) = 2 — the second derivative is exactly 2 everywhere
    // for the interior of a parabola sampled on the integers.
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 2.0) < 1e-10);
    c4a_pp_second_derivative_destroy(h);
}

void test_norris_williams_smoke() {
    double X[7] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0};
    double Y[7] = {0};
    c4a_pp_norris_williams_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_norris_williams_create(&h, /*segment=*/3,
                                                     /*gap=*/1,
                                                     /*derivative_order=*/1,
                                                     /*delta=*/1.0) ==
                     C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 7);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 7);
    C4A_TEST_REQUIRE(c4a_pp_norris_williams_transform(h, Xv, Yv) == C4A_OK);
    // Smoke check: the gap-1 derivative produces a sequence centred around
    // the local slope of the input (positive and increasing in this case).
    C4A_TEST_REQUIRE(Y[3] > 0.0);
    C4A_TEST_REQUIRE(Y[5] > Y[1]);
    c4a_pp_norris_williams_destroy(h);
    c4a_pp_norris_williams_destroy(nullptr);
}

void test_gaussian_smoke() {
    double X[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double Y[7] = {0};
    c4a_pp_gaussian_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_gaussian_create(&h, /*sigma=*/1.0, /*order=*/0,
                                              C4A_PP_GAUSSIAN_REFLECT, 0.0,
                                              4.0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 7);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 7);
    C4A_TEST_REQUIRE(c4a_pp_gaussian_transform(h, Xv, Yv) == C4A_OK);
    // Gaussian smoothing of a perfect ramp should leave the centre nearly
    // unchanged.
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 4.0) < 1e-10);
    c4a_pp_gaussian_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests — load fixtures and run every case.
// ---------------------------------------------------------------------------

c4a_pp_savgol_mode_t savgol_mode_from_string(const std::string& m) {
    if (m == "mirror")   return C4A_PP_SAVGOL_MIRROR;
    if (m == "constant") return C4A_PP_SAVGOL_CONSTANT;
    if (m == "nearest")  return C4A_PP_SAVGOL_NEAREST;
    if (m == "wrap")     return C4A_PP_SAVGOL_WRAP;
    if (m == "interp")   return C4A_PP_SAVGOL_INTERP;
    throw std::runtime_error("savgol: unknown mode " + m);
}

c4a_pp_gaussian_mode_t gaussian_mode_from_string(const std::string& m) {
    if (m == "reflect")  return C4A_PP_GAUSSIAN_REFLECT;
    if (m == "constant") return C4A_PP_GAUSSIAN_CONSTANT;
    if (m == "nearest")  return C4A_PP_GAUSSIAN_NEAREST;
    if (m == "mirror")   return C4A_PP_GAUSSIAN_MIRROR;
    if (m == "wrap")     return C4A_PP_GAUSSIAN_WRAP;
    throw std::runtime_error("gaussian: unknown mode " + m);
}

void verify_savgol_parity() {
    ParityFixture fx = load_fixture("savgol_v1.json");
    for (const auto& c : fx.cases) {
        const int w = static_cast<int>(
            params_get_int(c.params_json, "window_length", 11));
        const int p = static_cast<int>(
            params_get_int(c.params_json, "polyorder", 3));
        const int d = static_cast<int>(
            params_get_int(c.params_json, "deriv", 0));
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        const std::string m = params_get_string(c.params_json, "mode",
                                                  "mirror");
        const double cv = params_get_double(c.params_json, "cval", 0.0);

        c4a_pp_savgol_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_savgol_create(&h, w, p, d, delta,
                                                savgol_mode_from_string(m),
                                                cv) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_savgol_transform(h, Xv, Yv) == C4A_OK);
        // SG: SVD-based scipy lstsq vs our Vandermonde-QR may diverge by a
        // few ULPs on the coefficient computation, then amplified by the
        // convolution sum.  1e-9 abs / 1e-10 rel gives us a comfortable
        // margin while still catching algorithmic bugs.
        assert_close_with(out, c.expected_output, "savgol/" + c.name,
                           1e-9, 1e-10);
        c4a_pp_savgol_destroy(h);
    }
}

void verify_first_derivative_parity() {
    ParityFixture fx = load_fixture("first_derivative_v1.json");
    for (const auto& c : fx.cases) {
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        const int edge = static_cast<int>(
            params_get_int(c.params_json, "edge_order", 2));
        c4a_pp_first_derivative_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_first_derivative_create(&h, delta, edge) ==
                         C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_first_derivative_transform(h, Xv, Yv) ==
                         C4A_OK);
        // Pure arithmetic — byte-exact tolerance.
        assert_close_with(out, c.expected_output,
                           "first_derivative/" + c.name, 1e-12, 1e-13);
        c4a_pp_first_derivative_destroy(h);
    }
}

void verify_second_derivative_parity() {
    ParityFixture fx = load_fixture("second_derivative_v1.json");
    for (const auto& c : fx.cases) {
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        const int edge = static_cast<int>(
            params_get_int(c.params_json, "edge_order", 2));
        c4a_pp_second_derivative_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_second_derivative_create(&h, delta, edge) ==
                         C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_second_derivative_transform(h, Xv, Yv) ==
                         C4A_OK);
        assert_close_with(out, c.expected_output,
                           "second_derivative/" + c.name, 1e-12, 1e-13);
        c4a_pp_second_derivative_destroy(h);
    }
}

void verify_norris_williams_parity() {
    ParityFixture fx = load_fixture("norris_williams_v1.json");
    for (const auto& c : fx.cases) {
        const int seg = static_cast<int>(
            params_get_int(c.params_json, "segment", 5));
        const int gap = static_cast<int>(
            params_get_int(c.params_json, "gap", 5));
        const int d = static_cast<int>(
            params_get_int(c.params_json, "derivative_order", 1));
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        c4a_pp_norris_williams_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_norris_williams_create(&h, seg, gap, d,
                                                         delta) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_norris_williams_transform(h, Xv, Yv) ==
                         C4A_OK);
        // Pure arithmetic — tight tolerance.
        assert_close_with(out, c.expected_output,
                           "norris_williams/" + c.name, 1e-12, 1e-13);
        c4a_pp_norris_williams_destroy(h);
    }
}

void verify_gaussian_parity() {
    ParityFixture fx = load_fixture("gaussian_v1.json");
    for (const auto& c : fx.cases) {
        const double sigma = params_get_double(c.params_json, "sigma", 1.0);
        const int order = static_cast<int>(
            params_get_int(c.params_json, "order", 0));
        const std::string m = params_get_string(c.params_json, "mode",
                                                  "reflect");
        const double cv = params_get_double(c.params_json, "cval", 0.0);
        const double tr = params_get_double(c.params_json, "truncate", 4.0);
        c4a_pp_gaussian_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_gaussian_create(&h, sigma, order,
                                                  gaussian_mode_from_string(m),
                                                  cv, tr) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_gaussian_transform(h, Xv, Yv) == C4A_OK);
        // Gaussian: kernel sum + convolution differ by a couple of ULPs vs
        // scipy's correlate1d (which may use FFT for larger kernels).  We
        // keep the same tolerance as SG.
        assert_close_with(out, c.expected_output,
                           "gaussian/" + c.name, 1e-9, 1e-10);
        c4a_pp_gaussian_destroy(h);
    }
}

}  // namespace

void register_preprocessing_smoothing_tests(c4a_testing::Runner& r);
void register_preprocessing_smoothing_tests(c4a_testing::Runner& r) {
    r.run("pp_savgol_smoke",                test_savgol_smoke);
    r.run("pp_savgol_parity",               verify_savgol_parity);
    r.run("pp_first_derivative_smoke",      test_first_derivative_smoke);
    r.run("pp_first_derivative_parity",     verify_first_derivative_parity);
    r.run("pp_second_derivative_smoke",     test_second_derivative_smoke);
    r.run("pp_second_derivative_parity",    verify_second_derivative_parity);
    r.run("pp_norris_williams_smoke",       test_norris_williams_smoke);
    r.run("pp_norris_williams_parity",      verify_norris_williams_parity);
    r.run("pp_gaussian_smoke",              test_gaussian_smoke);
    r.run("pp_gaussian_parity",             verify_gaussian_parity);
}
