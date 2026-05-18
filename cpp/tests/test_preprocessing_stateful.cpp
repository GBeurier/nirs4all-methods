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
//     C4A_ERR_NOT_FITTED.
//
// Baseline also gets an inverse-transform test, and Derivate gets a
// dedicated `c4a_pp_derivate_output_cols(2, 200) == 198` shape-helper
// test. Total contributions: 4 smoke + 4 parity + 4 not-fitted + 1
// inverse + 1 output-cols = 14 new tests.

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
// Minimal JSON fixture parser. The shape is a Phase 3 variant of the Phase 2
// fixture: fit input + transform input + cases with per-case output shape
// (because Derivate's output column count varies with the `order` param).
// ---------------------------------------------------------------------------

struct ParityCase {
    std::string                 name;
    std::string                 params_json;
    std::int64_t                output_rows = 0;
    std::int64_t                output_cols = 0;
    std::vector<double>         expected_output;
};

struct ParityFixture {
    std::int64_t              fit_rows = 0;
    std::int64_t              fit_cols = 0;
    std::vector<double>       fit_input;
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
            if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
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
    fx.fit_rows = parse_int64(body, find_value_for_key(body, "fit_rows", 0, body.size()));
    fx.fit_cols = parse_int64(body, find_value_for_key(body, "fit_cols", 0, body.size()));
    fx.rows     = parse_int64(body, find_value_for_key(body, "rows", 0, body.size()));
    fx.cols     = parse_int64(body, find_value_for_key(body, "cols", 0, body.size()));

    std::size_t fit_pos = find_value_for_key(body, "fit_input_hex", 0, body.size());
    parse_hex_double_array(body, fit_pos, fx.fit_input);
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
        c.output_rows = parse_int64(body, find_value_for_key(body, "output_rows", lo, hi));
        c.output_cols = parse_int64(body, find_value_for_key(body, "output_cols", lo, hi));
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

void assert_close(const std::vector<double>& got,
                  const std::vector<double>& want,
                  const std::string& tag,
                  double abs_tol = 1e-10,
                  double rel_tol = 1e-11) {
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
          << " abs_diff=" << diff << " rel_diff=" << (ref > 0 ? diff / ref : diff);
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
    c4a_pp_msc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_msc_create(&h) == C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);

    int fitted = 1;
    C4A_TEST_REQUIRE(c4a_pp_msc_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 0);

    c4a_matrix_view_t Xfv = make_rowmajor_view(Xfit, 3, 4);
    C4A_TEST_REQUIRE(c4a_pp_msc_fit(h, Xfv) == C4A_OK);

    C4A_TEST_REQUIRE(c4a_pp_msc_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 1);

    c4a_matrix_view_t Xv = make_rowmajor_view(Xt, 2, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 4);
    C4A_TEST_REQUIRE(c4a_pp_msc_transform(h, Xv, Yv) == C4A_OK);
    // The Xfit columns are scalar multiples of [1, 2, 3, 4] so each MSC
    // (a, b) per column should be exact; the resulting MSC-transformed
    // rows are the per-row mean (after scatter correction). Just sanity-
    // check that none of the outputs is NaN/Inf.
    for (int k = 0; k < 8; ++k) {
        C4A_TEST_REQUIRE(std::isfinite(Y[k]));
    }
    c4a_pp_msc_destroy(h);
    c4a_pp_msc_destroy(nullptr);  // null-safe
}

void test_emsc_smoke() {
    double X[12] = {
        1.0, 2.0, 3.0, 4.0,
        1.5, 2.5, 3.5, 4.5,
        2.0, 3.0, 4.0, 5.0,
    };
    double Y[12] = {0};
    c4a_pp_emsc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_emsc_create(&h, /*degree=*/2) == C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);

    c4a_matrix_view_t Xv = make_rowmajor_view(X, 3, 4);
    C4A_TEST_REQUIRE(c4a_pp_emsc_fit(h, Xv) == C4A_OK);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 3, 4);
    C4A_TEST_REQUIRE(c4a_pp_emsc_transform(h, Xv, Yv) == C4A_OK);
    for (int k = 0; k < 12; ++k) {
        C4A_TEST_REQUIRE(std::isfinite(Y[k]));
    }
    c4a_pp_emsc_destroy(h);
}

void test_baseline_smoke() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_baseline_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_baseline_create(&h) == C4A_OK);

    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_baseline_fit(h, Xv) == C4A_OK);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, Xv, Yv) == C4A_OK);

    // Column means: (1+4)/2=2.5, (2+5)/2=3.5, (3+6)/2=4.5.
    // Row 0: [1-2.5, 2-3.5, 3-4.5] = [-1.5, -1.5, -1.5]
    C4A_TEST_REQUIRE(std::fabs(Y[0] + 1.5) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[1] + 1.5) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[2] + 1.5) < 1e-15);
    // Row 1: [4-2.5, 5-3.5, 6-4.5] = [1.5, 1.5, 1.5]
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 1.5) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[5] - 1.5) < 1e-15);
    c4a_pp_baseline_destroy(h);
}

void test_baseline_inverse() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    double Z[6] = {0};
    c4a_pp_baseline_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_baseline_create(&h) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    c4a_matrix_view_t Zv = make_rowmajor_view(Z, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_baseline_fit(h, Xv) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, Xv, Yv) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_baseline_inverse_transform(h, Yv, Zv) == C4A_OK);
    for (int k = 0; k < 6; ++k) {
        C4A_TEST_REQUIRE(std::fabs(Z[k] - X[k]) < 1e-15);
    }
    c4a_pp_baseline_destroy(h);
}

void test_derivate_smoke() {
    double X[8] = {1.0, 2.0, 4.0, 7.0,    /* diff1: [1, 2, 3], diff2: [1, 1] */
                   2.0, 3.0, 5.0, 8.0};
    double Y[6] = {0};  // output cols = 4 - 1 = 3, rows = 2
    c4a_pp_derivate_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_derivate_create(&h, /*order=*/1,
                                              /*delta=*/1.0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_derivate_fit(h, Xv) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_derivate_transform(h, Xv, Yv) == C4A_OK);
    // Row 0 diffs: [2-1, 4-2, 7-4] = [1, 2, 3]
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 1.0) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[1] - 2.0) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[2] - 3.0) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 1.0) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[5] - 3.0) < 1e-15);
    c4a_pp_derivate_destroy(h);
}

void test_derivate_output_cols() {
    C4A_TEST_REQUIRE(c4a_pp_derivate_output_cols(2, 200) == 198);
    C4A_TEST_REQUIRE(c4a_pp_derivate_output_cols(1, 10)  == 9);
    C4A_TEST_REQUIRE(c4a_pp_derivate_output_cols(3, 200) == 197);
    // Degenerate: order >= input_cols → 0 (out-of-range).
    C4A_TEST_REQUIRE(c4a_pp_derivate_output_cols(10, 10) == 0);
    C4A_TEST_REQUIRE(c4a_pp_derivate_output_cols(1, 1)   == 0);
}

// ---------------------------------------------------------------------------
// NOT_FITTED tests
// ---------------------------------------------------------------------------

void test_msc_not_fitted() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_msc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_msc_create(&h) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_msc_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    C4A_TEST_REQUIRE(c4a_pp_msc_inverse_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    c4a_pp_msc_destroy(h);
}

void test_emsc_not_fitted() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_emsc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_emsc_create(&h, 2) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_emsc_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    c4a_pp_emsc_destroy(h);
}

void test_baseline_not_fitted() {
    double X[6] = {1, 2, 3, 4, 5, 6};
    double Y[6] = {0};
    c4a_pp_baseline_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_baseline_create(&h) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    C4A_TEST_REQUIRE(c4a_pp_baseline_inverse_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    c4a_pp_baseline_destroy(h);
}

void test_derivate_not_fitted() {
    double X[8] = {1, 2, 4, 7, 2, 3, 5, 8};
    double Y[6] = {0};
    c4a_pp_derivate_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_derivate_create(&h, 1, 1.0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_derivate_transform(h, Xv, Yv) == C4A_ERR_NOT_FITTED);
    c4a_pp_derivate_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests
// ---------------------------------------------------------------------------

void verify_msc_parity() {
    ParityFixture fx = load_fixture("msc_v1.json");
    for (const auto& c : fx.cases) {
        const std::string variant = params_get_string(c.params_json, "variant", "");
        c4a_pp_msc_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_msc_create(&h) == C4A_OK);
        std::vector<double> fit_in = fx.fit_input;
        c4a_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        C4A_TEST_REQUIRE(c4a_pp_msc_fit(h, Xfv) == C4A_OK);

        if (variant == "inverse") {
            // The reference for this case is inverse_transform(transform(test)).
            // First apply transform to the test matrix, then inverse_transform.
            std::vector<double> in = fx.input;
            std::vector<double> mid(in.size(), 0.0);
            std::vector<double> out(in.size(), 0.0);
            c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.rows, fx.cols);
            c4a_matrix_view_t Mv = make_rowmajor_view(mid.data(),
                                                       fx.rows, fx.cols);
            c4a_matrix_view_t Ov = make_rowmajor_view(out.data(),
                                                       fx.rows, fx.cols);
            C4A_TEST_REQUIRE(c4a_pp_msc_transform(h, Xv, Mv) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_pp_msc_inverse_transform(h, Mv, Ov) == C4A_OK);
            assert_close(out, c.expected_output, "msc/" + c.name);
        } else {
            std::vector<double> in = fx.input;
            std::vector<double> out(in.size(), 0.0);
            c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.rows, fx.cols);
            c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                       fx.rows, fx.cols);
            C4A_TEST_REQUIRE(c4a_pp_msc_transform(h, Xv, Yv) == C4A_OK);
            assert_close(out, c.expected_output, "msc/" + c.name);
        }
        c4a_pp_msc_destroy(h);
    }
}

void verify_emsc_parity() {
    ParityFixture fx = load_fixture("emsc_v1.json");
    for (const auto& c : fx.cases) {
        const int degree = static_cast<int>(params_get_int(c.params_json,
                                                            "degree", 2));
        c4a_pp_emsc_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_emsc_create(&h, degree) == C4A_OK);
        std::vector<double> fit_in = fx.fit_input;
        c4a_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        C4A_TEST_REQUIRE(c4a_pp_emsc_fit(h, Xfv) == C4A_OK);

        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_emsc_transform(h, Xv, Yv) == C4A_OK);
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
        c4a_pp_emsc_destroy(h);
    }
}

void verify_baseline_parity() {
    ParityFixture fx = load_fixture("baseline_center_v1.json");
    for (const auto& c : fx.cases) {
        const std::string variant = params_get_string(c.params_json, "variant", "");
        c4a_pp_baseline_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_baseline_create(&h) == C4A_OK);
        std::vector<double> fit_in = fx.fit_input;
        c4a_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        C4A_TEST_REQUIRE(c4a_pp_baseline_fit(h, Xfv) == C4A_OK);

        if (variant == "test") {
            std::vector<double> in = fx.input;
            std::vector<double> out(in.size(), 0.0);
            c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.rows, fx.cols);
            c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                       fx.rows, fx.cols);
            C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, Xv, Yv) == C4A_OK);
            assert_close(out, c.expected_output, "baseline/" + c.name,
                         1e-12, 1e-13);
        } else if (variant == "inverse") {
            // transform(fit_X) then inverse_transform.
            std::vector<double> in = fx.fit_input;
            std::vector<double> mid(in.size(), 0.0);
            std::vector<double> out(in.size(), 0.0);
            c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.fit_rows, fx.fit_cols);
            c4a_matrix_view_t Mv = make_rowmajor_view(mid.data(),
                                                       fx.fit_rows, fx.fit_cols);
            c4a_matrix_view_t Ov = make_rowmajor_view(out.data(),
                                                       fx.fit_rows, fx.fit_cols);
            C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, Xv, Mv) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_pp_baseline_inverse_transform(h, Mv, Ov) == C4A_OK);
            assert_close(out, c.expected_output, "baseline/" + c.name,
                         1e-12, 1e-13);
        } else {
            // default: fit on fit_X then transform on fit_X.
            std::vector<double> in = fx.fit_input;
            std::vector<double> out(in.size(), 0.0);
            c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                       fx.fit_rows, fx.fit_cols);
            c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                       fx.fit_rows, fx.fit_cols);
            C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, Xv, Yv) == C4A_OK);
            assert_close(out, c.expected_output, "baseline/" + c.name,
                         1e-12, 1e-13);
        }
        c4a_pp_baseline_destroy(h);
    }
}

void verify_derivate_parity() {
    ParityFixture fx = load_fixture("derivate_v1.json");
    for (const auto& c : fx.cases) {
        const int order  = static_cast<int>(params_get_int(c.params_json,
                                                            "order", 1));
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        c4a_pp_derivate_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_derivate_create(&h, order, delta) == C4A_OK);
        std::vector<double> fit_in = fx.fit_input;
        c4a_matrix_view_t Xfv = make_rowmajor_view(fit_in.data(),
                                                    fx.fit_rows, fx.fit_cols);
        C4A_TEST_REQUIRE(c4a_pp_derivate_fit(h, Xfv) == C4A_OK);

        std::vector<double> in = fx.input;
        const std::int64_t out_size = c.output_rows * c.output_cols;
        std::vector<double> out(static_cast<std::size_t>(out_size), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),
                                                   fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(),
                                                   c.output_rows, c.output_cols);
        C4A_TEST_REQUIRE(c4a_pp_derivate_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "derivate/" + c.name,
                     1e-12, 1e-13);
        c4a_pp_derivate_destroy(h);
    }
}

/* Refit contract: calling _fit twice on the same handle must replace prior
 * state. We verify by fitting on X1, transforming X3 to capture output_A,
 * then fitting again on X2 (different stats), and transforming the same X3
 * to capture output_B. output_A must differ from output_B because the fit
 * state was overwritten. */
void test_msc_refit_replaces_state() {
    c4a_pp_msc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_msc_create(&h) == C4A_OK);

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
    c4a_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    c4a_matrix_view_init_rowmajor(&vX1, X1, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&vX2, X2, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&vX3, X3, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&voutA, outA, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&voutB, outB, 2, 3, C4A_DTYPE_F64);

    C4A_TEST_REQUIRE(c4a_pp_msc_fit(h, vX1) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_msc_transform(h, vX3, voutA) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_msc_fit(h, vX2) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_msc_transform(h, vX3, voutB) == C4A_OK);
    /* outputs must differ — different references produce different regressions */
    bool any_diff = false;
    for (int i = 0; i < 6; ++i) {
        if (std::fabs(outA[i] - outB[i]) > 1e-9) { any_diff = true; break; }
    }
    C4A_TEST_REQUIRE(any_diff);
    c4a_pp_msc_destroy(h);
}

void test_emsc_refit_replaces_state() {
    c4a_pp_emsc_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_emsc_create(&h, 2) == C4A_OK);
    double X1[8] = {1.0, 2.0, 3.0, 4.0,
                    2.0, 3.0, 4.0, 5.0};
    double X2[8] = {10.0,  5.0, 20.0, 15.0,
                    -1.0, -2.0, -3.0, -4.0};
    double X3[8] = {3.0, 4.0, 5.0, 6.0,
                    2.5, 3.5, 4.5, 5.5};
    double outA[8], outB[8];
    c4a_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    c4a_matrix_view_init_rowmajor(&vX1, X1, 2, 4, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&vX2, X2, 2, 4, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&vX3, X3, 2, 4, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&voutA, outA, 2, 4, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&voutB, outB, 2, 4, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(c4a_pp_emsc_fit(h, vX1) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_emsc_transform(h, vX3, voutA) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_emsc_fit(h, vX2) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_emsc_transform(h, vX3, voutB) == C4A_OK);
    bool any_diff = false;
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(outA[i] - outB[i]) > 1e-9) { any_diff = true; break; }
    }
    C4A_TEST_REQUIRE(any_diff);
    c4a_pp_emsc_destroy(h);
}

void test_baseline_refit_replaces_state() {
    c4a_pp_baseline_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_baseline_create(&h) == C4A_OK);
    double X1[6] = {0.0,  0.0,  0.0,
                    0.0,  0.0,  0.0};
    double X2[6] = {10.0, 20.0, 30.0,
                    40.0, 50.0, 60.0};
    double X3[6] = {1.0,  2.0,  3.0,
                    4.0,  5.0,  6.0};
    double outA[6], outB[6];
    c4a_matrix_view_t vX1, vX2, vX3, voutA, voutB;
    c4a_matrix_view_init_rowmajor(&vX1, X1, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&vX2, X2, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&vX3, X3, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&voutA, outA, 2, 3, C4A_DTYPE_F64);
    c4a_matrix_view_init_rowmajor(&voutB, outB, 2, 3, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(c4a_pp_baseline_fit(h, vX1) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, vX3, voutA) == C4A_OK);
    /* mean(X1) is all zeros, so outA == X3 */
    for (int i = 0; i < 6; ++i) C4A_TEST_REQUIRE(std::fabs(outA[i] - X3[i]) < 1e-15);
    C4A_TEST_REQUIRE(c4a_pp_baseline_fit(h, vX2) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_pp_baseline_transform(h, vX3, voutB) == C4A_OK);
    /* mean(X2) = [25, 35, 45], so outB[0] = X3[0] - 25 = -24, etc. */
    const double expected[6] = { 1.0 - 25.0, 2.0 - 35.0, 3.0 - 45.0,
                                  4.0 - 25.0, 5.0 - 35.0, 6.0 - 45.0 };
    for (int i = 0; i < 6; ++i) C4A_TEST_REQUIRE(std::fabs(outB[i] - expected[i]) < 1e-15);
    c4a_pp_baseline_destroy(h);
}

}  // namespace

void register_preprocessing_stateful_tests(c4a_testing::Runner& r);
void register_preprocessing_stateful_tests(c4a_testing::Runner& r) {
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
