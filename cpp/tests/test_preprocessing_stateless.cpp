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
// Minimal JSON fixture parser (hand-rolled — same approach as the Phase 1
// RNG test). We only need to extract a few scalar fields and the
// big-endian hex-encoded double arrays.
// ---------------------------------------------------------------------------

struct ParityCase {
    std::string                 name;
    std::string                 params_json;  // raw JSON of params dict
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

// Locate the start of a value for `"key" :` starting search at `from`. Returns
// the index of the first non-ws character after the colon. Restricted to
// [from, end) so we can scope lookups to a substring.
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

// Parse a hex-encoded big-endian double sitting at `s[pos]` (must be a
// JSON string with 16 hex chars). Advances `pos` past the closing quote.
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

// Parse a JSON array of hex-encoded doubles into `out`. `pos` must be at the
// '[' character; on success points one past the matching ']'.
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

// Locate the substring containing the cases array body
// (`"cases" : [ ... ]`). Returns (start_of_body, end_of_body).
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
    // Find the matching ']' by counting nested depth.
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

// Iterate over case objects inside `body`. Each case is `{ ... }`; we
// return the substring bounds for each one in document order.
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

// Parse a JSON string literal at position `pos`, advancing past the closing
// quote and returning the decoded contents (no escape handling beyond '\\"').
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

// Parse a JSON object/string/number value as a substring (best-effort: we
// only need raw access to the `params` object body and the case `name`
// string). Returns (begin, end) bounds covering the value.
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
    // Number / boolean / null — read until comma, whitespace, or close.
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

// ---------------------------------------------------------------------------
// Helpers for parsing case params (raw JSON-ish substring → typed value).
// ---------------------------------------------------------------------------

bool params_get_bool(const std::string& json, const std::string& key,
                     bool default_value) {
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) return default_value;
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    if (json.compare(p, 4, "true") == 0)  return true;
    if (json.compare(p, 5, "false") == 0) return false;
    return default_value;
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
// Tolerance check: byte equality OR (abs <= 1e-12 OR rel <= 1e-13).
// ---------------------------------------------------------------------------

void assert_close(const std::vector<double>& got,
                  const std::vector<double>& want,
                  const std::string& tag) {
    if (got.size() != want.size()) {
        std::ostringstream m;
        m << tag << " size mismatch: got " << got.size()
          << " want " << want.size();
        throw std::runtime_error(m.str());
    }
    constexpr double kAbsTol = 1e-12;
    constexpr double kRelTol = 1e-13;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double w = want[i];
        std::uint64_t gb, wb;
        std::memcpy(&gb, &g, sizeof(double));
        std::memcpy(&wb, &w, sizeof(double));
        if (gb == wb) continue;
        const double diff = std::fabs(g - w);
        const double ref  = std::max(std::fabs(g), std::fabs(w));
        if (diff <= kAbsTol) continue;
        if (ref > 0.0 && diff / ref <= kRelTol) continue;
        std::ostringstream m;
        m << tag << " mismatch at i=" << i
          << " got=" << g << " want=" << w
          << " abs_diff=" << diff << " rel_diff=" << (ref > 0 ? diff / ref : diff);
        throw std::runtime_error(m.str());
    }
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
