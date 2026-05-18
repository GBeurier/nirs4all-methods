// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 12 Y outlier filter
// (`c4a_filter_y_outlier_*`). The single operator has four methods (IQR,
// Z-score, percentile, MAD); each contributes one smoke test and one parity
// test that verifies exact mask equality + integer count equality + 1e-12
// tolerance on the exclusion rate against the frozen NumPy reference at
// parity/python_generator/src/c4a_parity_filters_ref/y_outlier.py.
//
// This test executable is registered as a standalone CTest target
// (`chemometrics4all_filters_tests`) to avoid touching the Phase 5b test
// harness. The fixture format used here is the Phase 12-specific schema
// documented in scripts/generate_phase12_fixtures.py, so we hand-roll a
// minimal parser rather than reusing fixture_parser.hpp (which is keyed to
// preprocessing matrices).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chemometrics4all/c4a.h"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined to locate the fixture files"
#endif

#include "harness.hpp"

namespace {

// FILTER_REQUIRE wraps the central harness's REQUIRE so each filter assertion
// throws c4a_testing::AssertFailed (caught by Runner::run upstream).
#define FILTER_REQUIRE(cond) C4A_TEST_REQUIRE(cond)

// ---------------------------------------------------------------------------
// Fixture parser (Phase 12 schema).
//
// Shape:
//   { "format":..., "n":<int>, "y_hex":[...], "cases":[
//       { "name":..., "params": {method, threshold, lower_percentile,
//                                 upper_percentile},
//         "expected_mask":[0|1, ...],
//         "expected_stats":{n_samples, n_kept, n_excluded,
//                            exclusion_rate:"<hex>"} },
//       ...
//     ] }
// ---------------------------------------------------------------------------

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

std::int64_t parse_int64_at(const std::string& s, std::size_t pos) {
    char* tail = nullptr;
    std::int64_t v = std::strtoll(s.c_str() + pos, &tail, 10);
    if (tail == s.c_str() + pos) {
        throw std::runtime_error("fixture: cannot parse integer");
    }
    return v;
}

double parse_hex_double_quoted(const std::string& s, std::size_t pos) {
    if (pos >= s.size() || s[pos] != '"') {
        throw std::runtime_error("fixture: expected '\"' starting hex double");
    }
    ++pos;
    if (pos + 16 > s.size()) {
        throw std::runtime_error("fixture: truncated hex double");
    }
    auto nibble = [](char c) -> std::uint8_t {
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
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

void parse_hex_double_array(const std::string& s, std::size_t& pos,
                             std::vector<double>& out) {
    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("fixture: '[' expected for hex array");
    }
    ++pos;
    for (;;) {
        skip_ws(s, pos);
        if (pos >= s.size()) {
            throw std::runtime_error("fixture: unterminated hex array");
        }
        if (s[pos] == ']') { ++pos; return; }
        out.push_back(parse_hex_double_quoted(s, pos));
        pos += 17;  // skip '"' + 16 hex
        if (pos >= s.size() || s[pos] != '"') {
            throw std::runtime_error("fixture: missing closing quote");
        }
        ++pos;
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') ++pos;
    }
}

void parse_int_array(const std::string& s, std::size_t& pos,
                      std::vector<int>& out) {
    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("fixture: '[' expected for int array");
    }
    ++pos;
    for (;;) {
        skip_ws(s, pos);
        if (pos >= s.size()) {
            throw std::runtime_error("fixture: unterminated int array");
        }
        if (s[pos] == ']') { ++pos; return; }
        char* tail = nullptr;
        long v = std::strtol(s.c_str() + pos, &tail, 10);
        if (tail == s.c_str() + pos) {
            throw std::runtime_error("fixture: bad int element");
        }
        out.push_back(static_cast<int>(v));
        pos = static_cast<std::size_t>(tail - s.c_str());
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') ++pos;
    }
}

std::pair<std::size_t, std::size_t> find_array_span(const std::string& s,
                                                     std::size_t pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("fixture: '[' expected");
    }
    const std::size_t start = pos + 1;
    int depth = 1;
    std::size_t i = start;
    bool in_string = false;
    bool escaped = false;
    while (i < s.size() && depth > 0) {
        const char c = s[i];
        if (escaped)         { escaped = false; }
        else if (in_string)  { if (c == '\\') escaped = true;
                                else if (c == '"') in_string = false; }
        else {
            if      (c == '"') in_string = true;
            else if (c == '[') ++depth;
            else if (c == ']') { --depth; if (depth == 0) return {start, i}; }
        }
        ++i;
    }
    throw std::runtime_error("fixture: unterminated array");
}

std::pair<std::size_t, std::size_t> find_object_span(const std::string& s,
                                                      std::size_t pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '{') {
        throw std::runtime_error("fixture: '{' expected");
    }
    const std::size_t start = pos + 1;
    int depth = 1;
    std::size_t i = start;
    bool in_string = false;
    bool escaped = false;
    while (i < s.size() && depth > 0) {
        const char c = s[i];
        if (escaped)         { escaped = false; }
        else if (in_string)  { if (c == '\\') escaped = true;
                                else if (c == '"') in_string = false; }
        else {
            if      (c == '"') in_string = true;
            else if (c == '{') ++depth;
            else if (c == '}') { --depth; if (depth == 0) return {start, i}; }
        }
        ++i;
    }
    throw std::runtime_error("fixture: unterminated object");
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
            if (depth == 0) { out.emplace_back(obj_start, i); break; }
        }
        if (depth != 0) {
            throw std::runtime_error("fixture: unterminated case");
        }
    }
    return out;
}

std::string parse_string_at(const std::string& s, std::size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') {
        throw std::runtime_error("fixture: expected string");
    }
    ++pos;
    const std::size_t start = pos;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) pos += 2;
        else ++pos;
    }
    if (pos >= s.size()) {
        throw std::runtime_error("fixture: unterminated string");
    }
    std::string out = s.substr(start, pos - start);
    ++pos;
    return out;
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

struct ParityCase {
    std::string         name;
    std::string         params_json;
    std::vector<int>    expected_mask;
    std::int64_t        expected_n_samples = 0;
    std::int64_t        expected_n_kept = 0;
    std::int64_t        expected_n_excluded = 0;
    double              expected_exclusion_rate = 0.0;
};

struct Fixture {
    std::int64_t            n = 0;
    std::vector<double>     y;
    std::vector<ParityCase> cases;
};

Fixture load_fixture(const std::string& path) {
    const std::string body = slurp(path);
    Fixture fx;
    fx.n = parse_int64_at(body, find_value_for_key(body, "n", 0, body.size()));
    std::size_t y_pos = find_value_for_key(body, "y_hex", 0, body.size());
    parse_hex_double_array(body, y_pos, fx.y);

    // Walk the cases array.
    std::size_t cases_pos = find_value_for_key(body, "cases", 0, body.size());
    auto [cases_lo, cases_hi] = find_array_span(body, cases_pos);
    auto spans = list_case_object_spans(body, cases_lo, cases_hi);
    for (const auto& [lo, hi] : spans) {
        ParityCase c;
        std::size_t np = find_value_for_key(body, "name", lo, hi);
        c.name = parse_string_at(body, np);

        // params is an object — slice the substring for keyword lookups.
        std::size_t pp = find_value_for_key(body, "params", lo, hi);
        auto [pb, pe] = find_object_span(body, pp);
        c.params_json = body.substr(pb, pe - pb);

        // expected_mask is an int array.
        std::size_t mp = find_value_for_key(body, "expected_mask", lo, hi);
        parse_int_array(body, mp, c.expected_mask);

        // expected_stats is an object — slice and parse individual fields.
        std::size_t sp = find_value_for_key(body, "expected_stats", lo, hi);
        auto [sb, se] = find_object_span(body, sp);
        c.expected_n_samples = parse_int64_at(
            body, find_value_for_key(body, "n_samples", sb, se));
        c.expected_n_kept = parse_int64_at(
            body, find_value_for_key(body, "n_kept", sb, se));
        c.expected_n_excluded = parse_int64_at(
            body, find_value_for_key(body, "n_excluded", sb, se));
        std::size_t er = find_value_for_key(body, "exclusion_rate", sb, se);
        c.expected_exclusion_rate = parse_hex_double_quoted(body, er);

        fx.cases.push_back(std::move(c));
    }
    return fx;
}

c4a_y_outlier_method_t method_from_string(const std::string& s) {
    if (s == "iqr")        return C4A_Y_OUTLIER_IQR;
    if (s == "zscore")     return C4A_Y_OUTLIER_ZSCORE;
    if (s == "percentile") return C4A_Y_OUTLIER_PERCENTILE;
    if (s == "mad")        return C4A_Y_OUTLIER_MAD;
    throw std::runtime_error("unknown method: " + s);
}

// ---------------------------------------------------------------------------
// Smoke tests (per method).
// ---------------------------------------------------------------------------

void test_y_outlier_iqr_smoke() {
    // Clearly outlying values on both ends.
    double y[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    std::uint8_t mask[10] = {0};
    c4a_filter_stats_t stats{};
    c4a_filter_y_outlier_handle_t* h = nullptr;
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_IQR,
                                                 1.5, 1.0, 99.0) == C4A_OK);
    FILTER_REQUIRE(h != nullptr);
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, 10) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 10, mask, &stats)
                    == C4A_OK);
    // 100 is the outlier; everyone else within bounds.
    FILTER_REQUIRE(stats.n_samples == 10);
    FILTER_REQUIRE(stats.n_excluded == 1);
    FILTER_REQUIRE(stats.n_kept == 9);
    FILTER_REQUIRE(mask[9] == 0);
    for (int i = 0; i < 9; ++i) FILTER_REQUIRE(mask[i] == 1);
    c4a_filter_y_outlier_destroy(h);
    c4a_filter_y_outlier_destroy(nullptr);  // NULL-safe
}

void test_y_outlier_zscore_smoke() {
    // Tight cluster around 0, single far-out value.
    double y[12] = {-1, -0.5, 0, 0.5, 1, -1.2, 0.3, -0.4, 0.7, -0.6, 0.1, 50};
    std::uint8_t mask[12] = {0};
    c4a_filter_stats_t stats{};
    c4a_filter_y_outlier_handle_t* h = nullptr;
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_ZSCORE,
                                                 3.0, 1.0, 99.0) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, 12) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 12, mask, &stats)
                    == C4A_OK);
    FILTER_REQUIRE(stats.n_samples == 12);
    FILTER_REQUIRE(stats.n_excluded >= 1);
    FILTER_REQUIRE(mask[11] == 0);   // y=50 is way outside 3 sigma.
    c4a_filter_y_outlier_destroy(h);
}

void test_y_outlier_percentile_smoke() {
    // Uniform 0..99 with two known outliers.
    double y[100];
    for (int i = 0; i < 100; ++i) y[i] = static_cast<double>(i);
    std::uint8_t mask[100] = {0};
    c4a_filter_stats_t stats{};
    c4a_filter_y_outlier_handle_t* h = nullptr;
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_PERCENTILE,
                                                 1.5, 5.0, 95.0) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, 100) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 100, mask, &stats)
                    == C4A_OK);
    FILTER_REQUIRE(stats.n_samples == 100);
    // 5..95 percentile excludes the tails.
    FILTER_REQUIRE(mask[0] == 0);
    FILTER_REQUIRE(mask[99] == 0);
    FILTER_REQUIRE(mask[50] == 1);
    c4a_filter_y_outlier_destroy(h);
}

void test_y_outlier_mad_smoke() {
    // Robust to a single extreme value.
    double y[11] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000};
    std::uint8_t mask[11] = {0};
    c4a_filter_stats_t stats{};
    c4a_filter_y_outlier_handle_t* h = nullptr;
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_MAD,
                                                 3.5, 1.0, 99.0) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, 11) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 11, mask, &stats)
                    == C4A_OK);
    FILTER_REQUIRE(stats.n_samples == 11);
    FILTER_REQUIRE(stats.n_excluded == 1);  // 1000 is the outlier
    FILTER_REQUIRE(mask[10] == 0);
    c4a_filter_y_outlier_destroy(h);
}

void test_y_outlier_create_param_validation() {
    c4a_filter_y_outlier_handle_t* h = nullptr;
    // Unknown method. Cast a known-bad integer code through the enum type
    // (well-defined per C++17 [expr.static.cast] for unscoped enums whose
    // underlying type contains the bit pattern). -Wconversion is silenced
    // locally because we are intentionally feeding an out-of-range value to
    // verify the create() rejection path.
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#endif
    FILTER_REQUIRE(c4a_filter_y_outlier_create(
        &h, static_cast<c4a_y_outlier_method_t>(99),
        1.5, 1.0, 99.0) == C4A_ERR_INVALID_ARGUMENT);
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
    FILTER_REQUIRE(h == nullptr);
    // Non-positive threshold.
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_IQR,
                                                 0.0, 1.0, 99.0)
                    == C4A_ERR_INVALID_ARGUMENT);
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_IQR,
                                                 -1.0, 1.0, 99.0)
                    == C4A_ERR_INVALID_ARGUMENT);
    // Invalid percentile order.
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_PERCENTILE,
                                                 1.5, 90.0, 10.0)
                    == C4A_ERR_INVALID_ARGUMENT);
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_PERCENTILE,
                                                 1.5, -1.0, 99.0)
                    == C4A_ERR_INVALID_ARGUMENT);
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_PERCENTILE,
                                                 1.5, 1.0, 101.0)
                    == C4A_ERR_INVALID_ARGUMENT);
    // NULL out pointer.
    FILTER_REQUIRE(c4a_filter_y_outlier_create(nullptr, C4A_Y_OUTLIER_IQR,
                                                 1.5, 1.0, 99.0)
                    == C4A_ERR_NULL_POINTER);
}

void test_y_outlier_fit_apply_param_validation() {
    c4a_filter_y_outlier_handle_t* h = nullptr;
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_IQR,
                                                 1.5, 1.0, 99.0) == C4A_OK);
    std::uint8_t mask[3] = {0};
    c4a_filter_stats_t stats{};
    double y[3] = {1, 2, 3};

    // _apply before _fit returns C4A_ERR_NOT_FITTED.
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 3, mask, &stats)
                    == C4A_ERR_NOT_FITTED);
    int fitted = -1;
    FILTER_REQUIRE(c4a_filter_y_outlier_is_fitted(h, &fitted) == C4A_OK);
    FILTER_REQUIRE(fitted == 0);

    // _fit validation: NULL handle.
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(nullptr, y, 3)
                    == C4A_ERR_NULL_POINTER);
    // Negative n.
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, -1)
                    == C4A_ERR_INVALID_ARGUMENT);
    // NULL y with n > 0.
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, nullptr, 3)
                    == C4A_ERR_NULL_POINTER);
    // n == 0 with NULL y is allowed (no-op fit; bounds at +/-inf).
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, nullptr, 0) == C4A_OK);

    // Do a valid fit so subsequent _apply calls succeed.
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, 3) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_is_fitted(h, &fitted) == C4A_OK);
    FILTER_REQUIRE(fitted == 1);

    // _apply validation: NULL handle.
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(nullptr, y, 3, mask, &stats)
                    == C4A_ERR_NULL_POINTER);
    // NULL mask.
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 3, nullptr, &stats)
                    == C4A_ERR_NULL_POINTER);
    // NULL stats.
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 3, mask, nullptr)
                    == C4A_ERR_NULL_POINTER);
    // Negative n.
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, -1, mask, &stats)
                    == C4A_ERR_INVALID_ARGUMENT);
    // NULL y with n > 0.
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, nullptr, 3, mask, &stats)
                    == C4A_ERR_NULL_POINTER);
    // n == 0 with NULL y is allowed.
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, nullptr, 0, mask, &stats)
                    == C4A_OK);
    FILTER_REQUIRE(stats.n_samples == 0);
    FILTER_REQUIRE(stats.n_kept == 0);
    FILTER_REQUIRE(stats.n_excluded == 0);
    FILTER_REQUIRE(stats.exclusion_rate == 0.0);

    // _is_fitted validation.
    FILTER_REQUIRE(c4a_filter_y_outlier_is_fitted(nullptr, &fitted)
                    == C4A_ERR_NULL_POINTER);
    FILTER_REQUIRE(c4a_filter_y_outlier_is_fitted(h, nullptr)
                    == C4A_ERR_NULL_POINTER);

    c4a_filter_y_outlier_destroy(h);
}

void test_y_outlier_nan_exclusion() {
    // NaN values must always be excluded.
    double y[5] = {1.0, std::nan(""), 2.0, std::nan(""), 3.0};
    std::uint8_t mask[5] = {0};
    c4a_filter_stats_t stats{};
    c4a_filter_y_outlier_handle_t* h = nullptr;
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_IQR,
                                                 1.5, 1.0, 99.0) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, 5) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 5, mask, &stats)
                    == C4A_OK);
    FILTER_REQUIRE(stats.n_samples == 5);
    FILTER_REQUIRE(stats.n_excluded == 2);  // the two NaNs
    FILTER_REQUIRE(stats.n_kept == 3);
    FILTER_REQUIRE(mask[1] == 0);
    FILTER_REQUIRE(mask[3] == 0);
    FILTER_REQUIRE(mask[0] == 1);
    FILTER_REQUIRE(mask[2] == 1);
    FILTER_REQUIRE(mask[4] == 1);
    c4a_filter_y_outlier_destroy(h);
}

void test_y_outlier_constant_input() {
    // Constant input: scale == 0, bounds collapse to +/-1e-10.
    double y[8] = {7, 7, 7, 7, 7, 7, 7, 7};
    std::uint8_t mask[8] = {0};
    c4a_filter_stats_t stats{};
    c4a_filter_y_outlier_handle_t* h = nullptr;
    FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, C4A_Y_OUTLIER_ZSCORE,
                                                 3.0, 1.0, 99.0) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, y, 8) == C4A_OK);
    FILTER_REQUIRE(c4a_filter_y_outlier_apply(h, y, 8, mask, &stats)
                    == C4A_OK);
    // All values exactly equal the mean -> all kept.
    FILTER_REQUIRE(stats.n_kept == 8);
    for (int i = 0; i < 8; ++i) FILTER_REQUIRE(mask[i] == 1);
    c4a_filter_y_outlier_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests (per method).
// ---------------------------------------------------------------------------

void verify_y_outlier_parity(const std::string& fixture_filename) {
    const Fixture fx = load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + fixture_filename);
    FILTER_REQUIRE(fx.n == static_cast<std::int64_t>(fx.y.size()));

    for (const auto& c : fx.cases) {
        const std::string method_str = params_get_string(c.params_json,
                                                          "method", "iqr");
        const c4a_y_outlier_method_t method = method_from_string(method_str);
        const double threshold = params_get_double(c.params_json,
                                                    "threshold", 1.5);
        const double lower_pct = params_get_double(c.params_json,
                                                    "lower_percentile", 1.0);
        const double upper_pct = params_get_double(c.params_json,
                                                    "upper_percentile", 99.0);

        c4a_filter_y_outlier_handle_t* h = nullptr;
        FILTER_REQUIRE(c4a_filter_y_outlier_create(&h, method, threshold,
                                                     lower_pct, upper_pct)
                        == C4A_OK);
        std::vector<std::uint8_t> mask(fx.y.size(), 0);
        c4a_filter_stats_t stats{};
        FILTER_REQUIRE(c4a_filter_y_outlier_fit(h, fx.y.data(), fx.n) == C4A_OK);
        FILTER_REQUIRE(c4a_filter_y_outlier_apply(
                          h, fx.y.data(), fx.n, mask.data(), &stats) == C4A_OK);

        const std::string tag = fixture_filename + "/" + c.name;

        // Stats — exact integer equality.
        if (stats.n_samples != c.expected_n_samples) {
            std::ostringstream m;
            m << tag << ": n_samples mismatch got=" << stats.n_samples
              << " want=" << c.expected_n_samples;
            throw std::runtime_error(m.str());
        }
        if (stats.n_kept != c.expected_n_kept) {
            std::ostringstream m;
            m << tag << ": n_kept mismatch got=" << stats.n_kept
              << " want=" << c.expected_n_kept;
            throw std::runtime_error(m.str());
        }
        if (stats.n_excluded != c.expected_n_excluded) {
            std::ostringstream m;
            m << tag << ": n_excluded mismatch got=" << stats.n_excluded
              << " want=" << c.expected_n_excluded;
            throw std::runtime_error(m.str());
        }
        // exclusion_rate — 1e-12 tolerance.
        const double dr = std::fabs(stats.exclusion_rate
                                     - c.expected_exclusion_rate);
        if (dr > 1e-12) {
            std::ostringstream m;
            m << tag << ": exclusion_rate diff=" << dr
              << " (got=" << stats.exclusion_rate
              << " want=" << c.expected_exclusion_rate << ")";
            throw std::runtime_error(m.str());
        }

        // Mask — exact byte equality.
        if (mask.size() != c.expected_mask.size()) {
            std::ostringstream m;
            m << tag << ": mask size mismatch got=" << mask.size()
              << " want=" << c.expected_mask.size();
            throw std::runtime_error(m.str());
        }
        for (std::size_t i = 0; i < mask.size(); ++i) {
            const std::uint8_t got = mask[i];
            const int          want = c.expected_mask[i];
            if (static_cast<int>(got) != want) {
                std::ostringstream m;
                m << tag << ": mask[" << i << "] got=" << static_cast<int>(got)
                  << " want=" << want;
                throw std::runtime_error(m.str());
            }
        }

        c4a_filter_y_outlier_destroy(h);
    }
}

void verify_iqr_parity()        {
    verify_y_outlier_parity("filter_y_outlier_iqr_v1.json");
}
void verify_zscore_parity()     {
    verify_y_outlier_parity("filter_y_outlier_zscore_v1.json");
}
void verify_percentile_parity() {
    verify_y_outlier_parity("filter_y_outlier_percentile_v1.json");
}
void verify_mad_parity()        {
    verify_y_outlier_parity("filter_y_outlier_mad_v1.json");
}

}  // namespace

void register_filters_y_tests(c4a_testing::Runner& r);
void register_filters_y_tests(c4a_testing::Runner& r) {
    r.run("filter_y_outlier_iqr_smoke",            test_y_outlier_iqr_smoke);
    r.run("filter_y_outlier_zscore_smoke",         test_y_outlier_zscore_smoke);
    r.run("filter_y_outlier_percentile_smoke",     test_y_outlier_percentile_smoke);
    r.run("filter_y_outlier_mad_smoke",            test_y_outlier_mad_smoke);
    r.run("filter_y_outlier_create_validation",    test_y_outlier_create_param_validation);
    r.run("filter_y_outlier_fit_apply_validation", test_y_outlier_fit_apply_param_validation);
    r.run("filter_y_outlier_nan_exclusion",        test_y_outlier_nan_exclusion);
    r.run("filter_y_outlier_constant_input",       test_y_outlier_constant_input);
    r.run("filter_y_outlier_iqr_parity",           verify_iqr_parity);
    r.run("filter_y_outlier_zscore_parity",        verify_zscore_parity);
    r.run("filter_y_outlier_percentile_parity",    verify_percentile_parity);
    r.run("filter_y_outlier_mad_parity",           verify_mad_parity);
}
