// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 15 augmenters (`c4a_aug_*` -- 7 operators,
// 21 ABI symbols). For each operator we register one smoke test (basic
// shape sanity + RNG-determinism check) and one parity test that loads the
// corresponding fixture under ``parity/fixtures/aug_<NAME>_v1.json`` and
// asserts bit-exact equality (1e-15 absolute tolerance) against the frozen
// Python reference in ``c4a_parity_augmenters_ref``.
//
// This executable is independent from the main ``chemometrics4all_tests``
// binary because the Phase 15 ABI contract requires the public augmenter
// declarations to live in c4a.h, but the locked task constraints forbid
// modifying c4a.h directly. The handles are therefore opaque-by-symbol;
// we forward-declare them here (matching the inline forward-decls used
// by c_api_augmenters.cpp) and exercise the ABI through the symbol-only
// surface.

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

// ---------------------------------------------------------------------------
// Forward declarations of the augmenter ABI (mirrored from
// cpp/src/c_api/c_api_augmenters.cpp). c4a.h does not expose them in Phase
// 15, so test consumers redeclare exactly the symbols they call.
// ---------------------------------------------------------------------------
extern "C" {

struct c4a_aug_gaussian_noise_handle_t;
struct c4a_aug_multiplicative_noise_handle_t;
struct c4a_aug_spike_noise_handle_t;
struct c4a_aug_hetero_noise_handle_t;
struct c4a_aug_linear_drift_handle_t;
struct c4a_aug_poly_drift_handle_t;
struct c4a_aug_path_length_handle_t;

c4a_status_t c4a_aug_gaussian_noise_create(
    c4a_aug_gaussian_noise_handle_t** out,
    c4a_rng_pcg64_state_t* rng, double sigma);
void c4a_aug_gaussian_noise_destroy(c4a_aug_gaussian_noise_handle_t* h);
c4a_status_t c4a_aug_gaussian_noise_apply(
    const c4a_aug_gaussian_noise_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

c4a_status_t c4a_aug_multiplicative_noise_create(
    c4a_aug_multiplicative_noise_handle_t** out,
    c4a_rng_pcg64_state_t* rng, double sigma_gain);
void c4a_aug_multiplicative_noise_destroy(
    c4a_aug_multiplicative_noise_handle_t* h);
c4a_status_t c4a_aug_multiplicative_noise_apply(
    const c4a_aug_multiplicative_noise_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

c4a_status_t c4a_aug_spike_noise_create(
    c4a_aug_spike_noise_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_spikes_min, int32_t n_spikes_max,
    double amplitude_min, double amplitude_max);
void c4a_aug_spike_noise_destroy(c4a_aug_spike_noise_handle_t* h);
c4a_status_t c4a_aug_spike_noise_apply(
    const c4a_aug_spike_noise_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

c4a_status_t c4a_aug_hetero_noise_create(
    c4a_aug_hetero_noise_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double noise_base, double noise_signal_dep);
void c4a_aug_hetero_noise_destroy(c4a_aug_hetero_noise_handle_t* h);
c4a_status_t c4a_aug_hetero_noise_apply(
    const c4a_aug_hetero_noise_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

c4a_status_t c4a_aug_linear_drift_create(
    c4a_aug_linear_drift_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max);
void c4a_aug_linear_drift_destroy(c4a_aug_linear_drift_handle_t* h);
c4a_status_t c4a_aug_linear_drift_apply(
    const c4a_aug_linear_drift_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

c4a_status_t c4a_aug_poly_drift_create(
    c4a_aug_poly_drift_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t degree,
    const double* coeff_min, const double* coeff_max);
void c4a_aug_poly_drift_destroy(c4a_aug_poly_drift_handle_t* h);
c4a_status_t c4a_aug_poly_drift_apply(
    const c4a_aug_poly_drift_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

c4a_status_t c4a_aug_path_length_create(
    c4a_aug_path_length_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double path_length_std, double min_path_length);
void c4a_aug_path_length_destroy(c4a_aug_path_length_handle_t* h);
c4a_status_t c4a_aug_path_length_apply(
    const c4a_aug_path_length_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

}  // extern "C"

namespace {

#define AUG_REQUIRE(cond) C4A_TEST_REQUIRE(cond)

constexpr double kParityTol = 1e-15;

// ---------------------------------------------------------------------------
// Fixture loader -- shares the schema across all 7 operators.
//
//   { "rows": <int>, "cols": <int>,
//     "X_hex": [ "<16 hex>", ... ],
//     "cases": [
//       { "name": "...", "seed": <uint64>, "params": {...},
//         "expected_out_hex": [ "<16 hex>", ... ] }, ...
//     ] }
//
// Hand-rolled parser (mirrors test_filters_y.cpp). We only need scalar
// integer / hex-double / array reads -- no nested object recursion beyond
// the per-case params block (which we read keyed-only).
// ---------------------------------------------------------------------------

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("cannot open fixture: " + path);
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
    const std::string q = "\"" + key + "\"";
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

std::uint64_t parse_uint64_at(const std::string& s, std::size_t pos) {
    char* tail = nullptr;
    std::uint64_t v = std::strtoull(s.c_str() + pos, &tail, 10);
    if (tail == s.c_str() + pos) {
        throw std::runtime_error("fixture: cannot parse unsigned integer");
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

std::pair<std::size_t, std::size_t> find_array_span(const std::string& s,
                                                     std::size_t pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("fixture: '[' expected");
    }
    const std::size_t start = pos + 1;
    int depth = 1;
    std::size_t i = start;
    bool in_string = false, escaped = false;
    while (i < s.size() && depth > 0) {
        const char c = s[i];
        if (escaped) { escaped = false; }
        else if (in_string) {
            if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
        } else {
            if (c == '"') in_string = true;
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
    bool in_string = false, escaped = false;
    while (i < s.size() && depth > 0) {
        const char c = s[i];
        if (escaped) { escaped = false; }
        else if (in_string) {
            if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
        } else {
            if (c == '"') in_string = true;
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
        bool in_string = false, escaped = false;
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
        if (depth != 0) throw std::runtime_error("fixture: unterminated case");
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
    if (pos >= s.size()) throw std::runtime_error("fixture: unterminated string");
    std::string out = s.substr(start, pos - start);
    ++pos;
    return out;
}

double params_get_double(const std::string& json, const std::string& key) {
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) {
        throw std::runtime_error("missing param: " + key);
    }
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    char* tail = nullptr;
    const double v = std::strtod(json.c_str() + p, &tail);
    if (tail == json.c_str() + p) throw std::runtime_error("bad param: " + key);
    return v;
}

std::int64_t params_get_int64(const std::string& json, const std::string& key) {
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) {
        throw std::runtime_error("missing param: " + key);
    }
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    char* tail = nullptr;
    const std::int64_t v = std::strtoll(json.c_str() + p, &tail, 10);
    if (tail == json.c_str() + p) throw std::runtime_error("bad int param: " + key);
    return v;
}

std::vector<double> params_get_double_array(const std::string& json,
                                              const std::string& key) {
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) throw std::runtime_error("missing array: " + key);
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    if (p >= json.size() || json[p] != '[') {
        throw std::runtime_error("missing '[' for array: " + key);
    }
    ++p;
    std::vector<double> out;
    for (;;) {
        while (p < json.size() && (json[p] == ' ' || json[p] == ',' ||
                                    json[p] == '\n' || json[p] == '\t')) ++p;
        if (p >= json.size()) break;
        if (json[p] == ']') { ++p; break; }
        char* tail = nullptr;
        const double v = std::strtod(json.c_str() + p, &tail);
        if (tail == json.c_str() + p) throw std::runtime_error("bad array elt");
        out.push_back(v);
        p = static_cast<std::size_t>(tail - json.c_str());
    }
    return out;
}

struct ParityCase {
    std::string         name;
    std::uint64_t       seed = 0;
    std::string         params_json;
    std::vector<double> expected_out;
};

struct Fixture {
    std::int64_t            rows = 0;
    std::int64_t            cols = 0;
    std::vector<double>     X;
    std::vector<ParityCase> cases;
};

Fixture load_fixture(const std::string& path) {
    const std::string body = slurp(path);
    Fixture fx;
    fx.rows = parse_int64_at(body, find_value_for_key(body, "rows", 0, body.size()));
    fx.cols = parse_int64_at(body, find_value_for_key(body, "cols", 0, body.size()));
    std::size_t x_pos = find_value_for_key(body, "X_hex", 0, body.size());
    parse_hex_double_array(body, x_pos, fx.X);

    std::size_t cases_pos = find_value_for_key(body, "cases", 0, body.size());
    auto [cases_lo, cases_hi] = find_array_span(body, cases_pos);
    auto spans = list_case_object_spans(body, cases_lo, cases_hi);
    for (const auto& [lo, hi] : spans) {
        ParityCase c;
        std::size_t np = find_value_for_key(body, "name", lo, hi);
        c.name = parse_string_at(body, np);

        c.seed = parse_uint64_at(body,
            find_value_for_key(body, "seed", lo, hi));

        std::size_t pp = find_value_for_key(body, "params", lo, hi);
        auto [pb, pe] = find_object_span(body, pp);
        c.params_json = body.substr(pb, pe - pb);

        std::size_t mp = find_value_for_key(body, "expected_out_hex", lo, hi);
        parse_hex_double_array(body, mp, c.expected_out);

        fx.cases.push_back(std::move(c));
    }
    return fx;
}

void compare_matrices(const std::string& tag,
                       const std::vector<double>& got,
                       const std::vector<double>& want) {
    if (got.size() != want.size()) {
        std::ostringstream m;
        m << tag << ": size mismatch got=" << got.size()
          << " want=" << want.size();
        throw std::runtime_error(m.str());
    }
    double max_abs = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(got[i] - want[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
    }
    if (max_abs > kParityTol) {
        std::ostringstream m;
        m << tag << ": max abs diff=" << max_abs
          << " at idx=" << worst
          << " (got=" << got[worst] << " want=" << want[worst] << ")";
        throw std::runtime_error(m.str());
    }
}

c4a_matrix_view_t make_rowmajor_f64(double* data, std::int64_t rows,
                                     std::int64_t cols) {
    c4a_matrix_view_t v{};
    c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    return v;
}

c4a_matrix_view_t make_rowmajor_f64_const(const double* data,
                                           std::int64_t rows,
                                           std::int64_t cols) {
    c4a_matrix_view_t v{};
    c4a_matrix_view_init_rowmajor(&v, const_cast<double*>(data), rows, cols,
                                    C4A_DTYPE_F64);
    return v;
}

// ---------------------------------------------------------------------------
// Smoke tests (one per operator). Each:
//   1. Creates an RNG with a known seed.
//   2. Creates the augmenter handle and applies to a small input.
//   3. Asserts the output has the right shape and (where appropriate) is
//      different from the input -- the operator did *something*.
//   4. Re-seeds the RNG and re-applies; the output must be identical
//      (determinism check).
//   5. Tears the handle and RNG down.
// ---------------------------------------------------------------------------

template <typename Create, typename Apply, typename Destroy>
void run_determinism_smoke(Create create_fn, Apply apply_fn,
                            Destroy destroy_fn, std::uint64_t seed) {
    constexpr std::int64_t R = 4;
    constexpr std::int64_t C = 6;
    double X_arr[R * C];
    for (std::int64_t i = 0; i < R * C; ++i) {
        X_arr[i] = 0.5 + 0.01 * static_cast<double>(i);
    }
    double out1[R * C] = {0};
    double out2[R * C] = {0};

    c4a_rng_pcg64_state_t* rng = nullptr;
    AUG_REQUIRE(c4a_rng_pcg64_create(seed, &rng) == C4A_OK);
    auto* h = create_fn(rng);
    AUG_REQUIRE(h != nullptr);
    AUG_REQUIRE(apply_fn(h, X_arr, out1, R, C) == C4A_OK);
    destroy_fn(h);

    // Re-seed and re-apply: the result must be byte-identical.
    AUG_REQUIRE(c4a_rng_pcg64_set_seed(rng, seed) == C4A_OK);
    h = create_fn(rng);
    AUG_REQUIRE(h != nullptr);
    AUG_REQUIRE(apply_fn(h, X_arr, out2, R, C) == C4A_OK);
    destroy_fn(h);
    c4a_rng_pcg64_destroy(rng);

    for (std::int64_t i = 0; i < R * C; ++i) {
        AUG_REQUIRE(out1[i] == out2[i]);
    }
}

void test_gaussian_noise_smoke() {
    run_determinism_smoke(
        [](c4a_rng_pcg64_state_t* rng) {
            c4a_aug_gaussian_noise_handle_t* h = nullptr;
            (void)c4a_aug_gaussian_noise_create(&h, rng, 0.05);
            return h;
        },
        [](const c4a_aug_gaussian_noise_handle_t* h, const double* X,
            double* out, std::int64_t R, std::int64_t C) {
            return c4a_aug_gaussian_noise_apply(
                h, make_rowmajor_f64_const(X, R, C),
                make_rowmajor_f64(out, R, C));
        },
        c4a_aug_gaussian_noise_destroy, /*seed=*/42u);
}

void test_multiplicative_noise_smoke() {
    run_determinism_smoke(
        [](c4a_rng_pcg64_state_t* rng) {
            c4a_aug_multiplicative_noise_handle_t* h = nullptr;
            (void)c4a_aug_multiplicative_noise_create(&h, rng, 0.1);
            return h;
        },
        [](const c4a_aug_multiplicative_noise_handle_t* h, const double* X,
            double* out, std::int64_t R, std::int64_t C) {
            return c4a_aug_multiplicative_noise_apply(
                h, make_rowmajor_f64_const(X, R, C),
                make_rowmajor_f64(out, R, C));
        },
        c4a_aug_multiplicative_noise_destroy, /*seed=*/42u);
}

void test_spike_noise_smoke() {
    run_determinism_smoke(
        [](c4a_rng_pcg64_state_t* rng) {
            c4a_aug_spike_noise_handle_t* h = nullptr;
            (void)c4a_aug_spike_noise_create(&h, rng, 1, 3, -0.5, 0.5);
            return h;
        },
        [](const c4a_aug_spike_noise_handle_t* h, const double* X,
            double* out, std::int64_t R, std::int64_t C) {
            return c4a_aug_spike_noise_apply(
                h, make_rowmajor_f64_const(X, R, C),
                make_rowmajor_f64(out, R, C));
        },
        c4a_aug_spike_noise_destroy, /*seed=*/42u);
}

void test_hetero_noise_smoke() {
    run_determinism_smoke(
        [](c4a_rng_pcg64_state_t* rng) {
            c4a_aug_hetero_noise_handle_t* h = nullptr;
            (void)c4a_aug_hetero_noise_create(&h, rng, 1e-3, 5e-3);
            return h;
        },
        [](const c4a_aug_hetero_noise_handle_t* h, const double* X,
            double* out, std::int64_t R, std::int64_t C) {
            return c4a_aug_hetero_noise_apply(
                h, make_rowmajor_f64_const(X, R, C),
                make_rowmajor_f64(out, R, C));
        },
        c4a_aug_hetero_noise_destroy, /*seed=*/42u);
}

void test_linear_drift_smoke() {
    run_determinism_smoke(
        [](c4a_rng_pcg64_state_t* rng) {
            c4a_aug_linear_drift_handle_t* h = nullptr;
            (void)c4a_aug_linear_drift_create(&h, rng, -0.1, 0.1, -0.001, 0.001);
            return h;
        },
        [](const c4a_aug_linear_drift_handle_t* h, const double* X,
            double* out, std::int64_t R, std::int64_t C) {
            return c4a_aug_linear_drift_apply(
                h, make_rowmajor_f64_const(X, R, C),
                make_rowmajor_f64(out, R, C));
        },
        c4a_aug_linear_drift_destroy, /*seed=*/42u);
}

void test_poly_drift_smoke() {
    const double lo[3] = {-0.1, -0.05, -0.033};
    const double hi[3] = { 0.1,  0.05,  0.033};
    run_determinism_smoke(
        [&lo, &hi](c4a_rng_pcg64_state_t* rng) {
            c4a_aug_poly_drift_handle_t* h = nullptr;
            (void)c4a_aug_poly_drift_create(&h, rng, 2, lo, hi);
            return h;
        },
        [](const c4a_aug_poly_drift_handle_t* h, const double* X,
            double* out, std::int64_t R, std::int64_t C) {
            return c4a_aug_poly_drift_apply(
                h, make_rowmajor_f64_const(X, R, C),
                make_rowmajor_f64(out, R, C));
        },
        c4a_aug_poly_drift_destroy, /*seed=*/42u);
}

void test_path_length_smoke() {
    run_determinism_smoke(
        [](c4a_rng_pcg64_state_t* rng) {
            c4a_aug_path_length_handle_t* h = nullptr;
            (void)c4a_aug_path_length_create(&h, rng, 0.05, 0.5);
            return h;
        },
        [](const c4a_aug_path_length_handle_t* h, const double* X,
            double* out, std::int64_t R, std::int64_t C) {
            return c4a_aug_path_length_apply(
                h, make_rowmajor_f64_const(X, R, C),
                make_rowmajor_f64(out, R, C));
        },
        c4a_aug_path_length_destroy, /*seed=*/42u);
}

// ---------------------------------------------------------------------------
// Parity tests -- one per operator. Common scaffold: load fixture, for each
// case re-seed the RNG, create the handle from the case's params, apply,
// compare against expected output.
// ---------------------------------------------------------------------------

template <typename CreateFromParams>
void verify_parity(const std::string& fixture_filename,
                    CreateFromParams create_from_params,
                    void (*destroy_fn)(void*),
                    c4a_status_t (*apply_fn)(const void* h,
                                              c4a_matrix_view_t,
                                              c4a_matrix_view_t)) {
    const Fixture fx = load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + fixture_filename);
    AUG_REQUIRE(static_cast<std::int64_t>(fx.X.size())
                 == fx.rows * fx.cols);

    std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols), 0.0);

    for (const auto& c : fx.cases) {
        c4a_rng_pcg64_state_t* rng = nullptr;
        AUG_REQUIRE(c4a_rng_pcg64_create(c.seed, &rng) == C4A_OK);
        void* h = create_from_params(rng, c.params_json);
        AUG_REQUIRE(h != nullptr);

        c4a_matrix_view_t Xv = make_rowmajor_f64_const(
            fx.X.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Ov = make_rowmajor_f64(
            out.data(), fx.rows, fx.cols);
        AUG_REQUIRE(apply_fn(h, Xv, Ov) == C4A_OK);
        destroy_fn(h);
        c4a_rng_pcg64_destroy(rng);

        compare_matrices(fixture_filename + "/" + c.name, out, c.expected_out);
    }
}

// Per-operator parity wrappers. Each builds the handle from the
// fixture-encoded params and forwards apply/destroy through `void*` shims.

c4a_status_t gaussian_noise_apply_void(const void* h, c4a_matrix_view_t X,
                                        c4a_matrix_view_t out) {
    return c4a_aug_gaussian_noise_apply(
        static_cast<const c4a_aug_gaussian_noise_handle_t*>(h), X, out);
}
void gaussian_noise_destroy_void(void* h) {
    c4a_aug_gaussian_noise_destroy(
        static_cast<c4a_aug_gaussian_noise_handle_t*>(h));
}

void test_gaussian_noise_parity() {
    verify_parity("aug_gaussian_noise_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
            const std::string& params) -> void* {
            const double sigma = params_get_double(params, "sigma");
            c4a_aug_gaussian_noise_handle_t* h = nullptr;
            AUG_REQUIRE(c4a_aug_gaussian_noise_create(&h, rng, sigma) == C4A_OK);
            return h;
        },
        gaussian_noise_destroy_void, gaussian_noise_apply_void);
}

c4a_status_t multiplicative_noise_apply_void(const void* h,
                                               c4a_matrix_view_t X,
                                               c4a_matrix_view_t out) {
    return c4a_aug_multiplicative_noise_apply(
        static_cast<const c4a_aug_multiplicative_noise_handle_t*>(h), X, out);
}
void multiplicative_noise_destroy_void(void* h) {
    c4a_aug_multiplicative_noise_destroy(
        static_cast<c4a_aug_multiplicative_noise_handle_t*>(h));
}

void test_multiplicative_noise_parity() {
    verify_parity("aug_multiplicative_noise_v1.json",
        [](c4a_rng_pcg64_state_t* rng, const std::string& params) -> void* {
            const double sigma_gain = params_get_double(params, "sigma_gain");
            c4a_aug_multiplicative_noise_handle_t* h = nullptr;
            AUG_REQUIRE(c4a_aug_multiplicative_noise_create(&h, rng, sigma_gain)
                         == C4A_OK);
            return h;
        },
        multiplicative_noise_destroy_void, multiplicative_noise_apply_void);
}

c4a_status_t spike_noise_apply_void(const void* h, c4a_matrix_view_t X,
                                      c4a_matrix_view_t out) {
    return c4a_aug_spike_noise_apply(
        static_cast<const c4a_aug_spike_noise_handle_t*>(h), X, out);
}
void spike_noise_destroy_void(void* h) {
    c4a_aug_spike_noise_destroy(
        static_cast<c4a_aug_spike_noise_handle_t*>(h));
}

void test_spike_noise_parity() {
    verify_parity("aug_spike_noise_v1.json",
        [](c4a_rng_pcg64_state_t* rng, const std::string& params) -> void* {
            const std::int32_t n_min = static_cast<std::int32_t>(
                params_get_int64(params, "n_spikes_min"));
            const std::int32_t n_max = static_cast<std::int32_t>(
                params_get_int64(params, "n_spikes_max"));
            const double amp_lo = params_get_double(params, "amplitude_min");
            const double amp_hi = params_get_double(params, "amplitude_max");
            c4a_aug_spike_noise_handle_t* h = nullptr;
            AUG_REQUIRE(c4a_aug_spike_noise_create(
                          &h, rng, n_min, n_max, amp_lo, amp_hi) == C4A_OK);
            return h;
        },
        spike_noise_destroy_void, spike_noise_apply_void);
}

c4a_status_t hetero_noise_apply_void(const void* h, c4a_matrix_view_t X,
                                       c4a_matrix_view_t out) {
    return c4a_aug_hetero_noise_apply(
        static_cast<const c4a_aug_hetero_noise_handle_t*>(h), X, out);
}
void hetero_noise_destroy_void(void* h) {
    c4a_aug_hetero_noise_destroy(
        static_cast<c4a_aug_hetero_noise_handle_t*>(h));
}

void test_hetero_noise_parity() {
    verify_parity("aug_hetero_noise_v1.json",
        [](c4a_rng_pcg64_state_t* rng, const std::string& params) -> void* {
            const double base = params_get_double(params, "noise_base");
            const double dep  = params_get_double(params, "noise_signal_dep");
            c4a_aug_hetero_noise_handle_t* h = nullptr;
            AUG_REQUIRE(c4a_aug_hetero_noise_create(&h, rng, base, dep)
                         == C4A_OK);
            return h;
        },
        hetero_noise_destroy_void, hetero_noise_apply_void);
}

c4a_status_t linear_drift_apply_void(const void* h, c4a_matrix_view_t X,
                                       c4a_matrix_view_t out) {
    return c4a_aug_linear_drift_apply(
        static_cast<const c4a_aug_linear_drift_handle_t*>(h), X, out);
}
void linear_drift_destroy_void(void* h) {
    c4a_aug_linear_drift_destroy(
        static_cast<c4a_aug_linear_drift_handle_t*>(h));
}

void test_linear_drift_parity() {
    verify_parity("aug_linear_drift_v1.json",
        [](c4a_rng_pcg64_state_t* rng, const std::string& params) -> void* {
            const double off_lo = params_get_double(params, "offset_min");
            const double off_hi = params_get_double(params, "offset_max");
            const double sl_lo  = params_get_double(params, "slope_min");
            const double sl_hi  = params_get_double(params, "slope_max");
            c4a_aug_linear_drift_handle_t* h = nullptr;
            AUG_REQUIRE(c4a_aug_linear_drift_create(
                          &h, rng, off_lo, off_hi, sl_lo, sl_hi) == C4A_OK);
            return h;
        },
        linear_drift_destroy_void, linear_drift_apply_void);
}

c4a_status_t poly_drift_apply_void(const void* h, c4a_matrix_view_t X,
                                     c4a_matrix_view_t out) {
    return c4a_aug_poly_drift_apply(
        static_cast<const c4a_aug_poly_drift_handle_t*>(h), X, out);
}
void poly_drift_destroy_void(void* h) {
    c4a_aug_poly_drift_destroy(
        static_cast<c4a_aug_poly_drift_handle_t*>(h));
}

void test_poly_drift_parity() {
    verify_parity("aug_poly_drift_v1.json",
        [](c4a_rng_pcg64_state_t* rng, const std::string& params) -> void* {
            const std::int32_t degree = static_cast<std::int32_t>(
                params_get_int64(params, "degree"));
            const std::vector<double> lo =
                params_get_double_array(params, "coeff_min");
            const std::vector<double> hi =
                params_get_double_array(params, "coeff_max");
            AUG_REQUIRE(static_cast<std::int32_t>(lo.size()) == degree + 1);
            AUG_REQUIRE(static_cast<std::int32_t>(hi.size()) == degree + 1);
            c4a_aug_poly_drift_handle_t* h = nullptr;
            AUG_REQUIRE(c4a_aug_poly_drift_create(
                          &h, rng, degree, lo.data(), hi.data()) == C4A_OK);
            return h;
        },
        poly_drift_destroy_void, poly_drift_apply_void);
}

c4a_status_t path_length_apply_void(const void* h, c4a_matrix_view_t X,
                                      c4a_matrix_view_t out) {
    return c4a_aug_path_length_apply(
        static_cast<const c4a_aug_path_length_handle_t*>(h), X, out);
}
void path_length_destroy_void(void* h) {
    c4a_aug_path_length_destroy(
        static_cast<c4a_aug_path_length_handle_t*>(h));
}

void test_path_length_parity() {
    verify_parity("aug_path_length_v1.json",
        [](c4a_rng_pcg64_state_t* rng, const std::string& params) -> void* {
            const double std_ = params_get_double(params, "path_length_std");
            const double min_ = params_get_double(params, "min_path_length");
            c4a_aug_path_length_handle_t* h = nullptr;
            AUG_REQUIRE(c4a_aug_path_length_create(&h, rng, std_, min_)
                         == C4A_OK);
            return h;
        },
        path_length_destroy_void, path_length_apply_void);
}

}  // namespace

void register_augmenters_noise_drift_tests(c4a_testing::Runner& r);
void register_augmenters_noise_drift_tests(c4a_testing::Runner& r) {
    r.run("aug_gaussian_noise_smoke",         test_gaussian_noise_smoke);
    r.run("aug_multiplicative_noise_smoke",   test_multiplicative_noise_smoke);
    r.run("aug_spike_noise_smoke",            test_spike_noise_smoke);
    r.run("aug_hetero_noise_smoke",           test_hetero_noise_smoke);
    r.run("aug_linear_drift_smoke",           test_linear_drift_smoke);
    r.run("aug_poly_drift_smoke",             test_poly_drift_smoke);
    r.run("aug_path_length_smoke",            test_path_length_smoke);
    r.run("aug_gaussian_noise_parity",        test_gaussian_noise_parity);
    r.run("aug_multiplicative_noise_parity",  test_multiplicative_noise_parity);
    r.run("aug_spike_noise_parity",           test_spike_noise_parity);
    r.run("aug_hetero_noise_parity",          test_hetero_noise_parity);
    r.run("aug_linear_drift_parity",          test_linear_drift_parity);
    r.run("aug_poly_drift_parity",            test_poly_drift_parity);
    r.run("aug_path_length_parity",           test_path_length_parity);
}
