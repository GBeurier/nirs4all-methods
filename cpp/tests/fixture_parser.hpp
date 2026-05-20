// SPDX-License-Identifier: CECILL-2.1
//
// Shared zero-dependency JSON fixture parser for the parity test suites.
//
// The phase fixtures all follow the same shape:
//
//   {
//     "format": "n4m_pp_*_v1",
//     "rows": <int>, "cols": <int>,
//     "input_hex":          [ "<16 hex chars>", ... ]   // mandatory
//     "fit_rows": <int>, "fit_cols": <int>,             // optional
//     "fit_input_hex":      [ "<16 hex chars>", ... ]   // optional
//     "cases": [
//       { "name": "...",
//         "params": { ... },
//         "output_rows": <int>, "output_cols": <int>,   // optional (only
//                                                       //  required for the
//                                                       //  phase-3 stateful
//                                                       //  fixtures where
//                                                       //  Derivate shrinks
//                                                       //  the output cols)
//         "output_hex":      [ "<16 hex chars>", ... ] },
//       ...
//     ]
//   }
//
// Phases 2-5+ all reuse this header — Phase 5a is the extraction point.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace n4m_testing {

// ---------------------------------------------------------------------------
// Slurp helper — load the full contents of a file into a string.
// ---------------------------------------------------------------------------

inline std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open fixture: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// Whitespace + integer / hex / array primitives.
// ---------------------------------------------------------------------------

inline void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
        ++i;
    }
}

inline std::size_t find_value_for_key(const std::string& s,
                                       const std::string& key,
                                       std::size_t from,
                                       std::size_t end) {
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

inline std::int64_t parse_int64(const std::string& s, std::size_t pos) {
    char* tail = nullptr;
    std::int64_t v = std::strtoll(s.c_str() + pos, &tail, 10);
    if (tail == s.c_str() + pos) {
        throw std::runtime_error("fixture: cannot parse integer");
    }
    return v;
}

inline double hex_to_double(const std::string& s, std::size_t pos) {
    // Decode a JSON-quoted 16-hex-char big-endian IEEE 754 double sitting at
    // `s[pos]` (no advancement — `pos` must point at the opening quote and
    // there must be at least 18 characters left).
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

inline double parse_hex_double(const std::string& s, std::size_t& pos) {
    const double d = hex_to_double(s, pos);
    pos += 1 /* quote */ + 16 /* hex */;
    if (pos >= s.size() || s[pos] != '"') {
        throw std::runtime_error("fixture: missing closing '\"' on hex double");
    }
    ++pos;
    return d;
}

inline void parse_hex_double_array(const std::string& s, std::size_t& pos,
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

// ---------------------------------------------------------------------------
// Cases array — substring extraction.
// ---------------------------------------------------------------------------

inline std::pair<std::size_t, std::size_t>
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

inline std::vector<std::pair<std::size_t, std::size_t>>
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

// ---------------------------------------------------------------------------
// String / value-span helpers (param JSON body).
// ---------------------------------------------------------------------------

inline std::string parse_string(const std::string& s, std::size_t& pos) {
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

inline std::pair<std::size_t, std::size_t>
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

// ---------------------------------------------------------------------------
// Typed `params` getters — best-effort substring lookups.
// ---------------------------------------------------------------------------

inline bool params_get_bool(const std::string& json, const std::string& key,
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

inline double params_get_double(const std::string& json,
                                 const std::string& key,
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

inline std::int64_t params_get_int(const std::string& json,
                                    const std::string& key,
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

inline std::string params_get_string(const std::string& json,
                                      const std::string& key,
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
// High-level loaders + tolerance assertion.
// ---------------------------------------------------------------------------

struct Case {
    std::string         name;
    std::string         params_json;
    std::int64_t        output_rows = 0;
    std::int64_t        output_cols = 0;
    std::vector<double> expected_output;
};

struct Fixture {
    std::int64_t        rows = 0;
    std::int64_t        cols = 0;
    std::vector<double> input;
    // Optional fit_input / fit_rows / fit_cols (Phase 3 stateful fixtures).
    bool                has_fit = false;
    std::int64_t        fit_rows = 0;
    std::int64_t        fit_cols = 0;
    std::vector<double> fit_input;
    std::vector<Case>   cases;
};

inline Fixture load_fixture(const std::string& path,
                             bool require_per_case_output_shape = false) {
    const std::string body = slurp(path);
    Fixture fx;
    fx.rows = parse_int64(body, find_value_for_key(body, "rows", 0, body.size()));
    fx.cols = parse_int64(body, find_value_for_key(body, "cols", 0, body.size()));

    if (body.find("\"fit_rows\"") != std::string::npos) {
        fx.has_fit  = true;
        fx.fit_rows = parse_int64(body, find_value_for_key(body, "fit_rows", 0, body.size()));
        fx.fit_cols = parse_int64(body, find_value_for_key(body, "fit_cols", 0, body.size()));
        std::size_t fit_pos = find_value_for_key(body, "fit_input_hex", 0, body.size());
        parse_hex_double_array(body, fit_pos, fx.fit_input);
    }
    std::size_t input_pos = find_value_for_key(body, "input_hex", 0, body.size());
    parse_hex_double_array(body, input_pos, fx.input);

    auto [cases_lo, cases_hi] = find_cases_array(body);
    auto spans = list_case_object_spans(body, cases_lo, cases_hi);
    for (const auto& [lo, hi] : spans) {
        Case c;
        std::size_t np = find_value_for_key(body, "name", lo, hi);
        c.name = parse_string(body, np);
        std::size_t pp = find_value_for_key(body, "params", lo, hi);
        auto [pb, pe] = read_value_span(body, pp);
        c.params_json = body.substr(pb, pe - pb);
        if (require_per_case_output_shape) {
            c.output_rows = parse_int64(body, find_value_for_key(body, "output_rows", lo, hi));
            c.output_cols = parse_int64(body, find_value_for_key(body, "output_cols", lo, hi));
        }
        std::size_t op = find_value_for_key(body, "output_hex", lo, hi);
        parse_hex_double_array(body, op, c.expected_output);
        fx.cases.push_back(std::move(c));
    }
    return fx;
}

// Tolerance check: byte equality OR (abs <= abs_tol OR rel <= rel_tol).
inline void assert_close(const std::vector<double>& got,
                          const std::vector<double>& want,
                          const std::string& tag,
                          double abs_tol = 1e-12,
                          double rel_tol = 1e-13) {
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

}  // namespace n4m_testing
