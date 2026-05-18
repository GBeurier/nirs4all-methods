// SPDX-License-Identifier: CECILL-2.1
//
// Bit-exact parity tests for the PCG64 RNG against NumPy's default_rng.
//
// The fixture lives at parity/fixtures/_rng_pcg64_stream_v1.json (path
// passed in via the -DC4A_PARITY_FIXTURE_DIR compile definition). For each
// seed in {0, 1, 42, 12345, 0xDEADBEEF} the fixture stores 4096 uint64
// draws and 4096 IEEE-754 binary64 doubles (encoded as big-endian hex to
// avoid JSON float-formatting precision loss). The C engine must match
// byte-for-byte on uint64 and bit-for-bit on standard_normal.
//
// Parser strategy: the fixture's JSON schema is small and stable, so we
// hand-roll a token-scanner that walks the file without dependencies. Any
// schema drift fails fast with a clear assertion.
//
// The standard_normal stream is captured from a *fresh* seeded generator
// (NumPy resets the PCG64 sub-state when reseeding) — see the fixture
// generator for details.

#include <cstdint>
#include <cstdio>
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
#  error "C4A_PARITY_FIXTURE_DIR must be defined to locate the fixture file"
#endif

namespace {

constexpr const char* kSeedKeys[] = {"0", "1", "42", "12345", "3735928559"};
constexpr int         kNumSeeds   = sizeof(kSeedKeys) / sizeof(kSeedKeys[0]);
constexpr std::size_t kNumSamples = 4096;

// ---------------------------------------------------------------------------
// Hand-rolled fixture parser. Scans for `"<seed>"` keys then extracts the
// uint64 and standard_normal arrays as flat vectors. Whitespace-insensitive,
// rejects any other malformed token configurations explicitly.
// ---------------------------------------------------------------------------

struct SeedData {
    std::vector<std::uint64_t> uint64_draws;
    std::vector<double>        normal_draws;
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

// Locate the array contents for `"key" : [ ... ]` starting at or after `from`.
// Returns the pair (start_of_first_element, one_past_close_bracket). Throws on
// failure. The search is scoped to the substring [from, end_pos).
std::pair<std::size_t, std::size_t>
find_array_for_key(const std::string& s, const std::string& key,
                   std::size_t from, std::size_t end_pos) {
    std::string quoted = "\"" + key + "\"";
    std::size_t pos    = s.find(quoted, from);
    if (pos == std::string::npos || pos >= end_pos) {
        throw std::runtime_error("fixture missing key: " + key);
    }
    pos += quoted.size();
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != ':') {
        throw std::runtime_error("fixture: ':' expected after key " + key);
    }
    ++pos;
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("fixture: '[' expected for key " + key);
    }
    ++pos;
    std::size_t close = s.find(']', pos);
    if (close == std::string::npos) {
        throw std::runtime_error("fixture: unterminated array for key " + key);
    }
    return {pos, close};
}

// Parse a uint64 array, comma-separated, possibly with whitespace.
std::vector<std::uint64_t> parse_uint64_array(const std::string& s,
                                               std::size_t begin,
                                               std::size_t end) {
    std::vector<std::uint64_t> out;
    out.reserve(kNumSamples);
    std::size_t i = begin;
    while (i < end) {
        skip_ws(s, i);
        if (i >= end) break;
        // strtoull handles any decimal uint64; the fixture writes them as
        // unsigned decimal integers without sign or 0x prefix.
        char* tail = nullptr;
        std::uint64_t v = std::strtoull(s.c_str() + i, &tail, 10);
        if (tail == s.c_str() + i) {
            throw std::runtime_error("fixture: cannot parse uint64");
        }
        out.push_back(v);
        i = static_cast<std::size_t>(tail - s.c_str());
        skip_ws(s, i);
        if (i < end && s[i] == ',') ++i;
    }
    return out;
}

// Parse a string array of big-endian hex-encoded doubles (16 hex chars per
// double, wrapped in double quotes). The encoding is documented in
// parity/python_generator/scripts/generate_rng_fixture.py.
std::vector<double> parse_double_hex_array(const std::string& s,
                                            std::size_t begin,
                                            std::size_t end) {
    std::vector<double> out;
    out.reserve(kNumSamples);
    std::size_t i = begin;
    while (i < end) {
        skip_ws(s, i);
        if (i >= end) break;
        if (s[i] != '"') {
            throw std::runtime_error(
                "fixture: expected '\"' starting hex double");
        }
        ++i;
        if (i + 16 > end) {
            throw std::runtime_error("fixture: truncated hex double");
        }
        // Decode 8 big-endian bytes from the 16-char hex run, then memcpy
        // them into a double. memcpy is the canonical strict-aliasing-safe
        // bit reinterpretation.
        std::uint8_t bytes[8];
        for (std::size_t b = 0; b < 8; ++b) {
            auto hex_nibble = [&](char c) -> std::uint8_t {
                if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
                if (c >= 'a' && c <= 'f')
                    return static_cast<std::uint8_t>(10 + c - 'a');
                if (c >= 'A' && c <= 'F')
                    return static_cast<std::uint8_t>(10 + c - 'A');
                throw std::runtime_error("fixture: bad hex digit");
            };
            std::uint8_t hi = hex_nibble(s[i + 2 * b]);
            std::uint8_t lo = hex_nibble(s[i + 2 * b + 1]);
            bytes[b]        = static_cast<std::uint8_t>((hi << 4) | lo);
        }
        // Big-endian bytes -> uint64 -> double via memcpy. Reverse the byte
        // order so the resulting uint64's native value matches the original
        // little-endian-stored double bit pattern of the host on x86_64.
        std::uint64_t bits = 0;
        for (std::size_t b = 0; b < 8; ++b) {
            bits = (bits << 8) | bytes[b];
        }
        double d;
        std::memcpy(&d, &bits, sizeof(d));
        out.push_back(d);
        i += 16;
        if (i >= end || s[i] != '"') {
            throw std::runtime_error(
                "fixture: missing closing '\"' on hex double");
        }
        ++i;
        skip_ws(s, i);
        if (i < end && s[i] == ',') ++i;
    }
    return out;
}

SeedData load_seed(const std::string& s, const std::string& seed_key) {
    // Locate the seed object: `"<seed>": { ... }`. We constrain the inner
    // searches to the bounds of this object so adjacent seeds can't be
    // mixed up.
    std::string quoted   = "\"" + seed_key + "\"";
    std::size_t key_pos  = s.find(quoted);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("fixture missing seed: " + seed_key);
    }
    std::size_t obj_open  = s.find('{', key_pos);
    if (obj_open == std::string::npos) {
        throw std::runtime_error("fixture: '{' missing for seed " + seed_key);
    }
    // Find matching '}' by depth-counting (the inner object has no nested
    // braces, but the counter keeps us safe if the schema ever grows).
    std::size_t obj_close = obj_open + 1;
    int depth = 1;
    while (obj_close < s.size() && depth > 0) {
        if (s[obj_close] == '{') ++depth;
        else if (s[obj_close] == '}') --depth;
        if (depth == 0) break;
        ++obj_close;
    }
    if (depth != 0) {
        throw std::runtime_error("fixture: '}' missing for seed " + seed_key);
    }

    auto [u_begin, u_end] =
        find_array_for_key(s, "uint64", obj_open, obj_close);
    auto [n_begin, n_end] =
        find_array_for_key(s, "standard_normal", obj_open, obj_close);

    SeedData d;
    d.uint64_draws = parse_uint64_array(s, u_begin, u_end);
    d.normal_draws = parse_double_hex_array(s, n_begin, n_end);
    return d;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

std::string fixture_path() {
    static const std::string p =
        std::string(C4A_PARITY_FIXTURE_DIR) + "/_rng_pcg64_stream_v1.json";
    return p;
}

const SeedData& cached_seed(const char* key) {
    static std::string body = slurp(fixture_path());
    // Lazily decode each seed once per process.
    static SeedData    cache[kNumSeeds];
    static bool        loaded[kNumSeeds] = {false, false, false, false, false};
    for (int i = 0; i < kNumSeeds; ++i) {
        if (std::strcmp(key, kSeedKeys[i]) == 0) {
            if (!loaded[i]) {
                cache[i]  = load_seed(body, kSeedKeys[i]);
                loaded[i] = true;
            }
            return cache[i];
        }
    }
    throw std::runtime_error("unknown seed key");
}

void verify_uint64_seed(const char* key) {
    const SeedData& d = cached_seed(key);
    C4A_TEST_REQUIRE(d.uint64_draws.size() == kNumSamples);

    std::uint64_t seed = std::strtoull(key, nullptr, 10);
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(seed, &rng) == C4A_OK);
    C4A_TEST_REQUIRE(rng != nullptr);

    std::vector<std::uint64_t> got(kNumSamples);
    C4A_TEST_REQUIRE(c4a_rng_pcg64_uint64_fill(rng, got.data(), kNumSamples)
                     == C4A_OK);

    for (std::size_t i = 0; i < kNumSamples; ++i) {
        if (got[i] != d.uint64_draws[i]) {
            std::ostringstream msg;
            msg << "uint64 mismatch seed=" << key << " i=" << i
                << " got=0x" << std::hex << got[i]
                << " want=0x" << d.uint64_draws[i];
            throw std::runtime_error(msg.str());
        }
    }

    c4a_rng_pcg64_destroy(rng);
}

void verify_standard_normal_seed(const char* key) {
    const SeedData& d = cached_seed(key);
    C4A_TEST_REQUIRE(d.normal_draws.size() == kNumSamples);

    std::uint64_t seed = std::strtoull(key, nullptr, 10);
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(seed, &rng) == C4A_OK);
    C4A_TEST_REQUIRE(rng != nullptr);

    std::vector<double> got(kNumSamples);
    C4A_TEST_REQUIRE(c4a_rng_pcg64_standard_normal_fill(rng, got.data(),
                                                          kNumSamples)
                     == C4A_OK);

    // Bit-exact: same hex pattern, identical FP representation. We assert
    // memcmp equality on the raw bytes — there's no ULP tolerance to grant.
    for (std::size_t i = 0; i < kNumSamples; ++i) {
        std::uint64_t got_bits, want_bits;
        std::memcpy(&got_bits, &got[i], sizeof(double));
        std::memcpy(&want_bits, &d.normal_draws[i], sizeof(double));
        if (got_bits != want_bits) {
            std::ostringstream msg;
            msg << "standard_normal bit mismatch seed=" << key << " i=" << i
                << " got_bits=0x" << std::hex << got_bits
                << " want_bits=0x" << want_bits
                << " (got=" << got[i] << " want=" << d.normal_draws[i] << ")";
            throw std::runtime_error(msg.str());
        }
    }

    c4a_rng_pcg64_destroy(rng);
}

void test_create_destroy_null_safe() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(123, nullptr) == C4A_ERR_NULL_POINTER);
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(123, &rng) == C4A_OK);
    C4A_TEST_REQUIRE(rng != nullptr);
    c4a_rng_pcg64_destroy(rng);
    c4a_rng_pcg64_destroy(nullptr);  // must not crash
}

void test_set_seed_resets_stream() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42, &rng) == C4A_OK);

    std::uint64_t a = 0;
    std::uint64_t b = 0;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_uint64(rng, &a) == C4A_OK);

    C4A_TEST_REQUIRE(c4a_rng_pcg64_set_seed(rng, 42) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_rng_pcg64_uint64(rng, &b) == C4A_OK);
    C4A_TEST_REQUIRE(a == b);

    c4a_rng_pcg64_destroy(rng);
}

void test_advance_matches_repeated_draws() {
    // Drawing N values then comparing should equal: create RNG, advance by
    // N, draw 1 — that single draw must match the (N+1)-th value of the
    // original stream.
    constexpr std::uint64_t kAdvance = 137;
    c4a_rng_pcg64_state_t*  ref      = nullptr;
    c4a_rng_pcg64_state_t*  adv      = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0xC0FFEE, &ref) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0xC0FFEE, &adv) == C4A_OK);

    std::uint64_t scratch = 0;
    for (std::uint64_t i = 0; i < kAdvance; ++i) {
        C4A_TEST_REQUIRE(c4a_rng_pcg64_uint64(ref, &scratch) == C4A_OK);
    }
    std::uint64_t ref_next = 0;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_uint64(ref, &ref_next) == C4A_OK);

    C4A_TEST_REQUIRE(c4a_rng_pcg64_advance(adv, kAdvance) == C4A_OK);
    std::uint64_t adv_next = 0;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_uint64(adv, &adv_next) == C4A_OK);

    C4A_TEST_REQUIRE(ref_next == adv_next);

    c4a_rng_pcg64_destroy(ref);
    c4a_rng_pcg64_destroy(adv);
}

}  // namespace

void register_rng_pcg64_tests(c4a_testing::Runner& r);
void register_rng_pcg64_tests(c4a_testing::Runner& r) {
    r.run("rng_pcg64_create_destroy_null_safe", test_create_destroy_null_safe);
    r.run("rng_pcg64_set_seed_resets_stream",   test_set_seed_resets_stream);
    r.run("rng_pcg64_advance_matches_repeats",  test_advance_matches_repeated_draws);

    for (int i = 0; i < kNumSeeds; ++i) {
        const char* key = kSeedKeys[i];
        std::string uname = std::string("rng_pcg64_uint64_parity_seed_") + key;
        std::string nname = std::string("rng_pcg64_standard_normal_parity_seed_") + key;
        // Capture key into the lambda by value (it lives in static storage,
        // so a raw pointer is fine, but we copy for explicitness).
        r.run(uname.c_str(), [key]() { verify_uint64_seed(key); });
        r.run(nname.c_str(), [key]() { verify_standard_normal_seed(key); });
    }
}
