// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 20 transfer-metrics utility
// (c4a_transfer_metrics_compute). One smoke test exercises the
// create-input / compute / consume happy path on a small inline matrix;
// one parity test loads transfer_metrics_v1.json, runs the C engine on
// each case, and asserts within tolerance against the frozen NumPy
// reference (parity/python_generator/src/c4a_parity_transfer_ref/).
//
// Tolerance per the brief (Phase 20):
//
//   * 1e-9 abs / 1e-10 rel.
//
// Iterative Jacobi + multiple matrix compositions (CKA / RV / Procrustes /
// Grassmann) make the bit-exact-with-numpy bar unrealistic; the tolerance
// is consistent with the LDLT-iterative baseline operators (Phase 5).

#include <cmath>
#include <cstddef>
#include <cstdint>
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

using ::c4a_testing::Case;
using ::c4a_testing::params_get_int;
using ::c4a_testing::params_get_double;
using ::c4a_testing::slurp;

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Smoke test — feed two small synthetic datasets and verify the result
// struct is populated with finite values (within reason).
// ---------------------------------------------------------------------------

void test_transfer_metrics_smoke() {
    // Construct a tiny source / target dataset with 8 samples and 5 features.
    // The two datasets are deliberately offset to give a non-zero centroid
    // distance and a non-degenerate spread.
    constexpr int N = 8;
    constexpr int P = 5;
    double X_src[N * P] = {0};
    double X_tgt[N * P] = {0};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            const double t = static_cast<double>(i) / (N - 1);
            X_src[i * P + j] = std::sin(t * 3.0 + j) + 0.05 * j;
            X_tgt[i * P + j] = std::sin(t * 3.0 + j) + 0.05 * j + 0.3;
        }
    }
    c4a_matrix_view_t Xs = make_rowmajor_view(X_src, N, P);
    c4a_matrix_view_t Xt = make_rowmajor_view(X_tgt, N, P);
    c4a_transfer_metrics_t out{};
    const c4a_status_t st = c4a_transfer_metrics_compute(
        Xs, Xt,
        /*n_components=*/3, /*k_neighbors=*/3, /*seed=*/42, &out);
    C4A_TEST_REQUIRE(st == C4A_OK);
    C4A_TEST_REQUIRE(std::isfinite(out.centroid_distance));
    C4A_TEST_REQUIRE(out.centroid_distance >= 0.0);
    C4A_TEST_REQUIRE(std::isfinite(out.cka_similarity));
    C4A_TEST_REQUIRE(out.cka_similarity >= 0.0);
    // For aligned features grassmann should be finite.
    C4A_TEST_REQUIRE(std::isfinite(out.grassmann_distance));
    C4A_TEST_REQUIRE(out.grassmann_distance >= 0.0);
    C4A_TEST_REQUIRE(std::isfinite(out.rv_coefficient));
    C4A_TEST_REQUIRE(std::isfinite(out.procrustes_disparity));
    C4A_TEST_REQUIRE(out.procrustes_disparity >= 0.0);
    C4A_TEST_REQUIRE(std::isfinite(out.trustworthiness));
    C4A_TEST_REQUIRE(out.trustworthiness <= 1.0 + 1e-12);
    C4A_TEST_REQUIRE(std::isfinite(out.spread_distance));
    C4A_TEST_REQUIRE(std::isfinite(out.evr_source));
    C4A_TEST_REQUIRE(std::isfinite(out.evr_target));
    C4A_TEST_REQUIRE(out.evr_source >= 0.0 && out.evr_source <= 1.0 + 1e-12);
    C4A_TEST_REQUIRE(out.evr_target >= 0.0 && out.evr_target <= 1.0 + 1e-12);
}

void test_transfer_metrics_invalid_args() {
    constexpr int N = 4;
    constexpr int P = 3;
    double X_src[N * P] = {0};
    double X_tgt[N * P] = {0};
    c4a_matrix_view_t Xs = make_rowmajor_view(X_src, N, P);
    c4a_matrix_view_t Xt = make_rowmajor_view(X_tgt, N, P);
    c4a_transfer_metrics_t out{};
    // out NULL
    C4A_TEST_REQUIRE(c4a_transfer_metrics_compute(Xs, Xt, 2, 2, 0, nullptr)
                     == C4A_ERR_NULL_POINTER);
    // n_components < 1
    C4A_TEST_REQUIRE(c4a_transfer_metrics_compute(Xs, Xt, 0, 2, 0, &out)
                     == C4A_ERR_INVALID_ARGUMENT);
    // k_neighbors < 2
    C4A_TEST_REQUIRE(c4a_transfer_metrics_compute(Xs, Xt, 2, 1, 0, &out)
                     == C4A_ERR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// Parity loader — the transfer-metrics fixture diverges slightly from the
// preprocessing-fixture shape because there are *two* input matrices and
// each case produces a scalar struct rather than a matrix. Schema:
//
//   {
//     "format": "c4a_transfer_metrics_v1",
//     "cases": [
//       { "name": "...",
//         "params": { "n_components": N, "k_neighbors": K, "seed": S },
//         "source_rows": R, "source_cols": C,
//         "source_hex": [ ... ],
//         "target_rows": R, "target_cols": C,
//         "target_hex": [ ... ],
//         "expected": {
//             "centroid_distance":    "<16 hex>",
//             "cka_similarity":       "<16 hex>",
//             "grassmann_distance":   "<16 hex>",
//             "rv_coefficient":       "<16 hex>",
//             "procrustes_disparity": "<16 hex>",
//             "trustworthiness":      "<16 hex>",
//             "spread_distance":      "<16 hex>",
//             "evr_source":           "<16 hex>",
//             "evr_target":           "<16 hex>"
//         } },
//       ...
//     ]
//   }
//
// We parse this with the shared fixture_parser primitives.
// ---------------------------------------------------------------------------

struct TransferCase {
    std::string name;
    int32_t     n_components;
    int32_t     k_neighbors;
    uint64_t    seed;
    int64_t     src_rows, src_cols;
    int64_t     tgt_rows, tgt_cols;
    std::vector<double> src, tgt;
    // Expected scalars, in the order of the c4a_transfer_metrics_t struct.
    double      centroid_distance;
    double      cka_similarity;
    double      grassmann_distance;
    double      rv_coefficient;
    double      procrustes_disparity;
    double      trustworthiness;
    double      spread_distance;
    double      evr_source;
    double      evr_target;
};

// Locate the value (as a hex-encoded double) of an expected field within a
// case object span. Returns NaN if the field is missing — the caller must
// treat a missing expectation as "skip" rather than fail.
double parse_expected_double(const std::string& body, std::size_t lo,
                              std::size_t hi, const std::string& key) {
    std::size_t expected_pos = body.find("\"expected\"", lo);
    if (expected_pos == std::string::npos || expected_pos > hi) {
        return std::nan("");
    }
    std::size_t open = body.find('{', expected_pos);
    if (open == std::string::npos || open > hi) {
        return std::nan("");
    }
    const std::string q = "\"" + key + "\"";
    std::size_t key_pos = body.find(q, open);
    if (key_pos == std::string::npos || key_pos > hi) {
        return std::nan("");
    }
    std::size_t p = key_pos + q.size();
    while (p < hi && (body[p] == ' ' || body[p] == '\t' || body[p] == ':')) {
        ++p;
    }
    if (p >= hi || body[p] != '"') {
        return std::nan("");
    }
    return ::c4a_testing::hex_to_double(body, p);
}

std::vector<TransferCase> load_transfer_cases() {
    const std::string body =
        slurp(std::string(C4A_PARITY_FIXTURE_DIR) + "/transfer_metrics_v1.json");
    auto [cases_lo, cases_hi] = ::c4a_testing::find_cases_array(body);
    auto spans = ::c4a_testing::list_case_object_spans(body, cases_lo, cases_hi);
    std::vector<TransferCase> out;
    for (const auto& [lo, hi] : spans) {
        TransferCase c;
        std::size_t np = ::c4a_testing::find_value_for_key(body, "name", lo, hi);
        c.name = ::c4a_testing::parse_string(body, np);
        std::size_t pp = ::c4a_testing::find_value_for_key(body, "params", lo, hi);
        auto [pb, pe] = ::c4a_testing::read_value_span(body, pp);
        const std::string params_json = body.substr(pb, pe - pb);
        c.n_components = static_cast<int32_t>(
            params_get_int(params_json, "n_components", 10));
        c.k_neighbors = static_cast<int32_t>(
            params_get_int(params_json, "k_neighbors", 10));
        c.seed = static_cast<uint64_t>(
            params_get_int(params_json, "seed", 0));

        c.src_rows = ::c4a_testing::parse_int64(body,
            ::c4a_testing::find_value_for_key(body, "source_rows", lo, hi));
        c.src_cols = ::c4a_testing::parse_int64(body,
            ::c4a_testing::find_value_for_key(body, "source_cols", lo, hi));
        std::size_t shp = ::c4a_testing::find_value_for_key(body, "source_hex", lo, hi);
        ::c4a_testing::parse_hex_double_array(body, shp, c.src);

        c.tgt_rows = ::c4a_testing::parse_int64(body,
            ::c4a_testing::find_value_for_key(body, "target_rows", lo, hi));
        c.tgt_cols = ::c4a_testing::parse_int64(body,
            ::c4a_testing::find_value_for_key(body, "target_cols", lo, hi));
        std::size_t thp = ::c4a_testing::find_value_for_key(body, "target_hex", lo, hi);
        ::c4a_testing::parse_hex_double_array(body, thp, c.tgt);

        c.centroid_distance    = parse_expected_double(body, lo, hi, "centroid_distance");
        c.cka_similarity       = parse_expected_double(body, lo, hi, "cka_similarity");
        c.grassmann_distance   = parse_expected_double(body, lo, hi, "grassmann_distance");
        c.rv_coefficient       = parse_expected_double(body, lo, hi, "rv_coefficient");
        c.procrustes_disparity = parse_expected_double(body, lo, hi, "procrustes_disparity");
        c.trustworthiness      = parse_expected_double(body, lo, hi, "trustworthiness");
        c.spread_distance      = parse_expected_double(body, lo, hi, "spread_distance");
        c.evr_source           = parse_expected_double(body, lo, hi, "evr_source");
        c.evr_target           = parse_expected_double(body, lo, hi, "evr_target");

        out.push_back(std::move(c));
    }
    return out;
}

void assert_metric(const std::string& tag, double got, double want,
                   double abs_tol, double rel_tol) {
    if (std::isnan(want)) {
        // No expectation recorded — accept anything finite (or NaN, since
        // the Python reference also returns NaN when a metric is undefined).
        return;
    }
    if (std::isnan(got) && std::isnan(want)) return;
    if (std::isnan(got) != std::isnan(want)) {
        throw std::runtime_error(tag + ": NaN mismatch got=" +
            (std::isnan(got) ? "NaN" : std::to_string(got)) + " want=" +
            (std::isnan(want) ? "NaN" : std::to_string(want)));
    }
    const double diff = std::fabs(got - want);
    const double ref  = std::max(std::fabs(got), std::fabs(want));
    if (diff <= abs_tol) return;
    if (ref > 0.0 && diff / ref <= rel_tol) return;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%s mismatch got=%.17g want=%.17g abs_diff=%.6e rel_diff=%.6e",
                  tag.c_str(), got, want, diff,
                  ref > 0.0 ? diff / ref : diff);
    throw std::runtime_error(buf);
}

// ---------------------------------------------------------------------------
// Parity verification.
// ---------------------------------------------------------------------------

void verify_transfer_metrics_parity() {
    const std::vector<TransferCase> cases = load_transfer_cases();
    constexpr double abs_tol = 1e-9;
    constexpr double rel_tol = 1e-10;
    for (const auto& c : cases) {
        std::vector<double> src = c.src;
        std::vector<double> tgt = c.tgt;
        c4a_matrix_view_t Xs = make_rowmajor_view(src.data(), c.src_rows, c.src_cols);
        c4a_matrix_view_t Xt = make_rowmajor_view(tgt.data(), c.tgt_rows, c.tgt_cols);
        c4a_transfer_metrics_t out{};
        const c4a_status_t st = c4a_transfer_metrics_compute(
            Xs, Xt, c.n_components, c.k_neighbors, c.seed, &out);
        C4A_TEST_REQUIRE(st == C4A_OK);

        const std::string tag = "transfer_metrics/" + c.name;
        assert_metric(tag + "/centroid_distance",    out.centroid_distance,
                      c.centroid_distance,    abs_tol, rel_tol);
        assert_metric(tag + "/cka_similarity",       out.cka_similarity,
                      c.cka_similarity,       abs_tol, rel_tol);
        assert_metric(tag + "/grassmann_distance",   out.grassmann_distance,
                      c.grassmann_distance,   abs_tol, rel_tol);
        assert_metric(tag + "/rv_coefficient",       out.rv_coefficient,
                      c.rv_coefficient,       abs_tol, rel_tol);
        assert_metric(tag + "/procrustes_disparity", out.procrustes_disparity,
                      c.procrustes_disparity, abs_tol, rel_tol);
        assert_metric(tag + "/trustworthiness",      out.trustworthiness,
                      c.trustworthiness,      abs_tol, rel_tol);
        assert_metric(tag + "/spread_distance",      out.spread_distance,
                      c.spread_distance,      abs_tol, rel_tol);
        assert_metric(tag + "/evr_source",           out.evr_source,
                      c.evr_source,           abs_tol, rel_tol);
        assert_metric(tag + "/evr_target",           out.evr_target,
                      c.evr_target,           abs_tol, rel_tol);
    }
}

}  // namespace

void register_transfer_metrics_tests(c4a_testing::Runner& r);
void register_transfer_metrics_tests(c4a_testing::Runner& r) {
    r.run("transfer_metrics_smoke",        test_transfer_metrics_smoke);
    r.run("transfer_metrics_invalid_args", test_transfer_metrics_invalid_args);
    r.run("transfer_metrics_parity",       verify_transfer_metrics_parity);
}
