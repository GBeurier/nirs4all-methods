// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 14 meta sample filters
// (HighLeverage, SpectralQuality, Composite). Each operator contributes
// one smoke test (exercise create / apply / destroy on a small inline
// matrix) and one parity test (load a JSON fixture produced by the
// matching generator under
// parity/python_generator/scripts/generate_phase14_fixtures.py, run the
// C engine, assert within tolerance against the frozen NumPy reference).
//
// Phase 14 lands its own test executable (n4m_tests_filters_meta)
// rather than appending to n4m_tests, because the latter's
// main.cpp is frozen during Phase 12 / Phase 14 parallel development. Once
// Phase 12 has landed, the two executables can be merged.
//
// Tolerances (per the Phase 14 brief):
//   * HighLeverage      : 1e-9 abs / 1e-10 rel  (matrix inverse conditioning)
//   * SpectralQuality   : exact mask equality   (pure thresholds)
//   * Composite         : exact mask equality   (boolean combination)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef N4M_PARITY_FIXTURE_DIR
#  error "N4M_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

using ParityFixture = ::n4m_testing::Fixture;
using ParityCase    = ::n4m_testing::Case;

ParityFixture load_fixture(const std::string& filename) {
    return ::n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/false);
}

using ::n4m_testing::params_get_double;
using ::n4m_testing::params_get_int;
using ::n4m_testing::params_get_string;
using ::n4m_testing::params_get_bool;

n4m_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st =
        n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

// Compare two boolean (uint8) masks for exact equality.
void require_mask_equals(const std::vector<std::uint8_t>& got,
                          const std::vector<double>& want,
                          const std::string& tag) {
    if (got.size() != want.size()) {
        throw std::runtime_error(tag + " mask size mismatch");
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const std::uint8_t want_bit = (want[i] != 0.0) ? 1 : 0;
        if (got[i] != want_bit) {
            throw std::runtime_error(tag + " mask mismatch at i=" +
                                      std::to_string(i));
        }
    }
}

// ---------------------------------------------------------------------------
// HighLeverage — smoke
// ---------------------------------------------------------------------------

void test_high_leverage_smoke() {
    // 8 samples × 3 features. The last row is a clear outlier — far enough
    // that even multiplier=1.5 catches it. Using multiplier=1.5 with
    // average training leverage = 3/8 = 0.375 → threshold = 0.5625; the
    // outlier leverage tops 0.875.
    double X[24] = {
        1.0, 1.0, 1.0,
        1.1, 0.9, 1.0,
        0.9, 1.1, 1.0,
        1.0, 1.05, 0.95,
        1.05, 0.95, 1.05,
        0.95, 0.95, 1.10,
        1.02, 0.98, 0.99,
        8.0, 8.0, 8.0,  // outlier
    };
    n4m_filter_leverage_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_filter_leverage_create(
        &h, /*method=*/0, /*multiplier=*/1.5,
        /*use_absolute=*/0, /*absolute=*/0.0,
        /*n_components=*/0, /*center=*/1) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);

    int fitted = 1;
    N4M_TEST_REQUIRE(n4m_filter_leverage_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 0);

    n4m_matrix_view_t Xv = make_rowmajor_view(X, 8, 3);
    N4M_TEST_REQUIRE(n4m_filter_leverage_fit(h, Xv) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_filter_leverage_is_fitted(h, &fitted) == N4M_OK);
    N4M_TEST_REQUIRE(fitted == 1);

    std::vector<std::uint8_t> mask(8, 0);
    n4m_filter_stats_t stats{};
    N4M_TEST_REQUIRE(n4m_filter_leverage_apply(h, Xv, mask.data(), &stats)
                     == N4M_OK);
    // The outlier row should be excluded.
    N4M_TEST_REQUIRE(mask[7] == 0);
    N4M_TEST_REQUIRE(stats.n_samples == 8);
    N4M_TEST_REQUIRE(stats.n_kept + stats.n_excluded == 8);
    N4M_TEST_REQUIRE(stats.n_excluded >= 1);

    const double thr = n4m_filter_leverage_threshold(h);
    N4M_TEST_REQUIRE(std::isfinite(thr) && thr > 0.0);

    n4m_filter_leverage_destroy(h);
    n4m_filter_leverage_destroy(nullptr);  // null-safe

    // Invalid parameters reject.
    N4M_TEST_REQUIRE(n4m_filter_leverage_create(
        &h, /*method=*/99, 2.0, 0, 0.0, 0, 1) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_filter_leverage_create(
        &h, 0, /*multiplier=*/-1.0, 0, 0.0, 0, 1) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_filter_leverage_create(
        &h, 0, 2.0, 1, /*absolute=*/1.5, 0, 1) == N4M_ERR_INVALID_ARGUMENT);

    // _apply before _fit returns N4M_ERR_NOT_FITTED.
    N4M_TEST_REQUIRE(n4m_filter_leverage_create(
        &h, 0, 2.0, 0, 0.0, 0, 1) == N4M_OK);
    std::vector<std::uint8_t> tmp(8, 0);
    n4m_filter_stats_t st2{};
    N4M_TEST_REQUIRE(n4m_filter_leverage_apply(h, Xv, tmp.data(), &st2)
                     == N4M_ERR_NOT_FITTED);
    n4m_filter_leverage_destroy(h);
}

// ---------------------------------------------------------------------------
// SpectralQuality — smoke
// ---------------------------------------------------------------------------

void test_spectral_quality_smoke() {
    // 5 samples × 4 features:
    //   row 0 — clean spectrum, should pass
    //   row 1 — half NaN (> default 10% threshold), should fail
    //   row 2 — contains +inf, should fail (check_inf=on)
    //   row 3 — all zeros, fails zero-ratio AND variance
    //   row 4 — constant 0.5 → zero variance, fails variance
    const double nan_val = std::nan("");
    const double inf_val = HUGE_VAL;
    double X[20] = {
        0.1, 0.2, 0.15, 0.18,
        nan_val, nan_val, 0.3, 0.4,
        0.1, inf_val, 0.2, 0.3,
        0.0, 0.0, 0.0, 0.0,
        0.5, 0.5, 0.5, 0.5,
    };

    n4m_filter_quality_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_filter_quality_create(
        &h,
        /*max_nan_ratio=*/0.1, /*max_zero_ratio=*/0.5,
        /*min_variance=*/1e-8,
        /*use_max=*/0, /*max_value=*/0.0,
        /*use_min=*/0, /*min_value=*/0.0,
        /*check_inf=*/1) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);

    n4m_matrix_view_t Xv = make_rowmajor_view(X, 5, 4);
    std::vector<std::uint8_t> mask(5, 0);
    n4m_filter_stats_t stats{};
    N4M_TEST_REQUIRE(n4m_filter_quality_apply(h, Xv, mask.data(), &stats)
                     == N4M_OK);
    N4M_TEST_REQUIRE(mask[0] == 1);  // clean
    N4M_TEST_REQUIRE(mask[1] == 0);  // NaN ratio
    N4M_TEST_REQUIRE(mask[2] == 0);  // inf
    N4M_TEST_REQUIRE(mask[3] == 0);  // zeros + variance
    N4M_TEST_REQUIRE(mask[4] == 0);  // variance
    N4M_TEST_REQUIRE(stats.n_samples == 5);
    N4M_TEST_REQUIRE(stats.n_kept == 1);
    N4M_TEST_REQUIRE(stats.n_excluded == 4);

    n4m_filter_quality_destroy(h);
    n4m_filter_quality_destroy(nullptr);  // null-safe

    // Invalid parameters reject.
    N4M_TEST_REQUIRE(n4m_filter_quality_create(
        &h, /*max_nan_ratio=*/-0.1, 0.5, 1e-8, 0, 0, 0, 0, 1)
        == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_filter_quality_create(
        &h, 0.1, /*max_zero_ratio=*/1.5, 1e-8, 0, 0, 0, 0, 1)
        == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_filter_quality_create(
        &h, 0.1, 0.5, /*min_variance=*/-1.0, 0, 0, 0, 0, 1)
        == N4M_ERR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// Composite — smoke
// ---------------------------------------------------------------------------

void test_composite_smoke() {
    // Build a HighLeverage filter + a SpectralQuality filter, then combine
    // them via ANY (keep iff both filters say keep). 4 samples, 3 features.
    const double nan_val = std::nan("");
    double X[12] = {
        0.10, 0.20, 0.15,    // clean + low leverage
        nan_val, nan_val, 0.5,  // bad quality, low leverage
        0.11, 0.18, 0.17,    // clean + low leverage
        5.00, 6.00, 7.00,    // high leverage
    };
    n4m_filter_leverage_handle_t* lev = nullptr;
    n4m_filter_quality_handle_t*  q   = nullptr;
    N4M_TEST_REQUIRE(n4m_filter_leverage_create(
        &lev, 0, 2.0, 0, 0.0, 0, 1) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_filter_quality_create(
        &q, 0.1, 0.5, 1e-8, 0, 0.0, 0, 0.0, 1) == N4M_OK);

    n4m_matrix_view_t Xv = make_rowmajor_view(X, 4, 3);
    N4M_TEST_REQUIRE(n4m_filter_leverage_fit(lev, Xv) == N4M_OK);

    n4m_filter_composite_handle_t* comp = nullptr;
    N4M_TEST_REQUIRE(n4m_filter_composite_create(&comp, N4M_COMPOSITE_ANY)
                     == N4M_OK);
    N4M_TEST_REQUIRE(n4m_filter_composite_add_leverage(comp, lev) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_filter_composite_add_quality(comp, q)   == N4M_OK);

    std::vector<std::uint8_t> mask(4, 0);
    n4m_filter_stats_t stats{};
    N4M_TEST_REQUIRE(n4m_filter_composite_apply(comp, Xv, mask.data(), &stats)
                     == N4M_OK);
    // ANY semantics: a sample is excluded if ANY filter excludes.
    // Row 1 fails the quality check → excluded.
    N4M_TEST_REQUIRE(mask[1] == 0);
    // Row 3 has extreme leverage → excluded.
    N4M_TEST_REQUIRE(mask[3] == 0);
    N4M_TEST_REQUIRE(stats.n_samples == 4);

    n4m_filter_composite_destroy(comp);
    n4m_filter_composite_destroy(nullptr);  // null-safe
    n4m_filter_leverage_destroy(lev);
    n4m_filter_quality_destroy(q);

    N4M_TEST_REQUIRE(n4m_filter_composite_create(&comp, 2)
        == N4M_ERR_INVALID_ARGUMENT);

    // Empty composite keeps every sample.
    N4M_TEST_REQUIRE(n4m_filter_composite_create(&comp, N4M_COMPOSITE_ANY)
                     == N4M_OK);
    std::vector<std::uint8_t> mask2(4, 0);
    n4m_filter_stats_t st2{};
    N4M_TEST_REQUIRE(n4m_filter_composite_apply(comp, Xv, mask2.data(), &st2)
                     == N4M_OK);
    for (auto m : mask2) N4M_TEST_REQUIRE(m == 1);
    N4M_TEST_REQUIRE(st2.n_kept == 4);
    n4m_filter_composite_destroy(comp);
}

// ---------------------------------------------------------------------------
// Parity verifiers.
// ---------------------------------------------------------------------------

void run_leverage_fixture(const std::string& filename) {
    ParityFixture fx = load_fixture(filename);
    for (const auto& c : fx.cases) {
        // method: "hat" or "pca"
        const std::string method = params_get_string(c.params_json, "method", "hat");
        const int method_id = (method == "pca") ? 1 : 0;
        const double mult = params_get_double(c.params_json, "threshold_multiplier", 2.0);
        // ``absolute_threshold`` is "null" in the JSON when the multiplier
        // mode is selected. A simple substring check distinguishes the
        // two states (the fixture writer always emits the key).
        const bool abs_null = (c.params_json.find("\"absolute_threshold\":null")
                                != std::string::npos) ||
                              (c.params_json.find("\"absolute_threshold\": null")
                                != std::string::npos);
        const int use_abs = abs_null ? 0 : 1;
        const double abs_thr =
            params_get_double(c.params_json, "absolute_threshold", 0.0);
        const int n_comp = static_cast<int>(
            params_get_int(c.params_json, "n_components", 0));
        const int center = static_cast<int>(
            params_get_int(c.params_json, "center", 1));

        n4m_filter_leverage_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_filter_leverage_create(
            &h, method_id, mult, use_abs, abs_thr, n_comp, center) == N4M_OK);
        std::vector<double> in = fx.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_filter_leverage_fit(h, Xv) == N4M_OK);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(fx.rows), 0);
        n4m_filter_stats_t stats{};
        N4M_TEST_REQUIRE(n4m_filter_leverage_apply(h, Xv, mask.data(), &stats)
                         == N4M_OK);
        // Compare mask bit-equal — leverage is a deterministic threshold of
        // a numerical scalar, so within the 1e-9 / 1e-10 contract the mask
        // should match every reference bit unless the test parameters land
        // exactly on the threshold (the fixture generator avoids that case).
        require_mask_equals(mask, c.expected_output,
                             filename + "/" + c.name);
        n4m_filter_leverage_destroy(h);
    }
}

void verify_high_leverage_parity() {
    // Tall matrix (rows > cols) — drives the direct hat path + an explicit
    // PCA case.
    run_leverage_fixture("filter_leverage_v1.json");
    // Wide matrix (rows < cols) — drives the auto-fallback to PCA.
    run_leverage_fixture("filter_leverage_wide_v1.json");
}

void verify_spectral_quality_parity() {
    ParityFixture fx = load_fixture("filter_quality_v1.json");
    for (const auto& c : fx.cases) {
        const double max_nan  = params_get_double(c.params_json, "max_nan_ratio", 0.1);
        const double max_zero = params_get_double(c.params_json, "max_zero_ratio", 0.5);
        const double min_var  = params_get_double(c.params_json, "min_variance", 1e-8);
        const int    has_max  = (c.params_json.find("\"max_value\"") != std::string::npos)
                              && (c.params_json.find("\"max_value\": null") == std::string::npos);
        const double max_val  = params_get_double(c.params_json, "max_value", 0.0);
        const int    has_min  = (c.params_json.find("\"min_value\"") != std::string::npos)
                              && (c.params_json.find("\"min_value\": null") == std::string::npos);
        const double min_val  = params_get_double(c.params_json, "min_value", 0.0);
        const int    chk_inf  = params_get_bool(c.params_json, "check_inf", true) ? 1 : 0;

        n4m_filter_quality_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_filter_quality_create(
            &h, max_nan, max_zero, min_var,
            has_max, max_val, has_min, min_val, chk_inf) == N4M_OK);
        std::vector<double> in = fx.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(fx.rows), 0);
        n4m_filter_stats_t stats{};
        N4M_TEST_REQUIRE(n4m_filter_quality_apply(h, Xv, mask.data(), &stats)
                         == N4M_OK);
        require_mask_equals(mask, c.expected_output, "quality/" + c.name);
        n4m_filter_quality_destroy(h);
    }
}

void verify_composite_parity() {
    // Composite parity drives both modes (ANY / ALL) against the same input
    // matrix with the same two sub-filters as the smoke test.
    ParityFixture fx = load_fixture("filter_composite_v1.json");
    for (const auto& c : fx.cases) {
        const std::string mode = params_get_string(c.params_json, "mode", "any");
        const n4m_composite_mode_t mode_id =
            (mode == "all") ? N4M_COMPOSITE_ALL : N4M_COMPOSITE_ANY;

        // Sub-filters: a HighLeverage and a SpectralQuality, both at defaults.
        n4m_filter_leverage_handle_t* lev = nullptr;
        n4m_filter_quality_handle_t*  q   = nullptr;
        N4M_TEST_REQUIRE(n4m_filter_leverage_create(
            &lev, 0, 2.0, 0, 0.0, 0, 1) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_filter_quality_create(
            &q, 0.1, 0.5, 1e-8, 0, 0.0, 0, 0.0, 1) == N4M_OK);
        std::vector<double> in = fx.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_filter_leverage_fit(lev, Xv) == N4M_OK);

        n4m_filter_composite_handle_t* comp = nullptr;
        N4M_TEST_REQUIRE(n4m_filter_composite_create(&comp, mode_id) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_filter_composite_add_leverage(comp, lev) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_filter_composite_add_quality(comp, q)   == N4M_OK);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(fx.rows), 0);
        n4m_filter_stats_t stats{};
        N4M_TEST_REQUIRE(n4m_filter_composite_apply(comp, Xv, mask.data(), &stats)
                         == N4M_OK);
        require_mask_equals(mask, c.expected_output, "composite/" + c.name);
        n4m_filter_composite_destroy(comp);
        n4m_filter_leverage_destroy(lev);
        n4m_filter_quality_destroy(q);
    }
}

}  // namespace

void register_filters_meta_tests(n4m_testing::Runner& r);
void register_filters_meta_tests(n4m_testing::Runner& r) {
    r.run("filter_leverage_smoke",       test_high_leverage_smoke);
    r.run("filter_quality_smoke",        test_spectral_quality_smoke);
    r.run("filter_composite_smoke",      test_composite_smoke);
    r.run("filter_leverage_parity",      verify_high_leverage_parity);
    r.run("filter_quality_parity",       verify_spectral_quality_parity);
    r.run("filter_composite_parity",     verify_composite_parity);
}
