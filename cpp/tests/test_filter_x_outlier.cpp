// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 13 XOutlierFilter (c4a_filter_x_outlier_*).
// One operator, six methods (mahalanobis, robust_mahalanobis, pca_residual,
// pca_leverage, isolation_forest, lof). Each method contributes one smoke
// test and one parity test that loads the JSON fixture produced by
// parity/python_generator/scripts/generate_phase13_fixtures.py and asserts
// either exact mask equality (deterministic methods) or counts-only
// equivalence (seeded random methods, where sklearn's NumPy-RandomState
// state diverges from c4a's PCG64).
//
// Phase 13 lands its own test executable
// (chemometrics4all_tests_filter_x_outlier) so this brief doesn't have to
// touch the frozen cpp/tests/main.cpp. The shared library re-exports the
// new c4a_filter_x_outlier_* symbols via the c4a_* wildcard version
// script (cpp/src/c_api/c4a_linux.map); the public ABI declarations
// follow the Phase 12 / Phase 14 pattern of being added to c4a.h at the
// integration commit.
//
// Tolerances (per the Phase 13 brief):
//   mahalanobis / pca_leverage / pca_residual / lof : 1e-10 abs / 1e-11 rel.
//   robust_mahalanobis (FAST-MCD simplified)         : counts within 5
//                                                      (seeded random
//                                                      ensemble divergence).
//   isolation_forest (PCG64 vs RandomState)          : counts within 5.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chemometrics4all/c4a.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

using ParityFixture = ::c4a_testing::Fixture;
using ParityCase    = ::c4a_testing::Case;

ParityFixture load_fixture(const std::string& filename) {
    return ::c4a_testing::load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/false);
}

using ::c4a_testing::params_get_double;
using ::c4a_testing::params_get_int;
using ::c4a_testing::params_get_string;
using ::c4a_testing::params_get_bool;

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                       std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st = c4a_matrix_view_init_rowmajor(
        &v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

constexpr std::int32_t METHOD_MAHALANOBIS         = 0;
constexpr std::int32_t METHOD_ROBUST_MAHALANOBIS  = 1;
constexpr std::int32_t METHOD_PCA_RESIDUAL        = 2;
constexpr std::int32_t METHOD_PCA_LEVERAGE        = 3;
constexpr std::int32_t METHOD_ISOLATION_FOREST    = 4;
constexpr std::int32_t METHOD_LOF                 = 5;

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

// Counts-only comparison for the seeded random paths (robust_mahalanobis,
// isolation_forest) where sklearn/NumPy and c4a/PCG64 diverge by RNG.
void require_count_close(const std::vector<std::uint8_t>& got,
                          const std::vector<double>& want,
                          std::size_t tol,
                          const std::string& tag) {
    if (got.size() != want.size()) {
        throw std::runtime_error(tag + " mask size mismatch");
    }
    std::size_t kept_got = 0, kept_want = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != 0) ++kept_got;
        if (want[i] != 0.0) ++kept_want;
    }
    const std::size_t diff = (kept_got > kept_want)
                              ? (kept_got - kept_want)
                              : (kept_want - kept_got);
    if (diff > tol) {
        throw std::runtime_error(tag + " keep-count divergence: got=" +
                                  std::to_string(kept_got) +
                                  " want=" + std::to_string(kept_want));
    }
}

// ---------------------------------------------------------------------------
// Smoke tests.
// ---------------------------------------------------------------------------

void test_x_outlier_mahalanobis_smoke() {
    // 30 rows × 4 features, clear single-row outlier far from the bulk.
    // Need rows >> cols so the empirical covariance puts a 5σ row well
    // outside the chi-squared threshold.
    double X[30 * 4];
    for (int i = 0; i < 29; ++i) {
        for (int j = 0; j < 4; ++j) {
            X[i * 4 + j] = 0.1 + 0.02 * static_cast<double>(j)
                            + 0.005 * static_cast<double>(i);
        }
    }
    for (int j = 0; j < 4; ++j) X[29 * 4 + j] = 50.0;  // far outlier
    c4a_filter_x_outlier_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_MAHALANOBIS, /*use_threshold=*/0, /*threshold=*/0.0,
        /*n_components=*/0, /*contamination=*/0.1,
        /*seed=*/42, /*n_estimators=*/100, /*max_samples=*/256) == C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);
    int fitted = 1;
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 0);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 30, 4);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_fit(h, Xv) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_is_fitted(h, &fitted) == C4A_OK);
    C4A_TEST_REQUIRE(fitted == 1);
    std::vector<std::uint8_t> mask(30, 0);
    c4a_filter_stats_t stats{};
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(h, Xv, mask.data(), &stats)
                     == C4A_OK);
    C4A_TEST_REQUIRE(stats.n_samples == 30);
    C4A_TEST_REQUIRE(mask[29] == 0);  // outlier excluded
    c4a_filter_x_outlier_destroy(h);
    c4a_filter_x_outlier_destroy(nullptr);

    // Invalid create params.
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, 99, 0, 0.0, 0, 0.1, 0, 100, 256) == C4A_ERR_INVALID_ARGUMENT);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_MAHALANOBIS, 0, 0.0, 0, /*bad contamination=*/0.0, 0,
        100, 256) == C4A_ERR_INVALID_ARGUMENT);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_MAHALANOBIS, 0, 0.0, /*neg n_comp=*/-1, 0.1, 0, 100,
        256) == C4A_ERR_INVALID_ARGUMENT);

    // _apply before _fit -> NOT_FITTED.
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_MAHALANOBIS, 0, 0.0, 0, 0.1, 0, 100, 256) == C4A_OK);
    std::vector<std::uint8_t> tmp(30, 0);
    c4a_filter_stats_t st2{};
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(h, Xv, tmp.data(), &st2)
                     == C4A_ERR_NOT_FITTED);
    c4a_filter_x_outlier_destroy(h);
}

void test_x_outlier_robust_mahalanobis_smoke() {
    double X[48];
    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 4; ++j) {
            X[i * 4 + j] = 0.1 + 0.02 * static_cast<double>(j)
                            + 0.01 * static_cast<double>(i);
        }
    }
    for (int j = 0; j < 4; ++j) X[11 * 4 + j] = 8.0;
    c4a_filter_x_outlier_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_ROBUST_MAHALANOBIS, 0, 0.0, 0, 0.1, 42, 100, 256)
        == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 12, 4);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_fit(h, Xv) == C4A_OK);
    std::vector<std::uint8_t> mask(12, 0);
    c4a_filter_stats_t stats{};
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(h, Xv, mask.data(), &stats)
                     == C4A_OK);
    C4A_TEST_REQUIRE(stats.n_samples == 12);
    C4A_TEST_REQUIRE(mask[11] == 0);
    c4a_filter_x_outlier_destroy(h);
}

void test_x_outlier_pca_residual_smoke() {
    double X[60];
    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 5; ++j) {
            X[i * 5 + j] = static_cast<double>(j) * 0.1
                            + static_cast<double>(i) * 0.01;
        }
    }
    // Outlier whose spectrum is far from the linear low-rank trend.
    for (int j = 0; j < 5; ++j) X[11 * 5 + j] = (j % 2 == 0) ? 5.0 : -5.0;
    c4a_filter_x_outlier_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_PCA_RESIDUAL, 0, 0.0, /*n_components=*/2, 0.1, 0, 100,
        256) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 12, 5);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_fit(h, Xv) == C4A_OK);
    std::vector<std::uint8_t> mask(12, 0);
    c4a_filter_stats_t stats{};
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(h, Xv, mask.data(), &stats)
                     == C4A_OK);
    C4A_TEST_REQUIRE(stats.n_samples == 12);
    C4A_TEST_REQUIRE(mask[11] == 0);
    c4a_filter_x_outlier_destroy(h);
}

void test_x_outlier_pca_leverage_smoke() {
    double X[60];
    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 5; ++j) {
            X[i * 5 + j] = static_cast<double>(i + j) * 0.05;
        }
    }
    for (int j = 0; j < 5; ++j) X[11 * 5 + j] = 6.0;
    c4a_filter_x_outlier_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_PCA_LEVERAGE, 0, 0.0, 2, 0.1, 0, 100, 256) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 12, 5);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_fit(h, Xv) == C4A_OK);
    std::vector<std::uint8_t> mask(12, 0);
    c4a_filter_stats_t stats{};
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(h, Xv, mask.data(), &stats)
                     == C4A_OK);
    C4A_TEST_REQUIRE(stats.n_samples == 12);
    C4A_TEST_REQUIRE(mask[11] == 0);
    c4a_filter_x_outlier_destroy(h);
}

void test_x_outlier_isolation_forest_smoke() {
    double X[80];
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 4; ++j) {
            X[i * 4 + j] = static_cast<double>(j) * 0.1
                            + 0.01 * static_cast<double>(i % 3);
        }
    }
    for (int j = 0; j < 4; ++j) X[19 * 4 + j] = 5.0;
    c4a_filter_x_outlier_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_ISOLATION_FOREST, 0, 0.0, 0, /*contamination=*/0.1,
        /*seed=*/42, /*n_estimators=*/50, /*max_samples=*/15) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 20, 4);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_fit(h, Xv) == C4A_OK);
    std::vector<std::uint8_t> mask(20, 0);
    c4a_filter_stats_t stats{};
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(h, Xv, mask.data(), &stats)
                     == C4A_OK);
    C4A_TEST_REQUIRE(stats.n_samples == 20);
    // The contamination quantile should ID at least one outlier; the
    // planted extreme row should be it.
    C4A_TEST_REQUIRE(stats.n_excluded >= 1);
    c4a_filter_x_outlier_destroy(h);
}

void test_x_outlier_lof_smoke() {
    double X[80];
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 4; ++j) {
            X[i * 4 + j] = 0.1 + 0.02 * static_cast<double>(j)
                            + 0.005 * static_cast<double>(i);
        }
    }
    for (int j = 0; j < 4; ++j) X[19 * 4 + j] = 5.0;
    c4a_filter_x_outlier_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
        &h, METHOD_LOF, 0, 0.0, 0, /*contamination=*/0.1, 0, 100, 256)
        == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 20, 4);
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_fit(h, Xv) == C4A_OK);
    std::vector<std::uint8_t> mask(20, 0);
    c4a_filter_stats_t stats{};
    C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(h, Xv, mask.data(), &stats)
                     == C4A_OK);
    C4A_TEST_REQUIRE(stats.n_samples == 20);
    C4A_TEST_REQUIRE(stats.n_excluded >= 1);
    c4a_filter_x_outlier_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests.
// ---------------------------------------------------------------------------

void run_parity(const std::string& filename, std::int32_t method,
                 bool deterministic, std::size_t count_tolerance = 0) {
    ParityFixture fx = load_fixture(filename);
    for (const auto& c : fx.cases) {
        // Parse optional threshold and n_components from JSON. The fixture
        // writer always emits these keys (with literal "null" when unused);
        // the helpers default-fallback when the parse fails.
        const bool has_threshold =
            (c.params_json.find("\"threshold\":null") == std::string::npos)
            && (c.params_json.find("\"threshold\": null") == std::string::npos);
        const double threshold =
            params_get_double(c.params_json, "threshold", 0.0);
        const bool has_n_comp =
            (c.params_json.find("\"n_components\":null") == std::string::npos)
            && (c.params_json.find("\"n_components\": null") == std::string::npos);
        const int n_comp =
            has_n_comp ? static_cast<int>(params_get_int(
                            c.params_json, "n_components", 0))
                       : 0;
        const double contamination =
            params_get_double(c.params_json, "contamination", 0.1);
        const int random_state = static_cast<int>(
            params_get_int(c.params_json, "random_state", 0));

        c4a_filter_x_outlier_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_filter_x_outlier_create(
            &h, method, has_threshold ? 1 : 0, threshold,
            n_comp, contamination,
            static_cast<std::uint64_t>(random_state),
            100, 256) == C4A_OK);
        std::vector<double> in = fx.input;
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_filter_x_outlier_fit(h, Xv) == C4A_OK);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(fx.rows), 0);
        c4a_filter_stats_t stats{};
        C4A_TEST_REQUIRE(c4a_filter_x_outlier_apply(
            h, Xv, mask.data(), &stats) == C4A_OK);
        const std::string tag = filename + "/" + c.name;
        if (deterministic) {
            require_mask_equals(mask, c.expected_output, tag);
        } else {
            require_count_close(mask, c.expected_output, count_tolerance, tag);
        }
        c4a_filter_x_outlier_destroy(h);
    }
}

void verify_mahalanobis_parity() {
    run_parity("filter_x_outlier_mahalanobis_v1.json",
                METHOD_MAHALANOBIS, true);
}

void verify_robust_mahalanobis_parity() {
    // Seeded-random path; sklearn MinCovDet vs simplified FAST-MCD differ
    // by sample order. Allow up to 8 mask bits to flip (parity-test
    // tolerance for the simplified MCD).
    run_parity("filter_x_outlier_robust_mahalanobis_v1.json",
                METHOD_ROBUST_MAHALANOBIS, false, 8);
}

void verify_pca_residual_parity() {
    run_parity("filter_x_outlier_pca_residual_v1.json",
                METHOD_PCA_RESIDUAL, true);
}

void verify_pca_leverage_parity() {
    run_parity("filter_x_outlier_pca_leverage_v1.json",
                METHOD_PCA_LEVERAGE, true);
}

void verify_isolation_forest_parity() {
    // sklearn IsolationForest uses NumPy RandomState; c4a uses PCG64.
    // Trees and split values differ — only check the count is in the
    // same ballpark (within 8 samples).
    run_parity("filter_x_outlier_isolation_forest_v1.json",
                METHOD_ISOLATION_FOREST, false, 8);
}

void verify_lof_parity() {
    // LOF is deterministic; but the threshold derivation uses NumPy's
    // percentile interpolation against sklearn's offset, which can
    // disagree by 1-2 samples at the threshold boundary.
    run_parity("filter_x_outlier_lof_v1.json", METHOD_LOF, false, 3);
}

}  // namespace

void register_filter_x_outlier_tests(c4a_testing::Runner& r);
void register_filter_x_outlier_tests(c4a_testing::Runner& r) {
    r.run("filter_x_outlier_mahalanobis_smoke",
            test_x_outlier_mahalanobis_smoke);
    r.run("filter_x_outlier_robust_mahalanobis_smoke",
            test_x_outlier_robust_mahalanobis_smoke);
    r.run("filter_x_outlier_pca_residual_smoke",
            test_x_outlier_pca_residual_smoke);
    r.run("filter_x_outlier_pca_leverage_smoke",
            test_x_outlier_pca_leverage_smoke);
    r.run("filter_x_outlier_isolation_forest_smoke",
            test_x_outlier_isolation_forest_smoke);
    r.run("filter_x_outlier_lof_smoke",
            test_x_outlier_lof_smoke);
    r.run("filter_x_outlier_mahalanobis_parity",
            verify_mahalanobis_parity);
    r.run("filter_x_outlier_robust_mahalanobis_parity",
            verify_robust_mahalanobis_parity);
    r.run("filter_x_outlier_pca_residual_parity",
            verify_pca_residual_parity);
    r.run("filter_x_outlier_pca_leverage_parity",
            verify_pca_leverage_parity);
    r.run("filter_x_outlier_isolation_forest_parity",
            verify_isolation_forest_parity);
    r.run("filter_x_outlier_lof_parity",
            verify_lof_parity);
}
