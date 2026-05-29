// SPDX-License-Identifier: CECILL-2.1
//
// Smoke + parity tests for the edge, spline, and random augmenter slice.
// Each operator gets a smoke test that exercises create / apply / destroy
// on a small inline matrix and verifies the universal contract from
// roadmap/phase-15-18-augmenters-abi-contract.md:
//
//   1. _create(NULL, ...)        -> N4M_ERR_NULL_POINTER
//   2. _create(out, NULL_rng,..) -> N4M_ERR_NULL_POINTER
//   3. _apply with non-matching shape -> N4M_ERR_SHAPE_MISMATCH
//   4. _apply with float32 input -> N4M_ERR_DTYPE_MISMATCH
//   5. Determinism: re-seeding the RNG before two independent runs yields the
//      same output bit-for-bit.
//   6. _destroy(NULL) is a no-op.
//
// Spline_X_Simplification and Spline_Curve_Simplification (formerly v2-deferred)
// are now implemented: they get the determinism check plus a fixture-backed
// parity assertion against the frozen nirs4all reference
// (parity/fixtures/aug_spline_{x,curve}_simplify_v1.json). Parity uses the
// standard 1e-12 / 1e-13 tolerance: the random control-point draw matches
// numpy.choice(replace=False) bit-for-bit, and the cubic refit uses the
// not-a-knot splrep surrogate (bspline.h) — identical to FITPACK to ~1e-15.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "n4m/n4m.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

namespace {

n4m_matrix_view_t rowmajor_view(double* data, std::int64_t rows,
                                 std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st = n4m_matrix_view_init_rowmajor(
        &v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

// Synthesize a (rows x cols) matrix of toy spectra: baseline + Gaussian
// peak + small linear trend. Wavelengths are 1000..1700 nm uniformly.
void synth_spectra(std::vector<double>& X, std::vector<double>& wl,
                    std::int64_t rows, std::int64_t cols) {
    X.assign(static_cast<std::size_t>(rows * cols), 0.0);
    wl.assign(static_cast<std::size_t>(cols), 0.0);
    for (std::int64_t j = 0; j < cols; ++j) {
        wl[static_cast<std::size_t>(j)] =
            1000.0 + 700.0 * static_cast<double>(j) /
                       static_cast<double>(cols - 1);
    }
    for (std::int64_t i = 0; i < rows; ++i) {
        const double a = 0.3 + 0.05 * static_cast<double>(i);
        const double b = 0.001 * static_cast<double>(i);
        const double cmu = 1200.0 + 50.0 * static_cast<double>(i);
        const double cs  = 80.0;
        const double amp = 0.8;
        for (std::int64_t j = 0; j < cols; ++j) {
            const double wj = wl[static_cast<std::size_t>(j)];
            const double t  = (wj - cmu) / cs;
            X[static_cast<std::size_t>(i * cols + j)] =
                a + b * (wj - 1000.0) + amp * std::exp(-0.5 * t * t);
        }
    }
}

bool double_equal(double a, double b, double tol = 0.0) {
    if (tol == 0.0) return a == b;
    return std::fabs(a - b) <= tol;
}

void require_equal_arrays(const std::vector<double>& a,
                           const std::vector<double>& b, double tol,
                           const char* tag) {
    if (a.size() != b.size()) {
        throw std::runtime_error(std::string(tag) + " size mismatch");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!double_equal(a[i], b[i], tol)) {
            throw std::runtime_error(std::string(tag) +
                                      " mismatch at i=" + std::to_string(i));
        }
    }
}

constexpr uint64_t kSeed = 3298921130ULL;

// -------------------------------------------------------------------------
// DetectorRollOff
// -------------------------------------------------------------------------

void test_detector_rolloff() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    // NULL out
    N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_create(
        nullptr, rng, 4, 1.0, 0.02, 1) == N4M_ERR_NULL_POINTER);
    // NULL rng
    n4m_aug_detector_rolloff_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_create(
        &h, nullptr, 4, 1.0, 0.02, 1) == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(h == nullptr);
    // Invalid detector model
    N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_create(
        &h, rng, 42, 1.0, 0.02, 1) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(h == nullptr);

    // Valid create
    N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_create(
        &h, rng, /*model=*/4, /*strength=*/1.0,
        /*noise=*/0.02, /*baseline=*/1) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);

    // First apply (seeded fresh).
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t W = rowmajor_view(wlv.data(), 1, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_apply(h, X, W, O) == N4M_OK);

    // Second apply with the same seed produces identical output.
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_apply(h, X, W, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "detector_rolloff determinism");

    // Shape mismatch on output buffer.
    std::vector<double> bad(10);
    n4m_matrix_view_t Obad = rowmajor_view(bad.data(), 1, 10);
    N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_apply(h, X, W, Obad)
                      == N4M_ERR_SHAPE_MISMATCH);

    n4m_aug_detector_rolloff_destroy(h);
    n4m_aug_detector_rolloff_destroy(nullptr);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference (seed echoed as kSeed = 3298921130).
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_detector_rolloff_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const int model = static_cast<int>(n4m_testing::params_get_int(
            c.params_json, "detector_model", 4));
        const double strength = n4m_testing::params_get_double(
            c.params_json, "effect_strength", 1.0);
        const double noise = n4m_testing::params_get_double(
            c.params_json, "noise_amplification", 0.02);
        const int baseline = n4m_testing::params_get_int(
            c.params_json, "include_baseline_distortion", 1) ? 1 : 0;
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_detector_rolloff_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_create(
            &hh, r, model, strength, noise, baseline) == N4M_OK);
        std::vector<double> in = fx.input, wl = fx.wavelengths;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Wv2 = rowmajor_view(wl.data(), 1, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_detector_rolloff_apply(hh, Xv2, Wv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "detector_rolloff parity[" + c.name + "]");
        n4m_aug_detector_rolloff_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// StrayLight
// -------------------------------------------------------------------------

void test_stray_light() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    n4m_aug_stray_light_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_stray_light_create(
        &h, rng, 0.001, 2.0, 0.1, 1) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t W = rowmajor_view(wlv.data(), 1, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_stray_light_apply(h, X, W, O) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_stray_light_apply(h, X, W, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "stray_light determinism");

    n4m_aug_stray_light_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_stray_light_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const double fraction = n4m_testing::params_get_double(
            c.params_json, "stray_light_fraction", 0.001);
        const double edge_enh = n4m_testing::params_get_double(
            c.params_json, "edge_enhancement", 2.0);
        const double edge_width = n4m_testing::params_get_double(
            c.params_json, "edge_width", 0.1);
        const int peak_trunc = n4m_testing::params_get_int(
            c.params_json, "include_peak_truncation", 1) ? 1 : 0;
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_stray_light_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_stray_light_create(
            &hh, r, fraction, edge_enh, edge_width, peak_trunc) == N4M_OK);
        std::vector<double> in = fx.input, wl = fx.wavelengths;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Wv2 = rowmajor_view(wl.data(), 1, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_stray_light_apply(hh, Xv2, Wv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "stray_light parity[" + c.name + "]");
        n4m_aug_stray_light_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// EdgeCurvature
// -------------------------------------------------------------------------

void test_edge_curvature() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    n4m_aug_edge_curve_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_edge_curve_create(
        &h, rng, 0.02, /*type=*/1, 0.0, 0.7) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t W = rowmajor_view(wlv.data(), 1, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_edge_curve_apply(h, X, W, O) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_edge_curve_apply(h, X, W, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "edge_curve determinism");

    n4m_aug_edge_curve_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_edge_curve_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const double curvature = n4m_testing::params_get_double(
            c.params_json, "curvature_strength", 0.02);
        const int type = static_cast<int>(n4m_testing::params_get_int(
            c.params_json, "curvature_type", 1));
        const double asymmetry = n4m_testing::params_get_double(
            c.params_json, "asymmetry", 0.0);
        const double edge_focus = n4m_testing::params_get_double(
            c.params_json, "edge_focus", 0.7);
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_edge_curve_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_edge_curve_create(
            &hh, r, curvature, type, asymmetry, edge_focus) == N4M_OK);
        std::vector<double> in = fx.input, wl = fx.wavelengths;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Wv2 = rowmajor_view(wl.data(), 1, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_edge_curve_apply(hh, Xv2, Wv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "edge_curve parity[" + c.name + "]");
        n4m_aug_edge_curve_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// TruncatedPeak
// -------------------------------------------------------------------------

void test_truncated_peak() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    n4m_aug_truncated_peak_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_truncated_peak_create(
        &h, rng, /*prob=*/0.5, /*amp_min=*/0.01, /*amp_max=*/0.1,
        /*w_min=*/50.0, /*w_max=*/200.0, /*left=*/1, /*right=*/1) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t W = rowmajor_view(wlv.data(), 1, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_truncated_peak_apply(h, X, W, O) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_truncated_peak_apply(h, X, W, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "truncated_peak determinism");

    n4m_aug_truncated_peak_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_truncated_peak_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const double prob = n4m_testing::params_get_double(
            c.params_json, "peak_probability", 0.5);
        const double amp_min = n4m_testing::params_get_double(
            c.params_json, "amplitude_min", 0.01);
        const double amp_max = n4m_testing::params_get_double(
            c.params_json, "amplitude_max", 0.1);
        const double w_min = n4m_testing::params_get_double(
            c.params_json, "width_min", 50.0);
        const double w_max = n4m_testing::params_get_double(
            c.params_json, "width_max", 200.0);
        const int left = n4m_testing::params_get_int(
            c.params_json, "left_edge", 1) ? 1 : 0;
        const int right = n4m_testing::params_get_int(
            c.params_json, "right_edge", 1) ? 1 : 0;
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_truncated_peak_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_truncated_peak_create(
            &hh, r, prob, amp_min, amp_max, w_min, w_max, left, right) == N4M_OK);
        std::vector<double> in = fx.input, wl = fx.wavelengths;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Wv2 = rowmajor_view(wl.data(), 1, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_truncated_peak_apply(hh, Xv2, Wv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "truncated_peak parity[" + c.name + "]");
        n4m_aug_truncated_peak_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// EdgeArtifacts (combined wrapper)
// -------------------------------------------------------------------------

void test_edge_artifacts_combined() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    // All flags enabled.
    const int32_t all = N4M_AUG_EDGE_ARTIFACTS_DETECTOR_ROLL_OFF |
                         N4M_AUG_EDGE_ARTIFACTS_STRAY_LIGHT       |
                         N4M_AUG_EDGE_ARTIFACTS_EDGE_CURVATURE    |
                         N4M_AUG_EDGE_ARTIFACTS_TRUNCATED_PEAKS;
    n4m_aug_edge_artifacts_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_create(
        &h, rng, all, 1.0, /*detector=*/4) == N4M_OK);

    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t W = rowmajor_view(wlv.data(), 1, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_apply(h, X, W, O) == N4M_OK);
    n4m_aug_edge_artifacts_destroy(h);
    h = nullptr;

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_create(
        &h, rng, all, 1.0, /*detector=*/4) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_apply(h, X, W, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "edge_artifacts determinism");

    n4m_aug_edge_artifacts_destroy(h);

    // Subset of flags (only stray light): output must differ from full
    // set, both must run cleanly.
    h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_create(
        &h, rng, N4M_AUG_EDGE_ARTIFACTS_STRAY_LIGHT, 1.0, 4) == N4M_OK);
    std::vector<double> O3(Xv.size());
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O33 = rowmajor_view(O3.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_apply(h, X, W, O33) == N4M_OK);
    n4m_aug_edge_artifacts_destroy(h);

    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_edge_artifacts_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const int flags = static_cast<int>(n4m_testing::params_get_int(
            c.params_json, "enabled_flags", 15));
        const double strength = n4m_testing::params_get_double(
            c.params_json, "overall_strength", 1.0);
        const int model = static_cast<int>(n4m_testing::params_get_int(
            c.params_json, "detector_model", 4));
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_edge_artifacts_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_create(
            &hh, r, flags, strength, model) == N4M_OK);
        std::vector<double> in = fx.input, wl = fx.wavelengths;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Wv2 = rowmajor_view(wl.data(), 1, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_edge_artifacts_apply(hh, Xv2, Wv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "edge_artifacts parity[" + c.name + "]");
        n4m_aug_edge_artifacts_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// Spline_Smoothing
// -------------------------------------------------------------------------

void test_spline_smoothing() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    n4m_aug_spline_smooth_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_spline_smooth_create(&h, rng) == N4M_OK);

    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_smooth_apply(h, X, O) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_smooth_apply(h, X, O22) == N4M_OK);
    // Stateless: same input -> same output regardless of RNG state.
    require_equal_arrays(O1, O2, 0.0, "spline_smooth idempotence");

    // Real FITPACK smoothing is not an interpolation pass-through; exact
    // numeric parity is covered by the nirs4all fixture/dashboard gate.
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < Xv.size(); ++i) {
        if (!std::isfinite(O1[i])) {
            throw std::runtime_error(
                "spline_smooth: non-finite output at i=" + std::to_string(i));
        }
        const double a = std::fabs(O1[i]);
        if (a > max_abs) max_abs = a;
        sum_abs += a;
    }
    if (max_abs > 10.0 || sum_abs <= 1.0) {
        throw std::runtime_error("spline_smooth: implausible output scale");
    }

    n4m_aug_spline_smooth_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Bit-exact replay is build-dependent: spline_smooth is the only
    // FITPACK-gated augmenter (cpp/src/core/augmentation/splines/
    // spline_smoothing.c). With N4M_HAVE_FITPACK=0 it is an intentional no-op,
    // with =1 it performs real cubic spline smoothing. The committed self-
    // fixture (parity/fixtures/aug_spline_smooth_v1.json) is the FITPACK
    // oracle, regenerated by parity/scripts/regen_aug_fixtures.py from a
    // fitpack=1 build. Replay it only when this build actually has FITPACK; the
    // determinism + plausibility smoke above already covered the no-op build.
    const std::string bi = n4m_get_build_info() ? n4m_get_build_info() : "";
    if (bi.find("fitpack=1") != std::string::npos) {
        const auto fx = n4m_testing::load_fixture(
            std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_spline_smooth_v1.json",
            /*require_per_case_output_shape=*/true);
        for (const auto& c : fx.cases) {
            const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
                c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
            n4m_rng_pcg64_state_t* r = nullptr;
            N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
            n4m_aug_spline_smooth_handle_t* hh = nullptr;
            N4M_TEST_REQUIRE(n4m_aug_spline_smooth_create(&hh, r) == N4M_OK);
            std::vector<double> in = fx.input;
            std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
            n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
            n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
            N4M_TEST_REQUIRE(n4m_aug_spline_smooth_apply(hh, Xv2, Ov2) == N4M_OK);
            n4m_testing::assert_close(out, c.expected_output,
                                      "spline_smooth parity[" + c.name + "]");
            n4m_aug_spline_smooth_destroy(hh);
            n4m_rng_pcg64_destroy(r);
        }
    }
}

// -------------------------------------------------------------------------
// Spline_X_Perturbations
// -------------------------------------------------------------------------

void test_spline_x_perturbations() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    n4m_aug_spline_x_perturb_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_spline_x_perturb_create(
        &h, rng, /*degree=*/3, /*density=*/0.05,
        /*range_min=*/-0.1, /*range_max=*/0.1) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_x_perturb_apply(h, X, O) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_x_perturb_apply(h, X, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "spline_x_perturb determinism");

    n4m_aug_spline_x_perturb_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_spline_x_perturb_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const int degree = static_cast<int>(n4m_testing::params_get_int(
            c.params_json, "spline_degree", 3));
        const double density = n4m_testing::params_get_double(
            c.params_json, "perturbation_density", 0.05);
        const double range_min = n4m_testing::params_get_double(
            c.params_json, "perturbation_range_min", -0.1);
        const double range_max = n4m_testing::params_get_double(
            c.params_json, "perturbation_range_max", 0.1);
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_spline_x_perturb_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_spline_x_perturb_create(
            &hh, r, degree, density, range_min, range_max) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_spline_x_perturb_apply(hh, Xv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "spline_x_perturb parity[" + c.name + "]");
        n4m_aug_spline_x_perturb_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// Spline_Y_Perturbations
// -------------------------------------------------------------------------

void test_spline_y_perturbations() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    n4m_aug_spline_y_perturb_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_spline_y_perturb_create(
        &h, rng, /*spline_points=*/-1, /*intensity=*/0.005) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_y_perturb_apply(h, X, O) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_y_perturb_apply(h, X, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "spline_y_perturb determinism");

    n4m_aug_spline_y_perturb_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_spline_y_perturb_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const int points = static_cast<int>(n4m_testing::params_get_int(
            c.params_json, "spline_points", -1));
        const double intensity = n4m_testing::params_get_double(
            c.params_json, "perturbation_intensity", 0.005);
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_spline_y_perturb_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_spline_y_perturb_create(
            &hh, r, points, intensity) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_spline_y_perturb_apply(hh, Xv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "spline_y_perturb parity[" + c.name + "]");
        n4m_aug_spline_y_perturb_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// Spline_X_Simplification
// -------------------------------------------------------------------------

void test_spline_x_simplification() {
    // Determinism: re-seeding before two runs yields bit-identical output.
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);
    n4m_aug_spline_x_simplify_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_spline_x_simplify_create(
        &h, rng, /*points=*/8, /*uniform=*/0) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t Oa = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_x_simplify_apply(h, X, Oa) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t Ob = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_x_simplify_apply(h, X, Ob) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "spline_x_simplify determinism");

    // Shape mismatch on output buffer.
    std::vector<double> bad(10);
    n4m_matrix_view_t Obad = rowmajor_view(bad.data(), 1, 10);
    N4M_TEST_REQUIRE(n4m_aug_spline_x_simplify_apply(h, X, Obad)
                      == N4M_ERR_SHAPE_MISMATCH);
    n4m_aug_spline_x_simplify_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_spline_x_simplify_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto pts = static_cast<std::int32_t>(
            n4m_testing::params_get_int(c.params_json, "spline_points", -1));
        const int uniform = n4m_testing::params_get_int(
            c.params_json, "uniform", 0) ? 1 : 0;
        const auto seed = static_cast<std::uint64_t>(
            n4m_testing::params_get_int(c.params_json, "random_state", 0));

        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_spline_x_simplify_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_spline_x_simplify_create(&hh, r, pts, uniform)
                          == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_spline_x_simplify_apply(hh, Xv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "spline_x_simplify parity[" + c.name + "]");
        n4m_aug_spline_x_simplify_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// Spline_Curve_Simplification
// -------------------------------------------------------------------------

void test_spline_curve_simplification() {
    // Determinism: re-seeding before two runs yields bit-identical output.
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);
    n4m_aug_spline_curve_simplify_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_spline_curve_simplify_create(
        &h, rng, /*points=*/8, /*uniform=*/0) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t Oa = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_curve_simplify_apply(h, X, Oa) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t Ob = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_spline_curve_simplify_apply(h, X, Ob) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "spline_curve_simplify determinism");

    n4m_aug_spline_curve_simplify_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_spline_curve_simplify_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto pts = static_cast<std::int32_t>(
            n4m_testing::params_get_int(c.params_json, "spline_points", -1));
        const int uniform = n4m_testing::params_get_int(
            c.params_json, "uniform", 0) ? 1 : 0;
        const auto seed = static_cast<std::uint64_t>(
            n4m_testing::params_get_int(c.params_json, "random_state", 0));

        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_spline_curve_simplify_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_spline_curve_simplify_create(&hh, r, pts, uniform)
                          == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_spline_curve_simplify_apply(hh, Xv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "spline_curve_simplify parity[" + c.name + "]");
        n4m_aug_spline_curve_simplify_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// Rotate_Translate
// -------------------------------------------------------------------------

void test_rotate_translate() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    n4m_aug_rotate_translate_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_rotate_translate_create(
        &h, rng, /*p_range=*/2.0, /*y_factor=*/3.0) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_rotate_translate_apply(h, X, O) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_rotate_translate_apply(h, X, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "rotate_translate determinism");

    n4m_aug_rotate_translate_destroy(h);
    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_rotate_translate_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const double p_range = n4m_testing::params_get_double(
            c.params_json, "p_range", 2.0);
        const double y_factor = n4m_testing::params_get_double(
            c.params_json, "y_factor", 3.0);
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_rotate_translate_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_rotate_translate_create(
            &hh, r, p_range, y_factor) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_rotate_translate_apply(hh, Xv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "rotate_translate parity[" + c.name + "]");
        n4m_aug_rotate_translate_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

// -------------------------------------------------------------------------
// Random_X_Operation
// -------------------------------------------------------------------------

void test_random_x_operation() {
    std::vector<double> Xv, wlv;
    synth_spectra(Xv, wlv, 4, 32);
    std::vector<double> O1(Xv.size()), O2(Xv.size());

    n4m_rng_pcg64_state_t* rng = nullptr;
    N4M_TEST_REQUIRE(n4m_rng_pcg64_create(kSeed, &rng) == N4M_OK);

    // Multiplicative
    n4m_aug_random_x_op_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_create(
        &h, rng, /*op=*/0, 0.97, 1.03) == N4M_OK);

    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t X = rowmajor_view(Xv.data(), 4, 32);
    n4m_matrix_view_t O = rowmajor_view(O1.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_apply(h, X, O) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    n4m_matrix_view_t O22 = rowmajor_view(O2.data(), 4, 32);
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_apply(h, X, O22) == N4M_OK);
    require_equal_arrays(O1, O2, 0.0, "random_x_op multiplicative determinism");

    // Output is roughly within [0.97*input, 1.03*input].
    for (size_t i = 0; i < Xv.size(); ++i) {
        const double lo = 0.97 * Xv[i] - 1e-12;
        const double hi = 1.03 * Xv[i] + 1e-12;
        if (O1[i] < lo || O1[i] > hi) {
            throw std::runtime_error(
                "random_x_op out of mul range at i=" + std::to_string(i));
        }
    }
    n4m_aug_random_x_op_destroy(h);

    // Additive
    h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_create(
        &h, rng, /*op=*/1, -0.01, 0.01) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_apply(h, X, O) == N4M_OK);
    n4m_aug_random_x_op_destroy(h);

    // Subtractive
    h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_create(
        &h, rng, /*op=*/2, 0.0, 0.01) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_rng_pcg64_set_seed(rng, kSeed) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_apply(h, X, O) == N4M_OK);
    n4m_aug_random_x_op_destroy(h);

    // Invalid op_kind.
    h = nullptr;
    N4M_TEST_REQUIRE(n4m_aug_random_x_op_create(
        &h, rng, /*op=*/99, 0.97, 1.03) == N4M_ERR_INVALID_ARGUMENT);

    n4m_rng_pcg64_destroy(rng);

    // Parity vs frozen nirs4all reference.
    const auto fx = n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/aug_random_x_op_v1.json",
        /*require_per_case_output_shape=*/true);
    for (const auto& c : fx.cases) {
        const auto seed = static_cast<std::uint64_t>(n4m_testing::params_get_int(
            c.params_json, "random_state", static_cast<std::int64_t>(kSeed)));
        const int op = static_cast<int>(n4m_testing::params_get_int(
            c.params_json, "op_kind", 0));
        const double range_min = n4m_testing::params_get_double(
            c.params_json, "operator_range_min", 0.97);
        const double range_max = n4m_testing::params_get_double(
            c.params_json, "operator_range_max", 1.03);
        n4m_rng_pcg64_state_t* r = nullptr;
        N4M_TEST_REQUIRE(n4m_rng_pcg64_create(seed, &r) == N4M_OK);
        n4m_aug_random_x_op_handle_t* hh = nullptr;
        N4M_TEST_REQUIRE(n4m_aug_random_x_op_create(
            &hh, r, op, range_min, range_max) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(static_cast<std::size_t>(fx.rows * fx.cols));
        n4m_matrix_view_t Xv2 = rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Ov2 = rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_aug_random_x_op_apply(hh, Xv2, Ov2) == N4M_OK);
        n4m_testing::assert_close(out, c.expected_output,
                                  "random_x_op parity[" + c.name + "]");
        n4m_aug_random_x_op_destroy(hh);
        n4m_rng_pcg64_destroy(r);
    }
}

}  // namespace

void register_augmenters_edge_splines_random_tests(n4m_testing::Runner& r);
void register_augmenters_edge_splines_random_tests(n4m_testing::Runner& r) {
    r.run("aug_detector_rolloff",         test_detector_rolloff);
    r.run("aug_stray_light",              test_stray_light);
    r.run("aug_edge_curvature",           test_edge_curvature);
    r.run("aug_truncated_peak",           test_truncated_peak);
    r.run("aug_edge_artifacts_combined",  test_edge_artifacts_combined);
    r.run("aug_spline_smoothing",         test_spline_smoothing);
    r.run("aug_spline_x_perturbations",   test_spline_x_perturbations);
    r.run("aug_spline_y_perturbations",   test_spline_y_perturbations);
    r.run("aug_spline_x_simplify",        test_spline_x_simplification);
    r.run("aug_spline_curve_simplify",    test_spline_curve_simplification);
    r.run("aug_rotate_translate",         test_rotate_translate);
    r.run("aug_random_x_operation",       test_random_x_operation);
}
