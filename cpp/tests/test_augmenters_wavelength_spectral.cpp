// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 16 wavelength + spectral augmenters
// (10 operators × 3 symbols = 30 new C ABI symbols).
//
// Each operator contributes one smoke test (exercise create/apply/destroy
// on a small inline matrix) and one parity test (load a JSON fixture
// produced by the matching generator under
// parity/python_generator/scripts/generate_phase16_fixtures.py, run the C
// engine, assert byte-for-byte against the frozen NumPy reference).
//
// Phase 16 lands its own test driver (chemometrics4all_tests_aug) — the
// canonical chemometrics4all_tests main.cpp is reserved for the post-merge
// integration commit, matching the Phase 12 / Phase 14 coordination
// pattern. Once Phases 15-18 are merged the two binaries fold into one.
//
// Tolerance: 1e-12 abs / 1e-12 rel (the locked PCG64 contract). The C
// engine consumes the same PCG64 stream as NumPy 1.26.4's
// `Generator.uniform / integers`, so the output is bit-equivalent up to
// IEEE-754 round-off in the linear-interp / convolution kernels.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

using ParityFixture = ::c4a_testing::Fixture;
using ::c4a_testing::assert_close;
using ::c4a_testing::params_get_double;
using ::c4a_testing::params_get_int;
using ::c4a_testing::params_get_string;

constexpr double kAbsTol = 1e-12;
constexpr double kRelTol = 1e-12;

ParityFixture load_fixture(const std::string& filename) {
    return ::c4a_testing::load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/true);
}

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

// Reseed the supplied RNG to the case's per-fixture seed. Every parity
// case carries `"seed": <int>` in its params payload.
void reseed_rng(c4a_rng_pcg64_state_t* rng, const std::string& params_json) {
    const std::int64_t seed = params_get_int(params_json, "seed", 0);
    C4A_TEST_REQUIRE(c4a_rng_pcg64_set_seed(rng,
                                              static_cast<std::uint64_t>(seed))
                     == C4A_OK);
}

// ---------------------------------------------------------------------------
// Smoke tests — one per operator. Each exercises the create / apply /
// destroy lifecycle on a 4 x 8 toy matrix.
// ---------------------------------------------------------------------------

void test_wavelength_shift_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[8 * 4] = {
        1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9,
        0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6,
        2.0, 2.1, 2.2, 2.3, 2.2, 2.1, 2.0, 1.9,
        1.5, 1.6, 1.7, 1.8, 1.9, 1.8, 1.7, 1.6,
    };
    double Y[8 * 4];
    c4a_aug_wavelength_shift_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_wavelength_shift_create(
        &h, rng, -1.0, 1.0, nullptr, 0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 8);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 8);
    C4A_TEST_REQUIRE(c4a_aug_wavelength_shift_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_wavelength_shift_destroy(h);
    c4a_aug_wavelength_shift_destroy(nullptr);

    // Invalid: NULL RNG.
    C4A_TEST_REQUIRE(c4a_aug_wavelength_shift_create(
        &h, nullptr, 0.0, 1.0, nullptr, 0) == C4A_ERR_NULL_POINTER);
    // Invalid: shift_hi < shift_lo.
    C4A_TEST_REQUIRE(c4a_aug_wavelength_shift_create(
        &h, rng, 2.0, 1.0, nullptr, 0) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_wavelength_stretch_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_wavelength_stretch_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_wavelength_stretch_create(
        &h, rng, 0.95, 1.05, nullptr, 0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_wavelength_stretch_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_wavelength_stretch_destroy(h);

    // stretch_lo <= 0 rejected.
    C4A_TEST_REQUIRE(c4a_aug_wavelength_stretch_create(
        &h, rng, 0.0, 1.0, nullptr, 0) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_local_warp_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 1.2, 1.4, 1.6,
                      1.6, 1.4, 1.2, 1.0,
                      0.5, 0.7, 0.9, 1.1,
                      1.1, 0.9, 0.7, 0.5 };
    double Y[16];
    c4a_aug_local_warp_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_local_warp_create(
        &h, rng, 5, 0.5, nullptr, 0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_local_warp_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_local_warp_destroy(h);

    // n_control_points < 2 rejected.
    C4A_TEST_REQUIRE(c4a_aug_local_warp_create(
        &h, rng, 1, 0.5, nullptr, 0) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_band_perturb_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_band_perturb_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_band_perturb_create(
        &h, rng, 2, 1, 3, 0.9, 1.1, -0.1, 0.1) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_band_perturb_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_band_perturb_destroy(h);

    // bw_hi <= bw_lo rejected.
    C4A_TEST_REQUIRE(c4a_aug_band_perturb_create(
        &h, rng, 2, 5, 5, 0.9, 1.1, -0.1, 0.1) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_band_mask_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_band_mask_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_band_mask_create(
        &h, rng, 1, 2, 1, 3, /*mode=zero*/0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_band_mask_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_band_mask_destroy(h);

    // Invalid mode rejected.
    C4A_TEST_REQUIRE(c4a_aug_band_mask_create(
        &h, rng, 1, 2, 1, 3, /*mode=*/99) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_channel_dropout_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_channel_dropout_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_channel_dropout_create(
        &h, rng, 0.2, /*mode=interp*/1) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_channel_dropout_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_channel_dropout_destroy(h);

    // Probability out of range rejected.
    C4A_TEST_REQUIRE(c4a_aug_channel_dropout_create(
        &h, rng, 1.5, 0) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_gauss_jitter_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_gauss_jitter_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_gauss_jitter_create(
        &h, rng, 0.5, 1.5, /*kernel_width=*/5) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_gauss_jitter_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_gauss_jitter_destroy(h);

    // sigma_lo <= 0 rejected.
    C4A_TEST_REQUIRE(c4a_aug_gauss_jitter_create(
        &h, rng, 0.0, 1.0, 5) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_unsharp_mask_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_unsharp_mask_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_unsharp_mask_create(
        &h, rng, 0.1, 0.5, 1.0, 5) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_unsharp_mask_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_unsharp_mask_destroy(h);

    // sigma <= 0 rejected.
    C4A_TEST_REQUIRE(c4a_aug_unsharp_mask_create(
        &h, rng, 0.1, 0.5, 0.0, 5) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_magnitude_warp_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_magnitude_warp_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_magnitude_warp_create(
        &h, rng, 3, 0.9, 1.1, nullptr, 0) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_magnitude_warp_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_magnitude_warp_destroy(h);

    // n_control_points < 2 rejected.
    C4A_TEST_REQUIRE(c4a_aug_magnitude_warp_create(
        &h, rng, 1, 0.9, 1.1, nullptr, 0) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

void test_local_clip_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    double X[16] = { 1.0, 2.0, 3.0, 4.0,
                      4.0, 3.0, 2.0, 1.0,
                      1.5, 2.5, 3.5, 4.5,
                      4.5, 3.5, 2.5, 1.5 };
    double Y[16];
    c4a_aug_local_clip_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_local_clip_create(
        &h, rng, 1, 1, 3) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 4, 4);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 4, 4);
    C4A_TEST_REQUIRE(c4a_aug_local_clip_apply(h, Xv, Yv) == C4A_OK);
    c4a_aug_local_clip_destroy(h);

    // width_hi <= width_lo rejected.
    C4A_TEST_REQUIRE(c4a_aug_local_clip_create(
        &h, rng, 1, 5, 5) == C4A_ERR_INVALID_ARGUMENT);
    c4a_rng_pcg64_destroy(rng);
}

// ---------------------------------------------------------------------------
// Parity verifiers — generic infrastructure parametrised by a single
// `apply` lambda. Each operator builds its handle from the case params,
// calls apply, and asserts against the fixture's expected output.
// ---------------------------------------------------------------------------

template <typename CreateApplyDestroy>
void run_parity_fixture(const std::string& filename,
                         CreateApplyDestroy&& body) {
    ParityFixture fx = load_fixture(filename);
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0u, &rng) == C4A_OK);
    for (const auto& c : fx.cases) {
        reseed_rng(rng, c.params_json);
        std::vector<double> in = fx.input;
        std::vector<double> out(static_cast<std::size_t>(c.output_rows) *
                                  static_cast<std::size_t>(c.output_cols), 0.0);
        c4a_matrix_view_t Xv =
            make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv =
            make_rowmajor_view(out.data(), c.output_rows, c.output_cols);
        body(rng, Xv, Yv, c.params_json);
        assert_close(out, c.expected_output, filename + "/" + c.name,
                     kAbsTol, kRelTol);
    }
    c4a_rng_pcg64_destroy(rng);
}

void verify_wavelength_shift_parity() {
    run_parity_fixture(
        "aug_wavelength_shift_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const double lo = params_get_double(params, "shift_lo", -1.0);
            const double hi = params_get_double(params, "shift_hi",  1.0);
            c4a_aug_wavelength_shift_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_wavelength_shift_create(
                &h, rng, lo, hi, nullptr, 0) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_wavelength_shift_apply(h, X, out)
                             == C4A_OK);
            c4a_aug_wavelength_shift_destroy(h);
        });
}

void verify_wavelength_stretch_parity() {
    run_parity_fixture(
        "aug_wavelength_stretch_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const double lo = params_get_double(params, "stretch_lo", 0.95);
            const double hi = params_get_double(params, "stretch_hi", 1.05);
            c4a_aug_wavelength_stretch_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_wavelength_stretch_create(
                &h, rng, lo, hi, nullptr, 0) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_wavelength_stretch_apply(h, X, out)
                             == C4A_OK);
            c4a_aug_wavelength_stretch_destroy(h);
        });
}

void verify_local_warp_parity() {
    run_parity_fixture(
        "aug_local_warp_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const std::int32_t k = static_cast<std::int32_t>(
                params_get_int(params, "n_control_points", 5));
            const double ms = params_get_double(params, "max_shift", 1.0);
            c4a_aug_local_warp_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_local_warp_create(
                &h, rng, k, ms, nullptr, 0) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_local_warp_apply(h, X, out) == C4A_OK);
            c4a_aug_local_warp_destroy(h);
        });
}

void verify_band_perturb_parity() {
    run_parity_fixture(
        "aug_band_perturb_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const std::int32_t nb = static_cast<std::int32_t>(
                params_get_int(params, "n_bands", 3));
            const std::int32_t bw_lo = static_cast<std::int32_t>(
                params_get_int(params, "bw_lo", 5));
            const std::int32_t bw_hi = static_cast<std::int32_t>(
                params_get_int(params, "bw_hi", 15));
            const double g_lo = params_get_double(params, "gain_lo", 0.9);
            const double g_hi = params_get_double(params, "gain_hi", 1.1);
            const double o_lo = params_get_double(params, "offset_lo", -0.01);
            const double o_hi = params_get_double(params, "offset_hi",  0.01);
            c4a_aug_band_perturb_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_band_perturb_create(
                &h, rng, nb, bw_lo, bw_hi, g_lo, g_hi, o_lo, o_hi) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_band_perturb_apply(h, X, out) == C4A_OK);
            c4a_aug_band_perturb_destroy(h);
        });
}

void verify_band_mask_parity() {
    run_parity_fixture(
        "aug_band_mask_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const std::int32_t nb_lo = static_cast<std::int32_t>(
                params_get_int(params, "n_bands_lo", 1));
            const std::int32_t nb_hi = static_cast<std::int32_t>(
                params_get_int(params, "n_bands_hi", 3));
            const std::int32_t bw_lo = static_cast<std::int32_t>(
                params_get_int(params, "bw_lo", 5));
            const std::int32_t bw_hi = static_cast<std::int32_t>(
                params_get_int(params, "bw_hi", 15));
            const std::string mode = params_get_string(params, "mode",
                                                        "zero");
            const std::int32_t mode_id = (mode == "interp") ? 1 : 0;
            c4a_aug_band_mask_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_band_mask_create(
                &h, rng, nb_lo, nb_hi, bw_lo, bw_hi, mode_id) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_band_mask_apply(h, X, out) == C4A_OK);
            c4a_aug_band_mask_destroy(h);
        });
}

void verify_channel_dropout_parity() {
    run_parity_fixture(
        "aug_channel_dropout_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const double p = params_get_double(params, "dropout_prob", 0.05);
            const std::string mode = params_get_string(params, "mode", "zero");
            const std::int32_t mode_id = (mode == "interp") ? 1 : 0;
            c4a_aug_channel_dropout_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_channel_dropout_create(
                &h, rng, p, mode_id) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_channel_dropout_apply(h, X, out)
                             == C4A_OK);
            c4a_aug_channel_dropout_destroy(h);
        });
}

void verify_gauss_jitter_parity() {
    run_parity_fixture(
        "aug_gauss_jitter_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const double lo = params_get_double(params, "sigma_lo", 0.5);
            const double hi = params_get_double(params, "sigma_hi", 1.5);
            const std::int32_t w = static_cast<std::int32_t>(
                params_get_int(params, "kernel_width", 9));
            c4a_aug_gauss_jitter_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_gauss_jitter_create(
                &h, rng, lo, hi, w) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_gauss_jitter_apply(h, X, out)
                             == C4A_OK);
            c4a_aug_gauss_jitter_destroy(h);
        });
}

void verify_unsharp_mask_parity() {
    run_parity_fixture(
        "aug_unsharp_mask_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const double a_lo = params_get_double(params, "amount_lo", 0.1);
            const double a_hi = params_get_double(params, "amount_hi", 0.5);
            const double sg   = params_get_double(params, "sigma",      1.0);
            const std::int32_t w = static_cast<std::int32_t>(
                params_get_int(params, "kernel_width", 11));
            c4a_aug_unsharp_mask_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_unsharp_mask_create(
                &h, rng, a_lo, a_hi, sg, w) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_unsharp_mask_apply(h, X, out)
                             == C4A_OK);
            c4a_aug_unsharp_mask_destroy(h);
        });
}

void verify_magnitude_warp_parity() {
    run_parity_fixture(
        "aug_magnitude_warp_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const std::int32_t k = static_cast<std::int32_t>(
                params_get_int(params, "n_control_points", 5));
            const double g_lo = params_get_double(params, "gain_lo", 0.9);
            const double g_hi = params_get_double(params, "gain_hi", 1.1);
            c4a_aug_magnitude_warp_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_magnitude_warp_create(
                &h, rng, k, g_lo, g_hi, nullptr, 0) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_magnitude_warp_apply(h, X, out)
                             == C4A_OK);
            c4a_aug_magnitude_warp_destroy(h);
        });
}

void verify_local_clip_parity() {
    run_parity_fixture(
        "aug_local_clip_v1.json",
        [](c4a_rng_pcg64_state_t* rng,
           c4a_matrix_view_t X, c4a_matrix_view_t out,
           const std::string& params) {
            const std::int32_t nr = static_cast<std::int32_t>(
                params_get_int(params, "n_regions", 1));
            const std::int32_t w_lo = static_cast<std::int32_t>(
                params_get_int(params, "width_lo", 5));
            const std::int32_t w_hi = static_cast<std::int32_t>(
                params_get_int(params, "width_hi", 15));
            c4a_aug_local_clip_handle_t* h = nullptr;
            C4A_TEST_REQUIRE(c4a_aug_local_clip_create(
                &h, rng, nr, w_lo, w_hi) == C4A_OK);
            C4A_TEST_REQUIRE(c4a_aug_local_clip_apply(h, X, out)
                             == C4A_OK);
            c4a_aug_local_clip_destroy(h);
        });
}

}  // namespace

void register_augmenters_wavelength_spectral_tests(c4a_testing::Runner& r);
void register_augmenters_wavelength_spectral_tests(c4a_testing::Runner& r) {
    r.run("wavelength_shift_smoke",      test_wavelength_shift_smoke);
    r.run("wavelength_stretch_smoke",    test_wavelength_stretch_smoke);
    r.run("local_warp_smoke",            test_local_warp_smoke);
    r.run("band_perturb_smoke",          test_band_perturb_smoke);
    r.run("band_mask_smoke",             test_band_mask_smoke);
    r.run("channel_dropout_smoke",       test_channel_dropout_smoke);
    r.run("gauss_jitter_smoke",          test_gauss_jitter_smoke);
    r.run("unsharp_mask_smoke",          test_unsharp_mask_smoke);
    r.run("magnitude_warp_smoke",        test_magnitude_warp_smoke);
    r.run("local_clip_smoke",            test_local_clip_smoke);

    r.run("wavelength_shift_parity",     verify_wavelength_shift_parity);
    r.run("wavelength_stretch_parity",   verify_wavelength_stretch_parity);
    r.run("local_warp_parity",           verify_local_warp_parity);
    r.run("band_perturb_parity",         verify_band_perturb_parity);
    r.run("band_mask_parity",            verify_band_mask_parity);
    r.run("channel_dropout_parity",      verify_channel_dropout_parity);
    r.run("gauss_jitter_parity",         verify_gauss_jitter_parity);
    r.run("unsharp_mask_parity",         verify_unsharp_mask_parity);
    r.run("magnitude_warp_parity",       verify_magnitude_warp_parity);
    r.run("local_clip_parity",           verify_local_clip_parity);
}
