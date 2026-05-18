// SPDX-License-Identifier: CECILL-2.1
//
// Phase 17 — mixup + physical + environmental augmenters smoke tests +
// fixture-driven parity for the deterministic cases.
//
// Each of the 10 augmenters gets:
//   * a `_smoke` test exercising create/apply/destroy lifecycle with a
//     small inline matrix.
//   * Where bit-exact parity is achievable (no stochastic dependence on
//     scipy.ndimage), a fixture-backed parity test.
//
// Stochastic augmenters that rely on the PCG64 stream are seeded via
// c4a_rng_pcg64_set_seed before each apply call so the test is fully
// reproducible.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "chemometrics4all/c4a.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined"
#endif

void register_augmenters_phase17_tests(c4a_testing::Runner& r);

namespace {

c4a_matrix_view_t make_view(double* p, int64_t r, int64_t c) {
    c4a_matrix_view_t v{};
    C4A_TEST_REQUIRE(c4a_matrix_view_init_rowmajor(&v, p, r, c, C4A_DTYPE_F64)
                     == C4A_OK);
    return v;
}

/* -------- Smoke: MixupAugmenter -------- */
void test_mixup_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_mixup_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_mixup_create(&h, rng, /*alpha=*/0.2) == C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double Y[6] = {0};
    auto Xv = make_view(X, 2, 3);
    auto Yv = make_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_aug_mixup_apply(h, Xv, Yv) == C4A_OK);
    /* MixupAugmenter combines row i with X[permutation[i]] via Beta(alpha,
     * alpha). With alpha=0.2, lambda concentrates near {0, 1}, so the
     * output mostly equals one of the two source rows. Each column j
     * value should be a convex combination of X[0][j] and X[1][j], i.e.
     * within the bound [min(X[*, j]), max(X[*, j])]. */
    for (int j = 0; j < 3; ++j) {
        const double lo = std::fmin(X[j], X[j + 3]) - 1e-12;
        const double hi = std::fmax(X[j], X[j + 3]) + 1e-12;
        C4A_TEST_REQUIRE(Y[j] >= lo && Y[j] <= hi);
        C4A_TEST_REQUIRE(Y[j + 3] >= lo && Y[j + 3] <= hi);
    }
    c4a_aug_mixup_destroy(h);
    c4a_aug_mixup_destroy(nullptr);  /* null-safe */
    c4a_rng_pcg64_destroy(rng);
}

void test_mixup_invalid_args() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(1u, &rng) == C4A_OK);
    c4a_aug_mixup_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_mixup_create(&h, rng, 0.0)
                     == C4A_ERR_INVALID_ARGUMENT);
    C4A_TEST_REQUIRE(h == nullptr);
    C4A_TEST_REQUIRE(c4a_aug_mixup_create(&h, nullptr, 0.2)
                     == C4A_ERR_NULL_POINTER);
    c4a_rng_pcg64_destroy(rng);
}

void test_mixup_determinism() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(123u, &rng) == C4A_OK);
    c4a_aug_mixup_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_mixup_create(&h, rng, 0.5) == C4A_OK);

    const int rows = 4, cols = 5;
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<double> X(n);
    for (size_t i = 0; i < n; ++i) X[i] = static_cast<double>(i);
    std::vector<double> Y1(n, 0.0);
    std::vector<double> Y2(n, 0.0);

    auto Xv = make_view(X.data(), rows, cols);
    auto Y1v = make_view(Y1.data(), rows, cols);
    auto Y2v = make_view(Y2.data(), rows, cols);

    c4a_rng_pcg64_set_seed(rng, 99u);
    C4A_TEST_REQUIRE(c4a_aug_mixup_apply(h, Xv, Y1v) == C4A_OK);

    c4a_rng_pcg64_set_seed(rng, 99u);
    C4A_TEST_REQUIRE(c4a_aug_mixup_apply(h, Xv, Y2v) == C4A_OK);

    for (size_t i = 0; i < n; ++i) {
        C4A_TEST_REQUIRE(Y1[i] == Y2[i]);
    }
    c4a_aug_mixup_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: LocalMixupAugmenter -------- */
void test_local_mixup_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_local_mixup_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_local_mixup_create(&h, rng, 0.2, 2) == C4A_OK);
    /* 4 rows, 3 cols, k=2 neighbors */
    double X[12] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    };
    double Y[12] = {0};
    auto Xv = make_view(X, 4, 3);
    auto Yv = make_view(Y, 4, 3);
    C4A_TEST_REQUIRE(c4a_aug_local_mixup_apply(h, Xv, Yv) == C4A_OK);
    /* All Y values must be finite and within input bounds. */
    for (int i = 0; i < 12; ++i) {
        C4A_TEST_REQUIRE(std::isfinite(Y[i]));
        C4A_TEST_REQUIRE(Y[i] >= 1.0 - 1e-12 && Y[i] <= 12.0 + 1e-12);
    }
    c4a_aug_local_mixup_destroy(h);
    c4a_aug_local_mixup_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: ScatterSimulationMSC -------- */
void test_scatter_sim_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_scatter_sim_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_create(&h, rng,
        /*a_low=*/-0.1, /*a_high=*/0.1,
        /*b_low=*/0.9, /*b_high=*/1.1) == C4A_OK);
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double Y[6] = {0};
    auto Xv = make_view(X, 2, 3);
    auto Yv = make_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_apply(h, Xv, Yv) == C4A_OK);
    /* With b > 0 and a in [-0.1, 0.1], Y should be in [0.7, 6.8] roughly. */
    for (int i = 0; i < 6; ++i) {
        C4A_TEST_REQUIRE(std::isfinite(Y[i]));
    }
    c4a_aug_scatter_sim_destroy(h);
    c4a_aug_scatter_sim_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

void test_scatter_sim_deterministic_pointwise() {
    /* With b_low = b_high = 1 and a_low = a_high = 0.5, the augmentation
     * collapses to out = 0.5 + X (no randomness in the output values). */
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_scatter_sim_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_create(&h, rng, 0.5, 0.5, 1.0, 1.0)
                     == C4A_OK);
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double Y[6] = {0};
    auto Xv = make_view(X, 2, 3);
    auto Yv = make_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 6; ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - (X[i] + 0.5)) < 1e-12);
    }
    c4a_aug_scatter_sim_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: ParticleSizeAugmenter -------- */
void test_particle_size_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    const int cols = 8;
    double wl[8] = {1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0};
    c4a_aug_particle_size_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_particle_size_create(&h, rng,
        /*mean=*/50.0, /*var=*/15.0,
        /*use_range=*/0, /*lo=*/0.0, /*hi=*/0.0,
        /*ref=*/50.0, /*wl_exp=*/1.5,
        /*strength=*/0.1,
        /*include_pl=*/1, /*pl_sens=*/0.5,
        wl, cols) == C4A_OK);
    double X[16] = {1};
    double Y[16] = {0};
    for (int i = 0; i < 16; ++i) X[i] = 0.5 + 0.001 * static_cast<double>(i);
    auto Xv = make_view(X, 2, cols);
    auto Yv = make_view(Y, 2, cols);
    C4A_TEST_REQUIRE(c4a_aug_particle_size_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 16; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));
    c4a_aug_particle_size_destroy(h);
    c4a_aug_particle_size_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: EMSCDistortionAugmenter -------- */
void test_emsc_distort_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    const int cols = 8;
    double wl[8] = {1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0};
    c4a_aug_emsc_distort_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_emsc_distort_create(&h, rng,
        /*mult_low=*/0.9, /*mult_high=*/1.1,
        /*add_low=*/-0.05, /*add_high=*/0.05,
        /*poly_order=*/2, /*poly_strength=*/0.02,
        /*correlation=*/0.3,
        wl, cols) == C4A_OK);
    double X[16] = {0};
    double Y[16] = {0};
    for (int i = 0; i < 16; ++i) X[i] = 0.5;
    auto Xv = make_view(X, 2, cols);
    auto Yv = make_view(Y, 2, cols);
    C4A_TEST_REQUIRE(c4a_aug_emsc_distort_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 16; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));
    c4a_aug_emsc_distort_destroy(h);
    c4a_aug_emsc_distort_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: BatchEffectAugmenter -------- */
void test_batch_effect_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_batch_effect_handle_t* h = nullptr;
    /* No wavelengths — integer x axis. */
    C4A_TEST_REQUIRE(c4a_aug_batch_effect_create(&h, rng,
        /*offset_std=*/0.02, /*slope_std=*/0.01, /*gain_std=*/0.03,
        /*variation_scope=*/0,
        /*wavelengths=*/nullptr, /*n_wavelengths=*/0) == C4A_OK);
    double X[12] = {0};
    double Y[12] = {0};
    for (int i = 0; i < 12; ++i) X[i] = 0.5 + 0.01 * static_cast<double>(i % 6);
    auto Xv = make_view(X, 3, 4);
    auto Yv = make_view(Y, 3, 4);
    C4A_TEST_REQUIRE(c4a_aug_batch_effect_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 12; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));
    c4a_aug_batch_effect_destroy(h);
    c4a_aug_batch_effect_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

void test_batch_effect_zero_variation() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_batch_effect_handle_t* h = nullptr;
    /* zero variation — should pass X through (gain ~ N(1,0) = 1). */
    C4A_TEST_REQUIRE(c4a_aug_batch_effect_create(&h, rng,
        0.0, 0.0, 0.0, /*scope=*/1, nullptr, 0) == C4A_OK);
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double Y[6] = {0};
    auto Xv = make_view(X, 2, 3);
    auto Yv = make_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_aug_batch_effect_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 6; ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - X[i]) < 1e-12);
    }
    c4a_aug_batch_effect_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: InstrumentalBroadeningAugmenter -------- */
void test_instrument_broaden_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    const int cols = 16;
    double wl[16];
    for (int j = 0; j < cols; ++j) wl[j] = 1000.0 + static_cast<double>(j) * 10.0;
    c4a_aug_instrument_broaden_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_instrument_broaden_create(&h, rng,
        /*fwhm=*/5.0, /*use_fwhm_range=*/0,
        /*fwhm_low=*/0.0, /*fwhm_high=*/0.0,
        /*variation_scope=*/0,
        wl, cols) == C4A_OK);
    double X[32] = {0};
    double Y[32] = {0};
    for (int i = 0; i < cols; ++i) X[i] = (i == cols/2) ? 1.0 : 0.0;
    for (int i = cols; i < 2*cols; ++i) X[i] = 0.5;
    auto Xv = make_view(X, 2, cols);
    auto Yv = make_view(Y, 2, cols);
    C4A_TEST_REQUIRE(c4a_aug_instrument_broaden_apply(h, Xv, Yv) == C4A_OK);
    /* Smoothing of a delta should spread mass; max output < 1.0 */
    double mx = 0.0;
    for (int i = 0; i < cols; ++i) if (Y[i] > mx) mx = Y[i];
    C4A_TEST_REQUIRE(mx < 1.0);
    c4a_aug_instrument_broaden_destroy(h);
    c4a_aug_instrument_broaden_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: DeadBandAugmenter -------- */
void test_dead_band_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_dead_band_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_dead_band_create(&h, rng,
        /*n_bands=*/1, /*width_lo=*/2, /*width_hi=*/3,
        /*noise_std=*/0.05, /*probability=*/1.0,
        /*variation_scope=*/0) == C4A_OK);
    double X[20] = {0};
    double Y[20] = {0};
    for (int i = 0; i < 20; ++i) X[i] = 1.0;
    auto Xv = make_view(X, 2, 10);
    auto Yv = make_view(Y, 2, 10);
    C4A_TEST_REQUIRE(c4a_aug_dead_band_apply(h, Xv, Yv) == C4A_OK);
    /* Some entries should be replaced with noise around 0. */
    int replaced = 0;
    for (int i = 0; i < 20; ++i) {
        if (std::fabs(Y[i] - 1.0) > 0.3) ++replaced;
    }
    C4A_TEST_REQUIRE(replaced > 0);
    c4a_aug_dead_band_destroy(h);
    c4a_aug_dead_band_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

void test_dead_band_zero_probability() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_dead_band_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_dead_band_create(&h, rng,
        1, 3, 5, 0.05, /*probability=*/0.0, 0) == C4A_OK);
    double X[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double Y[10] = {0};
    auto Xv = make_view(X, 1, 10);
    auto Yv = make_view(Y, 1, 10);
    C4A_TEST_REQUIRE(c4a_aug_dead_band_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 10; ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - X[i]) < 1e-12);
    }
    c4a_aug_dead_band_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: TemperatureAugmenter -------- */
void test_temperature_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    const int cols = 20;
    double wl[20];
    for (int j = 0; j < cols; ++j) wl[j] = 1400.0 + static_cast<double>(j) * 30.0;
    c4a_aug_temperature_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_temperature_create(&h, rng,
        /*temperature_delta=*/5.0,
        /*use_temp_range=*/0, 0.0, 0.0,
        /*enable_shift=*/1, /*enable_intensity=*/1, /*enable_broadening=*/1,
        /*region_specific=*/1,
        wl, cols) == C4A_OK);
    double X[40] = {0};
    double Y[40] = {0};
    for (int i = 0; i < 40; ++i) X[i] = 0.5;
    auto Xv = make_view(X, 2, cols);
    auto Yv = make_view(Y, 2, cols);
    C4A_TEST_REQUIRE(c4a_aug_temperature_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 40; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));
    c4a_aug_temperature_destroy(h);
    c4a_aug_temperature_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

void test_temperature_zero_delta_passthrough() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    const int cols = 8;
    double wl[8] = {1400.0, 1450.0, 1500.0, 1550.0,
                    1600.0, 1650.0, 1700.0, 1750.0};
    c4a_aug_temperature_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_temperature_create(&h, rng,
        /*delta=*/0.0, 0, 0.0, 0.0,
        1, 1, 1, 1,
        wl, cols) == C4A_OK);
    double X[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double Y[8] = {0};
    auto Xv = make_view(X, 1, cols);
    auto Yv = make_view(Y, 1, cols);
    C4A_TEST_REQUIRE(c4a_aug_temperature_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 8; ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - X[i]) < 1e-12);
    }
    c4a_aug_temperature_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Smoke: MoistureAugmenter -------- */
void test_moisture_smoke() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    const int cols = 20;
    double wl[20];
    for (int j = 0; j < cols; ++j) wl[j] = 1400.0 + static_cast<double>(j) * 30.0;
    c4a_aug_moisture_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_moisture_create(&h, rng,
        /*aw_delta=*/0.1,
        /*use_aw_range=*/0, 0.0, 0.0,
        /*ref_aw=*/0.5,
        /*free_frac=*/0.3, /*bound_shift=*/25.0,
        /*moisture_content=*/0.10,
        /*enable_shift=*/1, /*enable_intensity=*/1,
        wl, cols) == C4A_OK);
    double X[40] = {0};
    double Y[40] = {0};
    for (int i = 0; i < 40; ++i) X[i] = 0.5;
    auto Xv = make_view(X, 2, cols);
    auto Yv = make_view(Y, 2, cols);
    C4A_TEST_REQUIRE(c4a_aug_moisture_apply(h, Xv, Yv) == C4A_OK);
    for (int i = 0; i < 40; ++i) C4A_TEST_REQUIRE(std::isfinite(Y[i]));
    c4a_aug_moisture_destroy(h);
    c4a_aug_moisture_destroy(nullptr);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- ABI: shape/stride/null guards -------- */
void test_apply_shape_mismatch() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(1u, &rng) == C4A_OK);
    c4a_aug_mixup_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_mixup_create(&h, rng, 0.2) == C4A_OK);
    double X[6] = {0};
    double Y[4] = {0};
    auto Xv = make_view(X, 2, 3);
    auto Yv = make_view(Y, 2, 2);
    C4A_TEST_REQUIRE(c4a_aug_mixup_apply(h, Xv, Yv) == C4A_ERR_SHAPE_MISMATCH);
    c4a_aug_mixup_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

void test_apply_null_handle() {
    double X[6] = {0};
    double Y[6] = {0};
    auto Xv = make_view(X, 2, 3);
    auto Yv = make_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_aug_mixup_apply(nullptr, Xv, Yv) == C4A_ERR_NULL_POINTER);
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_apply(nullptr, Xv, Yv) == C4A_ERR_NULL_POINTER);
    C4A_TEST_REQUIRE(c4a_aug_local_mixup_apply(nullptr, Xv, Yv) == C4A_ERR_NULL_POINTER);
}

void test_create_null_out() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(1u, &rng) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_aug_mixup_create(nullptr, rng, 0.2) == C4A_ERR_NULL_POINTER);
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_create(nullptr, rng, 0, 1, 0, 1) == C4A_ERR_NULL_POINTER);
    c4a_rng_pcg64_destroy(rng);
}

void test_inplace_mixup() {
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(42u, &rng) == C4A_OK);
    c4a_aug_mixup_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_mixup_create(&h, rng, 0.5) == C4A_OK);
    double X[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto Xv = make_view(X, 4, 3);
    /* In-place: same buffer for X and out. */
    C4A_TEST_REQUIRE(c4a_aug_mixup_apply(h, Xv, Xv) == C4A_OK);
    for (int i = 0; i < 12; ++i) C4A_TEST_REQUIRE(std::isfinite(X[i]));
    c4a_aug_mixup_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

/* -------- Parity: deterministic cases from aug_phase17_v1.json -------- */

/* The fixture has a non-standard schema with optional `output_hex` and
 * `wavelengths_hex` keys; we parse it directly without using the shared
 * fixture loader. */
struct DetCase {
    std::string name;
    std::int64_t rows;
    std::int64_t cols;
    std::vector<double> input;
    std::vector<double> wavelengths;
    std::vector<double> expected;
};

/* Read a hex-double array referenced by `key`. */
static void read_hex_array(const std::string& s, const std::string& key,
                            std::vector<double>& out) {
    std::size_t pos = c4a_testing::find_value_for_key(s, key, 0, s.size());
    c4a_testing::parse_hex_double_array(s, pos, out);
}

static DetCase load_det_case(const std::string& wanted_name) {
    DetCase dc;
    const std::string path =
        std::string(C4A_PARITY_FIXTURE_DIR) + "/aug_phase17_v1.json";
    const std::string s = c4a_testing::slurp(path);
    /* Parse rows / cols. */
    dc.rows = c4a_testing::parse_int64(
        s, c4a_testing::find_value_for_key(s, "rows", 0, s.size()));
    dc.cols = c4a_testing::parse_int64(
        s, c4a_testing::find_value_for_key(s, "cols", 0, s.size()));
    read_hex_array(s, "input_hex", dc.input);
    read_hex_array(s, "wavelengths_hex", dc.wavelengths);

    /* Find the named case and its output_hex. */
    auto cases_span = c4a_testing::find_cases_array(s);
    auto case_spans = c4a_testing::list_case_object_spans(
        s, cases_span.first, cases_span.second);
    for (const auto& span : case_spans) {
        const std::string sub = s.substr(span.first, span.second - span.first);
        const std::string name = c4a_testing::params_get_string(sub, "name", "");
        if (name != wanted_name) continue;
        dc.name = name;
        std::size_t pos = c4a_testing::find_value_for_key(
            sub, "output_hex", 0, sub.size());
        c4a_testing::parse_hex_double_array(sub, pos, dc.expected);
        return dc;
    }
    throw std::runtime_error("fixture: case not found: " + wanted_name);
}

void test_parity_scatter_sim_constant() {
    DetCase dc = load_det_case("scatter_sim_constant");
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0u, &rng) == C4A_OK);
    c4a_aug_scatter_sim_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_create(&h, rng, 0.5, 0.5, 1.0, 1.0)
                     == C4A_OK);
    std::vector<double> Y(static_cast<size_t>(dc.rows) * static_cast<size_t>(dc.cols));
    auto Xv = make_view(dc.input.data(), dc.rows, dc.cols);
    auto Yv = make_view(Y.data(), dc.rows, dc.cols);
    C4A_TEST_REQUIRE(c4a_aug_scatter_sim_apply(h, Xv, Yv) == C4A_OK);
    for (size_t i = 0; i < Y.size(); ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - dc.expected[i]) < 1e-12);
    }
    c4a_aug_scatter_sim_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

void test_parity_batch_effect_zero() {
    DetCase dc = load_det_case("batch_effect_zero_batch");
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0u, &rng) == C4A_OK);
    c4a_aug_batch_effect_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_batch_effect_create(&h, rng,
        0.0, 0.0, 0.0, 1, nullptr, 0) == C4A_OK);
    std::vector<double> Y(static_cast<size_t>(dc.rows) * static_cast<size_t>(dc.cols));
    auto Xv = make_view(dc.input.data(), dc.rows, dc.cols);
    auto Yv = make_view(Y.data(), dc.rows, dc.cols);
    C4A_TEST_REQUIRE(c4a_aug_batch_effect_apply(h, Xv, Yv) == C4A_OK);
    for (size_t i = 0; i < Y.size(); ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - dc.expected[i]) < 1e-12);
    }
    c4a_aug_batch_effect_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

void test_parity_instrument_broaden_fixed() {
    DetCase dc = load_det_case("instrument_broaden_fixed_fwhm5");
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0u, &rng) == C4A_OK);
    c4a_aug_instrument_broaden_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_instrument_broaden_create(&h, rng,
        5.0, 0, 0.0, 0.0, 0,
        dc.wavelengths.data(),
        static_cast<int64_t>(dc.wavelengths.size())) == C4A_OK);
    std::vector<double> Y(static_cast<size_t>(dc.rows) * static_cast<size_t>(dc.cols));
    auto Xv = make_view(dc.input.data(), dc.rows, dc.cols);
    auto Yv = make_view(Y.data(), dc.rows, dc.cols);
    C4A_TEST_REQUIRE(c4a_aug_instrument_broaden_apply(h, Xv, Yv) == C4A_OK);
    /* Gaussian filter parity vs scipy.ndimage.gaussian_filter1d ~ 1e-9 abs. */
    for (size_t i = 0; i < Y.size(); ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - dc.expected[i]) < 1e-9);
    }
    c4a_aug_instrument_broaden_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

void test_parity_temperature_zero_delta() {
    DetCase dc = load_det_case("temperature_zero_delta");
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0u, &rng) == C4A_OK);
    c4a_aug_temperature_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_temperature_create(&h, rng,
        0.0, 0, 0.0, 0.0, 1, 1, 1, 1,
        dc.wavelengths.data(),
        static_cast<int64_t>(dc.wavelengths.size())) == C4A_OK);
    std::vector<double> Y(static_cast<size_t>(dc.rows) * static_cast<size_t>(dc.cols));
    auto Xv = make_view(dc.input.data(), dc.rows, dc.cols);
    auto Yv = make_view(Y.data(), dc.rows, dc.cols);
    C4A_TEST_REQUIRE(c4a_aug_temperature_apply(h, Xv, Yv) == C4A_OK);
    for (size_t i = 0; i < Y.size(); ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - dc.expected[i]) < 1e-12);
    }
    c4a_aug_temperature_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

void test_parity_dead_band_zero_prob() {
    DetCase dc = load_det_case("dead_band_zero_probability");
    c4a_rng_pcg64_state_t* rng = nullptr;
    C4A_TEST_REQUIRE(c4a_rng_pcg64_create(0u, &rng) == C4A_OK);
    c4a_aug_dead_band_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_aug_dead_band_create(&h, rng,
        1, 5, 10, 0.05, 0.0, 0) == C4A_OK);
    std::vector<double> Y(static_cast<size_t>(dc.rows) * static_cast<size_t>(dc.cols));
    auto Xv = make_view(dc.input.data(), dc.rows, dc.cols);
    auto Yv = make_view(Y.data(), dc.rows, dc.cols);
    C4A_TEST_REQUIRE(c4a_aug_dead_band_apply(h, Xv, Yv) == C4A_OK);
    for (size_t i = 0; i < Y.size(); ++i) {
        C4A_TEST_REQUIRE(std::fabs(Y[i] - dc.expected[i]) < 1e-12);
    }
    c4a_aug_dead_band_destroy(h);
    c4a_rng_pcg64_destroy(rng);
}

}  // namespace

void register_augmenters_phase17_tests(c4a_testing::Runner& r) {
    r.run("aug_phase17/mixup_smoke",                  test_mixup_smoke);
    r.run("aug_phase17/mixup_invalid_args",           test_mixup_invalid_args);
    r.run("aug_phase17/mixup_determinism",            test_mixup_determinism);
    r.run("aug_phase17/mixup_inplace",                test_inplace_mixup);
    r.run("aug_phase17/local_mixup_smoke",            test_local_mixup_smoke);
    r.run("aug_phase17/scatter_sim_smoke",            test_scatter_sim_smoke);
    r.run("aug_phase17/scatter_sim_deterministic",    test_scatter_sim_deterministic_pointwise);
    r.run("aug_phase17/particle_size_smoke",          test_particle_size_smoke);
    r.run("aug_phase17/emsc_distort_smoke",           test_emsc_distort_smoke);
    r.run("aug_phase17/batch_effect_smoke",           test_batch_effect_smoke);
    r.run("aug_phase17/batch_effect_zero_variation",  test_batch_effect_zero_variation);
    r.run("aug_phase17/instrument_broaden_smoke",     test_instrument_broaden_smoke);
    r.run("aug_phase17/dead_band_smoke",              test_dead_band_smoke);
    r.run("aug_phase17/dead_band_zero_probability",   test_dead_band_zero_probability);
    r.run("aug_phase17/temperature_smoke",            test_temperature_smoke);
    r.run("aug_phase17/temperature_zero_delta",       test_temperature_zero_delta_passthrough);
    r.run("aug_phase17/moisture_smoke",               test_moisture_smoke);
    r.run("aug_phase17/apply_shape_mismatch",         test_apply_shape_mismatch);
    r.run("aug_phase17/apply_null_handle",            test_apply_null_handle);
    r.run("aug_phase17/create_null_out",              test_create_null_out);
    /* Fixture-driven parity for deterministic cases. */
    r.run("aug_phase17/parity_scatter_sim_constant",    test_parity_scatter_sim_constant);
    r.run("aug_phase17/parity_batch_effect_zero",       test_parity_batch_effect_zero);
    r.run("aug_phase17/parity_instrument_broaden_fixed", test_parity_instrument_broaden_fixed);
    r.run("aug_phase17/parity_temperature_zero_delta",  test_parity_temperature_zero_delta);
    r.run("aug_phase17/parity_dead_band_zero_prob",     test_parity_dead_band_zero_prob);
}
