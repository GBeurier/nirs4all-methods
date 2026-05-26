/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * bench/cpp/n4m_donor_bench.cpp — donor-operator timing tool.
 *
 * The cross-binding orchestrator times the ~73 PLS-family methods, but the
 * ~150 donor operators (augmentation, preprocessing, filters, splitters)
 * are only parity-validated by `n4m_tests` and carry no timing. This tool
 * fills that gap: given a method id + a dashboard cell size, it generates a
 * fresh deterministic matrix at that size, runs the operator through the
 * public C ABI, and times it with the same adaptive protocol the Python
 * bench scripts use (benchmarks/cross_binding/scripts/_common.py). It prints
 * one JSON object (the adaptive-v1 timing record) on stdout; the donor
 * pipeline wraps that into a dashboard CSV row.
 *
 * Why dashboard-size matrices and not the parity fixtures: the fixtures are
 * tiny (e.g. 12x24) regression inputs — timing them measures create/destroy
 * and wrapper overhead, not the algorithm. We time at the dashboard sizes
 * (250x50, 50x250) instead, keeping the fixtures as the parity gate only.
 *
 * Each operator is a `DonorOp`: a `make(rng, n, p)` that builds the handle
 * with curated default params (lifted from the parity / smoke tests), a
 * `run(handle, BenchCtx&)` that performs the *timed* operation, and a
 * `destroy`. BenchCtx carries the input view, a 1xP wavelength axis (edge
 * augmenters take it as an explicit apply() argument), the output buffer and
 * scratch — so a single uniform table covers every apply shape.
 *
 * Usage:
 *   n4m_donor_bench --method aug_gaussian_noise --n 250 --p 50 \
 *                   --runs 40 --seed 1234567890
 *   n4m_donor_bench --list      # registered method ids, one per line
 */
#include "n4m/n4m.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

/* ----- adaptive timing protocol (mirror of _common.py) ----------------- */
constexpr double kWarmupAbortMs = 5.0 * 60.0 * 1000.0;  /* 5 min  */
constexpr double kProbeAbortMs = 60.0 * 1000.0;         /* 60 s   */
constexpr int kDefaultMaxRuns = 40;

struct AdaptiveTarget {
    int total;
    const char* statistic;
    const char* decision;
};

AdaptiveTarget adaptive_target(double probe_ms, int max_runs) {
    max_runs = std::max(2, max_runs);
    if (probe_ms > kProbeAbortMs) return {2, "single", "probe_gt_60s"};
    if (probe_ms > 30000.0) return {std::min(max_runs, 2), "single", "probe_gt_30s"};
    if (probe_ms > 5000.0) return {std::min(max_runs, 3), "mean", "probe_gt_5s"};
    if (probe_ms > 1000.0) return {std::min(max_runs, 10), "median", "probe_gt_1s"};
    if (probe_ms > 100.0) return {std::min(max_runs, 20), "median", "probe_gt_100ms"};
    return {std::min(max_runs, kDefaultMaxRuns), "median", "probe_le_100ms"};
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

/* ----- deterministic input generation ---------------------------------- */
uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void fill_matrix(std::vector<double>& buf, uint64_t seed) {
    uint64_t state = seed ? seed : 0x1234567890ULL;
    for (double& x : buf) {
        const uint64_t r = splitmix64(state);
        const double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
        x = 0.2 + 0.8 * u;  /* spectra-like band [0.2, 1.0) */
    }
}

/* A NIRS-like wavelength axis (linspace 1000..2500 nm) of length p. */
std::vector<double> wavelength_axis(int64_t p) {
    std::vector<double> wl(static_cast<size_t>(p));
    if (p == 1) { wl[0] = 1000.0; return wl; }
    const double step = 1500.0 / static_cast<double>(p - 1);
    for (int64_t j = 0; j < p; ++j) wl[static_cast<size_t>(j)] = 1000.0 + step * static_cast<double>(j);
    return wl;
}

/* ----- bench context handed to each op's timed run --------------------- */
struct BenchCtx {
    n4m_matrix_view_t X;    /* n x p input  */
    n4m_matrix_view_t wl;   /* 1 x p wavelength axis (edge augmenters use it) */
    n4m_matrix_view_t out;  /* n x p output */
};

/* ----- donor operator registry ----------------------------------------- */
struct DonorOp {
    const char* name;
    void* (*make)(n4m_rng_pcg64_state_t* rng, int64_t n, int64_t p);
    n4m_status_t (*run)(void* h, BenchCtx& c);
    void (*destroy)(void* h);
};

/* Convenience macros to keep the table readable. Each op is one CREATE_*
 * (with curated params), a RUN_* (the timed apply), and a DESTROY_. */
#define DESTROY(fn, T) [](void* h) { fn(static_cast<T*>(h)); }
#define RUN_APPLY(fn, T) \
    [](void* h, BenchCtx& c) { return fn(static_cast<T*>(h), c.X, c.out); }
#define RUN_APPLY_WL(fn, T) \
    [](void* h, BenchCtx& c) { return fn(static_cast<T*>(h), c.X, c.wl, c.out); }

const DonorOp kOps[] = {
    /* ---- noise / drift (Phase 15) -------------------------------------- */
    {"aug_gaussian_noise",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_gaussian_noise_handle_t* h = nullptr;
         return n4m_aug_gaussian_noise_create(&h, r, 0.01) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_gaussian_noise_apply, n4m_aug_gaussian_noise_handle_t),
     DESTROY(n4m_aug_gaussian_noise_destroy, n4m_aug_gaussian_noise_handle_t)},

    {"aug_multiplicative_noise",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_multiplicative_noise_handle_t* h = nullptr;
         return n4m_aug_multiplicative_noise_create(&h, r, 0.05) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_multiplicative_noise_apply, n4m_aug_multiplicative_noise_handle_t),
     DESTROY(n4m_aug_multiplicative_noise_destroy, n4m_aug_multiplicative_noise_handle_t)},

    {"aug_spike_noise",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_spike_noise_handle_t* h = nullptr;
         return n4m_aug_spike_noise_create(&h, r, 1, 3, -0.5, 0.5) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_spike_noise_apply, n4m_aug_spike_noise_handle_t),
     DESTROY(n4m_aug_spike_noise_destroy, n4m_aug_spike_noise_handle_t)},

    {"aug_hetero_noise",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_hetero_noise_handle_t* h = nullptr;
         return n4m_aug_hetero_noise_create(&h, r, 1e-3, 5e-3) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_hetero_noise_apply, n4m_aug_hetero_noise_handle_t),
     DESTROY(n4m_aug_hetero_noise_destroy, n4m_aug_hetero_noise_handle_t)},

    {"aug_linear_drift",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_linear_drift_handle_t* h = nullptr;
         return n4m_aug_linear_drift_create(&h, r, -0.1, 0.1, -0.001, 0.001) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_linear_drift_apply, n4m_aug_linear_drift_handle_t),
     DESTROY(n4m_aug_linear_drift_destroy, n4m_aug_linear_drift_handle_t)},

    {"aug_poly_drift",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         static const double lo[3] = {-0.1, -0.05, -0.033};
         static const double hi[3] = {0.1, 0.05, 0.033};
         n4m_aug_poly_drift_handle_t* h = nullptr;
         return n4m_aug_poly_drift_create(&h, r, 2, lo, hi) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_poly_drift_apply, n4m_aug_poly_drift_handle_t),
     DESTROY(n4m_aug_poly_drift_destroy, n4m_aug_poly_drift_handle_t)},

    {"aug_path_length",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_path_length_handle_t* h = nullptr;
         return n4m_aug_path_length_create(&h, r, 0.05, 0.5) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_path_length_apply, n4m_aug_path_length_handle_t),
     DESTROY(n4m_aug_path_length_destroy, n4m_aug_path_length_handle_t)},

    /* ---- wavelength / spectral (Phase 16) ------------------------------ */
    {"aug_wavelength_shift",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_wavelength_shift_handle_t* h = nullptr;
         return n4m_aug_wavelength_shift_create(&h, r, -2.0, 2.0, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_wavelength_shift_apply, n4m_aug_wavelength_shift_handle_t),
     DESTROY(n4m_aug_wavelength_shift_destroy, n4m_aug_wavelength_shift_handle_t)},

    {"aug_wavelength_stretch",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_wavelength_stretch_handle_t* h = nullptr;
         return n4m_aug_wavelength_stretch_create(&h, r, 0.98, 1.02, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_wavelength_stretch_apply, n4m_aug_wavelength_stretch_handle_t),
     DESTROY(n4m_aug_wavelength_stretch_destroy, n4m_aug_wavelength_stretch_handle_t)},

    {"aug_local_warp",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_local_warp_handle_t* h = nullptr;
         return n4m_aug_local_warp_create(&h, r, 5, 2.0, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_local_warp_apply, n4m_aug_local_warp_handle_t),
     DESTROY(n4m_aug_local_warp_destroy, n4m_aug_local_warp_handle_t)},

    {"aug_band_perturb",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_band_perturb_handle_t* h = nullptr;
         return n4m_aug_band_perturb_create(&h, r, 3, 5, 15, 0.9, 1.1, -0.05, 0.05) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_band_perturb_apply, n4m_aug_band_perturb_handle_t),
     DESTROY(n4m_aug_band_perturb_destroy, n4m_aug_band_perturb_handle_t)},

    {"aug_band_mask",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_band_mask_handle_t* h = nullptr;
         return n4m_aug_band_mask_create(&h, r, 1, 3, 5, 15, 0) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_band_mask_apply, n4m_aug_band_mask_handle_t),
     DESTROY(n4m_aug_band_mask_destroy, n4m_aug_band_mask_handle_t)},

    {"aug_channel_dropout",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_channel_dropout_handle_t* h = nullptr;
         return n4m_aug_channel_dropout_create(&h, r, 0.1, 0) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_channel_dropout_apply, n4m_aug_channel_dropout_handle_t),
     DESTROY(n4m_aug_channel_dropout_destroy, n4m_aug_channel_dropout_handle_t)},

    {"aug_gauss_jitter",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_gauss_jitter_handle_t* h = nullptr;
         return n4m_aug_gauss_jitter_create(&h, r, 0.5, 2.0, 5) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_gauss_jitter_apply, n4m_aug_gauss_jitter_handle_t),
     DESTROY(n4m_aug_gauss_jitter_destroy, n4m_aug_gauss_jitter_handle_t)},

    {"aug_unsharp_mask",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_unsharp_mask_handle_t* h = nullptr;
         return n4m_aug_unsharp_mask_create(&h, r, 0.5, 1.5, 1.0, 5) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_unsharp_mask_apply, n4m_aug_unsharp_mask_handle_t),
     DESTROY(n4m_aug_unsharp_mask_destroy, n4m_aug_unsharp_mask_handle_t)},

    {"aug_magnitude_warp",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_magnitude_warp_handle_t* h = nullptr;
         return n4m_aug_magnitude_warp_create(&h, r, 5, 0.9, 1.1, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_magnitude_warp_apply, n4m_aug_magnitude_warp_handle_t),
     DESTROY(n4m_aug_magnitude_warp_destroy, n4m_aug_magnitude_warp_handle_t)},

    {"aug_local_clip",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_local_clip_handle_t* h = nullptr;
         return n4m_aug_local_clip_create(&h, r, 2, 3, 10) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_local_clip_apply, n4m_aug_local_clip_handle_t),
     DESTROY(n4m_aug_local_clip_destroy, n4m_aug_local_clip_handle_t)},

    /* ---- mixup / physical / environmental (Phase 17) ------------------- */
    {"aug_mixup",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_mixup_handle_t* h = nullptr;
         return n4m_aug_mixup_create(&h, r, 0.4) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_mixup_apply, n4m_aug_mixup_handle_t),
     DESTROY(n4m_aug_mixup_destroy, n4m_aug_mixup_handle_t)},

    {"aug_local_mixup",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_local_mixup_handle_t* h = nullptr;
         return n4m_aug_local_mixup_create(&h, r, 0.4, 5) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_local_mixup_apply, n4m_aug_local_mixup_handle_t),
     DESTROY(n4m_aug_local_mixup_destroy, n4m_aug_local_mixup_handle_t)},

    {"aug_scatter_sim",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_scatter_sim_handle_t* h = nullptr;
         return n4m_aug_scatter_sim_create(&h, r, 0.8, 1.2, -0.05, 0.05) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_scatter_sim_apply, n4m_aug_scatter_sim_handle_t),
     DESTROY(n4m_aug_scatter_sim_destroy, n4m_aug_scatter_sim_handle_t)},

    {"aug_particle_size",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_particle_size_handle_t* h = nullptr;
         return n4m_aug_particle_size_create(&h, r, 50.0, 10.0, 0, 10.0, 100.0, 50.0,
                                              1.0, 0.1, 0, 0.1, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_particle_size_apply, n4m_aug_particle_size_handle_t),
     DESTROY(n4m_aug_particle_size_destroy, n4m_aug_particle_size_handle_t)},

    {"aug_emsc_distort",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_emsc_distort_handle_t* h = nullptr;
         return n4m_aug_emsc_distort_create(&h, r, 0.95, 1.05, -0.02, 0.02, 2, 0.1, 0.5,
                                            wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_emsc_distort_apply, n4m_aug_emsc_distort_handle_t),
     DESTROY(n4m_aug_emsc_distort_destroy, n4m_aug_emsc_distort_handle_t)},

    {"aug_batch_effect",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_batch_effect_handle_t* h = nullptr;
         return n4m_aug_batch_effect_create(&h, r, 0.01, 0.001, 0.02, 0, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_batch_effect_apply, n4m_aug_batch_effect_handle_t),
     DESTROY(n4m_aug_batch_effect_destroy, n4m_aug_batch_effect_handle_t)},

    {"aug_instrument_broaden",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_instrument_broaden_handle_t* h = nullptr;
         return n4m_aug_instrument_broaden_create(&h, r, 5.0, 0, 3.0, 8.0, 0, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_instrument_broaden_apply, n4m_aug_instrument_broaden_handle_t),
     DESTROY(n4m_aug_instrument_broaden_destroy, n4m_aug_instrument_broaden_handle_t)},

    {"aug_dead_band",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_dead_band_handle_t* h = nullptr;
         return n4m_aug_dead_band_create(&h, r, 2, 3, 10, 0.01, 0.5, 0) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_dead_band_apply, n4m_aug_dead_band_handle_t),
     DESTROY(n4m_aug_dead_band_destroy, n4m_aug_dead_band_handle_t)},

    {"aug_temperature",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_temperature_handle_t* h = nullptr;
         return n4m_aug_temperature_create(&h, r, 5.0, 0, -10.0, 10.0, 1, 1, 1, 0,
                                           wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_temperature_apply, n4m_aug_temperature_handle_t),
     DESTROY(n4m_aug_temperature_destroy, n4m_aug_temperature_handle_t)},

    {"aug_moisture",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_moisture_handle_t* h = nullptr;
         return n4m_aug_moisture_create(&h, r, 0.1, 0, 0.2, 0.9, 0.5, 0.5, 0.01, 0.15, 1, 1,
                                        wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_moisture_apply, n4m_aug_moisture_handle_t),
     DESTROY(n4m_aug_moisture_destroy, n4m_aug_moisture_handle_t)},

    /* ---- edge / spline / random (Phase 18). Edge ops take the wavelength
     *      axis as an explicit apply() argument. The two simplification
     *      splines are v2-deferred (apply → NOT_IMPLEMENTED) and omitted. -- */
    {"aug_detector_rolloff",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_detector_rolloff_handle_t* h = nullptr;
         return n4m_aug_detector_rolloff_create(&h, r, 0, 0.5, 1.0, 0) == N4M_OK ? h : nullptr; },
     RUN_APPLY_WL(n4m_aug_detector_rolloff_apply, n4m_aug_detector_rolloff_handle_t),
     DESTROY(n4m_aug_detector_rolloff_destroy, n4m_aug_detector_rolloff_handle_t)},

    {"aug_stray_light",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_stray_light_handle_t* h = nullptr;
         return n4m_aug_stray_light_create(&h, r, 0.001, 2.0, 0.1, 1) == N4M_OK ? h : nullptr; },
     RUN_APPLY_WL(n4m_aug_stray_light_apply, n4m_aug_stray_light_handle_t),
     DESTROY(n4m_aug_stray_light_destroy, n4m_aug_stray_light_handle_t)},

    {"aug_edge_curve",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_edge_curve_handle_t* h = nullptr;
         return n4m_aug_edge_curve_create(&h, r, 0.1, 0, 0.0, 0.5) == N4M_OK ? h : nullptr; },
     RUN_APPLY_WL(n4m_aug_edge_curve_apply, n4m_aug_edge_curve_handle_t),
     DESTROY(n4m_aug_edge_curve_destroy, n4m_aug_edge_curve_handle_t)},

    {"aug_truncated_peak",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_truncated_peak_handle_t* h = nullptr;
         return n4m_aug_truncated_peak_create(&h, r, 0.5, 0.1, 0.5, 2.0, 10.0, 1, 1) == N4M_OK ? h : nullptr; },
     RUN_APPLY_WL(n4m_aug_truncated_peak_apply, n4m_aug_truncated_peak_handle_t),
     DESTROY(n4m_aug_truncated_peak_destroy, n4m_aug_truncated_peak_handle_t)},

    {"aug_edge_artifacts",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_edge_artifacts_handle_t* h = nullptr;
         return n4m_aug_edge_artifacts_create(&h, r, 0xF, 1.0, 0) == N4M_OK ? h : nullptr; },
     RUN_APPLY_WL(n4m_aug_edge_artifacts_apply, n4m_aug_edge_artifacts_handle_t),
     DESTROY(n4m_aug_edge_artifacts_destroy, n4m_aug_edge_artifacts_handle_t)},

    {"aug_spline_smooth",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_spline_smooth_handle_t* h = nullptr;
         return n4m_aug_spline_smooth_create(&h, r) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_spline_smooth_apply, n4m_aug_spline_smooth_handle_t),
     DESTROY(n4m_aug_spline_smooth_destroy, n4m_aug_spline_smooth_handle_t)},

    {"aug_spline_x_perturb",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_spline_x_perturb_handle_t* h = nullptr;
         return n4m_aug_spline_x_perturb_create(&h, r, 3, 0.1, -0.05, 0.05) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_spline_x_perturb_apply, n4m_aug_spline_x_perturb_handle_t),
     DESTROY(n4m_aug_spline_x_perturb_destroy, n4m_aug_spline_x_perturb_handle_t)},

    {"aug_spline_y_perturb",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_spline_y_perturb_handle_t* h = nullptr;
         return n4m_aug_spline_y_perturb_create(&h, r, 0, 0.1) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_spline_y_perturb_apply, n4m_aug_spline_y_perturb_handle_t),
     DESTROY(n4m_aug_spline_y_perturb_destroy, n4m_aug_spline_y_perturb_handle_t)},

    {"aug_rotate_translate",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_rotate_translate_handle_t* h = nullptr;
         return n4m_aug_rotate_translate_create(&h, r, 0.1, 0.1) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_rotate_translate_apply, n4m_aug_rotate_translate_handle_t),
     DESTROY(n4m_aug_rotate_translate_destroy, n4m_aug_rotate_translate_handle_t)},

    {"aug_random_x_op",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_random_x_op_handle_t* h = nullptr;
         return n4m_aug_random_x_op_create(&h, r, 0, 0.9, 1.1) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_random_x_op_apply, n4m_aug_random_x_op_handle_t),
     DESTROY(n4m_aug_random_x_op_destroy, n4m_aug_random_x_op_handle_t)},
};

#undef DESTROY
#undef RUN_APPLY
#undef RUN_APPLY_WL

const DonorOp* find_op(const std::string& name) {
    for (const DonorOp& op : kOps) {
        if (name == op.name) return &op;
    }
    return nullptr;
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

[[noreturn]] void die(const char* msg) {
    std::fprintf(stderr, "n4m_donor_bench: %s\n", msg);
    std::exit(2);
}

int run_list() {
    for (const DonorOp& op : kOps) std::printf("%s\n", op.name);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string method;
    long n = 250, p = 50, max_runs = kDefaultMaxRuns, threads = 1;
    unsigned long long seed = 1234567890ULL;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) die(what);
            return argv[++i];
        };
        if (a == "--list") return run_list();
        else if (a == "--method") method = next("--method needs a value");
        else if (a == "--n") n = std::strtol(next("--n needs a value"), nullptr, 10);
        else if (a == "--p") p = std::strtol(next("--p needs a value"), nullptr, 10);
        else if (a == "--runs") max_runs = std::strtol(next("--runs needs a value"), nullptr, 10);
        else if (a == "--seed") seed = std::strtoull(next("--seed needs a value"), nullptr, 10);
        else if (a == "--threads") threads = std::strtol(next("--threads needs a value"), nullptr, 10);
        else die(("unknown argument: " + a).c_str());
    }

    if (method.empty()) die("--method is required (or use --list)");
    if (n <= 0 || p <= 0) die("--n and --p must be positive");
    const DonorOp* op = find_op(method);
    if (op == nullptr) die(("unknown method: " + method + " (try --list)").c_str());

    /* Thread count is honoured via the OMP_NUM_THREADS / *_NUM_THREADS env
     * vars the orchestrator sets per cell; the augmenter ABI takes no
     * context, so we only record `threads` in the output for the cell key. */

    std::vector<double> X(static_cast<size_t>(n) * static_cast<size_t>(p));
    std::vector<double> out(X.size());
    std::vector<double> wl = wavelength_axis(p);
    fill_matrix(X, seed);

    BenchCtx ctx;
    if (n4m_matrix_view_init_rowmajor(&ctx.X, X.data(), n, p, N4M_DTYPE_F64) != N4M_OK ||
        n4m_matrix_view_init_rowmajor(&ctx.out, out.data(), n, p, N4M_DTYPE_F64) != N4M_OK ||
        n4m_matrix_view_init_rowmajor(&ctx.wl, wl.data(), 1, p, N4M_DTYPE_F64) != N4M_OK) {
        die("matrix view init failed");
    }

    n4m_rng_pcg64_state_t* rng = nullptr;
    if (n4m_rng_pcg64_create(seed, &rng) != N4M_OK) die("rng create failed");
    void* handle = op->make(rng, n, p);
    if (handle == nullptr) {
        n4m_rng_pcg64_destroy(rng);
        std::printf("{\"ok\":false,\"algorithm\":\"%s\",\"n\":%ld,\"p\":%ld,"
                    "\"threads\":%ld,\"reason\":\"operator create failed\"}\n",
                    method.c_str(), n, p, threads);
        return 1;
    }

    auto one_run = [&]() -> bool { return op->run(handle, ctx) == N4M_OK; };

    double t0 = now_ms();
    bool ok = one_run();
    double warmup_ms = now_ms() - t0;

    std::vector<double> samples;
    const char* statistic = "single";
    const char* decision = "max_runs_1_warmup_only";

    if (ok && warmup_ms <= kWarmupAbortMs && max_runs >= 2) {
        t0 = now_ms();
        ok = one_run();
        const double probe_ms = now_ms() - t0;
        samples.push_back(probe_ms);
        const AdaptiveTarget tgt = adaptive_target(probe_ms, static_cast<int>(max_runs));
        statistic = tgt.statistic;
        decision = tgt.decision;
        const int target_samples = std::max(1, tgt.total - 1);
        for (int i = 1; ok && i < target_samples; ++i) {
            t0 = now_ms();
            ok = one_run();
            samples.push_back(now_ms() - t0);
        }
    } else if (warmup_ms > kWarmupAbortMs) {
        samples.push_back(warmup_ms);
        decision = "warmup_gt_5min";
    } else {
        samples.push_back(warmup_ms);
    }

    op->destroy(handle);
    n4m_rng_pcg64_destroy(rng);

    if (!ok) {
        std::printf("{\"ok\":false,\"algorithm\":\"%s\",\"n\":%ld,\"p\":%ld,"
                    "\"threads\":%ld,\"reason\":\"run returned non-OK\"}\n",
                    method.c_str(), n, p, threads);
        return 1;
    }

    if (samples.size() == 1) statistic = "single";
    const double reported = (std::strcmp(statistic, "mean") == 0) ? mean(samples) : median(samples);
    const int n_runs = static_cast<int>(samples.size());
    const char* ver = n4m_get_version_string();

    std::printf(
        "{\"ok\":true,\"algorithm\":\"%s\",\"n\":%ld,\"p\":%ld,\"threads\":%ld,"
        "\"n_runs\":%d,\"total_runs\":%d,\"max_runs\":%ld,"
        "\"reported_ms\":%.6f,\"median_ms\":%.6f,\"sample_median_ms\":%.6f,"
        "\"mean_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"warmup_ms\":%.6f,"
        "\"warmup_included\":false,\"timing_statistic\":\"%s\","
        "\"timing_decision\":\"%s\",\"libn4m\":\"%s\"}\n",
        method.c_str(), n, p, threads, n_runs, n_runs + 1,
        max_runs, reported, reported, median(samples), mean(samples),
        *std::min_element(samples.begin(), samples.end()),
        *std::max_element(samples.begin(), samples.end()), warmup_ms,
        statistic, decision, ver ? ver : "unknown");
    return 0;
}
