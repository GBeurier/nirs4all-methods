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

/* ----- adaptive timing protocol ---------------------------------------- *
 *
 * Warm-up (one discarded run) + a timed probe, then enough timed runs for a
 * stable median: many reps for sub-millisecond ops, few for slow ones. The
 * count follows a fixed time budget (~target wall-clock per cell), so a ~2 ms
 * op gets ~100+ reps, a ~1 s op ~5-6, and anything over 30 s runs once. This
 * keeps the cheap-op medians robust while bounding total benchmark time. */
constexpr double kWarmupAbortMs = 5.0 * 60.0 * 1000.0;  /* 5 min  */
constexpr int kDefaultMaxRuns = 200;                    /* cap for ultra-fast ops */
constexpr double kTimeBudgetMs = 400.0;                 /* ~per-cell measurement budget */

struct AdaptiveTarget {
    int total;            /* total timed executions incl. the probe */
    const char* statistic;
    const char* decision;
};

AdaptiveTarget adaptive_target(double probe_ms, int max_runs) {
    max_runs = std::max(1, max_runs);
    if (probe_ms > 30000.0) return {std::min(max_runs, 1), "single", "probe_gt_30s"};
    if (probe_ms > 5000.0)  return {std::min(max_runs, 3), "mean",   "probe_gt_5s"};
    /* time-budget: target ≈ budget / probe, clamped to [6, max_runs]. */
    int target = static_cast<int>(std::lround(kTimeBudgetMs / probe_ms));
    target = std::max(6, std::min(target, max_runs));
    return {target, "median", "time_budget"};
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
    n4m_matrix_view_t out;     /* n x p output (F64) */
    n4m_matrix_view_t out_i32; /* n x p output (I32; discretizers)           */
    n4m_matrix_view_t Y;       /* n x 1 target matrix (Y-aware splitters)    */
    int64_t n;              /* rows */
    int64_t p;              /* cols */
    const double* y;        /* n-length target vector (y-outlier filters)    */
    uint8_t* mask;          /* n-length keep-mask scratch (filters)          */
    const int64_t* groups;  /* n-length group ids (group-aware splitters)    */
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
         return n4m_aug_particle_size_create(&h, r, 50.0, 15.0, 0, 5.0, 500.0, 50.0,
                                              1.5, 0.1, 1, 0.5, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_particle_size_apply, n4m_aug_particle_size_handle_t),
     DESTROY(n4m_aug_particle_size_destroy, n4m_aug_particle_size_handle_t)},

    {"aug_emsc_distort",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_emsc_distort_handle_t* h = nullptr;
         return n4m_aug_emsc_distort_create(&h, r, 0.9, 1.1, -0.05, 0.05, 2, 0.02, 0.3,
                                            wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_emsc_distort_apply, n4m_aug_emsc_distort_handle_t),
     DESTROY(n4m_aug_emsc_distort_destroy, n4m_aug_emsc_distort_handle_t)},

    {"aug_batch_effect",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_batch_effect_handle_t* h = nullptr;
         return n4m_aug_batch_effect_create(&h, r, 0.02, 0.01, 0.03, 0, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_batch_effect_apply, n4m_aug_batch_effect_handle_t),
     DESTROY(n4m_aug_batch_effect_destroy, n4m_aug_batch_effect_handle_t)},

    {"aug_instrument_broaden",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_instrument_broaden_handle_t* h = nullptr;
         return n4m_aug_instrument_broaden_create(&h, r, 3.0, 0, 3.0, 8.0, 0, wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_instrument_broaden_apply, n4m_aug_instrument_broaden_handle_t),
     DESTROY(n4m_aug_instrument_broaden_destroy, n4m_aug_instrument_broaden_handle_t)},

    {"aug_dead_band",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_dead_band_handle_t* h = nullptr;
         return n4m_aug_dead_band_create(&h, r, 1, 5, 10, 0.05, 1.0, 0) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_dead_band_apply, n4m_aug_dead_band_handle_t),
     DESTROY(n4m_aug_dead_band_destroy, n4m_aug_dead_band_handle_t)},

    {"aug_temperature",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_temperature_handle_t* h = nullptr;
         return n4m_aug_temperature_create(&h, r, 5.0, 0, -5.0, 5.0, 1, 1, 1, 1,
                                           wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_temperature_apply, n4m_aug_temperature_handle_t),
     DESTROY(n4m_aug_temperature_destroy, n4m_aug_temperature_handle_t)},

    {"aug_moisture",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_aug_moisture_handle_t* h = nullptr;
         return n4m_aug_moisture_create(&h, r, 0.1, 0, 0.0, 1.0, 0.5, 0.3, 25.0, 0.10, 1, 1,
                                        wl.data(), p) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_moisture_apply, n4m_aug_moisture_handle_t),
     DESTROY(n4m_aug_moisture_destroy, n4m_aug_moisture_handle_t)},

    /* ---- edge / spline / random (Phase 18). Edge ops take the wavelength
     *      axis as an explicit apply() argument. The two simplification
     *      splines are v2-deferred (apply → NOT_IMPLEMENTED) and omitted. -- */
    {"aug_detector_rolloff",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_detector_rolloff_handle_t* h = nullptr;
         return n4m_aug_detector_rolloff_create(&h, r, 4, 1.0, 0.02, 1) == N4M_OK ? h : nullptr; },
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
         return n4m_aug_edge_curve_create(&h, r, 0.02, 0, 0.0, 0.7) == N4M_OK ? h : nullptr; },
     RUN_APPLY_WL(n4m_aug_edge_curve_apply, n4m_aug_edge_curve_handle_t),
     DESTROY(n4m_aug_edge_curve_destroy, n4m_aug_edge_curve_handle_t)},

    {"aug_truncated_peak",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_truncated_peak_handle_t* h = nullptr;
         return n4m_aug_truncated_peak_create(&h, r, 0.5, 0.01, 0.1, 50.0, 200.0, 1, 1) == N4M_OK ? h : nullptr; },
     RUN_APPLY_WL(n4m_aug_truncated_peak_apply, n4m_aug_truncated_peak_handle_t),
     DESTROY(n4m_aug_truncated_peak_destroy, n4m_aug_truncated_peak_handle_t)},

    {"aug_edge_artifacts",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_edge_artifacts_handle_t* h = nullptr;
         return n4m_aug_edge_artifacts_create(&h, r, 0xF, 1.0, 4) == N4M_OK ? h : nullptr; },
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
         return n4m_aug_spline_x_perturb_create(&h, r, 3, 0.05, -0.1, 0.1) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_spline_x_perturb_apply, n4m_aug_spline_x_perturb_handle_t),
     DESTROY(n4m_aug_spline_x_perturb_destroy, n4m_aug_spline_x_perturb_handle_t)},

    {"aug_spline_y_perturb",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_spline_y_perturb_handle_t* h = nullptr;
         return n4m_aug_spline_y_perturb_create(&h, r, -1, 0.005) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_spline_y_perturb_apply, n4m_aug_spline_y_perturb_handle_t),
     DESTROY(n4m_aug_spline_y_perturb_destroy, n4m_aug_spline_y_perturb_handle_t)},

    {"aug_rotate_translate",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_rotate_translate_handle_t* h = nullptr;
         return n4m_aug_rotate_translate_create(&h, r, 2.0, 3.0) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_rotate_translate_apply, n4m_aug_rotate_translate_handle_t),
     DESTROY(n4m_aug_rotate_translate_destroy, n4m_aug_rotate_translate_handle_t)},

    {"aug_random_x_op",
     [](n4m_rng_pcg64_state_t* r, int64_t, int64_t) -> void* {
         n4m_aug_random_x_op_handle_t* h = nullptr;
         return n4m_aug_random_x_op_create(&h, r, 0, 0.97, 1.03) == N4M_OK ? h : nullptr; },
     RUN_APPLY(n4m_aug_random_x_op_apply, n4m_aug_random_x_op_handle_t),
     DESTROY(n4m_aug_random_x_op_destroy, n4m_aug_random_x_op_handle_t)},

    /* ---- filters (Phase 12-14). Timed operation is fit + apply (the full
     *      cost of learning the bounds then producing the keep-mask). The
     *      X-outlier variants share one ABI op selected by `method`; the
     *      Y-outlier variants operate on a target vector. CompositeFilter
     *      (needs externally-owned sub-filters) and the y-outlier validation
     *      scenarios (constant_input / nan_exclusion) are left parity-only. */
#define X_OUTLIER(name, method, ncomp)                                            \
    {name,                                                                        \
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {                      \
         n4m_filter_x_outlier_handle_t* h = nullptr;                              \
         return n4m_filter_x_outlier_create(&h, method, 0, 0.0, ncomp, 0.1,       \
                                            42u, 100, 256) == N4M_OK ? h : nullptr; }, \
     [](void* h, BenchCtx& c) {                                                   \
         auto* H = static_cast<n4m_filter_x_outlier_handle_t*>(h);                \
         n4m_filter_stats_t st{};                                                 \
         n4m_status_t s = n4m_filter_x_outlier_fit(H, c.X);                        \
         return s == N4M_OK ? n4m_filter_x_outlier_apply(H, c.X, c.mask, &st) : s; }, \
     DESTROY(n4m_filter_x_outlier_destroy, n4m_filter_x_outlier_handle_t)}
    X_OUTLIER("filter_x_outlier_mahalanobis", N4M_X_OUTLIER_MAHALANOBIS, 0),
    X_OUTLIER("filter_x_outlier_robust_mahalanobis", N4M_X_OUTLIER_ROBUST_MAHALANOBIS, 0),
    X_OUTLIER("filter_x_outlier_pca_residual", N4M_X_OUTLIER_PCA_RESIDUAL, 5),
    X_OUTLIER("filter_x_outlier_pca_leverage", N4M_X_OUTLIER_PCA_LEVERAGE, 5),
    X_OUTLIER("filter_x_outlier_isolation_forest", N4M_X_OUTLIER_ISOLATION_FOREST, 0),
    X_OUTLIER("filter_x_outlier_lof", N4M_X_OUTLIER_LOF, 0),
#undef X_OUTLIER

#define Y_OUTLIER(name, method)                                                   \
    {name,                                                                        \
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {                      \
         n4m_filter_y_outlier_handle_t* h = nullptr;                              \
         return n4m_filter_y_outlier_create(&h, method, 3.0, 5.0, 95.0)           \
                == N4M_OK ? h : nullptr; },                                       \
     [](void* h, BenchCtx& c) {                                                   \
         auto* H = static_cast<n4m_filter_y_outlier_handle_t*>(h);                \
         n4m_filter_stats_t st{};                                                 \
         n4m_status_t s = n4m_filter_y_outlier_fit(H, c.y, c.n);                   \
         return s == N4M_OK ? n4m_filter_y_outlier_apply(H, c.y, c.n, c.mask, &st) : s; }, \
     DESTROY(n4m_filter_y_outlier_destroy, n4m_filter_y_outlier_handle_t)}
    Y_OUTLIER("filter_y_outlier_iqr", N4M_Y_OUTLIER_IQR),
    Y_OUTLIER("filter_y_outlier_zscore", N4M_Y_OUTLIER_ZSCORE),
    Y_OUTLIER("filter_y_outlier_percentile", N4M_Y_OUTLIER_PERCENTILE),
    Y_OUTLIER("filter_y_outlier_mad", N4M_Y_OUTLIER_MAD),
#undef Y_OUTLIER

    {"filter_leverage",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_filter_leverage_handle_t* h = nullptr;
         return n4m_filter_leverage_create(&h, 0, 3.0, 0, 0.0, 0, 1) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_filter_leverage_handle_t*>(h);
         n4m_filter_stats_t st{};
         n4m_status_t s = n4m_filter_leverage_fit(H, c.X);
         return s == N4M_OK ? n4m_filter_leverage_apply(H, c.X, c.mask, &st) : s; },
     DESTROY(n4m_filter_leverage_destroy, n4m_filter_leverage_handle_t)},

    {"filter_quality",  /* stateless: apply only, no fit */
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_filter_quality_handle_t* h = nullptr;
         return n4m_filter_quality_create(&h, 0.1, 0.5, 1e-6, 0, 0.0, 0, 0.0, 1) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_filter_stats_t st{};
         return n4m_filter_quality_apply(static_cast<n4m_filter_quality_handle_t*>(h),
                                         c.X, c.mask, &st); },
     DESTROY(n4m_filter_quality_destroy, n4m_filter_quality_handle_t)},

    /* ---- splitters (Phase 11). Timed operation is one split() (or
     *      split_fold(0) for the k-fold variants), including the heap
     *      train/test index allocation it returns and the matching
     *      result_destroy — that allocation IS part of producing a split. */
    {"split_kennard_stone",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_kennard_stone_handle_t* h = nullptr;
         return n4m_split_kennard_stone_create(&h, 0.25) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_kennard_stone_split(
             static_cast<n4m_split_kennard_stone_handle_t*>(h), c.X, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_kennard_stone_destroy, n4m_split_kennard_stone_handle_t)},

    {"split_spxy",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_spxy_handle_t* h = nullptr;
         return n4m_split_spxy_create(&h, 0.25) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_spxy_split(
             static_cast<n4m_split_spxy_handle_t*>(h), c.X, c.Y, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_spxy_destroy, n4m_split_spxy_handle_t)},

    {"split_spxy_fold",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_spxy_fold_handle_t* h = nullptr;
         return n4m_split_spxy_fold_create(&h, 5, N4M_SPLIT_Y_METRIC_EUCLIDEAN) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_spxy_fold_split_fold(
             static_cast<n4m_split_spxy_fold_handle_t*>(h), c.X, c.Y, 0, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_spxy_fold_destroy, n4m_split_spxy_fold_handle_t)},

    {"split_spxy_g_fold",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_spxy_g_fold_handle_t* h = nullptr;
         return n4m_split_spxy_g_fold_create(&h, 5, N4M_SPLIT_Y_METRIC_EUCLIDEAN,
                                             N4M_SPLIT_AGGREGATION_MEAN) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_spxy_g_fold_split_fold(
             static_cast<n4m_split_spxy_g_fold_handle_t*>(h), c.X, c.Y,
             c.groups, c.n, 0, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_spxy_g_fold_destroy, n4m_split_spxy_g_fold_handle_t)},

    {"split_kmeans",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_kmeans_handle_t* h = nullptr;
         return n4m_split_kmeans_create(&h, 0.25, 42u, 100) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_kmeans_split(
             static_cast<n4m_split_kmeans_handle_t*>(h), c.X, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_kmeans_destroy, n4m_split_kmeans_handle_t)},

    {"split_kbins_stratified",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_kbins_stratified_handle_t* h = nullptr;
         return n4m_split_kbins_stratified_create(&h, 0.25, 42u, 5, N4M_SPLIT_KBINS_UNIFORM) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_kbins_stratified_split(
             static_cast<n4m_split_kbins_stratified_handle_t*>(h), c.Y, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_kbins_stratified_destroy, n4m_split_kbins_stratified_handle_t)},

    {"split_bsgk",  /* BinnedStratifiedGroupKFold */
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_binned_strat_group_kfold_handle_t* h = nullptr;
         return n4m_split_binned_strat_group_kfold_create(&h, 5, 5, N4M_SPLIT_KBINS_UNIFORM, 1, 42u) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_binned_strat_group_kfold_split_fold(
             static_cast<n4m_split_binned_strat_group_kfold_handle_t*>(h),
             c.Y, c.groups, c.n, 0, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_binned_strat_group_kfold_destroy, n4m_split_binned_strat_group_kfold_handle_t)},

    {"split_systematic_circular",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_systematic_circular_handle_t* h = nullptr;
         return n4m_split_systematic_circular_create(&h, 0.25, 42u) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_systematic_circular_split(
             static_cast<n4m_split_systematic_circular_handle_t*>(h), c.Y, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_systematic_circular_destroy, n4m_split_systematic_circular_handle_t)},

    {"split_split_splitter",  /* SPlit (data twinning) */
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_split_split_splitter_handle_t* h = nullptr;
         return n4m_split_split_splitter_create(&h, 0.25, 42u) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         n4m_split_result_t res{};
         n4m_status_t s = n4m_split_split_splitter_split(
             static_cast<n4m_split_split_splitter_handle_t*>(h), c.X, &res);
         n4m_split_result_destroy(&res); return s; },
     DESTROY(n4m_split_split_splitter_destroy, n4m_split_split_splitter_handle_t)},

    /* ---- preprocessing, dim-preserving (output is n x p, so it reuses the
     *      n x p out buffer). Timed op = transform, or fit + transform for
     *      stateful ops. The dim-changing pp ops (crop/derivate/wavelet/PCA/
     *      resample/...) are handled with output_cols-aware runs below. */
#define PP_T(base)  [](void* h, BenchCtx& c) {                                    \
        return n4m_pp_##base##_transform(static_cast<n4m_pp_##base##_handle_t*>(h), c.X, c.out); }
#define PP_FT(base) [](void* h, BenchCtx& c) {                                    \
        auto* H = static_cast<n4m_pp_##base##_handle_t*>(h);                      \
        n4m_status_t s = n4m_pp_##base##_fit(H, c.X);                             \
        return s == N4M_OK ? n4m_pp_##base##_transform(H, c.X, c.out) : s; }
#define PP_D(base) DESTROY(n4m_pp_##base##_destroy, n4m_pp_##base##_handle_t)
#define PP_MAKE(base, ...) [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* { \
        n4m_pp_##base##_handle_t* h = nullptr;                                    \
        return n4m_pp_##base##_create(&h, ##__VA_ARGS__) == N4M_OK ? h : nullptr; }
#define PP_MAKE0(base) [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {     \
        n4m_pp_##base##_handle_t* h = nullptr;                                    \
        return n4m_pp_##base##_create(&h) == N4M_OK ? h : nullptr; }

    {"pp_snv",  PP_MAKE(snv, 1, 1, 0), PP_T(snv), PP_D(snv)},
    {"pp_rnv",  PP_MAKE(rnv, 1, 1, 0.25), PP_T(rnv), PP_D(rnv)},
    {"pp_lsnv", PP_MAKE(lsnv, 11, 0, 0.0), PP_T(lsnv), PP_D(lsnv)},
    {"pp_area", PP_MAKE(area, 0), PP_T(area), PP_D(area)},
    {"pp_normalize", PP_MAKE(normalize, 0.0, 1.0), PP_T(normalize), PP_D(normalize)},
    {"pp_simple_scale", PP_MAKE0(simple_scale), PP_T(simple_scale), PP_D(simple_scale)},
    {"pp_detrend", PP_MAKE(detrend, 1), PP_T(detrend), PP_D(detrend)},
    {"pp_gaussian", PP_MAKE(gaussian, 2.0, 0, N4M_PP_GAUSSIAN_REFLECT, 0.0, 4.0), PP_T(gaussian), PP_D(gaussian)},
    {"pp_savgol", PP_MAKE(savgol, 11, 2, 0, 1.0, N4M_PP_SAVGOL_MIRROR, 0.0), PP_T(savgol), PP_D(savgol)},
    {"pp_first_derivative", PP_MAKE(first_derivative, 1.0, 2), PP_T(first_derivative), PP_D(first_derivative)},
    {"pp_second_derivative", PP_MAKE(second_derivative, 1.0, 2), PP_T(second_derivative), PP_D(second_derivative)},
    {"pp_norris_williams", PP_MAKE(norris_williams, 5, 3, 1, 1.0), PP_T(norris_williams), PP_D(norris_williams)},
    {"pp_frac_to_pct", PP_MAKE0(frac_to_pct), PP_T(frac_to_pct), PP_D(frac_to_pct)},
    {"pp_pct_to_frac", PP_MAKE0(pct_to_frac), PP_T(pct_to_frac), PP_D(pct_to_frac)},
    {"pp_from_absorbance", PP_MAKE(from_absorbance, 0), PP_T(from_absorbance), PP_D(from_absorbance)},
    {"pp_to_absorbance", PP_MAKE(to_absorbance, 0, 1e-6, 1), PP_T(to_absorbance), PP_D(to_absorbance)},
    {"pp_kubelka_munk", PP_MAKE(kubelka_munk, 0, 1e-6), PP_T(kubelka_munk), PP_D(kubelka_munk)},
    {"pp_range_disc",  /* discretizer → int32 bin indices */
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         static const double bins[3] = {0.4, 0.6, 0.8};
         n4m_pp_range_disc_handle_t* h = nullptr;
         return n4m_pp_range_disc_create(&h, bins, 3) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         return n4m_pp_range_disc_transform(static_cast<n4m_pp_range_disc_handle_t*>(h), c.X, c.out_i32); },
     PP_D(range_disc)},
    /* baseline-correction family (iterative) */
    {"pp_airpls", PP_MAKE(airpls, 1e5, 15, 1e-3), PP_T(airpls), PP_D(airpls)},
    {"pp_arpls",  PP_MAKE(arpls, 1e5, 50, 1e-3), PP_T(arpls), PP_D(arpls)},
    {"pp_asls",   PP_MAKE(asls, 1e5, 0.01, 10, 1e-3), PP_T(asls), PP_D(asls)},
    {"pp_iasls",  PP_MAKE(iasls, 1e5, 0.01, 2, 10, 1e-3), PP_T(iasls), PP_D(iasls)},
    {"pp_modpoly",  PP_MAKE(modpoly, 2, 100, 1e-3), PP_T(modpoly), PP_D(modpoly)},
    {"pp_imodpoly", PP_MAKE(imodpoly, 2, 100, 1e-3), PP_T(imodpoly), PP_D(imodpoly)},
    {"pp_beads", PP_MAKE(beads, 0.5, 5.0, 4.0, 20, 1e-2), PP_T(beads), PP_D(beads)},
    {"pp_snip", PP_MAKE(snip, 20), PP_T(snip), PP_D(snip)},
    {"pp_rolling_ball", PP_MAKE(rolling_ball, 10, 5), PP_T(rolling_ball), PP_D(rolling_ball)},
    {"pp_wavelet_denoise",  /* dim-preserving denoise (n×p out) */
     PP_MAKE(wavelet_denoise, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION,
             3, N4M_PP_WAVELET_THRESHOLD_SOFT, N4M_PP_WAVELET_NOISE_MEDIAN),
     PP_T(wavelet_denoise), PP_D(wavelet_denoise)},
    /* stateful, X-only fit */
    {"pp_msc", PP_MAKE0(msc), PP_FT(msc), PP_D(msc)},
    {"pp_emsc", PP_MAKE(emsc, 2), PP_FT(emsc), PP_D(emsc)},
    {"pp_baseline", PP_MAKE0(baseline), PP_FT(baseline), PP_D(baseline)},
    {"pp_kbins_disc", PP_MAKE(kbins_disc, 5, 0),  /* discretizer → int32 bins */
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_kbins_disc_handle_t*>(h);
         n4m_status_t s = n4m_pp_kbins_disc_fit(H, c.X);
         return s == N4M_OK ? n4m_pp_kbins_disc_transform(H, c.X, c.out_i32) : s; },
     PP_D(kbins_disc)},
    {"pp_log", PP_MAKE(log, 10.0, 0.0, 1, 1e-6), PP_FT(log), PP_D(log)},
    /* stateful, fit needs an extra vector */
    {"pp_osc",  PP_MAKE(osc, 2, 1),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_osc_handle_t*>(h);
         n4m_status_t s = n4m_pp_osc_fit(H, c.X, c.y, c.n);
         return s == N4M_OK ? n4m_pp_osc_transform(H, c.X, c.out) : s; },
     PP_D(osc)},
    /* pp_epo: EPO needs a calibrated difference/clutter direction set for
     * fit(); a synthetic spectrum yields a degenerate projection, so it is
     * left parity-only until a representative `d` is wired in. */

    /* ---- preprocessing, dim-changing. transform() writes n × out_cols, so
     *      each run queries the op's output-column count and transforms into
     *      a correctly-sized scratch buffer. That alloc is part of the timed
     *      op (these transforms produce a freshly-shaped array). Stateful
     *      ops fit() first. */
    {"pp_crop",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t p) -> void* {
         n4m_pp_crop_handle_t* h = nullptr;
         return n4m_pp_crop_create(&h, 1, p - 1) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_crop_handle_t*>(h);
         int64_t oc = n4m_pp_crop_output_cols(H, c.p);
         if (oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_crop_transform(H, c.X, ov); },
     PP_D(crop)},

    {"pp_derivate",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_pp_derivate_handle_t* h = nullptr;
         return n4m_pp_derivate_create(&h, 1, 1.0) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_derivate_handle_t*>(h);
         n4m_status_t fs = n4m_pp_derivate_fit(H, c.X);
         if (fs != N4M_OK) return fs;
         int64_t oc = n4m_pp_derivate_output_cols(1, c.p);
         if (oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_derivate_transform(H, c.X, ov); },
     PP_D(derivate)},

    {"pp_haar",
     PP_MAKE0(haar),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_haar_handle_t*>(h);
         int64_t oc = 0;
         if (n4m_pp_haar_output_cols(c.p, &oc) != N4M_OK || oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_haar_transform(H, c.X, ov); },
     PP_D(haar)},

    {"pp_resample",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         n4m_pp_resample_handle_t* h = nullptr;
         return n4m_pp_resample_create(&h, 128) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_resample_handle_t*>(h);
         int64_t oc = n4m_pp_resample_output_cols(H, c.p);
         if (oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_resample_transform(H, c.X, ov); },
     PP_D(resample)},

    {"pp_resampler",
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t p) -> void* {
         auto wl = wavelength_axis(p);
         n4m_pp_resampler_handle_t* h = nullptr;
         return n4m_pp_resampler_create(&h, wl.data(), p, 0, 0.0, 0.0, 0, 0.0, 0, 1) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_resampler_handle_t*>(h);
         /* resampler fits the source→target wavelength map from the source axis */
         n4m_status_t s = n4m_pp_resampler_fit(H, static_cast<const double*>(c.wl.data), c.p);
         if (s != N4M_OK) return s;
         int64_t oc = n4m_pp_resampler_output_cols(H);
         if (oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_resampler_transform(H, c.X, ov); },
     PP_D(resampler)},

    {"pp_wavelet",
     PP_MAKE(wavelet, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_wavelet_handle_t*>(h);
         int64_t oc = 0;
         if (n4m_pp_wavelet_output_cols(H, c.p, &oc) != N4M_OK || oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_wavelet_transform(H, c.X, ov); },
     PP_D(wavelet)},

    {"pp_wavelet_features",
     PP_MAKE(wavelet_features, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION, 3),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_wavelet_features_handle_t*>(h);
         int64_t oc = 0;
         if (n4m_pp_wavelet_features_output_cols(H, c.p, &oc) != N4M_OK || oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_wavelet_features_transform(H, c.X, ov); },
     PP_D(wavelet_features)},

    {"pp_wavelet_pca",
     PP_MAKE(wavelet_pca, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION, 3, 5.0),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_wavelet_pca_handle_t*>(h);
         n4m_status_t s = n4m_pp_wavelet_pca_fit(H, c.X);
         if (s != N4M_OK) return s;
         int64_t oc = 0;
         if (n4m_pp_wavelet_pca_output_cols(H, &oc) != N4M_OK || oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_wavelet_pca_transform(H, c.X, ov); },
     PP_D(wavelet_pca)},

    {"pp_wavelet_svd",
     PP_MAKE(wavelet_svd, N4M_PP_WAVELET_HAAR, N4M_PP_WAVELET_BOUNDARY_PERIODIZATION, 3, 5.0),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_wavelet_svd_handle_t*>(h);
         n4m_status_t s = n4m_pp_wavelet_svd_fit(H, c.X);
         if (s != N4M_OK) return s;
         int64_t oc = 0;
         if (n4m_pp_wavelet_svd_output_cols(H, &oc) != N4M_OK || oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_wavelet_svd_transform(H, c.X, ov); },
     PP_D(wavelet_svd)},

    {"pp_flex_pca",
     PP_MAKE(flex_pca, 5.0),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_flex_pca_handle_t*>(h);
         n4m_status_t s = n4m_pp_flex_pca_fit(H, c.X);
         if (s != N4M_OK) return s;
         int64_t oc = 0;
         if (n4m_pp_flex_pca_output_cols(H, &oc) != N4M_OK || oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_flex_pca_transform(H, c.X, ov); },
     PP_D(flex_pca)},

    {"pp_flex_svd",
     PP_MAKE(flex_svd, 5.0),
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_flex_svd_handle_t*>(h);
         n4m_status_t s = n4m_pp_flex_svd_fit(H, c.X);
         if (s != N4M_OK) return s;
         int64_t oc = 0;
         if (n4m_pp_flex_svd_output_cols(H, &oc) != N4M_OK || oc <= 0) return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_flex_svd_transform(H, c.X, ov); },
     PP_D(flex_svd)},

    {"pp_fck_static",  /* FCK static transformer (n_kernels = n_orders × n_scales) */
     [](n4m_rng_pcg64_state_t*, int64_t, int64_t) -> void* {
         static const double orders[3] = {0.0, 1.0, 2.0};
         static const double scales[2] = {1.0, 2.0};
         n4m_pp_fck_static_handle_t* h = nullptr;
         return n4m_pp_fck_static_create(&h, 5, orders, 3, scales, 2) == N4M_OK ? h : nullptr; },
     [](void* h, BenchCtx& c) {
         auto* H = static_cast<n4m_pp_fck_static_handle_t*>(h);
         int32_t oc = 0;  /* 6 kernels (3 orders × 2 scales) */
         if (n4m_pp_fck_static_output_cols(6, static_cast<int32_t>(c.p), &oc) != N4M_OK || oc <= 0)
             return N4M_ERR_INVALID_ARGUMENT;
         std::vector<double> ob(static_cast<size_t>(c.n) * static_cast<size_t>(oc));
         n4m_matrix_view_t ov;
         n4m_matrix_view_init_rowmajor(&ov, ob.data(), c.n, oc, N4M_DTYPE_F64);
         return n4m_pp_fck_static_transform(H, c.X, ov); },
     PP_D(fck_static)},
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
    std::string dump_path;  /* --dump-output: write c.out of the first fresh
                             * run (raw little-endian f64, n*p) for binding-vs-
                             * C++ transcription validation. Valid only for ops
                             * whose run writes c.out (augmenters + dim-
                             * preserving preprocessing). */
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
        else if (a == "--dump-output") dump_path = next("--dump-output needs a value");
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
    std::vector<double> y(static_cast<size_t>(n));
    std::vector<uint8_t> mask(static_cast<size_t>(n), 0);
    std::vector<int32_t> out_i32(X.size(), 0);
    std::vector<int64_t> groups(static_cast<size_t>(n));
    fill_matrix(X, seed);
    fill_matrix(y, seed ^ 0xABCDEFULL);  /* deterministic target vector */
    for (int64_t i = 0; i < n; ++i) groups[static_cast<size_t>(i)] = i % 8;  /* 8 groups */

    BenchCtx ctx;
    ctx.n = n;
    ctx.p = p;
    ctx.y = y.data();
    ctx.mask = mask.data();
    ctx.groups = groups.data();
    if (n4m_matrix_view_init_rowmajor(&ctx.X, X.data(), n, p, N4M_DTYPE_F64) != N4M_OK ||
        n4m_matrix_view_init_rowmajor(&ctx.out, out.data(), n, p, N4M_DTYPE_F64) != N4M_OK ||
        n4m_matrix_view_init_rowmajor(&ctx.wl, wl.data(), 1, p, N4M_DTYPE_F64) != N4M_OK ||
        n4m_matrix_view_init_rowmajor(&ctx.Y, y.data(), n, 1, N4M_DTYPE_F64) != N4M_OK ||
        n4m_matrix_view_init_rowmajor(&ctx.out_i32, out_i32.data(), n, p, N4M_DTYPE_I32) != N4M_OK) {
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

    /* Dump the first fresh run's output for transcription validation — before
     * the timing loop advances the handle RNG, so stochastic augmenters match
     * a single fresh Python `PCG64(seed)` apply. */
    if (ok && !dump_path.empty()) {
        if (std::FILE* f = std::fopen(dump_path.c_str(), "wb")) {
            std::fwrite(out.data(), sizeof(double), out.size(), f);
            std::fclose(f);
        } else {
            die(("could not open --dump-output path: " + dump_path).c_str());
        }
    }

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
