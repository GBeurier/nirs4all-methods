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
 * Usage:
 *   n4m_donor_bench --method aug_gaussian_noise --n 250 --p 50 \
 *                   --runs 40 --seed 1234567890
 *   n4m_donor_bench --list      # print the registered method ids, one per line
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

/* (target_total, statistic, decision) chosen from the probe run, exactly as
 * _common.adaptive_target() does. */
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
/* splitmix64 → reproducible standard-normal-ish F64 fill. The exact values
 * don't matter for timing; determinism (per seed) does. */
uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void fill_matrix(std::vector<double>& buf, uint64_t seed) {
    uint64_t state = seed ? seed : 0x1234567890ULL;
    for (double& x : buf) {
        /* uniform in [0,1), then shifted to a spectra-like band [0.2, 1.0) */
        const uint64_t r = splitmix64(state);
        const double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
        x = 0.2 + 0.8 * u;
    }
}

/* ----- donor operator registry ----------------------------------------- */
/* Each op exposes the universal augmenter triple via type-erased function
 * pointers. `make` constructs the handle with curated default params (taken
 * from the parity fixtures / smoke tests); `apply` is the timed operation;
 * `destroy` releases the handle. Captureless lambdas decay to function
 * pointers, so no std::function / allocation is involved on the hot path. */
struct DonorOp {
    const char* name;
    void* (*make)(n4m_rng_pcg64_state_t* rng);
    n4m_status_t (*apply)(void* h, n4m_matrix_view_t X, n4m_matrix_view_t out);
    void (*destroy)(void* h);
};

const DonorOp kOps[] = {
    {"aug_gaussian_noise",
     [](n4m_rng_pcg64_state_t* rng) -> void* {
         n4m_aug_gaussian_noise_handle_t* h = nullptr;
         if (n4m_aug_gaussian_noise_create(&h, rng, 0.01) != N4M_OK) return nullptr;
         return h;
     },
     [](void* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
         return n4m_aug_gaussian_noise_apply(
             static_cast<n4m_aug_gaussian_noise_handle_t*>(h), X, out);
     },
     [](void* h) { n4m_aug_gaussian_noise_destroy(static_cast<n4m_aug_gaussian_noise_handle_t*>(h)); }},

    {"aug_multiplicative_noise",
     [](n4m_rng_pcg64_state_t* rng) -> void* {
         n4m_aug_multiplicative_noise_handle_t* h = nullptr;
         if (n4m_aug_multiplicative_noise_create(&h, rng, 0.05) != N4M_OK) return nullptr;
         return h;
     },
     [](void* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
         return n4m_aug_multiplicative_noise_apply(
             static_cast<n4m_aug_multiplicative_noise_handle_t*>(h), X, out);
     },
     [](void* h) { n4m_aug_multiplicative_noise_destroy(static_cast<n4m_aug_multiplicative_noise_handle_t*>(h)); }},

    {"aug_spike_noise",
     [](n4m_rng_pcg64_state_t* rng) -> void* {
         n4m_aug_spike_noise_handle_t* h = nullptr;
         if (n4m_aug_spike_noise_create(&h, rng, 1, 3, -0.5, 0.5) != N4M_OK) return nullptr;
         return h;
     },
     [](void* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
         return n4m_aug_spike_noise_apply(
             static_cast<n4m_aug_spike_noise_handle_t*>(h), X, out);
     },
     [](void* h) { n4m_aug_spike_noise_destroy(static_cast<n4m_aug_spike_noise_handle_t*>(h)); }},

    {"aug_hetero_noise",
     [](n4m_rng_pcg64_state_t* rng) -> void* {
         n4m_aug_hetero_noise_handle_t* h = nullptr;
         if (n4m_aug_hetero_noise_create(&h, rng, 1e-3, 5e-3) != N4M_OK) return nullptr;
         return h;
     },
     [](void* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
         return n4m_aug_hetero_noise_apply(
             static_cast<n4m_aug_hetero_noise_handle_t*>(h), X, out);
     },
     [](void* h) { n4m_aug_hetero_noise_destroy(static_cast<n4m_aug_hetero_noise_handle_t*>(h)); }},

    {"aug_linear_drift",
     [](n4m_rng_pcg64_state_t* rng) -> void* {
         n4m_aug_linear_drift_handle_t* h = nullptr;
         if (n4m_aug_linear_drift_create(&h, rng, -0.1, 0.1, -0.001, 0.001) != N4M_OK) return nullptr;
         return h;
     },
     [](void* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
         return n4m_aug_linear_drift_apply(
             static_cast<n4m_aug_linear_drift_handle_t*>(h), X, out);
     },
     [](void* h) { n4m_aug_linear_drift_destroy(static_cast<n4m_aug_linear_drift_handle_t*>(h)); }},

    {"aug_poly_drift",
     [](n4m_rng_pcg64_state_t* rng) -> void* {
         static const double lo[3] = {-0.1, -0.05, -0.033};
         static const double hi[3] = {0.1, 0.05, 0.033};
         n4m_aug_poly_drift_handle_t* h = nullptr;
         if (n4m_aug_poly_drift_create(&h, rng, 2, lo, hi) != N4M_OK) return nullptr;
         return h;
     },
     [](void* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
         return n4m_aug_poly_drift_apply(
             static_cast<n4m_aug_poly_drift_handle_t*>(h), X, out);
     },
     [](void* h) { n4m_aug_poly_drift_destroy(static_cast<n4m_aug_poly_drift_handle_t*>(h)); }},

    {"aug_path_length",
     [](n4m_rng_pcg64_state_t* rng) -> void* {
         n4m_aug_path_length_handle_t* h = nullptr;
         if (n4m_aug_path_length_create(&h, rng, 0.05, 0.5) != N4M_OK) return nullptr;
         return h;
     },
     [](void* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
         return n4m_aug_path_length_apply(
             static_cast<n4m_aug_path_length_handle_t*>(h), X, out);
     },
     [](void* h) { n4m_aug_path_length_destroy(static_cast<n4m_aug_path_length_handle_t*>(h)); }},
};

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
     * context, and these ops are element-wise single-threaded anyway, so we
     * only record `threads` in the output for the dashboard cell key. */

    /* One input matrix + one output buffer, reused across timed runs. The
     * timed operation is the operator's apply(): create/destroy is one-time
     * setup, not part of the per-call cost the dashboard reports. */
    std::vector<double> X(static_cast<size_t>(n) * static_cast<size_t>(p));
    std::vector<double> out(X.size());
    fill_matrix(X, seed);

    n4m_matrix_view_t Xv, outv;
    if (n4m_matrix_view_init_rowmajor(&Xv, X.data(), n, p, N4M_DTYPE_F64) != N4M_OK ||
        n4m_matrix_view_init_rowmajor(&outv, out.data(), n, p, N4M_DTYPE_F64) != N4M_OK) {
        die("matrix view init failed");
    }

    n4m_rng_pcg64_state_t* rng = nullptr;
    if (n4m_rng_pcg64_create(seed, &rng) != N4M_OK) die("rng create failed");
    void* handle = op->make(rng);
    if (handle == nullptr) {
        n4m_rng_pcg64_destroy(rng);
        die("operator create failed");
    }

    auto one_run = [&]() -> bool { return op->apply(handle, Xv, outv) == N4M_OK; };

    /* warm-up (discarded) */
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
                    "\"threads\":%ld,\"reason\":\"apply returned non-OK\"}\n",
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
