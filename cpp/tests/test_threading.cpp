// SPDX-License-Identifier: CECILL-2.1
//
// Multi-context threading test.
//
// The n4m.h contract (§4) is: "a single n4m_context_t* must not be used
// concurrently from two threads — use one context per thread. Across contexts
// the library is thread-safe." This test fits the SAME PLS model from several
// std::threads, each with its OWN context/config/model, and asserts that every
// thread's fitted coefficients are BIT-IDENTICAL to the single-threaded
// baseline — guarding both determinism and the cross-context thread-safety
// promise. It also checks that last_error is per-context (an error provoked on
// one thread's context never leaks into another's).

#include "n4m/n4m.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "harness.hpp"

void register_threading_tests(n4m_testing::Runner& r);

namespace {

constexpr std::int64_t kRows = 6;
constexpr std::int64_t kFeatures = 4;
constexpr std::int32_t kComponents = 2;
constexpr std::size_t kThreads = 6;

const double kX[kRows * kFeatures] = {
    0.11, 0.52, 0.33, 0.04,
    0.45, 0.16, 0.67, 0.28,
    0.79, 0.81, 0.22, 0.13,
    0.24, 0.55, 0.96, 0.37,
    0.68, 0.39, 0.18, 0.92,
    0.51, 0.27, 0.74, 0.66,
};
const double kY[kRows] = {1.0, 2.0, 3.0, 2.5, 1.5, 2.2};

// Fit the fixed problem with a private context; return its coefficients (bytes
// preserved). On any ABI failure, throws via N4M_TEST_REQUIRE.
std::vector<double> fit_coefficients() {
    n4m_context_t* ctx = nullptr;
    n4m_config_t*  cfg = nullptr;
    n4m_model_t*   model = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, kComponents) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);

    n4m_matrix_view_t X{}, Y{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &X, const_cast<double*>(kX), kRows, kFeatures, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &Y, const_cast<double*>(kY), kRows, 1, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, &X, &Y, &model) == N4M_OK);
    N4M_TEST_REQUIRE(model != nullptr);

    n4m_array_t* arr = nullptr;
    N4M_TEST_REQUIRE(n4m_model_get_array(ctx, model, N4M_MODEL_COEFFICIENTS, &arr) == N4M_OK);
    std::int64_t rows = 0, cols = 0;
    N4M_TEST_REQUIRE(n4m_array_shape(arr, &rows, &cols) == N4M_OK);
    n4m_matrix_view_t view{};
    N4M_TEST_REQUIRE(n4m_array_view(arr, &view) == N4M_OK);
    const auto* p = static_cast<const double*>(view.data);
    std::vector<double> coef(p, p + rows * cols);
    n4m_array_free(arr);

    n4m_model_destroy(model);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return coef;
}

bool bit_identical(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint64_t ab, bb;
        std::memcpy(&ab, &a[i], sizeof(double));
        std::memcpy(&bb, &b[i], sizeof(double));
        if (ab != bb) return false;
    }
    return true;
}

void test_multi_context_fit_is_bit_identical() {
    const std::vector<double> baseline = fit_coefficients();
    N4M_TEST_REQUIRE(!baseline.empty());

    std::vector<std::vector<double>> results(kThreads);
    std::vector<std::string> errors(kThreads);
    std::vector<std::thread> pool;
    pool.reserve(kThreads);

    // Start barrier: every worker registers as ready and spins until released,
    // so all kThreads are inside fit_coefficients() concurrently. Without this a
    // scheduler (or a library-wide lock) could serialize the fits and the
    // bit-identity assertion would pass without ever exercising true overlap.
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> go{false};

    for (std::size_t t = 0; t < kThreads; ++t) {
        pool.emplace_back([t, &results, &errors, &ready, &go]() {
            ready.fetch_add(std::size_t{1}, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                results[t] = fit_coefficients();
            } catch (const std::exception& e) {
                errors[t] = e.what();
            } catch (...) {
                errors[t] = "non-std exception in worker";
            }
        });
    }
    while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);   // release all workers at once
    for (auto& th : pool) th.join();

    for (std::size_t t = 0; t < kThreads; ++t) {
        if (!errors[t].empty()) {
            throw std::runtime_error("thread " + std::to_string(t) + " failed: " + errors[t]);
        }
        if (!bit_identical(results[t], baseline)) {
            throw std::runtime_error("thread " + std::to_string(t) +
                " coefficients are not bit-identical to the single-threaded baseline");
        }
    }
}

// Provoke an error on one thread's context and a success on another's,
// concurrently, and assert last_error is per-context (no cross-talk).
void test_last_error_is_per_context() {
    std::string err_good;
    std::string err_bad;
    bool good_ok = false;
    bool bad_rejected = false;

    std::thread good([&]() {
        n4m_context_t* ctx = nullptr;
        n4m_config_t*  cfg = nullptr;
        n4m_context_create(&ctx);
        n4m_config_create(&cfg);
        n4m_config_set_n_components(cfg, kComponents);
        n4m_matrix_view_t X{}, Y{};
        n4m_matrix_view_init_rowmajor(&X, const_cast<double*>(kX), kRows, kFeatures, N4M_DTYPE_F64);
        n4m_matrix_view_init_rowmajor(&Y, const_cast<double*>(kY), kRows, 1, N4M_DTYPE_F64);
        n4m_model_t* m = nullptr;
        good_ok = (n4m_model_fit(ctx, cfg, &X, &Y, &m) == N4M_OK);
        err_good = n4m_context_last_error(ctx);  // expect empty on its own context
        n4m_model_destroy(m);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    });

    std::thread bad([&]() {
        n4m_context_t* ctx = nullptr;
        n4m_config_t*  cfg = nullptr;
        n4m_context_create(&ctx);
        n4m_config_create(&cfg);
        n4m_config_set_n_components(cfg, 1);
        std::vector<std::int32_t> xi(kRows * kFeatures, 1);
        std::vector<double> y(kRows, 1.0);
        n4m_matrix_view_t X{}, Y{};
        n4m_matrix_view_init_rowmajor(&X, xi.data(), kRows, kFeatures, N4M_DTYPE_I32);
        n4m_matrix_view_init_rowmajor(&Y, y.data(), kRows, 1, N4M_DTYPE_F64);
        n4m_model_t* m = nullptr;
        bad_rejected = (n4m_model_fit(ctx, cfg, &X, &Y, &m) == N4M_ERR_DTYPE_MISMATCH);
        err_bad = n4m_context_last_error(ctx);  // expect a non-empty message
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    });

    good.join();
    bad.join();

    N4M_TEST_REQUIRE(good_ok);
    N4M_TEST_REQUIRE(bad_rejected);
    // The successful context never picked up the failing context's error.
    N4M_TEST_REQUIRE(err_good.empty());
    N4M_TEST_REQUIRE(!err_bad.empty());
}

// Deterministic per-context error isolation, same thread: two contexts, an
// error on one must not appear on the other. This proves the error buffer is a
// per-context member (not thread-local or global) without relying on thread
// interleaving — if it were thread-local/global, ctx_good (same thread) would
// observe ctx_bad's message.
void test_error_isolation_same_thread() {
    n4m_context_t* ctx_good = nullptr;
    n4m_context_t* ctx_bad  = nullptr;
    n4m_config_t*  cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx_good) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_context_create(&ctx_bad) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, 1) == N4M_OK);

    // Clean fit on ctx_good.
    n4m_matrix_view_t Xg{}, Yg{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Xg, const_cast<double*>(kX), kRows, kFeatures, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Yg, const_cast<double*>(kY), kRows, 1, N4M_DTYPE_F64) == N4M_OK);
    n4m_model_t* mg = nullptr;
    N4M_TEST_REQUIRE(n4m_model_fit(ctx_good, cfg, &Xg, &Yg, &mg) == N4M_OK);

    // Failing fit on ctx_bad (I32 X -> DTYPE_MISMATCH), AFTER the good fit.
    std::vector<std::int32_t> xi(static_cast<std::size_t>(kRows * kFeatures), 1);
    std::vector<double> y(static_cast<std::size_t>(kRows), 1.0);
    n4m_matrix_view_t Xb{}, Yb{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Xb, xi.data(), kRows, kFeatures, N4M_DTYPE_I32) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Yb, y.data(), kRows, 1, N4M_DTYPE_F64) == N4M_OK);
    n4m_model_t* mb = nullptr;
    N4M_TEST_REQUIRE(n4m_model_fit(ctx_bad, cfg, &Xb, &Yb, &mb) == N4M_ERR_DTYPE_MISMATCH);

    // ctx_bad holds the error; ctx_good (same thread) is untouched.
    const std::string bad_err  = n4m_context_last_error(ctx_bad);
    const std::string good_err = n4m_context_last_error(ctx_good);
    N4M_TEST_REQUIRE(!bad_err.empty());
    N4M_TEST_REQUIRE(good_err.empty());

    n4m_model_destroy(mb);
    n4m_model_destroy(mg);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx_bad);
    n4m_context_destroy(ctx_good);
}

}  // namespace

void register_threading_tests(n4m_testing::Runner& r) {
    r.run("threading/multi_context_fit_bit_identical", test_multi_context_fit_is_bit_identical);
    r.run("threading/last_error_is_per_context",       test_last_error_is_per_context);
    r.run("threading/error_isolation_same_thread",     test_error_isolation_same_thread);
}
