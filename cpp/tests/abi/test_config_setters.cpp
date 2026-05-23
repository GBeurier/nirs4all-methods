// SPDX-License-Identifier: CECILL-2.1
//
// Setter / getter round-trip tests. Every setter:
//   1. Returns N4M_OK for a valid value.
//   2. Round-trips through the matching getter.
//   3. Rejects out-of-range values with N4M_ERR_INVALID_ARGUMENT.
//   4. Is idempotent on rejection (state unchanged).
//   5. Returns N4M_ERR_NULL_POINTER on a NULL config.

#include <limits>

#include "n4m/n4m.h"
#include "harness.hpp"

namespace {

n4m_config_t* fresh_cfg() {
    n4m_config_t* c = nullptr;
    n4m_config_create(&c);
    return c;
}

}  // namespace

TEST(config_setters, defaults_match_spec) {
    auto* cfg = fresh_cfg();
    n4m_algorithm_t a = N4M_ALGO_PCR;       // start with wrong value
    n4m_solver_t s    = N4M_SOLVER_SVD;
    n4m_deflation_t d = N4M_DEFLATION_XY;
    int32_t nc = 0;
    int32_t cx = 9, sx = 9, cy = 9, sy = 9;
    double tol = 0.0;
    int32_t mi = 0;
    int32_t ss = 9, sd = 9;
    n4m_dtype_t dt = N4M_DTYPE_F32;
    CHECK_EQ(n4m_config_get_algorithm(cfg, &a), N4M_OK);
    CHECK_EQ(n4m_config_get_solver(cfg, &s),    N4M_OK);
    CHECK_EQ(n4m_config_get_deflation(cfg, &d), N4M_OK);
    CHECK_EQ(n4m_config_get_n_components(cfg, &nc), N4M_OK);
    CHECK_EQ(n4m_config_get_center_x(cfg, &cx), N4M_OK);
    CHECK_EQ(n4m_config_get_scale_x(cfg, &sx),  N4M_OK);
    CHECK_EQ(n4m_config_get_center_y(cfg, &cy), N4M_OK);
    CHECK_EQ(n4m_config_get_scale_y(cfg, &sy),  N4M_OK);
    CHECK_EQ(n4m_config_get_tol(cfg, &tol),     N4M_OK);
    CHECK_EQ(n4m_config_get_max_iter(cfg, &mi), N4M_OK);
    CHECK_EQ(n4m_config_get_store_scores(cfg, &ss), N4M_OK);
    CHECK_EQ(n4m_config_get_store_diagnostics(cfg, &sd), N4M_OK);
    CHECK_EQ(n4m_config_get_dtype(cfg, &dt),    N4M_OK);
    CHECK_EQ(a, N4M_ALGO_PLS_REGRESSION);
    CHECK_EQ(s, N4M_SOLVER_NIPALS);
    CHECK_EQ(d, N4M_DEFLATION_REGRESSION);
    CHECK_EQ(nc, 2);
    CHECK_EQ(cx, 1); CHECK_EQ(sx, 1); CHECK_EQ(cy, 1); CHECK_EQ(sy, 1);
    CHECK_EQ(tol, 1e-6);
    CHECK_EQ(mi, 500);
    CHECK_EQ(ss, 0); CHECK_EQ(sd, 0);
    CHECK_EQ(dt, N4M_DTYPE_F64);
    n4m_config_destroy(cfg);
}

TEST(config_setters, algorithm_round_trip) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(n4m_config_set_algorithm(cfg, N4M_ALGO_OPLS), N4M_OK);
    n4m_algorithm_t a = N4M_ALGO_PCR;
    CHECK_EQ(n4m_config_get_algorithm(cfg, &a), N4M_OK);
    CHECK_EQ(a, N4M_ALGO_OPLS);
    n4m_config_destroy(cfg);
}

TEST(config_setters, algorithm_rejects_out_of_range) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(n4m_config_set_algorithm(cfg, N4M_ALGO_OPLS), N4M_OK);
    const auto invalid_algorithm = static_cast<n4m_algorithm_t>(N4M_ALGO_PCR + 1);
    CHECK_EQ(n4m_config_set_algorithm(cfg, invalid_algorithm),
              N4M_ERR_INVALID_ARGUMENT);
    // State unchanged on rejection
    n4m_algorithm_t a;
    CHECK_EQ(n4m_config_get_algorithm(cfg, &a), N4M_OK);
    CHECK_EQ(a, N4M_ALGO_OPLS);
    n4m_config_destroy(cfg);
}

TEST(config_setters, n_components_rejects_non_positive) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(n4m_config_set_n_components(cfg, 5), N4M_OK);
    CHECK_EQ(n4m_config_set_n_components(cfg, 0),  N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n4m_config_set_n_components(cfg, -1), N4M_ERR_INVALID_ARGUMENT);
    int32_t nc;
    CHECK_EQ(n4m_config_get_n_components(cfg, &nc), N4M_OK);
    CHECK_EQ(nc, 5);
    n4m_config_destroy(cfg);
}

TEST(config_setters, tol_rejects_nan_inf_nonpositive) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(n4m_config_set_tol(cfg, 1e-3), N4M_OK);
    CHECK_EQ(n4m_config_set_tol(cfg, std::numeric_limits<double>::quiet_NaN()),
              N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n4m_config_set_tol(cfg, std::numeric_limits<double>::infinity()),
              N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n4m_config_set_tol(cfg, -1.0), N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n4m_config_set_tol(cfg, 0.0),  N4M_ERR_INVALID_ARGUMENT);
    double t;
    CHECK_EQ(n4m_config_get_tol(cfg, &t), N4M_OK);
    CHECK_EQ(t, 1e-3);
    n4m_config_destroy(cfg);
}

TEST(config_setters, bool_setters_reject_non_01) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(n4m_config_set_center_x(cfg, 1), N4M_OK);
    CHECK_EQ(n4m_config_set_center_x(cfg, 2), N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n4m_config_set_center_x(cfg, -1), N4M_ERR_INVALID_ARGUMENT);
    int32_t v;
    CHECK_EQ(n4m_config_get_center_x(cfg, &v), N4M_OK);
    CHECK_EQ(v, 1);
    n4m_config_destroy(cfg);
}

TEST(config_setters, null_config_returns_null_pointer) {
    CHECK_EQ(n4m_config_set_algorithm(nullptr, N4M_ALGO_PLS_REGRESSION),
              N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_config_set_tol(nullptr, 1e-3), N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_config_set_n_components(nullptr, 5), N4M_ERR_NULL_POINTER);
}

TEST(config_setters, clone_deep_copy) {
    auto* src = fresh_cfg();
    n4m_config_set_n_components(src, 7);
    n4m_config_set_algorithm(src, N4M_ALGO_OPLS_DA);
    n4m_config_t* dst = nullptr;
    CHECK_EQ(n4m_config_clone(src, &dst), N4M_OK);
    int32_t nc;
    CHECK_EQ(n4m_config_get_n_components(dst, &nc), N4M_OK);
    CHECK_EQ(nc, 7);
    // Mutate dst; src unchanged.
    n4m_config_set_n_components(dst, 99);
    CHECK_EQ(n4m_config_get_n_components(src, &nc), N4M_OK);
    CHECK_EQ(nc, 7);
    n4m_config_destroy(src);
    n4m_config_destroy(dst);
}

TEST(config_setters, hook_pointers_round_trip_and_clear) {
    auto* cfg = fresh_cfg();
    n4m_pipeline_t* pipe = nullptr;
    n4m_pipeline_create(&pipe);
    CHECK_EQ(n4m_config_set_pipeline(cfg, pipe), N4M_OK);
    const n4m_pipeline_t* got = nullptr;
    CHECK_EQ(n4m_config_get_pipeline(cfg, &got), N4M_OK);
    CHECK_EQ(got, pipe);
    // NULL clears
    CHECK_EQ(n4m_config_set_pipeline(cfg, nullptr), N4M_OK);
    CHECK_EQ(n4m_config_get_pipeline(cfg, &got), N4M_OK);
    CHECK_EQ(got, nullptr);
    n4m_pipeline_destroy(pipe);
    n4m_config_destroy(cfg);
}
