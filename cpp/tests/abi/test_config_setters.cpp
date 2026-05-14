// SPDX-License-Identifier: CeCILL-2.1
//
// Setter / getter round-trip tests. Every setter:
//   1. Returns P4A_OK for a valid value.
//   2. Round-trips through the matching getter.
//   3. Rejects out-of-range values with P4A_ERR_INVALID_ARGUMENT.
//   4. Is idempotent on rejection (state unchanged).
//   5. Returns P4A_ERR_NULL_POINTER on a NULL config.

#include <limits>

#include "pls4all/p4a.h"
#include "harness.hpp"

namespace {

p4a_config_t* fresh_cfg() {
    p4a_config_t* c = nullptr;
    p4a_config_create(&c);
    return c;
}

}  // namespace

TEST(config_setters, defaults_match_spec) {
    auto* cfg = fresh_cfg();
    p4a_algorithm_t a = P4A_ALGO_PCR;       // start with wrong value
    p4a_solver_t s    = P4A_SOLVER_SVD;
    p4a_deflation_t d = P4A_DEFLATION_XY;
    int32_t nc = 0;
    int32_t cx = 9, sx = 9, cy = 9, sy = 9;
    double tol = 0.0;
    int32_t mi = 0;
    int32_t ss = 9, sd = 9;
    p4a_dtype_t dt = P4A_DTYPE_F32;
    CHECK_EQ(p4a_config_get_algorithm(cfg, &a), P4A_OK);
    CHECK_EQ(p4a_config_get_solver(cfg, &s),    P4A_OK);
    CHECK_EQ(p4a_config_get_deflation(cfg, &d), P4A_OK);
    CHECK_EQ(p4a_config_get_n_components(cfg, &nc), P4A_OK);
    CHECK_EQ(p4a_config_get_center_x(cfg, &cx), P4A_OK);
    CHECK_EQ(p4a_config_get_scale_x(cfg, &sx),  P4A_OK);
    CHECK_EQ(p4a_config_get_center_y(cfg, &cy), P4A_OK);
    CHECK_EQ(p4a_config_get_scale_y(cfg, &sy),  P4A_OK);
    CHECK_EQ(p4a_config_get_tol(cfg, &tol),     P4A_OK);
    CHECK_EQ(p4a_config_get_max_iter(cfg, &mi), P4A_OK);
    CHECK_EQ(p4a_config_get_store_scores(cfg, &ss), P4A_OK);
    CHECK_EQ(p4a_config_get_store_diagnostics(cfg, &sd), P4A_OK);
    CHECK_EQ(p4a_config_get_dtype(cfg, &dt),    P4A_OK);
    CHECK_EQ(a, P4A_ALGO_PLS_REGRESSION);
    CHECK_EQ(s, P4A_SOLVER_NIPALS);
    CHECK_EQ(d, P4A_DEFLATION_REGRESSION);
    CHECK_EQ(nc, 2);
    CHECK_EQ(cx, 1); CHECK_EQ(sx, 1); CHECK_EQ(cy, 1); CHECK_EQ(sy, 1);
    CHECK_EQ(tol, 1e-6);
    CHECK_EQ(mi, 500);
    CHECK_EQ(ss, 0); CHECK_EQ(sd, 0);
    CHECK_EQ(dt, P4A_DTYPE_F64);
    p4a_config_destroy(cfg);
}

TEST(config_setters, algorithm_round_trip) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(p4a_config_set_algorithm(cfg, P4A_ALGO_OPLS), P4A_OK);
    p4a_algorithm_t a = P4A_ALGO_PCR;
    CHECK_EQ(p4a_config_get_algorithm(cfg, &a), P4A_OK);
    CHECK_EQ(a, P4A_ALGO_OPLS);
    p4a_config_destroy(cfg);
}

TEST(config_setters, algorithm_rejects_out_of_range) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(p4a_config_set_algorithm(cfg, P4A_ALGO_OPLS), P4A_OK);
    CHECK_EQ(p4a_config_set_algorithm(cfg, (p4a_algorithm_t)999),
              P4A_ERR_INVALID_ARGUMENT);
    // State unchanged on rejection
    p4a_algorithm_t a;
    CHECK_EQ(p4a_config_get_algorithm(cfg, &a), P4A_OK);
    CHECK_EQ(a, P4A_ALGO_OPLS);
    p4a_config_destroy(cfg);
}

TEST(config_setters, n_components_rejects_non_positive) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(p4a_config_set_n_components(cfg, 5), P4A_OK);
    CHECK_EQ(p4a_config_set_n_components(cfg, 0),  P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(p4a_config_set_n_components(cfg, -1), P4A_ERR_INVALID_ARGUMENT);
    int32_t nc;
    CHECK_EQ(p4a_config_get_n_components(cfg, &nc), P4A_OK);
    CHECK_EQ(nc, 5);
    p4a_config_destroy(cfg);
}

TEST(config_setters, tol_rejects_nan_inf_nonpositive) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(p4a_config_set_tol(cfg, 1e-3), P4A_OK);
    CHECK_EQ(p4a_config_set_tol(cfg, std::numeric_limits<double>::quiet_NaN()),
              P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(p4a_config_set_tol(cfg, std::numeric_limits<double>::infinity()),
              P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(p4a_config_set_tol(cfg, -1.0), P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(p4a_config_set_tol(cfg, 0.0),  P4A_ERR_INVALID_ARGUMENT);
    double t;
    CHECK_EQ(p4a_config_get_tol(cfg, &t), P4A_OK);
    CHECK_EQ(t, 1e-3);
    p4a_config_destroy(cfg);
}

TEST(config_setters, bool_setters_reject_non_01) {
    auto* cfg = fresh_cfg();
    CHECK_EQ(p4a_config_set_center_x(cfg, 1), P4A_OK);
    CHECK_EQ(p4a_config_set_center_x(cfg, 2), P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(p4a_config_set_center_x(cfg, -1), P4A_ERR_INVALID_ARGUMENT);
    int32_t v;
    CHECK_EQ(p4a_config_get_center_x(cfg, &v), P4A_OK);
    CHECK_EQ(v, 1);
    p4a_config_destroy(cfg);
}

TEST(config_setters, null_config_returns_null_pointer) {
    CHECK_EQ(p4a_config_set_algorithm(nullptr, P4A_ALGO_PLS_REGRESSION),
              P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_config_set_tol(nullptr, 1e-3), P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_config_set_n_components(nullptr, 5), P4A_ERR_NULL_POINTER);
}

TEST(config_setters, clone_deep_copy) {
    auto* src = fresh_cfg();
    p4a_config_set_n_components(src, 7);
    p4a_config_set_algorithm(src, P4A_ALGO_OPLS_DA);
    p4a_config_t* dst = nullptr;
    CHECK_EQ(p4a_config_clone(src, &dst), P4A_OK);
    int32_t nc;
    CHECK_EQ(p4a_config_get_n_components(dst, &nc), P4A_OK);
    CHECK_EQ(nc, 7);
    // Mutate dst; src unchanged.
    p4a_config_set_n_components(dst, 99);
    CHECK_EQ(p4a_config_get_n_components(src, &nc), P4A_OK);
    CHECK_EQ(nc, 7);
    p4a_config_destroy(src);
    p4a_config_destroy(dst);
}

TEST(config_setters, hook_pointers_round_trip_and_clear) {
    auto* cfg = fresh_cfg();
    p4a_pipeline_t* pipe = nullptr;
    p4a_pipeline_create(&pipe);
    CHECK_EQ(p4a_config_set_pipeline(cfg, pipe), P4A_OK);
    const p4a_pipeline_t* got = nullptr;
    CHECK_EQ(p4a_config_get_pipeline(cfg, &got), P4A_OK);
    CHECK_EQ(got, pipe);
    // NULL clears
    CHECK_EQ(p4a_config_set_pipeline(cfg, nullptr), P4A_OK);
    CHECK_EQ(p4a_config_get_pipeline(cfg, &got), P4A_OK);
    CHECK_EQ(got, nullptr);
    p4a_pipeline_destroy(pipe);
    p4a_config_destroy(cfg);
}
