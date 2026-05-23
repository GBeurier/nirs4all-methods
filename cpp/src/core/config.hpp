// SPDX-License-Identifier: CECILL-2.1
//
// Internal Config — POD-style storage for every knob declared on the
// public n4m_config_t handle. Setter validation lives in the c_api_config
// wrappers; the Config class itself only holds values.

#pragma once

#include <cstdint>

#include "n4m/n4m.h"

namespace n4m::core {

class Config {
  public:
    Config() noexcept;

    Config(const Config&) noexcept = default;
    Config& operator=(const Config&) noexcept = default;
    Config(Config&&) noexcept = default;
    Config& operator=(Config&&) noexcept = default;

    // Algorithm / solver / deflation
    n4m_algorithm_t algorithm{N4M_ALGO_PLS_REGRESSION};
    n4m_solver_t    solver{N4M_SOLVER_NIPALS};
    n4m_deflation_t deflation{N4M_DEFLATION_REGRESSION};

    // Geometry / numerics
    std::int32_t n_components{2};
    std::int32_t center_x{1};
    std::int32_t scale_x{1};
    std::int32_t center_y{1};
    std::int32_t scale_y{1};
    double       tol{1e-6};
    std::int32_t max_iter{500};
    std::int32_t store_scores{0};
    std::int32_t store_diagnostics{0};
    n4m_dtype_t  dtype{N4M_DTYPE_F64};

    // Sparse PLS soft-threshold lambda. 0.0 means no sparsity. The
    // interpretation depends on `sparse_simpls_legacy`:
    //   * default (legacy=0): Chun & Keles 2010 spls — `sparsity_lambda`
    //     is the relative `eta` in [0, 1) used by `ust(z, eta)` =
    //     sign(z) * max(|z| - eta * max(|z|), 0) inside the sparse
    //     direction vector, then a standard SIMPLS on the active set.
    //   * legacy=1: absolute soft-threshold of the SIMPLS direction at
    //     every component — `w[i] := sign(w[i]) * max(|w[i]| - lambda,
    //     0)`, renormalized.
    double sparsity_lambda{0.0};

    // Switch `fit_pls_sparse_simpls` to the pre-0.97.4 behaviour
    // (per-component absolute soft-threshold of the SIMPLS direction).
    // 0 (default) uses the Chun & Keles 2010 spls algorithm that matches
    // the R `spls` package.
    std::int32_t sparse_simpls_legacy{0};

    // RBF / polynomial kernel parameters consumed by the kernel-algorithm
    // PLS solver when kernel_type != N4M_KERNEL_LINEAR.
    std::int32_t kernel_type{0};      // 0=linear (default), 1=RBF, 2=polynomial
    double       kernel_gamma{0.0};   // RBF: kernel(x,y) = exp(-gamma * |x-y|^2); 0 means 1/n_features.
    double       kernel_coef0{1.0};   // polynomial: (gamma * x.y + coef0)^degree.
    std::int32_t kernel_degree{3};    // polynomial degree.

    // Domain-invariant PLS (di-PLS) regularization weight. Interpretation
    // depends on `di_pls_legacy`:
    //   * default (legacy=0): Nikzad-Langerodi 2018 algorithm matching
    //     `diPLSlib.models.DIPLS` — `di_lambda` is the per-component L2
    //     penalty on the convex relaxation of the source-target covariance
    //     difference, applied via `w = solve(I + (l/(y'y))*D, w_pls)`.
    //     Predictions subtract the target mean.
    //   * legacy=1: pre-0.97.4 pls4all behaviour — SIMPLS direction with a
    //     scalar `lambda/(1+lambda)` projection along (mu_source-mu_target).
    //     Predictions subtract the source mean.
    double di_lambda{0.0};

    // Switch `fit_di_pls` to the pre-0.97.4 SIMPLS-direction projection
    // path. 0 (default) uses the diPLSlib algorithm and target-mean rescale
    // so predictions match `diPLSlib.models.DIPLS(rescale='Target')`.
    std::int32_t di_pls_legacy{0};

    // Recursive PLS forgetting factor in (0, 1]; 1.0 = full memory.
    double recursive_forgetting{1.0};

    // Switch `fit_robust_pls` to the pre-0.97.4 Huber-IRLS algorithm
    // (NIPALS/SIMPLS on sqrt(w)-prescaled centered data with Huber weights on
    // Y-residuals and MAD-based scale). 0 (default) uses Partial Robust
    // M-regression (Serneels et al. 2005) matching R `chemometrics::prm`
    // bit-for-bit (median centering, leverage-and-residual Fair weights with
    // fairct=4, IRLS up to 30 outer iterations on a univariate SIMPLS kernel,
    // intercept = median(y - X@b)).
    std::int32_t robust_pls_legacy{0};

    // Switch `approximate_press` to the pre-0.97.4 Eastment-Krzanowski
    // leverage-inflated training-residual approximation
    // (PRESS_k ≈ Σ_i (y_i − ŷ_i^{(k)})^2 · (1−k/n)^{-2}). 0 (default) runs
    // true leave-one-out PRESS by refitting SIMPLS on the n−1 retained
    // samples for every fold, matching R `pls::plsr(validation='LOO',
    // method='simpls', scale=FALSE)$validation$PRESS` bit-for-bit.
    std::int32_t approximate_press_legacy{0};

    // Composability hooks (non-owning).
    const n4m_pipeline_t*        pipeline{nullptr};
    const n4m_operator_bank_t*   operator_bank{nullptr};
    const n4m_gating_strategy_t* gating_strategy{nullptr};
};

}  // namespace n4m::core

// Opaque alias for the C ABI.
struct n4m_config_s : public ::n4m::core::Config {};
