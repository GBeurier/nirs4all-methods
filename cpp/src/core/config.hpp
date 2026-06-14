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

    // RNG engine for stochastic methods (variable selectors, ensemble PLS,
    // randomized SVD). 0 = N4M_RNG_SPLITMIX64 (default; reproduces n4m's
    // historical streams bit-for-bit). Other values select an external
    // library's exact RNG for reference parity — see n4m_rng_kind_t.
    std::int32_t rng_kind{0};

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

    // Direct (closed-form) Ridge regression knobs (n4m_estimators_ridge_fit). Penalty
    // `ridge_lambda` is the sklearn `alpha`; used when no per-call lambda is
    // supplied. `ridge_fit_intercept` (default true) toggles intercept fitting
    // (and centering): the penalty is never applied to the intercept.
    // Data-only — no public setter, to keep the new ABI surface to the single
    // n4m_estimators_ridge_fit symbol; the lambda is threaded through the fit call.
    double       ridge_lambda{1.0};
    std::int32_t ridge_fit_intercept{1};

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

    // AOM sweep route policy. 0 = auto exact operator-moment routes when
    // guarded; 1 = force legacy materialized-chain scoring; 2 = reject
    // candidate screens that would require materialized fallbacks.
    std::int32_t aom_moment_policy{N4M_AOM_MOMENT_AUTO};

    // AOM PLS candidate scoring policy. 0 = exact CV RMSE; 1 = score-only
    // moment PLS1 GCV proxy, intended as a fast first-pass screen.
    std::int32_t aom_pls_score_mode{N4M_AOM_PLS_SCORE_CV};

    // AOM sweep output policy. 0 = return selected OOF/final predictions and
    // coefficients; 1 = return score/rank outputs only.
    std::int32_t aom_score_only{0};

    // CUDA PLS1 moment CV scheduling policy. 0 = process independent fold jobs
    // through one reusable stream/workspace; 1 = allow bounded stream-parallel
    // fold batches on the selected single CUDA device.
    std::int32_t cuda_pls_parallel_folds{0};

    // CUDA PLS1 moment device-route threshold in feature count. The default
    // keeps the historical conservative guard; lower values are for explicit
    // crossover campaigns without recompiling.
    std::int32_t cuda_pls_min_device_features{1024};

    // CUDA PLS1 moment many-design batching policy. 0 (default) processes
    // independent fold/moment jobs through the reusable sequential workspace;
    // 1 selects the experimental tiled/strided-batched path that fuses the
    // per-component p^2 work with cuBLAS strided-batched calls while keeping
    // fold-level exact CV scores. Off by default; the N4M_CUDA_PLS_MANY_BATCHED
    // env var remains an equivalent opt-in fallback.
    std::int32_t cuda_pls_many_batched{0};

    // Composability hooks (non-owning).
    const n4m_pipeline_t*        pipeline{nullptr};
    const n4m_operator_bank_t*   operator_bank{nullptr};
    const n4m_gating_strategy_t* gating_strategy{nullptr};
};

}  // namespace n4m::core

// Opaque alias for the C ABI.
struct n4m_config_s : public ::n4m::core::Config {};
