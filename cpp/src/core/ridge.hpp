// SPDX-License-Identifier: CECILL-2.1
//
// Internal kernel for direct (closed-form) multi-output Ridge regression,
// sklearn / glmnet-compatible.
//
//   beta = (Xc' Xc + lambda I_p)^-1 Xc' Yc   (column-centered X, Y)
//   intercept = y_mean - x_mean . beta        (penalty NOT applied to intercept)
//
// This is a genuine closed-form ridge — distinct from fit_ridge_pls
// (extra_pls.cpp), which is ridge-AUGMENTED SIMPLS rank-truncated by
// n_components. AUTO solver selection:
//   * p <= n  -> PRIMAL: QR on the augmented [Xc; sqrt(lambda) I_p] / [Yc; 0]
//                (textbook-stable, reuses the shipped Householder QR).
//   * p >  n  -> DUAL  : K = Xc Xc' (n x n), solve (K + lambda I_n) A = Yc,
//                then B = Xc' A  (O(n^3 + n^2 p) instead of O(p^3)).
// Both paths produce identical coefficients up to round-off.
//
// Standardization: center X and Y by column mean (always when the
// corresponding cfg center flag is set, and only when fit_intercept is on).
// Optional column scaling of X via cfg.scale_x; a zero-variance column gets
// scale 1.0 (sklearn convention) -> a 0 coefficient, not NaN. Coefficients and
// intercept are reported on the ORIGINAL (de-scaled) X scale.
// NOTE: cfg.scale_y is intentionally IGNORED (no-op) for direct ridge — Y is
// never scaled, matching sklearn.linear_model.Ridge (which scales neither the
// target nor, by default, X). Only x_scale is meaningful here.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

enum class RidgeSolver { kAuto = 0, kPrimal = 1, kDual = 2 };

struct RidgeResult {
    std::int64_t n_samples{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    double lambda{0.0};
    RidgeSolver solver_used{RidgeSolver::kAuto};

    std::vector<double> coefficients;   // p x q row-major (original X scale)
    std::vector<double> intercept;      // 1 x q
    std::vector<double> x_mean;         // 1 x p
    std::vector<double> x_scale;        // 1 x p (1.0 when scale_x off / zero-var)
    std::vector<double> y_mean;         // 1 x q
    std::vector<double> predictions;    // n x q row-major (original scale)

    double rmse{0.0};
};

// Fit direct ridge. `lambda` is the L2 penalty (sklearn `alpha`); must be
// finite and >= 0. `solver` chooses the path (kAuto picks primal/dual by
// shape). center/scale come from cfg: center_x/center_y (gated by
// fit_intercept) and scale_x; cfg.scale_y is ignored (Y is never scaled).
// `fit_intercept` controls whether an intercept term is computed (when
// false the intercept is zero and centering of X/Y is skipped).
[[nodiscard]] n4m_status_t fit_ridge(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    double lambda,
    RidgeSolver solver,
    bool fit_intercept,
    RidgeResult& out);

}  // namespace n4m::core
