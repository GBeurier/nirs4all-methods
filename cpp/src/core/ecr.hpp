// SPDX-License-Identifier: CECILL-2.1
//
// Internal kernel for Elastic Component Regression (ECR), a continuous
// bridge between PCR (alpha=0) and PLS (alpha=1). Reference:
//   Liu Z., Cai W., Shao X. (2009/2010). "Elastic component regression
//   for spectral data analysis." Anal. Chim. Acta / Chemom. Intell. Lab.
//   Syst. Implemented in MATLAB by Hongdong Li (libPLS 1.95 `ecr.m`).
//
// For each component i:
//   H = (1 - alpha) X'X + alpha * X'y y'X
//   w = dominant eigenvector of H (power method)
//   t = X w
//   p = X'^t / (t'^t)
//   r = y' t / (t' t)
//   B = W (P' W)^{-1} R'
//   X <- X - t p',  y <- y - t r'
//
// Single-target (q == 1) is supported by the libPLS reference; the
// kernel implements the same convention.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

struct EcrResult {
    std::int64_t n_samples{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    double alpha{0.5};

    std::vector<double> predictions;       // n × q row-major (back-scaled)
    std::vector<double> coefficients;      // p × q row-major (original X scale)
    std::vector<double> x_mean;            // p
    std::vector<double> x_scale;           // p (1.0 when method=center)
    std::vector<double> y_mean;            // q
    std::vector<double> y_scale;           // q (1.0 when method=center)

    std::vector<double> weights_w;         // p × n_components row-major
    std::vector<double> loadings_p;        // p × n_components row-major
    std::vector<double> y_loadings;        // q × n_components row-major
    std::vector<double> wstar;             // p × n_components (X · wstar = T)
    std::vector<double> r2x;               // 1 × n_components (%)
    std::vector<double> r2y;               // 1 × n_components (%)

    double rmse{0.0};
};

[[nodiscard]] n4m_status_t fit_ecr(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    double alpha,
    EcrResult& out);

}  // namespace n4m::core
