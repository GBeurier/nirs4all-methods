// SPDX-License-Identifier: CECILL-2.1
//
// PSO-PLS (§48) — Binary Particle Swarm Optimization for variable
// selection. Reference: Kennedy J. & Eberhart R. C. (1997)
// "A discrete binary version of the particle swarm algorithm",
// IEEE Conf. on Systems, Man, and Cybernetics, vol. 5: 4104-4108.
//
// Each particle is a binary mask over the p features; fitness is the
// CV-RMSE of a PLS regression on the selected subset. Velocity is
// continuous and updated by inertia + cognitive + social terms; the
// position is re-binarised via a sigmoid stochastic threshold.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct PsoSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_swarm{0};
    std::int32_t n_iterations{0};
    std::uint64_t seed{0};
    double w{0.0};
    double c1{0.0};
    double c2{0.0};
    double v_max{0.0};
    double best_rmse{0.0};

    std::vector<double> inclusion_frequencies;   // length p
    std::vector<double> best_rmse_by_iteration;  // length n_iterations
    std::vector<double> mean_rmse_by_iteration;  // length n_iterations
    std::vector<std::int64_t> selected_indices;  // gbest mask, ascending
};

[[nodiscard]] n4m_status_t select_by_pso(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t n_swarm,
    std::int32_t n_iterations,
    double w,
    double c1,
    double c2,
    double v_max,
    std::uint64_t seed,
    PsoSelectionResult& out);

}  // namespace n4m::core
