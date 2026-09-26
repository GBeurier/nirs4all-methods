// SPDX-License-Identifier: CECILL-2.1
//
// AOM operator superblocks and the strict-chain Ridge-PLS selector.
//
// A superblock concatenates the outputs of a strict-linear operator bank,
// centers the columns and optionally scales each block to unit RMS, then fits
// a Ridge, PLS or Ridge-PLS head. Variants weight the blocks by kernel-target
// alignment (MKL) or keep an actively screened subset. The head grid is
// selected by train-fold CV (all preprocessing refitted per fold) and the
// final coefficients are folded back to the input space.
//
// The chain selector applies each strict-linear chain sequentially and
// selects chain x components x lambda of a Ridge-PLS head by train-fold CV.
//
// Both return an affine input-space predictor ("input_coefficients",
// "intercept") with the fit diagnostics of the n4m reference.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/method_result.hpp"
#include "core/operator_entry.hpp"

namespace n4m::core {

enum class SuperblockHead : std::int32_t { kRidge = 0, kPls = 1, kRidgePls = 2 };
enum class BlockSelection : std::int32_t { kAll = 0, kMkl = 1, kActive = 2 };
enum class ActiveScore : std::int32_t { kNorm = 0, kKta = 1, kBlend = 2 };

struct SuperblockOptions {
    SuperblockHead head{SuperblockHead::kRidge};
    BlockSelection selection{BlockSelection::kAll};
    bool rms_scaling{true};
    bool center_x{true};
    bool center_y{true};                   // Ridge and PLS heads
    std::vector<double> lambdas;           // Ridge alphas or Ridge-PLS lambdas
    std::vector<std::int32_t> components;  // PLS / Ridge-PLS component counts
    std::int32_t mkl_top_k{6};
    std::int32_t active_top_m{20};
    double active_diversity_threshold{0.98};
    ActiveScore active_score{ActiveScore::kNorm};
    std::int32_t active_max_per_family{0};  // 0: no cap
    bool keep_identity{true};
};

// `fold_ids` gives the CV fold (0 .. k-1) of every row of X.
[[nodiscard]] n4m_status_t fit_aom_superblock(Context& ctx,
                                              const std::vector<OperatorEntry>& operators,
                                              const n4m_matrix_view_t& X,
                                              const n4m_matrix_view_t& Y,
                                              const std::vector<std::int32_t>& fold_ids,
                                              const SuperblockOptions& options,
                                              MethodResult& out);

struct ChainRidgePlsOptions {
    std::vector<double> lambdas;
    std::vector<std::int32_t> components;
    bool center_x{true};
    bool center_y{true};
};

[[nodiscard]] n4m_status_t fit_aom_chain_ridge_pls(
    Context& ctx,
    const std::vector<std::vector<OperatorEntry>>& chains,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::vector<std::int32_t>& fold_ids,
    const ChainRidgePlsOptions& options,
    MethodResult& out);

}  // namespace n4m::core
