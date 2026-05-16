// SPDX-License-Identifier: CeCILL-2.1
//
// VIP_SPA variable-selection primitive: VIP scoring + threshold mask
// followed by Successive Projections Algorithm on the surviving subset.
//
// Reference: Favilla et al. 2013 (VIP) + Araujo et al. 2001 (SPA).
// Algorithmic composition: auswahl.VIP_SPA (LSX-UniWue).

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace pls4all::core {

struct VipSpaSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t top_k{0};        // caller-requested upper bound
    std::int32_t n_selected{0};   // actual count (= len(selected_indices))
    std::int32_t n_candidates{0};
    double vip_threshold{0.0};

    std::vector<double> vip_scores;
    std::vector<unsigned char> vip_mask;
    std::vector<double> selection_scores;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] p4a_status_t select_by_vip_spa(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    double vip_threshold,
    std::int32_t top_k,
    VipSpaSelectionResult& out);

}  // namespace pls4all::core
