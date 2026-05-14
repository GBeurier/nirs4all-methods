// SPDX-License-Identifier: CeCILL-2.1
//
// Internal GatingStrategy — for AOM-PLS / POP-PLS (Phase 6). Phase 0 only
// stores the mode; gating coefficients are added in Phase 6.

#pragma once

#include "pls4all/p4a.h"

namespace pls4all::core {

class GatingStrategy {
  public:
    explicit GatingStrategy(p4a_gating_mode_t mode = P4A_GATING_HARD) noexcept
        : mode_(mode) {}

    [[nodiscard]] p4a_gating_mode_t mode() const noexcept { return mode_; }

  private:
    p4a_gating_mode_t mode_;
};

[[nodiscard]] inline bool gating_mode_is_valid(p4a_gating_mode_t m) noexcept {
    return m >= P4A_GATING_HARD && m <= P4A_GATING_PER_TARGET;
}

}  // namespace pls4all::core

struct p4a_gating_strategy_s : public ::pls4all::core::GatingStrategy {
    using ::pls4all::core::GatingStrategy::GatingStrategy;
};
