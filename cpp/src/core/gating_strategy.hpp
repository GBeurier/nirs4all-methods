// SPDX-License-Identifier: CECILL-2.1
//
// Internal GatingStrategy — for AOM-PLS / POP-PLS (Phase 6). Phase 0 only
// stores the mode; gating coefficients are added in Phase 6.

#pragma once

#include "pls4all/p4a.h"

namespace n4m::core {

class GatingStrategy {
  public:
    explicit GatingStrategy(n4m_gating_mode_t mode = N4M_GATING_HARD) noexcept
        : mode_(mode) {}

    [[nodiscard]] n4m_gating_mode_t mode() const noexcept { return mode_; }

  private:
    n4m_gating_mode_t mode_;
};

[[nodiscard]] inline bool gating_mode_is_valid(n4m_gating_mode_t m) noexcept {
    return m >= N4M_GATING_HARD && m <= N4M_GATING_PER_TARGET;
}

}  // namespace n4m::core

struct n4m_gating_strategy_s : public ::n4m::core::GatingStrategy {
    using ::n4m::core::GatingStrategy::GatingStrategy;
};
