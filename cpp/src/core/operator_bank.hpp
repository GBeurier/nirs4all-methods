// SPDX-License-Identifier: CECILL-2.1
//
// Internal OperatorBank — unordered list of operators used by AOM-PLS
// (Phase 6) for soft / hard / sparse gating.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"
#include "core/operator_entry.hpp"

namespace n4m::core {

class OperatorBank {
  public:
    OperatorBank() = default;

    OperatorBank(const OperatorBank&) = default;
    OperatorBank& operator=(const OperatorBank&) = default;
    OperatorBank(OperatorBank&&) noexcept = default;
    OperatorBank& operator=(OperatorBank&&) noexcept = default;

    void add(n4m_operator_kind_t kind, const double* params, std::int32_t n_params) {
        entries_.emplace_back(kind, params, n_params);
    }
    [[nodiscard]] std::int32_t size() const noexcept {
        return static_cast<std::int32_t>(entries_.size());
    }
    [[nodiscard]] const std::vector<OperatorEntry>& entries() const noexcept {
        return entries_;
    }

  private:
    std::vector<OperatorEntry> entries_;
};

}  // namespace n4m::core

struct n4m_operator_bank_s : public ::n4m::core::OperatorBank {};
