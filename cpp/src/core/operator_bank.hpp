// SPDX-License-Identifier: CeCILL-2.1
//
// Internal OperatorBank — unordered list of operators used by AOM-PLS
// (Phase 6) for soft / hard / sparse gating.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"
#include "core/operator_entry.hpp"

namespace pls4all::core {

class OperatorBank {
  public:
    OperatorBank() = default;

    OperatorBank(const OperatorBank&) = default;
    OperatorBank& operator=(const OperatorBank&) = default;
    OperatorBank(OperatorBank&&) noexcept = default;
    OperatorBank& operator=(OperatorBank&&) noexcept = default;

    void add(p4a_operator_kind_t kind, const double* params, std::int32_t n_params) {
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

}  // namespace pls4all::core

struct p4a_operator_bank_s : public ::pls4all::core::OperatorBank {};
