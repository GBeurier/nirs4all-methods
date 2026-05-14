// SPDX-License-Identifier: CeCILL-2.1
//
// Internal Pipeline — ordered list of operators with explicit fit/transform.
// The lifecycle is fully implemented at Phase 0; fit/transform are stubbed
// to P4A_ERR_NOT_IMPLEMENTED and land in Phase 3.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"
#include "core/operator_entry.hpp"

namespace pls4all::core {

class Pipeline {
  public:
    Pipeline() = default;

    Pipeline(const Pipeline&) = default;
    Pipeline& operator=(const Pipeline&) = default;
    Pipeline(Pipeline&&) noexcept = default;
    Pipeline& operator=(Pipeline&&) noexcept = default;

    void add_operator(p4a_operator_kind_t kind, const double* params, std::int32_t n_params) {
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

struct p4a_pipeline_s : public ::pls4all::core::Pipeline {};
