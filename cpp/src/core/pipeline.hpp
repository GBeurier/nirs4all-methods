// SPDX-License-Identifier: CeCILL-2.1
//
// Internal Pipeline — ordered list of operators with explicit fit/transform.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"
#include "core/context.hpp"
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
        fitted_ = false;
        states_.clear();
    }
    [[nodiscard]] std::int32_t size() const noexcept {
        return static_cast<std::int32_t>(entries_.size());
    }
    [[nodiscard]] const std::vector<OperatorEntry>& entries() const noexcept {
        return entries_;
    }
    [[nodiscard]] bool fitted() const noexcept {
        return fitted_;
    }

    [[nodiscard]] p4a_status_t fit(Context& ctx,
                                   const p4a_matrix_view_t& X,
                                   const p4a_matrix_view_t* Y);
    [[nodiscard]] p4a_status_t transform(Context& ctx,
                                         const p4a_matrix_view_t& X,
                                         p4a_matrix_view_t& out) const;

  private:
    struct OperatorState {
        p4a_operator_kind_t kind{P4A_OP_IDENTITY};
        std::int64_t n_features{0};
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> extra;
    };

    std::vector<OperatorEntry> entries_;
    std::vector<OperatorState> states_;
    std::int64_t n_features_{0};
    bool fitted_{false};
};

}  // namespace pls4all::core

struct p4a_pipeline_s : public ::pls4all::core::Pipeline {};
