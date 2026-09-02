// SPDX-License-Identifier: CECILL-2.1
//
// Internal Pipeline — ordered list of operators with explicit fit/transform.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"
#include "core/common/context.hpp"
#include "core/operator_entry.hpp"

namespace n4m::core {

class Pipeline {
  public:
    Pipeline() = default;

    Pipeline(const Pipeline&) = default;
    Pipeline& operator=(const Pipeline&) = default;
    Pipeline(Pipeline&&) noexcept = default;
    Pipeline& operator=(Pipeline&&) noexcept = default;

    void add_operator(n4m_operator_kind_t kind, const double* params, std::int32_t n_params) {
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

    // Validate/canonicalize the only pipeline slice that a fitted Model may
    // currently own and serialize: SNV(no params) -> Savitzky-Golay smooth.
    [[nodiscard]] n4m_status_t model_snv_savgol_config(
        Context& ctx,
        std::int32_t& out_window,
        std::int32_t& out_poly_degree) const;

    // Restore the fitted state of that exact stateless slice from N4MM v2.
    // No training statistics are needed: SNV is row-local and SG coefficients
    // are determined entirely by the canonical window/poly configuration.
    [[nodiscard]] n4m_status_t restore_model_snv_savgol(
        Context& ctx,
        std::int64_t n_features,
        std::int32_t window,
        std::int32_t poly_degree);

    [[nodiscard]] n4m_status_t fit(Context& ctx,
                                   const n4m_matrix_view_t& X,
                                   const n4m_matrix_view_t* Y);
    [[nodiscard]] n4m_status_t transform(Context& ctx,
                                         const n4m_matrix_view_t& X,
                                         n4m_matrix_view_t& out) const;

  private:
    struct OperatorState {
        n4m_operator_kind_t kind{N4M_OP_IDENTITY};
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

}  // namespace n4m::core

struct n4m_pipeline_s : public ::n4m::core::Pipeline {};
