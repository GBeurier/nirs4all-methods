// SPDX-License-Identifier: CECILL-2.1
//
// Shared definition of a single preprocessing-operator entry, used by both
// the OperatorBank (AOM-PLS, Phase 6) and the Pipeline (Phase 3 leakage-safe
// CV). The struct itself is internal — callers go through the opaque
// n4m_operator_bank_t / n4m_pipeline_t handles.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

namespace n4m::core {

struct OperatorEntry {
    n4m_operator_kind_t kind{N4M_OP_IDENTITY};
    std::vector<double> params;  // owned copy of the caller's parameters

    OperatorEntry() = default;
    OperatorEntry(n4m_operator_kind_t k, const double* src, std::int32_t n)
        : kind(k), params(src && n > 0 ? std::vector<double>(src, src + n) : std::vector<double>()) {}
};

[[nodiscard]] inline bool operator_kind_is_valid(n4m_operator_kind_t k) noexcept {
    return k >= N4M_OP_IDENTITY && k <= N4M_OP_FCK;
}

}  // namespace n4m::core
