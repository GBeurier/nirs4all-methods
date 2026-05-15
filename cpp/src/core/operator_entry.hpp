// SPDX-License-Identifier: CeCILL-2.1
//
// Shared definition of a single preprocessing-operator entry, used by both
// the OperatorBank (AOM-PLS, Phase 6) and the Pipeline (Phase 3 leakage-safe
// CV). The struct itself is internal — callers go through the opaque
// p4a_operator_bank_t / p4a_pipeline_t handles.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

namespace pls4all::core {

struct OperatorEntry {
    p4a_operator_kind_t kind{P4A_OP_IDENTITY};
    std::vector<double> params;  // owned copy of the caller's parameters

    OperatorEntry() = default;
    OperatorEntry(p4a_operator_kind_t k, const double* src, std::int32_t n)
        : kind(k), params(src && n > 0 ? std::vector<double>(src, src + n) : std::vector<double>()) {}
};

[[nodiscard]] inline bool operator_kind_is_valid(p4a_operator_kind_t k) noexcept {
    return k >= P4A_OP_IDENTITY && k <= P4A_OP_FINITE_DIFFERENCE;
}

}  // namespace pls4all::core
