// SPDX-License-Identifier: CECILL-2.1
#pragma once

#include <cstddef>
#include <vector>

#include "n4m/n4m.h"

namespace n4m::core {
// Fitted spectral representations. Parameters and learned arrays are owned here;
// bindings only marshal views. LVSE's basis includes channel scaling.
struct SpectralEncoding {
    int kind{0};  // 0: LVSE, 1: GCU
    int width{64}, rank{4}, iterations{60};
    double overlap{0.0}, tolerance{1e-3};
    bool standardize{true}, snv{false};
    std::size_t features{0}, outputs{0};
    std::vector<double> center, scale, basis;

    n4m_status_t fit(const std::vector<double>& x, std::size_t n, std::size_t p);
    n4m_status_t
    transform(const std::vector<double>& x, std::size_t n, std::vector<double>& out) const;
};
}  // namespace n4m::core
