// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "n4m/component_analysis.h"
#include <algorithm>
#include <cmath>
#include <new>
#include <vector>
extern "C" N4M_API n4m_status_t n4m_component_normalize_loss_v1(
    const double* values, const double* baselines, int64_t n, double* output) {
    if (!values || !baselines || !output) return N4M_ERR_NULL_POINTER;
    if (n <= 0) return N4M_ERR_INVALID_ARGUMENT;
    if (n > 100000000) return N4M_ERR_UNSUPPORTED;
    try {
        std::vector<double> result(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            if (!std::isfinite(values[i]) || values[i] < 0 ||
                !std::isfinite(baselines[i]) || baselines[i] <= 0)
                return N4M_ERR_INVALID_ARGUMENT;
            result[static_cast<size_t>(i)] = values[i] / baselines[i];
            if (!std::isfinite(result[static_cast<size_t>(i)])) return N4M_ERR_INVALID_ARGUMENT;
        }
        std::copy(result.begin(), result.end(), output);
        return N4M_OK;
    } catch (const std::bad_alloc&) { return N4M_ERR_OUT_OF_MEMORY; }
      catch (...) { return N4M_ERR_INTERNAL; }
}
