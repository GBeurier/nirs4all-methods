// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#pragma once

#include "core/estimator/spec.hpp"

namespace n4m::estimator {

// One source block: training-fitted sqrt(sum(X^2) / n), no centering.
std::unique_ptr<Adapter> make_tr_block_rms(const MethodSpec& spec);

}  // namespace n4m::estimator
