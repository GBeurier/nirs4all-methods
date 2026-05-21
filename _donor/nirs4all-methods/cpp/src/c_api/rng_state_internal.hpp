// SPDX-License-Identifier: CECILL-2.1
//
// Internal-only definition of `n4m_rng_pcg64_state_t` shared between the
// translation units that wrap the PCG64 engine (`c_api_rng.cpp`) and the
// stochastic operators that consume a borrowed RNG handle (e.g. the Phase
// 18 augmenters). Keeps the engine pointer accessible to those TUs without
// re-publishing the engine type through the public ABI.
//
// This header is NEVER installed and never included by `n4m/n4m.h`.

#pragma once

#include "n4m/n4m.h"

#include "core/common/rng_pcg64.h"

// The public ABI forward-declares `n4m_rng_pcg64_state_t` in n4m.h. We pin
// its layout here so any internal consumer can navigate to the engine
// pointer via the documented `engine` field.
struct n4m_rng_pcg64_state_t {
    n4m_rng_pcg64 engine;
};
