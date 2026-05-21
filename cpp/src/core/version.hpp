/* SPDX-License-Identifier: CECILL-2.1 */
/* Compat shim — forwards to the unified common-core header at
 * cpp/src/core/common/version.hpp post the M3 + M2.5 unification.
 * This file exists so existing pls4all sources that #include "core/version.hpp"
 * keep working without churn. Delete after the M5+M6 header reorganisation. */
#pragma once
#include "core/common/version.hpp"
