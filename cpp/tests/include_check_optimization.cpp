// SPDX-License-Identifier: CECILL-2.1
//
// Compile-only guard: proves n4m/optimization.h is independently includable as
// C++ (no prior include of n4m/n4m.h). Contributes no runtime test; if the
// header is not self-contained this TU fails to compile.

#include "n4m/optimization.h"

extern "C" void n4m_include_check_optimization_cpp(void);
extern "C" void n4m_include_check_optimization_cpp(void) {}
