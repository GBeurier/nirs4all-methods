/* SPDX-License-Identifier: CECILL-2.1 */
/* Compile-only guard: proves n4m/optimization.h is independently includable as
 * C (extern-C-clean, no C++ constructs leak). Contributes no runtime test. */

#include "n4m/optimization.h"

void n4m_include_check_optimization_c(void);
void n4m_include_check_optimization_c(void) {}
