/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#include "n4m/explain.h"
_Static_assert(sizeof(n4m_explain_options_v1_t) == 72, "explain options layout");
int n4m_explain_c_header_check(void);
int n4m_explain_c_header_check(void) {
    n4m_explain_options_v1_t options;
    return (int)n4m_explain_options_init_v1(&options);
}
