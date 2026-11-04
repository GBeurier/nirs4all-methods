/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#include "n4m/fanova.h"
int n4m_fanova_c_header_check(void);
int n4m_fanova_c_header_check(void) {
    n4m_fanova_options_v1_t options;
    return (int)n4m_fanova_options_init_v1(&options);
}
