// SPDX-License-Identifier: CECILL-2.1
#pragma once
#include "n4m/n4m.h"
namespace n4m::core {
n4m_status_t compress_linear_stack(const n4m_matrix_view_t&,
                                   const n4m_matrix_view_t&,
                                   const n4m_matrix_view_t&,
                                   const n4m_matrix_view_t&,
                                   n4m_matrix_view_t&,
                                   n4m_matrix_view_t&);
}
