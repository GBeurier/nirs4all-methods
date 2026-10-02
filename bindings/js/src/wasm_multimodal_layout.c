/* SPDX-License-Identifier: CECILL-2.1 */
/* Compile-time guards for the thin wasm32 descriptor offsets. */
#include <stddef.h>
#include "n4m/multimodal.h"
_Static_assert(sizeof(void*) == 4, "multimodal JS binding requires wasm32");
_Static_assert(sizeof(n4m_multimodal_source_spec_v1_t) == 96, "source spec layout");
_Static_assert(offsetof(n4m_multimodal_source_spec_v1_t, weight) == 40, "weight offset");
_Static_assert(offsetof(n4m_multimodal_source_spec_v1_t, n_components) == 48, "component offset");
_Static_assert(offsetof(n4m_multimodal_source_spec_v1_t, numeric_column) == 80, "numeric column offset");
_Static_assert(sizeof(n4m_multimodal_recipe_v1_t) == 40, "recipe layout");
_Static_assert(offsetof(n4m_multimodal_recipe_v1_t, alpha) == 16, "alpha offset");
_Static_assert(sizeof(n4m_multimodal_source_view_v1_t) == 56, "raw view layout");
_Static_assert(offsetof(n4m_multimodal_source_view_v1_t, numeric_data) == 36, "numeric pointer offset");
_Static_assert(offsetof(n4m_multimodal_source_view_v1_t, categorical_offsets) == 52, "categorical offsets pointer");
