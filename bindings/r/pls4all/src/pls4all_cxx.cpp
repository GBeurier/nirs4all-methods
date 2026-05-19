// SPDX-License-Identifier: CECILL-2.1
//
// Minimal top-level C++ TU that exists solely to force R's build system
// into C++ link mode. R-exts (§1.6) warns that R only auto-selects the
// C++ linker when at least one C++ source lives directly under src/;
// our actual C++ implementation is under src/libp4a/, so without this
// sentinel some platforms would link with `cc` instead of `c++`,
// leaving libstdc++ symbols unresolved at runtime.
//
// The sentinel does not export anything visible to R.

namespace pls4all_cran {

// Reserved for future internal C++ utilities consumed by the R wrappers.
// Kept extern so the linker keeps the TU alive.
extern const char* pls4all_cran_marker;
const char* pls4all_cran_marker = "pls4all-cran";

} // namespace pls4all_cran
