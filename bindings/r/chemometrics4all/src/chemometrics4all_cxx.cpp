// SPDX-License-Identifier: CECILL-2.1
//
// Top-level C++ sentinel for R's build system. The vendored libc4a core
// contains C++ translation units, so the package shared object must be linked
// with the C++ linker on every platform.
extern "C" int chemometrics4all_cxx_link_sentinel(void) {
    return 0;
}
