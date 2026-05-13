// SPDX-License-Identifier: CeCILL-2.1
//
// pls4all_core placeholder translation unit. The real implementation lands in
// PR 4 (status / version / context / config) and subsequent PRs. This file
// exists only so PR 2 (cmake-skeleton) produces a linkable static library.
//
// It defines a single internal symbol that the linker keeps; pls4all_c stays
// otherwise empty until the C ABI wrappers are added.

namespace pls4all::internal {
namespace {

[[maybe_unused]] void phase0_core_placeholder() noexcept {
    // Intentionally empty — anonymous namespace, file-local.
}

}  // anonymous namespace
}  // namespace pls4all::internal
