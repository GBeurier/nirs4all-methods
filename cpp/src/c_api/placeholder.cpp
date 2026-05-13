// SPDX-License-Identifier: CeCILL-2.1
//
// pls4all C ABI placeholder translation unit. PRs 4-7 replace it with the real
// extern "C" wrappers (c_api_version, c_api_context, c_api_config,
// c_api_matrix, c_api_error, c_api_operator_bank_stub, c_api_pipeline_stub,
// c_api_model_stub). For PR 2 we only need a TU that produces a linkable
// object so libp4a can build with zero P4A_API symbols exported.

namespace pls4all::internal {
namespace {

[[maybe_unused]] void phase0_c_api_placeholder() noexcept {
    // Intentionally empty — anonymous namespace, file-local.
}

}  // anonymous namespace
}  // namespace pls4all::internal
