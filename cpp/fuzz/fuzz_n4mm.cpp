// SPDX-License-Identifier: CECILL-2.1
//
// Bounded hostile-input driver for the authoritative public N4MM ABI.

#include "n4m/n4m.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr std::size_t kMaxInputBytes = 1024U * 1024U;

[[noreturn]] void contract_violation() {
    std::abort();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size > kMaxInputBytes) {
        return 0;
    }

    std::uint32_t format = 0;
    std::uint32_t abi_major = 0;
    std::uint32_t abi_minor = 0;
    std::uint32_t abi_patch = 0;
    (void)n4m_serialization_inspect(
        data, size, &format, &abi_major, &abi_minor, &abi_patch);

    n4m_serialized_model_info_v1_t model_info{};
    const n4m_status_t model_status =
        n4m_serialization_inspect_model_v1(data, size, &model_info);

    n4m_serialized_pipeline_info_v1_t pipeline_info{};
    (void)n4m_serialization_inspect_pipeline_v1(
        data, size, &pipeline_info, sizeof(pipeline_info));

    // The allocation-bearing importer only receives bytes already accepted by
    // the authoritative complete-payload inspector. The harness does not copy,
    // decode, repair checksums, or otherwise reimplement the N4MM parser.
    if (model_status != N4M_OK) {
        return 0;
    }

    n4m_context_t* context = nullptr;
    if (n4m_context_create(&context) != N4M_OK || context == nullptr) {
        return 0;
    }
    n4m_model_t* model = nullptr;
    const n4m_status_t import_status =
        n4m_model_import_from_buffer(context, data, size, &model);
    if ((import_status == N4M_OK) != (model != nullptr)) {
        n4m_model_destroy(model);
        n4m_context_destroy(context);
        contract_violation();
    }
    n4m_model_destroy(model);
    n4m_context_destroy(context);
    return 0;
}
