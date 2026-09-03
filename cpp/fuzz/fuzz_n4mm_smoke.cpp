// SPDX-License-Identifier: CECILL-2.1
//
// Portable finite smoke for the libFuzzer entry-point contract.

#include <array>
#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

int main() {
    const std::array<std::uint8_t, 1> empty_storage{};
    (void)LLVMFuzzerTestOneInput(empty_storage.data(), 0U);

    const std::array<std::uint8_t, 4> magic_only{'N', '4', 'M', 'M'};
    (void)LLVMFuzzerTestOneInput(magic_only.data(), magic_only.size());

    std::array<std::uint8_t, 28> corrupt_payload{};
    corrupt_payload[0] = 'N';
    corrupt_payload[1] = '4';
    corrupt_payload[2] = 'M';
    corrupt_payload[3] = 'M';
    (void)LLVMFuzzerTestOneInput(corrupt_payload.data(), corrupt_payload.size());
    return 0;
}
