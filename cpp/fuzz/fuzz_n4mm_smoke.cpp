// SPDX-License-Identifier: CECILL-2.1
//
// Portable finite smoke for the libFuzzer entry-point contract.

#include "n4m/n4m.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }
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

    for (int index = 1; index < argc; ++index) {
        std::ifstream input(argv[index], std::ios::binary);
        if (!input) {
            return 1;
        }
        const std::vector<std::uint8_t> bytes{
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        n4m_serialized_model_info_v1_t info{};
        if (bytes.empty() ||
            n4m_serialization_inspect_model_v1(bytes.data(), bytes.size(),
                                                &info) != N4M_OK) {
            return 1;
        }
        (void)LLVMFuzzerTestOneInput(bytes.data(), bytes.size());
    }
    return 0;
}
