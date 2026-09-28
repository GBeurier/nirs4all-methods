// SPDX-License-Identifier: CECILL-2.1
// Generate deterministic, valid N4MM seeds through the public C ABI.

#include "n4m/n4m.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>
#include <vector>

namespace {

bool write_seed(n4m_context_t* context,
                const std::filesystem::path& path,
                std::int32_t n_targets,
                const std::vector<double>& coefficients,
                const std::vector<double>& intercept) {
    n4m_linear_predictor_spec_t spec{};
    spec.source_training_samples = 17;
    spec.n_features = 2;
    spec.n_targets = n_targets;
    spec.coefficients = coefficients.data();
    spec.intercept = intercept.data();

    n4m_model_t* model = nullptr;
    if (n4m_model_import_linear_predictor(context, &spec, &model) != N4M_OK ||
        model == nullptr) {
        n4m_model_destroy(model);
        return false;
    }

    std::size_t size = 0;
    bool ok = n4m_model_export_size(model, &size) == N4M_OK && size > 0;
    std::vector<std::uint8_t> bytes(ok ? size : 0);
    std::size_t written = 0;
    if (ok) {
        ok = n4m_model_export_to_buffer(model, bytes.data(), bytes.size(),
                                        &written) == N4M_OK &&
             written == bytes.size();
    }
    n4m_serialized_model_info_v1_t info{};
    if (ok) {
        ok = n4m_serialization_inspect_model_v1(bytes.data(), bytes.size(),
                                                &info) == N4M_OK;
    }
    n4m_model_destroy(model);
    if (!ok) {
        return false;
    }

    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: n4m_fuzz_n4mm_seed OUTPUT_DIRECTORY\n";
        return 2;
    }
    const std::filesystem::path output_dir(argv[1]);
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    if (error) {
        std::cerr << "cannot create seed directory: " << error.message() << '\n';
        return 1;
    }

    n4m_context_t* context = nullptr;
    if (n4m_context_create(&context) != N4M_OK || context == nullptr) {
        return 1;
    }
    const bool ok =
        write_seed(context, output_dir / "valid-affine-1.n4mm", 1,
                   {2.0, -1.0}, {1.5}) &&
        write_seed(context, output_dir / "valid-affine-2.n4mm", 2,
                   {2.0, 0.5, -1.0, 3.0}, {1.5, -2.0});
    n4m_context_destroy(context);
    return ok ? 0 : 1;
}
