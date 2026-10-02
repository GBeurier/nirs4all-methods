// SPDX-License-Identifier: CECILL-2.1
#include "core/estimator/multimodal_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

#include "core/common/context.hpp"
#include "n4m/n4m_version.h"

namespace n4m::multimodal {
namespace {
using Pipeline = n4m_multimodal_pipeline_s;
using Source = Pipeline::Source;
constexpr std::size_t kMaxBytes = 64u * 1024u * 1024u;
constexpr std::size_t kMaxIdentity = 1024u * 1024u;
constexpr std::int64_t kMaxWidth = 1024 * 1024;
constexpr std::int64_t kMaxElements = 16 * 1024 * 1024;
constexpr std::size_t kMaxCategories = 65536;
constexpr const char* kScaler = "preprocessing.scaling.standard_scale";
constexpr const char* kPca = "preprocessing.feature_selection.flexible_pca";
constexpr const char* kRidge = "models.regularized.ridge";

n4m_status_t error(n4m_context_t* ctx, const char* text,
                   n4m_status_t status = N4M_ERR_INVALID_ARGUMENT) {
    if (ctx != nullptr) ctx->set_error(text);
    return status;
}

bool utf8(const unsigned char* bytes, std::size_t size) {
    for (std::size_t i = 0; i < size;) {
        const unsigned char first = bytes[i++];
        if (first < 0x80) continue;
        std::uint32_t value = 0;
        int continuation = 0;
        std::uint32_t minimum = 0;
        if (first >= 0xc2 && first <= 0xdf) { value = first & 0x1f; continuation = 1; minimum = 0x80; }
        else if (first >= 0xe0 && first <= 0xef) { value = first & 0x0f; continuation = 2; minimum = 0x800; }
        else if (first >= 0xf0 && first <= 0xf4) { value = first & 7; continuation = 3; minimum = 0x10000; }
        else return false;
        if (static_cast<std::size_t>(continuation) > size - i) return false;
        for (int j = 0; j < continuation; ++j) {
            const unsigned char next = bytes[i++];
            if ((next & 0xc0) != 0x80) return false;
            value = (value << 6) | (next & 0x3f);
        }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
    }
    return true;
}

bool text(const char* value, std::string& out) {
    if (value == nullptr) return false;
    std::size_t n = 0;
    while (n <= 256 && value[n] != '\0') ++n;
    if (n == 0 || n > 256 || !utf8(reinterpret_cast<const unsigned char*>(value), n)) return false;
    out.assign(value, n);
    return true;
}

bool identity(const void* data, std::size_t size, std::string& out) {
    if (size == 0 || size > kMaxIdentity || data == nullptr ||
        !utf8(static_cast<const unsigned char*>(data), size)) return false;
    out.assign(static_cast<const char*>(data), size);
    return true;
}

std::int64_t width(const Source& source) {
    std::int64_t result = 1;
    for (auto dimension : source.shape) {
        if (dimension <= 0 || dimension > kMaxWidth / result) return 0;
        result *= dimension;
    }
    return result;
}

Source plan_copy(const Source& source) {
    Source out{};
    out.name = source.name; out.representation = source.representation;
    out.dtype = source.dtype; out.identity = source.identity; out.shape = source.shape;
    out.encoder = source.encoder; out.weight = source.weight;
    out.n_components = source.n_components; out.random_state = source.random_state;
    out.numeric_column = source.numeric_column; out.categorical_column = source.categorical_column;
    out.with_mean = source.with_mean; out.with_std = source.with_std;
    out.whiten = source.whiten; out.ignore_unknown = source.ignore_unknown;
    return out;
}

n4m_status_t params(n4m_context_t* ctx, const char* method,
                    std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)>& out) {
    std::int32_t index = -1;
    n4m_status_t status = n4m_method_find(method, &index);
    if (status != N4M_OK) return error(ctx, "required multimodal encoder is unavailable", status);
    n4m_params_t* raw = nullptr;
    status = n4m_params_create(ctx, index, &raw);
    out.reset(raw);
    return status;
}

n4m_status_t new_estimator(n4m_context_t* ctx, const Source* source, double alpha,
                          Pipeline::Estimator& out) {
    const char* method = source == nullptr ? kRidge :
        source->encoder == N4M_MULTIMODAL_TENSOR_PCA ? kPca : kScaler;
    std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)> values(nullptr, n4m_params_destroy);
    n4m_status_t status = params(ctx, method, values);
    if (status != N4M_OK) return status;
    if (source == nullptr) {
        status = n4m_params_set_double(values.get(), "alpha", alpha);
        if (status == N4M_OK) status = n4m_params_set_bool(values.get(), "center_x", 1);
        if (status == N4M_OK) status = n4m_params_set_bool(values.get(), "center_y", 1);
        if (status == N4M_OK) status = n4m_params_set_bool(values.get(), "scale_x", 0);
    } else if (source->encoder == N4M_MULTIMODAL_TENSOR_PCA) {
        status = n4m_params_set_double(values.get(), "n_components", static_cast<double>(source->n_components));
    } else {
        status = n4m_params_set_bool(values.get(), "with_mean", source->with_mean);
        if (status == N4M_OK) status = n4m_params_set_bool(values.get(), "with_std", source->with_std);
    }
    if (status != N4M_OK) return error(ctx, "invalid multimodal encoder parameters", status);
    n4m_estimator_t* raw = nullptr;
    status = n4m_estimator_create(ctx, method, values.get(), &raw);
    out.reset(raw);
    return status;
}

n4m_matrix_view_t matrix(std::vector<double>& values, std::int64_t rows, std::int64_t cols) {
    n4m_matrix_view_t out{};
    n4m_matrix_view_init_rowmajor(&out, values.data(), rows, cols, N4M_DTYPE_F64);
    return out;
}

n4m_status_t input(n4m_context_t* ctx, const Source& source,
                   const n4m_multimodal_source_view_v1_t& view,
                   std::vector<double>& numbers, std::vector<std::string>& categories,
                   std::int64_t& rows) {
    if (view.struct_size < sizeof(view) || view.rank != static_cast<std::int32_t>(source.shape.size() + 1) ||
        view.shape == nullptr || view.strides == nullptr || view.shape[0] < 0 ||
        view.name == nullptr || source.name != view.name || view.representation_id == nullptr ||
        source.representation != view.representation_id || view.dtype == nullptr || source.dtype != view.dtype ||
        view.identity_bytes != source.identity.size() || view.identity_utf8 == nullptr ||
        std::memcmp(view.identity_utf8, source.identity.data(), source.identity.size()) != 0) {
        return error(ctx, "multimodal source identity, dtype, shape or order mismatch", N4M_ERR_SHAPE_MISMATCH);
    }
    for (std::size_t d = 0; d < source.shape.size(); ++d) {
        if (view.shape[d + 1] != source.shape[d]) return error(ctx, "multimodal raw shape mismatch", N4M_ERR_SHAPE_MISMATCH);
    }
    rows = view.shape[0];
    const bool mixed = source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER;
    const std::int64_t cols = mixed ? 1 : width(source);
    if (cols <= 0 || rows > kMaxElements / cols ||
        (view.numeric_dtype != N4M_DTYPE_F64 && view.numeric_dtype != N4M_DTYPE_F32) ||
        (rows != 0 && view.numeric_data == nullptr)) return error(ctx, "invalid or oversized multimodal numeric tensor");
    if (!mixed && ((source.dtype == "float64" && view.numeric_dtype != N4M_DTYPE_F64) ||
                   (source.dtype == "float32" && view.numeric_dtype != N4M_DTYPE_F32))) {
        return error(ctx, "multimodal numeric dtype mismatch", N4M_ERR_SHAPE_MISMATCH);
    }
    std::int64_t max_offset = 0;
    for (std::int32_t d = 0; d < (mixed ? 1 : view.rank); ++d) {
        if (view.strides[d] < 0 || (view.shape[d] > 1 && view.strides[d] == 0) ||
            (view.shape[d] > 1 && view.strides[d] > (std::numeric_limits<std::int64_t>::max() - max_offset) / (view.shape[d] - 1))) {
            return error(ctx, "invalid multimodal tensor strides");
        }
        if (view.shape[d] > 0) max_offset += (view.shape[d] - 1) * view.strides[d];
    }
    const auto item_bytes = view.numeric_dtype == N4M_DTYPE_F64 ? sizeof(double) : sizeof(float);
    if (static_cast<std::uint64_t>(max_offset) >
        static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / item_bytes) {
        return error(ctx, "multimodal tensor byte strides overflow");
    }
    numbers.resize(static_cast<std::size_t>(rows * cols));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            std::int64_t offset = row * view.strides[0];
            std::int64_t remaining = col;
            if (!mixed) {
                for (std::int32_t d = view.rank - 1; d >= 1; --d) {
                    offset += (remaining % view.shape[d]) * view.strides[d];
                    remaining /= view.shape[d];
                }
            }
            const double value = view.numeric_dtype == N4M_DTYPE_F64
                ? static_cast<const double*>(view.numeric_data)[offset]
                : static_cast<double>(static_cast<const float*>(view.numeric_data)[offset]);
            if (!std::isfinite(value)) return error(ctx, "multimodal numeric source contains NaN or Inf");
            numbers[static_cast<std::size_t>(row * cols + col)] = value;
        }
    }
    categories.clear();
    if (mixed) {
        if (view.utf8_bytes > kMaxBytes || view.categorical_offsets == nullptr ||
            (view.utf8_bytes != 0 && view.categorical_utf8 == nullptr) || view.categorical_offsets[0] != 0 ||
            view.categorical_offsets[rows] != view.utf8_bytes) return error(ctx, "invalid categorical UTF-8 offsets");
        categories.reserve(static_cast<std::size_t>(rows));
        const auto* bytes = static_cast<const unsigned char*>(view.categorical_utf8);
        for (std::int64_t row = 0; row < rows; ++row) {
            const auto begin = view.categorical_offsets[row];
            const auto end = view.categorical_offsets[row + 1];
            if (end < begin || end > view.utf8_bytes || end - begin > kMaxIdentity ||
                !utf8(bytes == nullptr ? nullptr : bytes + begin, static_cast<std::size_t>(end - begin))) {
                return error(ctx, "invalid categorical UTF-8 cell");
            }
            categories.emplace_back(end == begin ? "" : reinterpret_cast<const char*>(bytes + begin),
                                    static_cast<std::size_t>(end - begin));
        }
    } else if (view.categorical_utf8 != nullptr || view.categorical_offsets != nullptr || view.utf8_bytes != 0) {
        return error(ctx, "numeric source unexpectedly contains categorical cells");
    }
    return N4M_OK;
}

n4m_status_t check_source_count(n4m_context_t* ctx, const Pipeline& pipeline, std::int32_t n,
                                const n4m_multimodal_source_view_v1_t* values) {
    if (n != static_cast<std::int32_t>(pipeline.sources.size()) || values == nullptr) {
        return error(ctx, "multimodal source count/order mismatch", N4M_ERR_SHAPE_MISMATCH);
    }
    return N4M_OK;
}

std::uint64_t checksum(const unsigned char* bytes, std::size_t size) {
    std::uint64_t out = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < size; ++i) { out ^= bytes[i]; out *= 0x100000001b3ULL; }
    return out;
}

struct Writer {
    std::vector<unsigned char>& bytes;
    void u64(std::uint64_t value) { for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<unsigned char>(value >> (8 * i))); }
    void u32(std::uint32_t value) { for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<unsigned char>(value >> (8 * i))); }
    void f64(double value) { std::uint64_t bits; std::memcpy(&bits, &value, 8); u64(bits); }
    void block(const void* data, std::size_t size) {
        u64(size);
        if (size != 0) {
            const auto* p = static_cast<const unsigned char*>(data);
            bytes.insert(bytes.end(), p, p + size);
        }
    }
    void string(const std::string& value) { block(value.data(), value.size()); }
};

struct Reader {
    const unsigned char* bytes; std::size_t size, pos = 0;
    bool integer(std::uint64_t& value, std::size_t count = 8) {
        if (count > size - pos) return false;
        value = 0;
        for (std::size_t i = 0; i < count; ++i) value |= static_cast<std::uint64_t>(bytes[pos++]) << (8 * i);
        return true;
    }
    bool number(double& value) { std::uint64_t bits; if (!integer(bits)) return false; std::memcpy(&value, &bits, 8); return std::isfinite(value); }
    bool block(const unsigned char*& data, std::size_t& length, std::size_t limit = kMaxBytes) {
        std::uint64_t n;
        if (!integer(n) || n > limit || n > size - pos) return false;
        length = static_cast<std::size_t>(n); data = bytes + pos; pos += length; return true;
    }
    bool string(std::string& out, std::size_t limit = kMaxIdentity) {
        const unsigned char* data; std::size_t length;
        if (!block(data, length, limit) || !utf8(data, length)) return false;
        out.assign(reinterpret_cast<const char*>(data), length); return true;
    }
};

void recipe_bytes(const Pipeline& pipeline, std::vector<unsigned char>& bytes) {
    Writer out{bytes};
    out.f64(pipeline.alpha); out.u32(static_cast<std::uint32_t>(pipeline.sources.size()));
    for (const auto& source : pipeline.sources) {
        out.string(source.name); out.string(source.representation); out.string(source.dtype); out.string(source.identity);
        out.u32(static_cast<std::uint32_t>(source.shape.size()));
        for (auto dimension : source.shape) out.u64(static_cast<std::uint64_t>(dimension));
        out.u32(source.encoder); out.f64(source.weight);
        out.u64(static_cast<std::uint64_t>(source.n_components)); out.u64(static_cast<std::uint64_t>(source.random_state));
        out.u64(static_cast<std::uint64_t>(source.numeric_column)); out.u64(static_cast<std::uint64_t>(source.categorical_column));
        out.u32(static_cast<std::uint32_t>(source.with_mean)); out.u32(static_cast<std::uint32_t>(source.with_std));
        out.u32(static_cast<std::uint32_t>(source.whiten)); out.u32(static_cast<std::uint32_t>(source.ignore_unknown));
    }
}

n4m_status_t state_bytes(n4m_context_t* ctx, const Pipeline::Estimator& state, std::vector<unsigned char>& out) {
    std::size_t size = 0, written = 0;
    n4m_status_t status = n4m_estimator_export_size(ctx, state.get(), 0, &size);
    if (status != N4M_OK) return status;
    if (size > kMaxBytes) return error(ctx, "multimodal nested state is oversized");
    out.resize(size);
    return n4m_estimator_export_to_buffer(ctx, state.get(), 0, out.data(), out.size(), &written);
}

bool integer_param(const n4m_params_t* params, const char* name, std::int64_t expected) {
    std::int64_t value = 0, count = 0;
    return n4m_params_get_int(params, name, &value, 1, &count) == N4M_OK && count == 1 && value == expected;
}
bool double_param(const n4m_params_t* params, const char* name, double expected) {
    double value = 0; std::int64_t count = 0;
    return n4m_params_get_double(params, name, &value, 1, &count) == N4M_OK && count == 1 && value == expected;
}

n4m_status_t validate_state(n4m_context_t* ctx, const Pipeline::Estimator& state,
                            const Source* source, double alpha, std::int64_t expected_width,
                            std::int64_t& output_width) {
    const char* method = source == nullptr ? kRidge : source->encoder == N4M_MULTIMODAL_TENSOR_PCA ? kPca : kScaler;
    std::int32_t actual = -1, expected = -1, retains = -1; std::uint64_t capabilities = 0;
    std::int64_t input_width = 0;
    n4m_status_t status = n4m_estimator_info(state.get(), &actual, &capabilities);
    if (status == N4M_OK) status = n4m_method_find(method, &expected);
    if (status == N4M_OK) status = n4m_estimator_n_features_in(state.get(), &input_width);
    if (status == N4M_OK) status = n4m_estimator_contains_training_rows(state.get(), &retains);
    if (status != N4M_OK) return status;
    if (actual != expected || input_width != expected_width || retains != 0) return error(ctx, "multimodal nested method/width/state mismatch", N4M_ERR_CORRUPT_BUFFER);
    n4m_params_t* raw = nullptr;
    status = n4m_estimator_get_params(ctx, state.get(), &raw);
    std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)> params_ptr(raw, n4m_params_destroy);
    if (status != N4M_OK) return status;
    bool valid = false;
    if (source == nullptr) {
        valid = double_param(raw, "alpha", alpha) && integer_param(raw, "center_x", 1) &&
            integer_param(raw, "center_y", 1) && integer_param(raw, "scale_x", 0);
        status = n4m_estimator_n_outputs(state.get(), &output_width);
        valid = valid && output_width == 1;
    } else {
        valid = source->encoder == N4M_MULTIMODAL_TENSOR_PCA
            ? double_param(raw, "n_components", static_cast<double>(source->n_components))
            : integer_param(raw, "with_mean", source->with_mean) && integer_param(raw, "with_std", source->with_std);
        status = n4m_estimator_transform_cols(state.get(), &output_width);
        valid = valid && output_width == (source->encoder == N4M_MULTIMODAL_TENSOR_PCA ? source->n_components : expected_width);
    }
    if (status != N4M_OK) return status;
    return valid ? N4M_OK : error(ctx, "multimodal nested parameters or output width mismatch", N4M_ERR_CORRUPT_BUFFER);
}
}  // namespace

n4m_status_t create(n4m_context_t* ctx, const n4m_multimodal_recipe_v1_t* recipe,
                    std::unique_ptr<Pipeline>& out) {
    if (ctx == nullptr || recipe == nullptr) return N4M_ERR_NULL_POINTER;
    if (recipe->struct_size < sizeof(*recipe) || recipe->n_sources < 1 || recipe->n_sources > 4 ||
        recipe->sources == nullptr || !std::isfinite(recipe->alpha) || recipe->alpha < 0 ||
        recipe->center_x != 1 || recipe->center_y != 1 || recipe->scale_x != 0) {
        return error(ctx, "multimodal profile requires early fusion and centered, unscaled Ridge");
    }
    auto pipeline = std::make_unique<Pipeline>(); pipeline->alpha = recipe->alpha;
    std::set<std::string> names;
    for (std::int32_t i = 0; i < recipe->n_sources; ++i) {
        const auto& spec = recipe->sources[i]; Source source{};
        if (spec.struct_size < sizeof(spec) || !text(spec.name, source.name) ||
            !text(spec.representation_id, source.representation) || !text(spec.dtype, source.dtype) ||
            !identity(spec.identity_utf8, spec.identity_bytes, source.identity) ||
            !names.insert(source.name).second || spec.ndim < 1 || spec.ndim > 7 || spec.shape == nullptr ||
            !std::isfinite(spec.weight) || spec.weight < 0) return error(ctx, "invalid multimodal source schema/recipe");
        source.shape.assign(spec.shape, spec.shape + spec.ndim);
        source.encoder = spec.encoder; source.weight = spec.weight;
        source.n_components = spec.n_components; source.random_state = spec.random_state;
        source.with_mean = spec.with_mean; source.with_std = spec.with_std;
        source.whiten = spec.whiten; source.ignore_unknown = spec.ignore_unknown;
        source.numeric_column = spec.numeric_column; source.categorical_column = spec.categorical_column;
        const char* required_names[] = {"nir", "image", "series", "metadata"};
        const char* required_representations[] = {"signal_1d", "rgb_image", "series_mv", "tabular_mixed"};
        const std::uint32_t required_encoders[] = {N4M_MULTIMODAL_STANDARD_SCALER,
            N4M_MULTIMODAL_TENSOR_PCA, N4M_MULTIMODAL_TENSOR_PCA, N4M_MULTIMODAL_COLUMN_TRANSFORMER};
        // Selection and fusion order are explicit. Encoder semantics belong to
        // the declared modality, rather than its position in the selected list.
        std::size_t modality = 0;
        while (modality < 4 && source.name != required_names[modality]) ++modality;
        if (modality == 4 || source.representation != required_representations[modality] ||
            source.encoder != required_encoders[modality]) return error(ctx, "multimodal profile source/encoder pattern mismatch");
        const auto cols = width(source);
        if (cols == 0) return error(ctx, "invalid or oversized multimodal source shape");
        if (source.encoder == N4M_MULTIMODAL_STANDARD_SCALER) {
            if (spec.ndim != 1 || spec.with_mean != 1 || spec.with_std != 1 || spec.whiten != 0 ||
                spec.n_components != 0 || spec.random_state != 0 || spec.ignore_unknown != 0 ||
                spec.numeric_column != -1 || spec.categorical_column != -1) return error(ctx, "unsupported StandardScaler recipe");
        } else if (source.encoder == N4M_MULTIMODAL_TENSOR_PCA) {
            if (spec.n_components <= 0 || spec.n_components > cols || spec.random_state < 0 ||
                static_cast<std::uint64_t>(spec.random_state) > std::numeric_limits<std::uint32_t>::max() ||
                spec.whiten != 0 || spec.with_mean != 0 || spec.with_std != 0 || spec.ignore_unknown != 0 ||
                spec.numeric_column != -1 || spec.categorical_column != -1) return error(ctx, "unsupported TensorPCA recipe");
        } else if (source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER) {
            if (spec.ndim != 1 || cols != 2 || spec.numeric_column != 0 || spec.categorical_column != 1 ||
                spec.with_mean != 1 || spec.with_std != 1 || spec.ignore_unknown != 1 || spec.whiten != 0 ||
                spec.n_components != 0 || spec.random_state != 0) return error(ctx, "unsupported mixed-column recipe");
        } else return error(ctx, "unsupported multimodal encoder kind", N4M_ERR_UNSUPPORTED);
        if (source.encoder != N4M_MULTIMODAL_COLUMN_TRANSFORMER && source.dtype != "float64" && source.dtype != "float32") {
            return error(ctx, "numeric multimodal source dtype must be float64 or float32");
        }
        pipeline->sources.push_back(std::move(source));
    }
    out = std::move(pipeline); return N4M_OK;
}

n4m_status_t features(n4m_context_t* ctx, const Pipeline& pipeline, std::int32_t n,
                      const n4m_multimodal_source_view_v1_t* views, std::vector<double>& out,
                      std::int64_t& rows, std::int64_t& cols) {
    if (!pipeline.fitted) return error(ctx, "multimodal pipeline is not fitted", N4M_ERR_NOT_FITTED);
    n4m_status_t status = check_source_count(ctx, pipeline, n, views);
    if (status != N4M_OK) return status;
    rows = -1; cols = pipeline.encoded_cols;
    std::vector<std::vector<double>> blocks;
    std::vector<std::int64_t> widths;
    for (std::size_t i = 0; i < pipeline.sources.size(); ++i) {
        const auto& source = pipeline.sources[i];
        std::vector<double> values; std::vector<std::string> cells; std::int64_t count = 0;
        status = input(ctx, source, views[i], values, cells, count);
        if (status != N4M_OK) return status;
        if (rows == -1) rows = count;
        if (rows != count) return error(ctx, "multimodal sources have unequal row counts", N4M_ERR_SHAPE_MISMATCH);
        std::int64_t encoded_width = 0;
        status = n4m_estimator_transform_cols(source.state.get(), &encoded_width);
        if (status != N4M_OK) return status;
        if (source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER) encoded_width += static_cast<std::int64_t>(source.categories.size());
        if (encoded_width <= 0 || rows > kMaxElements / encoded_width) return error(ctx, "multimodal encoded tensor is oversized");
        std::vector<double> encoded(static_cast<std::size_t>(rows * encoded_width), 0.0);
        std::vector<double> numeric(static_cast<std::size_t>(rows * (source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER ? 1 : encoded_width)));
        auto x = matrix(values, rows, source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER ? 1 : width(source));
        auto z = matrix(numeric, rows, source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER ? 1 : encoded_width);
        if (rows != 0) status = n4m_estimator_transform(ctx, source.state.get(), &x, &z);
        if (status != N4M_OK) return status;
        if (source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER) {
            for (std::int64_t row = 0; row < rows; ++row) {
                encoded[static_cast<std::size_t>(row * encoded_width)] = numeric[static_cast<std::size_t>(row)];
                const auto category = std::lower_bound(source.categories.begin(), source.categories.end(), cells[static_cast<std::size_t>(row)]);
                if (category != source.categories.end() && *category == cells[static_cast<std::size_t>(row)]) {
                    encoded[static_cast<std::size_t>(row * encoded_width + 1 + (category - source.categories.begin()))] = 1.0;
                }
            }
        } else encoded = std::move(numeric);
        for (double& value : encoded) {
            value *= source.weight;
            if (!std::isfinite(value)) return error(ctx, "multimodal weighted feature is not finite");
        }
        widths.push_back(encoded_width); blocks.push_back(std::move(encoded));
    }
    if (cols <= 0 || rows > kMaxElements / cols) return error(ctx, "multimodal fused tensor is oversized");
    out.resize(static_cast<std::size_t>(rows * cols));
    std::int64_t start = 0;
    for (std::size_t block = 0; block < blocks.size(); ++block) {
        for (std::int64_t row = 0; row < rows; ++row) {
            std::copy_n(blocks[block].data() + row * widths[block], static_cast<std::size_t>(widths[block]),
                        out.data() + row * cols + start);
        }
        start += widths[block];
    }
    return start == cols ? N4M_OK : error(ctx, "multimodal fused state width mismatch", N4M_ERR_CORRUPT_BUFFER);
}

n4m_status_t fit(n4m_context_t* ctx, Pipeline& pipeline, std::int32_t n,
                 const n4m_multimodal_source_view_v1_t* views, const n4m_matrix_view_t* y) {
    n4m_status_t status = check_source_count(ctx, pipeline, n, views);
    if (status != N4M_OK) return status;
    if (y == nullptr || y->cols != 1) return error(ctx, "multimodal profile requires one numeric target", N4M_ERR_SHAPE_MISMATCH);
    Pipeline next{}; next.alpha = pipeline.alpha;
    for (const auto& source : pipeline.sources) next.sources.push_back(plan_copy(source));
    std::int64_t rows = -1;
    for (std::size_t i = 0; i < next.sources.size(); ++i) {
        auto& source = next.sources[i];
        std::vector<double> values; std::vector<std::string> cells; std::int64_t count = 0;
        status = input(ctx, source, views[i], values, cells, count);
        if (status != N4M_OK) return status;
        if (rows == -1) rows = count;
        if (count <= 0 || count != rows || y->rows != rows) return error(ctx, "multimodal fit row counts mismatch", N4M_ERR_SHAPE_MISMATCH);
        if (source.encoder == N4M_MULTIMODAL_TENSOR_PCA && (rows < 2 || source.n_components > std::min(rows, width(source)))) {
            return error(ctx, "TensorPCA component count exceeds training rows/features");
        }
        if (source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER) {
            std::set<std::string> vocabulary(cells.begin(), cells.end());
            if (vocabulary.empty() || vocabulary.size() > kMaxCategories) return error(ctx, "categorical training vocabulary is oversized");
            source.categories.assign(vocabulary.begin(), vocabulary.end());
        }
        status = new_estimator(ctx, &source, next.alpha, source.state);
        if (status != N4M_OK) return status;
        auto x = matrix(values, rows, source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER ? 1 : width(source));
        n4m_fit_inputs_v1_t inputs{}; inputs.struct_size = sizeof(inputs); inputs.X = &x;
        status = n4m_estimator_fit(ctx, source.state.get(), &inputs);
        if (status != N4M_OK) return status;
        std::int64_t encoded = 0;
        status = n4m_estimator_transform_cols(source.state.get(), &encoded);
        if (status != N4M_OK) return status;
        next.encoded_cols += encoded + static_cast<std::int64_t>(source.categories.size());
        if (next.encoded_cols > kMaxWidth) return error(ctx, "multimodal fused width is oversized");
    }
    next.fitted = true;
    std::vector<double> fused; std::int64_t cols = 0;
    status = features(ctx, next, n, views, fused, rows, cols);
    if (status != N4M_OK) return status;
    status = new_estimator(ctx, nullptr, next.alpha, next.model);
    if (status != N4M_OK) return status;
    auto x = matrix(fused, rows, cols);
    n4m_fit_inputs_v1_t inputs{}; inputs.struct_size = sizeof(inputs); inputs.X = &x; inputs.Y = y;
    status = n4m_estimator_fit(ctx, next.model.get(), &inputs);
    if (status == N4M_OK) pipeline = std::move(next);
    return status;
}

n4m_status_t save(n4m_context_t* ctx, const Pipeline& pipeline, std::vector<unsigned char>& bytes) {
    if (!pipeline.fitted) return error(ctx, "multimodal pipeline is not fitted", N4M_ERR_NOT_FITTED);
    bytes = {'N', '4', 'M', 'F'}; Writer out{bytes};
    out.u32(1); out.u32(N4M_ABI_VERSION_MAJOR); out.u32(N4M_ABI_VERSION_MINOR); out.u32(N4M_ABI_VERSION_PATCH);
    std::vector<unsigned char> recipe; recipe_bytes(pipeline, recipe); out.block(recipe.data(), recipe.size());
    const auto limit = static_cast<std::size_t>(std::min<std::uint64_t>(ctx->max_state_bytes(), kMaxBytes));
    if (bytes.size() + 8 > limit) return error(ctx, "multimodal state exceeds byte limit");
    for (const auto& source : pipeline.sources) {
        std::vector<unsigned char> state; n4m_status_t status = state_bytes(ctx, source.state, state);
        if (status != N4M_OK) return status;
        std::size_t required = 16 + state.size();
        for (const auto& category : source.categories) required += 8 + category.size();
        if (required > limit - bytes.size() - 8) return error(ctx, "multimodal state exceeds byte limit");
        out.block(state.data(), state.size()); out.u64(source.categories.size());
        for (const auto& category : source.categories) out.string(category);
        if (bytes.size() + 8 > std::min<std::uint64_t>(ctx->max_state_bytes(), kMaxBytes)) return error(ctx, "multimodal state exceeds byte limit");
    }
    std::vector<unsigned char> model; n4m_status_t status = state_bytes(ctx, pipeline.model, model);
    if (status != N4M_OK) return status;
    if (model.size() + 8 > limit - bytes.size() - 8) return error(ctx, "multimodal state exceeds byte limit");
    out.block(model.data(), model.size());
    if (bytes.size() + 8 > std::min<std::uint64_t>(ctx->max_state_bytes(), kMaxBytes)) return error(ctx, "multimodal state exceeds byte limit");
    out.u64(checksum(bytes.data(), bytes.size())); return N4M_OK;
}

n4m_status_t load(n4m_context_t* ctx, const n4m_multimodal_recipe_v1_t* expected,
                  const void* buffer, std::size_t size, std::unique_ptr<Pipeline>& out) {
    if (ctx == nullptr || buffer == nullptr) return N4M_ERR_NULL_POINTER;
    if (size < 36 || size > std::min<std::uint64_t>(ctx->max_state_bytes(), kMaxBytes)) return error(ctx, "invalid multimodal state byte size", N4M_ERR_CORRUPT_BUFFER);
    const auto* bytes = static_cast<const unsigned char*>(buffer);
    Reader footer{bytes + size - 8, 8}; std::uint64_t stored;
    if (std::memcmp(bytes, "N4MF", 4) != 0 || !footer.integer(stored) || stored != checksum(bytes, size - 8)) {
        return error(ctx, "multimodal state magic/checksum mismatch", N4M_ERR_CORRUPT_BUFFER);
    }
    Reader in{bytes + 4, size - 12}; std::uint64_t format, major, minor, patch;
    if (!in.integer(format, 4) || !in.integer(major, 4) || !in.integer(minor, 4) || !in.integer(patch, 4) ||
        format != 1 || major != N4M_ABI_VERSION_MAJOR) return error(ctx, "unsupported multimodal state format/ABI", N4M_ERR_UNSUPPORTED);
    std::unique_ptr<Pipeline> pipeline;
    n4m_status_t status = create(ctx, expected, pipeline);
    if (status != N4M_OK) return status;
    std::vector<unsigned char> expected_bytes; recipe_bytes(*pipeline, expected_bytes);
    const unsigned char* recipe; std::size_t recipe_size;
    if (!in.block(recipe, recipe_size) || recipe_size != expected_bytes.size() ||
        std::memcmp(recipe, expected_bytes.data(), recipe_size) != 0) {
        return error(ctx, "multimodal state contradicts expected recipe/source schema", N4M_ERR_CORRUPT_BUFFER);
    }
    for (auto& source : pipeline->sources) {
        const unsigned char* state; std::size_t state_size; std::uint64_t n_categories;
        if (!in.block(state, state_size) || !in.integer(n_categories) || n_categories > kMaxCategories ||
            (source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER ? n_categories == 0 : n_categories != 0)) {
            return error(ctx, "invalid multimodal branch state", N4M_ERR_CORRUPT_BUFFER);
        }
        n4m_estimator_t* raw = nullptr;
        status = n4m_estimator_import_from_buffer(ctx, state, state_size, &raw); source.state.reset(raw);
        if (status != N4M_OK) return status;
        std::int64_t encoded = 0;
        status = validate_state(ctx, source.state, &source, pipeline->alpha,
                                source.encoder == N4M_MULTIMODAL_COLUMN_TRANSFORMER ? 1 : width(source), encoded);
        if (status != N4M_OK) return status;
        for (std::uint64_t i = 0; i < n_categories; ++i) {
            std::string category;
            if (!in.string(category) || (!source.categories.empty() && source.categories.back() >= category)) {
                return error(ctx, "invalid or duplicate categorical state", N4M_ERR_CORRUPT_BUFFER);
            }
            source.categories.push_back(std::move(category));
        }
        pipeline->encoded_cols += encoded + static_cast<std::int64_t>(n_categories);
        if (pipeline->encoded_cols > kMaxWidth) return error(ctx, "multimodal state width is oversized", N4M_ERR_CORRUPT_BUFFER);
    }
    const unsigned char* state; std::size_t state_size;
    if (!in.block(state, state_size) || in.pos != in.size) return error(ctx, "truncated/trailing multimodal state", N4M_ERR_CORRUPT_BUFFER);
    n4m_estimator_t* raw = nullptr;
    status = n4m_estimator_import_from_buffer(ctx, state, state_size, &raw); pipeline->model.reset(raw);
    if (status != N4M_OK) return status;
    std::int64_t outputs = 0;
    status = validate_state(ctx, pipeline->model, nullptr, pipeline->alpha, pipeline->encoded_cols, outputs);
    if (status != N4M_OK) return status;
    pipeline->fitted = true; out = std::move(pipeline); return N4M_OK;
}
}  // namespace n4m::multimodal
