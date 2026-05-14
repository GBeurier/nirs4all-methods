// SPDX-License-Identifier: CeCILL-2.1
//
// extern "C" wrappers for the live fitted-model implementation.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "pls4all/p4a.h"
#include "pls4all/p4a_version.h"

#include "core/context.hpp"
#include "core/model.hpp"

namespace {

inline ::pls4all::core::Context* as_core(p4a_context_t* ctx) noexcept {
    return static_cast<::pls4all::core::Context*>(ctx);
}

inline const ::pls4all::core::Model* as_core(const p4a_model_t* model) noexcept {
    return static_cast<const ::pls4all::core::Model*>(model);
}

inline const ::pls4all::core::Array* as_core(const p4a_array_t* arr) noexcept {
    return static_cast<const ::pls4all::core::Array*>(arr);
}

void set_error(p4a_context_t* ctx, const char* message) noexcept {
    if (ctx != nullptr) {
        as_core(ctx)->set_error(message);
    }
}

[[nodiscard]] bool add_checked(std::size_t& total, std::size_t add) noexcept {
    if (add > std::numeric_limits<std::size_t>::max() - total) {
        return false;
    }
    total += add;
    return true;
}

[[nodiscard]] bool add_vector_size(std::size_t& total, const std::vector<double>& values) noexcept {
    if (!add_checked(total, sizeof(std::uint64_t))) {
        return false;
    }
    if (values.size() > (std::numeric_limits<std::size_t>::max() - total) / sizeof(double)) {
        return false;
    }
    total += values.size() * sizeof(double);
    return true;
}

[[nodiscard]] bool checked_product(std::uint64_t a,
                                   std::uint64_t b,
                                   std::uint64_t& out) noexcept {
    if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

[[nodiscard]] bool output_element_count(std::int64_t rows,
                                        std::int64_t cols,
                                        std::size_t& out) noexcept {
    out = 0;
    if (rows < 0 || cols < 0) {
        return false;
    }
    const auto urows = static_cast<std::uint64_t>(rows);
    const auto ucols = static_cast<std::uint64_t>(cols);
    std::uint64_t product = 0;
    if (!checked_product(urows, ucols, product) ||
        product > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    out = static_cast<std::size_t>(product);
    return true;
}

[[nodiscard]] bool serialized_size(const ::pls4all::core::Model& model,
                                   std::size_t& out) noexcept {
    std::size_t total = 0;
    if (!add_checked(total, 4U)) return false;                         // magic
    if (!add_checked(total, 4U * sizeof(std::uint32_t))) return false;  // format + ABI triple
    if (!add_checked(total, 3U * sizeof(std::uint32_t))) return false;  // enums
    if (!add_checked(total, 4U * sizeof(std::uint64_t))) return false;  // dimensions
    if (!add_checked(total, 5U * sizeof(std::uint32_t))) return false;  // flags
    if (!add_checked(total, sizeof(double))) return false;              // tol
    if (!add_checked(total, sizeof(std::uint32_t))) return false;       // max_iter

    if (!add_vector_size(total, model.x_mean)) return false;
    if (!add_vector_size(total, model.x_scale)) return false;
    if (!add_vector_size(total, model.y_mean)) return false;
    if (!add_vector_size(total, model.y_scale)) return false;
    if (!add_vector_size(total, model.coefficients)) return false;
    if (!add_vector_size(total, model.weights_w)) return false;
    if (!add_vector_size(total, model.loadings_p)) return false;
    if (!add_vector_size(total, model.y_loadings_q)) return false;
    if (!add_vector_size(total, model.rotations_r)) return false;
    if (!add_vector_size(total, model.scores_t)) return false;
    if (!add_vector_size(total, model.y_scores_u)) return false;
    if (!add_checked(total, sizeof(std::uint64_t))) return false;       // checksum
    out = total;
    return true;
}

[[nodiscard]] std::uint64_t fnv1a64(const unsigned char* data, std::size_t size) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(data[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

struct Writer {
    unsigned char* data;
    std::size_t capacity;
    std::size_t pos{0};
    bool ok{true};

    void bytes(const void* src, std::size_t n) noexcept {
        if (!ok || n > capacity - pos) {
            ok = false;
            return;
        }
        std::memcpy(data + pos, src, n);
        pos += n;
    }

    void u32(std::uint32_t value) noexcept {
        unsigned char b[4] = {
            static_cast<unsigned char>(value & 0xffU),
            static_cast<unsigned char>((value >> 8U) & 0xffU),
            static_cast<unsigned char>((value >> 16U) & 0xffU),
            static_cast<unsigned char>((value >> 24U) & 0xffU),
        };
        bytes(b, sizeof(b));
    }

    void u64(std::uint64_t value) noexcept {
        unsigned char b[8] = {};
        for (std::size_t i = 0; i < sizeof(b); ++i) {
            b[i] = static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
        }
        bytes(b, sizeof(b));
    }

    void f64(double value) noexcept {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
};

[[nodiscard]] std::uint64_t decode_u64_le(const unsigned char* data) noexcept {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
        value |= static_cast<std::uint64_t>(data[i]) << (8U * i);
    }
    return value;
}

struct Reader {
    const unsigned char* data;
    std::size_t size;
    std::size_t pos{0};

    [[nodiscard]] bool bytes(void* dst, std::size_t n) noexcept {
        if (n > size - pos) {
            return false;
        }
        std::memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }

    [[nodiscard]] bool u32(std::uint32_t& out) noexcept {
        unsigned char b[4] = {};
        if (!bytes(b, sizeof(b))) {
            return false;
        }
        out = static_cast<std::uint32_t>(b[0]) |
              (static_cast<std::uint32_t>(b[1]) << 8U) |
              (static_cast<std::uint32_t>(b[2]) << 16U) |
              (static_cast<std::uint32_t>(b[3]) << 24U);
        return true;
    }

    [[nodiscard]] bool u64(std::uint64_t& out) noexcept {
        unsigned char b[8] = {};
        if (!bytes(b, sizeof(b))) {
            return false;
        }
        out = decode_u64_le(b);
        return true;
    }

    [[nodiscard]] bool f64(double& out) noexcept {
        std::uint64_t bits = 0;
        if (!u64(bits)) {
            return false;
        }
        std::memcpy(&out, &bits, sizeof(out));
        return true;
    }
};

void write_vector(Writer& w, const std::vector<double>& values) noexcept {
    w.u64(static_cast<std::uint64_t>(values.size()));
    for (double value : values) {
        w.f64(value);
    }
}

[[nodiscard]] bool read_vector(Reader& r,
                               std::uint64_t expected,
                               std::vector<double>& out) {
    std::uint64_t count = 0;
    if (!r.u64(count) || count != expected) {
        return false;
    }
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(count);
    out.clear();
    out.resize(n);
    std::fill(out.begin(), out.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (!r.f64(out[i])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool write_model_to_buffer(const ::pls4all::core::Model& model,
                                         void* buffer,
                                         std::size_t buffer_size,
                                         std::size_t& written) noexcept {
    auto* out = static_cast<unsigned char*>(buffer);
    Writer w{out, buffer_size};
    const char magic[4] = {'P', '4', 'A', 'M'};
    w.bytes(magic, sizeof(magic));
    w.u32(P4A_SERIALIZATION_FORMAT_VERSION);
    w.u32(P4A_ABI_VERSION_MAJOR);
    w.u32(P4A_ABI_VERSION_MINOR);
    w.u32(P4A_ABI_VERSION_PATCH);

    w.u32(static_cast<std::uint32_t>(model.algorithm));
    w.u32(static_cast<std::uint32_t>(model.solver));
    w.u32(static_cast<std::uint32_t>(model.deflation));
    w.u64(static_cast<std::uint64_t>(model.n_samples));
    w.u64(static_cast<std::uint64_t>(model.n_features));
    w.u64(static_cast<std::uint64_t>(model.n_targets));
    w.u64(static_cast<std::uint64_t>(model.n_components));
    w.u32(static_cast<std::uint32_t>(model.center_x));
    w.u32(static_cast<std::uint32_t>(model.scale_x));
    w.u32(static_cast<std::uint32_t>(model.center_y));
    w.u32(static_cast<std::uint32_t>(model.scale_y));
    w.u32(static_cast<std::uint32_t>(model.store_scores));
    w.f64(model.tol);
    w.u32(static_cast<std::uint32_t>(model.max_iter));

    write_vector(w, model.x_mean);
    write_vector(w, model.x_scale);
    write_vector(w, model.y_mean);
    write_vector(w, model.y_scale);
    write_vector(w, model.coefficients);
    write_vector(w, model.weights_w);
    write_vector(w, model.loadings_p);
    write_vector(w, model.y_loadings_q);
    write_vector(w, model.rotations_r);
    write_vector(w, model.scores_t);
    write_vector(w, model.y_scores_u);
    if (!w.ok || w.pos + sizeof(std::uint64_t) > buffer_size) {
        return false;
    }
    const std::uint64_t checksum = fnv1a64(out, w.pos);
    w.u64(checksum);
    if (!w.ok) {
        return false;
    }
    written = w.pos;
    return true;
}

[[nodiscard]] p4a_status_t inspect_header(const void* buffer,
                                          std::size_t buffer_size,
                                          std::uint32_t& format_version,
                                          std::uint32_t& abi_major,
                                          std::uint32_t& abi_minor,
                                          std::uint32_t& abi_patch) noexcept {
    format_version = 0;
    abi_major = 0;
    abi_minor = 0;
    abi_patch = 0;
    if (buffer == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    if (buffer_size < 20U) {
        return P4A_ERR_CORRUPT_BUFFER;
    }
    auto* data = static_cast<const unsigned char*>(buffer);
    if (data[0] != static_cast<unsigned char>('P') ||
        data[1] != static_cast<unsigned char>('4') ||
        data[2] != static_cast<unsigned char>('A') ||
        data[3] != static_cast<unsigned char>('M')) {
        return P4A_ERR_CORRUPT_BUFFER;
    }
    Reader r{data + 4U, buffer_size - 4U};
    return (r.u32(format_version) &&
            r.u32(abi_major) &&
            r.u32(abi_minor) &&
            r.u32(abi_patch)) ? P4A_OK : P4A_ERR_CORRUPT_BUFFER;
}

[[nodiscard]] bool valid_flag(std::uint32_t value) noexcept {
    return value == 0U || value == 1U;
}

[[nodiscard]] bool import_model_from_buffer(const void* buffer,
                                            std::size_t buffer_size,
                                            std::unique_ptr<p4a_model_s>& out) {
    out.reset();
    if (buffer_size < 28U) {
        return false;
    }
    auto* data = static_cast<const unsigned char*>(buffer);
    const std::size_t payload_size = buffer_size - sizeof(std::uint64_t);
    const std::uint64_t stored_checksum = decode_u64_le(data + payload_size);
    if (stored_checksum != fnv1a64(data, payload_size)) {
        return false;
    }

    std::uint32_t format = 0;
    std::uint32_t abi_major = 0;
    std::uint32_t abi_minor = 0;
    std::uint32_t abi_patch = 0;
    if (inspect_header(buffer, buffer_size, format, abi_major, abi_minor, abi_patch) != P4A_OK) {
        return false;
    }
    if (format != P4A_SERIALIZATION_FORMAT_VERSION) {
        return false;
    }

    Reader r{data + 20U, payload_size - 20U};
    std::uint32_t algorithm = 0;
    std::uint32_t solver = 0;
    std::uint32_t deflation = 0;
    std::uint64_t n_samples = 0;
    std::uint64_t n_features = 0;
    std::uint64_t n_targets = 0;
    std::uint64_t n_components = 0;
    std::uint32_t center_x = 0;
    std::uint32_t scale_x = 0;
    std::uint32_t center_y = 0;
    std::uint32_t scale_y = 0;
    std::uint32_t store_scores = 0;
    double tol = 0.0;
    std::uint32_t max_iter = 0;
    if (!r.u32(algorithm) || !r.u32(solver) || !r.u32(deflation) ||
        !r.u64(n_samples) || !r.u64(n_features) || !r.u64(n_targets) ||
        !r.u64(n_components) || !r.u32(center_x) || !r.u32(scale_x) ||
        !r.u32(center_y) || !r.u32(scale_y) || !r.u32(store_scores) ||
        !r.f64(tol) || !r.u32(max_iter)) {
        return false;
    }
    const bool supported_pls =
        algorithm == static_cast<std::uint32_t>(P4A_ALGO_PLS_REGRESSION) &&
        (solver == static_cast<std::uint32_t>(P4A_SOLVER_NIPALS) ||
         solver == static_cast<std::uint32_t>(P4A_SOLVER_SIMPLS) ||
         solver == static_cast<std::uint32_t>(P4A_SOLVER_KERNEL_ALGORITHM) ||
         solver == static_cast<std::uint32_t>(P4A_SOLVER_WIDE_KERNEL) ||
         solver == static_cast<std::uint32_t>(P4A_SOLVER_SVD));
    const bool supported_pcr =
        algorithm == static_cast<std::uint32_t>(P4A_ALGO_PCR) &&
        solver == static_cast<std::uint32_t>(P4A_SOLVER_SVD);
    if ((!supported_pls && !supported_pcr) ||
        deflation != static_cast<std::uint32_t>(P4A_DEFLATION_REGRESSION)) {
        return false;
    }
    if (n_samples == 0U || n_features == 0U || n_targets == 0U ||
        n_components == 0U ||
        n_samples > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        n_features > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        n_targets > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        n_components > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        !valid_flag(center_x) || !valid_flag(scale_x) ||
        !valid_flag(center_y) || !valid_flag(scale_y) ||
        !valid_flag(store_scores) || !(tol > 0.0) || max_iter == 0U) {
        return false;
    }

    std::uint64_t p_times_q = 0;
    std::uint64_t p_times_a = 0;
    std::uint64_t q_times_a = 0;
    std::uint64_t n_times_a = 0;
    if (!checked_product(n_features, n_targets, p_times_q) ||
        !checked_product(n_features, n_components, p_times_a) ||
        !checked_product(n_targets, n_components, q_times_a) ||
        !checked_product(n_samples, n_components, n_times_a)) {
        return false;
    }

    auto model = std::make_unique<p4a_model_s>();
    model->algorithm = static_cast<p4a_algorithm_t>(algorithm);
    model->solver = static_cast<p4a_solver_t>(solver);
    model->deflation = P4A_DEFLATION_REGRESSION;
    model->n_samples = static_cast<std::int64_t>(n_samples);
    model->n_features = static_cast<std::int32_t>(n_features);
    model->n_targets = static_cast<std::int32_t>(n_targets);
    model->n_components = static_cast<std::int32_t>(n_components);
    model->center_x = static_cast<std::int32_t>(center_x);
    model->scale_x = static_cast<std::int32_t>(scale_x);
    model->center_y = static_cast<std::int32_t>(center_y);
    model->scale_y = static_cast<std::int32_t>(scale_y);
    model->store_scores = static_cast<std::int32_t>(store_scores);
    model->tol = tol;
    model->max_iter = static_cast<std::int32_t>(max_iter);

    if (!read_vector(r, n_features, model->x_mean) ||
        !read_vector(r, n_features, model->x_scale) ||
        !read_vector(r, n_targets, model->y_mean) ||
        !read_vector(r, n_targets, model->y_scale) ||
        !read_vector(r, p_times_q, model->coefficients) ||
        !read_vector(r, p_times_a, model->weights_w) ||
        !read_vector(r, p_times_a, model->loadings_p) ||
        !read_vector(r, q_times_a, model->y_loadings_q) ||
        !read_vector(r, p_times_a, model->rotations_r)) {
        return false;
    }
    const std::uint64_t expected_scores = store_scores == 0U ? 0U : n_times_a;
    if (!read_vector(r, expected_scores, model->scores_t) ||
        !read_vector(r, expected_scores, model->y_scores_u) ||
        r.pos != r.size) {
        return false;
    }

    (void)abi_major;
    (void)abi_minor;
    (void)abi_patch;
    out = std::move(model);
    return true;
}

}  // namespace

extern "C" {

P4A_API p4a_status_t p4a_model_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_model_t** out_model) {
    if (out_model != nullptr) {
        *out_model = nullptr;
    }
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr || out_model == nullptr) {
        set_error(ctx, "null pointer in p4a_model_fit");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        std::unique_ptr<::pls4all::core::Model> fitted;
        p4a_status_t status = P4A_ERR_UNSUPPORTED;
        if (cfg->algorithm == P4A_ALGO_PCR) {
            status = ::pls4all::core::fit_pcr_svd(
                *as_core(ctx), *cfg, *X, *Y, fitted);
        } else if (cfg->solver == P4A_SOLVER_KERNEL_ALGORITHM ||
                   cfg->solver == P4A_SOLVER_WIDE_KERNEL) {
            status = ::pls4all::core::fit_pls_regression_kernel(
                *as_core(ctx), *cfg, *X, *Y, fitted);
        } else if (cfg->solver == P4A_SOLVER_SIMPLS) {
            status = ::pls4all::core::fit_pls_regression_simpls(
                *as_core(ctx), *cfg, *X, *Y, fitted);
        } else if (cfg->solver == P4A_SOLVER_SVD) {
            status = ::pls4all::core::fit_pls_regression_svd(
                *as_core(ctx), *cfg, *X, *Y, fitted);
        } else {
            status = ::pls4all::core::fit_pls_regression_nipals(
                *as_core(ctx), *cfg, *X, *Y, fitted);
        }
        if (status != P4A_OK) {
            return status;
        }
        auto raw = std::make_unique<p4a_model_s>();
        *static_cast<::pls4all::core::Model*>(raw.get()) = std::move(*fitted);
        *out_model = raw.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_model_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_model_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_model_destroy(p4a_model_t* model) {
    try {
        delete model;
    } catch (...) {
    }
}

P4A_API p4a_status_t p4a_model_predict(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_matrix_view_t* out) {
    if (ctx == nullptr || model == nullptr || X == nullptr || out == nullptr) {
        set_error(ctx, "null pointer in p4a_model_predict");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        return ::pls4all::core::predict_into(*as_core(ctx), *as_core(model), *X, *out);
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_model_predict");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_model_predict");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_transform(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_matrix_view_t* out_scores) {
    if (ctx == nullptr || model == nullptr || X == nullptr || out_scores == nullptr) {
        set_error(ctx, "null pointer in p4a_model_transform");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        return ::pls4all::core::transform_into(*as_core(ctx), *as_core(model), *X, *out_scores);
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_model_transform");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_model_transform");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_predict_alloc(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_array_t** out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (ctx == nullptr || model == nullptr || X == nullptr || out == nullptr) {
        set_error(ctx, "null pointer in p4a_model_predict_alloc");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        auto arr = std::make_unique<p4a_array_s>();
        arr->dtype = P4A_DTYPE_F64;
        arr->rows = X->rows;
        arr->cols = as_core(model)->n_targets;
        std::size_t n_values = 0;
        if (!output_element_count(arr->rows, arr->cols, n_values)) {
            set_error(ctx, "predict output shape is invalid or too large");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        arr->values.clear();
        arr->values.resize(n_values);
        std::fill(arr->values.begin(), arr->values.end(), 0.0);
        p4a_matrix_view_t view{};
        view.data = arr->values.data();
        view.rows = arr->rows;
        view.cols = arr->cols;
        view.row_stride = arr->cols > 0 ? arr->cols : 1;
        view.col_stride = 1;
        view.dtype = P4A_DTYPE_F64;
        const p4a_status_t status = ::pls4all::core::predict_into(
            *as_core(ctx), *as_core(model), *X, view);
        if (status != P4A_OK) {
            return status;
        }
        *out = arr.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_model_predict_alloc");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_model_predict_alloc");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_transform_alloc(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_array_t** out_scores) {
    if (out_scores != nullptr) {
        *out_scores = nullptr;
    }
    if (ctx == nullptr || model == nullptr || X == nullptr || out_scores == nullptr) {
        set_error(ctx, "null pointer in p4a_model_transform_alloc");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        auto arr = std::make_unique<p4a_array_s>();
        arr->dtype = P4A_DTYPE_F64;
        arr->rows = X->rows;
        arr->cols = as_core(model)->n_components;
        std::size_t n_values = 0;
        if (!output_element_count(arr->rows, arr->cols, n_values)) {
            set_error(ctx, "transform output shape is invalid or too large");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        arr->values.clear();
        arr->values.resize(n_values);
        std::fill(arr->values.begin(), arr->values.end(), 0.0);
        p4a_matrix_view_t view{};
        view.data = arr->values.data();
        view.rows = arr->rows;
        view.cols = arr->cols;
        view.row_stride = arr->cols > 0 ? arr->cols : 1;
        view.col_stride = 1;
        view.dtype = P4A_DTYPE_F64;
        const p4a_status_t status = ::pls4all::core::transform_into(
            *as_core(ctx), *as_core(model), *X, view);
        if (status != P4A_OK) {
            return status;
        }
        *out_scores = arr.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_model_transform_alloc");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_model_transform_alloc");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_get_array(
    p4a_context_t* ctx, const p4a_model_t* model,
    p4a_model_array_t which, p4a_array_t** out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (ctx == nullptr || model == nullptr || out == nullptr) {
        set_error(ctx, "null pointer in p4a_model_get_array");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        std::unique_ptr<::pls4all::core::Array> tmp;
        const p4a_status_t status = ::pls4all::core::model_array(
            *as_core(ctx), *as_core(model), which, tmp);
        if (status != P4A_OK) {
            return status;
        }
        auto arr = std::make_unique<p4a_array_s>();
        *static_cast<::pls4all::core::Array*>(arr.get()) = std::move(*tmp);
        *out = arr.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_model_get_array");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_model_get_array");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_get_n_components(const p4a_model_t* model,
                                                int32_t* out) {
    if (model == nullptr || out == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out = as_core(model)->n_components;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_get_n_features(const p4a_model_t* model,
                                              int32_t* out) {
    if (model == nullptr || out == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out = as_core(model)->n_features;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_get_n_targets(const p4a_model_t* model,
                                             int32_t* out) {
    if (model == nullptr || out == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out = as_core(model)->n_targets;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_export_size(const p4a_model_t* model,
                                           size_t* out_size) {
    if (model == nullptr || out_size == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        std::size_t size = 0;
        if (!serialized_size(*as_core(model), size)) {
            *out_size = 0;
            return P4A_ERR_INVALID_ARGUMENT;
        }
        *out_size = size;
        return P4A_OK;
    } catch (...) {
        *out_size = 0;
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_export_to_buffer(
    const p4a_model_t* model, void* buffer, size_t buffer_size,
    size_t* out_written) {
    if (out_written != nullptr) {
        *out_written = 0;
    }
    if (model == nullptr || buffer == nullptr || out_written == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        std::size_t required = 0;
        if (!serialized_size(*as_core(model), required)) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (buffer_size < required) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        std::size_t written = 0;
        if (!write_model_to_buffer(*as_core(model), buffer, buffer_size, written) ||
            written != required) {
            return P4A_ERR_INTERNAL;
        }
        *out_written = written;
        return P4A_OK;
    } catch (...) {
        if (out_written != nullptr) {
            *out_written = 0;
        }
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_model_import_from_buffer(
    p4a_context_t* ctx, const void* buffer, size_t buffer_size,
    p4a_model_t** out_model) {
    if (out_model != nullptr) {
        *out_model = nullptr;
    }
    if (ctx == nullptr || buffer == nullptr || out_model == nullptr) {
        set_error(ctx, "null pointer in p4a_model_import_from_buffer");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        std::uint32_t format = 0;
        std::uint32_t abi_major = 0;
        std::uint32_t abi_minor = 0;
        std::uint32_t abi_patch = 0;
        p4a_status_t status = inspect_header(
            buffer, buffer_size, format, abi_major, abi_minor, abi_patch);
        if (status != P4A_OK) {
            set_error(ctx, "corrupt model buffer header");
            return status;
        }
        if (format != P4A_SERIALIZATION_FORMAT_VERSION) {
            set_error(ctx, "model serialization format version is not supported");
            return P4A_ERR_VERSION_INCOMPATIBLE;
        }
        std::unique_ptr<p4a_model_s> imported;
        if (!import_model_from_buffer(buffer, buffer_size, imported)) {
            set_error(ctx, "corrupt model buffer payload");
            return P4A_ERR_CORRUPT_BUFFER;
        }
        if (abi_major != P4A_ABI_VERSION_MAJOR ||
            abi_minor != P4A_ABI_VERSION_MINOR ||
            abi_patch != P4A_ABI_VERSION_PATCH) {
            as_core(ctx)->set_errorf(
                "model writer ABI was %u.%u.%u; current ABI is %u.%u.%u",
                abi_major, abi_minor, abi_patch,
                static_cast<unsigned>(P4A_ABI_VERSION_MAJOR),
                static_cast<unsigned>(P4A_ABI_VERSION_MINOR),
                static_cast<unsigned>(P4A_ABI_VERSION_PATCH));
        } else {
            as_core(ctx)->clear_error();
        }
        *out_model = imported.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_model_import_from_buffer");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_model_import_from_buffer");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_serialization_inspect(
    const void* buffer, size_t buffer_size,
    uint32_t* out_format_version,
    uint32_t* out_writer_abi_major,
    uint32_t* out_writer_abi_minor,
    uint32_t* out_writer_abi_patch) {
    if (out_format_version != nullptr) *out_format_version = 0;
    if (out_writer_abi_major != nullptr) *out_writer_abi_major = 0;
    if (out_writer_abi_minor != nullptr) *out_writer_abi_minor = 0;
    if (out_writer_abi_patch != nullptr) *out_writer_abi_patch = 0;
    if (out_format_version == nullptr || out_writer_abi_major == nullptr ||
        out_writer_abi_minor == nullptr || out_writer_abi_patch == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    std::uint32_t format = 0;
    std::uint32_t abi_major = 0;
    std::uint32_t abi_minor = 0;
    std::uint32_t abi_patch = 0;
    const p4a_status_t status = inspect_header(
        buffer, buffer_size, format, abi_major, abi_minor, abi_patch);
    if (status != P4A_OK) {
        return status;
    }
    *out_format_version = format;
    *out_writer_abi_major = abi_major;
    *out_writer_abi_minor = abi_minor;
    *out_writer_abi_patch = abi_patch;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_array_dtype(const p4a_array_t* arr,
                                     p4a_dtype_t* out) {
    if (arr == nullptr || out == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out = as_core(arr)->dtype;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_array_shape(const p4a_array_t* arr,
                                     int64_t* rows, int64_t* cols) {
    if (arr == nullptr || rows == nullptr || cols == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *rows = as_core(arr)->rows;
        *cols = as_core(arr)->cols;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_array_view(const p4a_array_t* arr,
                                    p4a_matrix_view_t* out) {
    if (arr == nullptr || out == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        const auto* core = as_core(arr);
        out->data = core->values.empty()
            ? nullptr
            : const_cast<double*>(core->values.data());
        out->rows = core->rows;
        out->cols = core->cols;
        out->row_stride = core->cols > 0 ? core->cols : 1;
        out->col_stride = 1;
        out->dtype = core->dtype;
        out->reserved0 = 0;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_array_free(p4a_array_t* arr) {
    try {
        delete arr;
    } catch (...) {
    }
}

}  // extern "C"
