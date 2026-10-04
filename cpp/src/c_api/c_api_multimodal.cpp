// SPDX-License-Identifier: CECILL-2.1
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <new>
#include <vector>

#include "core/common/context.hpp"
#include "core/estimator/multimodal_pipeline.hpp"
#include "n4m/multimodal.h"

namespace {
template <typename Fn>
n4m_status_t guarded(n4m_context_t* ctx, Fn&& function) noexcept {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    try { ctx->clear_error(); return function(); }
    catch (const std::bad_alloc&) { ctx->set_error("out of memory"); return N4M_ERR_OUT_OF_MEMORY; }
    catch (const std::exception& e) { ctx->set_error(e.what()); return N4M_ERR_INTERNAL; }
    catch (...) { ctx->set_error("internal error"); return N4M_ERR_INTERNAL; }
}

n4m_status_t output(n4m_context_t* ctx, n4m_matrix_view_t* view, std::int64_t rows, std::int64_t cols) {
    if (view == nullptr) return N4M_ERR_NULL_POINTER;
    if (view->dtype != N4M_DTYPE_F64 || view->rows != rows || view->cols != cols ||
        n4m_matrix_view_validate(view) != N4M_OK) {
        ctx->set_error("multimodal output must have the requested float64 shape/strides");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rows > 0 && cols > 0) {
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
        const auto row = static_cast<std::uint64_t>(rows - 1), col = static_cast<std::uint64_t>(cols - 1);
        const auto rs = static_cast<std::uint64_t>(view->row_stride), cs = static_cast<std::uint64_t>(view->col_stride);
        if ((row && rs > limit / row) || (col && cs > (limit - row * rs) / col)) {
            ctx->set_error("multimodal output byte strides overflow"); return N4M_ERR_STRIDE_INVALID;
        }
    }
    return N4M_OK;
}
}

N4M_API n4m_status_t n4m_multimodal_pipeline_create(n4m_context_t* ctx,
    const n4m_multimodal_recipe_v1_t* recipe, n4m_multimodal_pipeline_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    return guarded(ctx, [&]() {
        std::unique_ptr<n4m_multimodal_pipeline_s> pipeline;
        const auto status = n4m::multimodal::create(ctx, recipe, pipeline);
        if (status == N4M_OK) *out = pipeline.release();
        return status;
    });
}

N4M_API void n4m_multimodal_pipeline_destroy(n4m_multimodal_pipeline_t* pipeline) { delete pipeline; }

N4M_API n4m_status_t n4m_multimodal_pipeline_fit(n4m_context_t* ctx,
    n4m_multimodal_pipeline_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, const n4m_matrix_view_t* y) {
    return guarded(ctx, [&]() {
        return pipeline == nullptr ? N4M_ERR_NULL_POINTER : n4m::multimodal::fit(ctx, *pipeline, count, sources, y);
    });
}

N4M_API n4m_status_t n4m_multimodal_pipeline_predict(n4m_context_t* ctx,
    const n4m_multimodal_pipeline_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, n4m_matrix_view_t* out) {
    return guarded(ctx, [&]() {
        if (pipeline == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<double> features; std::int64_t rows = 0, cols = 0;
        auto status = n4m::multimodal::features(ctx, *pipeline, count, sources, features, rows, cols);
        if (status != N4M_OK) return status;
        status = output(ctx, out, rows, 1);
        if (status != N4M_OK || rows == 0) return status;
        n4m_matrix_view_t x{};
        n4m_matrix_view_init_rowmajor(&x, features.data(), rows, cols, N4M_DTYPE_F64);
        return n4m_estimator_predict(ctx, pipeline->model.get(), &x, out);
    });
}

N4M_API n4m_status_t n4m_multimodal_pipeline_transform_cols(
    const n4m_multimodal_pipeline_t* pipeline, int64_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = 0;
    if (!pipeline->fitted) return N4M_ERR_NOT_FITTED;
    *out = pipeline->encoded_cols; return N4M_OK;
}

N4M_API n4m_status_t n4m_multimodal_pipeline_transform(n4m_context_t* ctx,
    const n4m_multimodal_pipeline_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, n4m_matrix_view_t* out) {
    return guarded(ctx, [&]() {
        if (pipeline == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<double> features; std::int64_t rows = 0, cols = 0;
        auto status = n4m::multimodal::features(ctx, *pipeline, count, sources, features, rows, cols);
        if (status != N4M_OK) return status;
        status = output(ctx, out, rows, cols);
        if (status != N4M_OK) return status;
        auto* values = static_cast<double*>(out->data);
        for (std::int64_t row = 0; row < rows; ++row) {
            for (std::int64_t col = 0; col < cols; ++col) {
                values[row * out->row_stride + col * out->col_stride] = features[static_cast<std::size_t>(row * cols + col)];
            }
        }
        return N4M_OK;
    });
}

N4M_API n4m_status_t n4m_multimodal_pipeline_export_size(n4m_context_t* ctx,
    const n4m_multimodal_pipeline_t* pipeline, size_t* out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = 0;
    return guarded(ctx, [&]() {
        if (pipeline == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<unsigned char> bytes;
        const auto status = n4m::multimodal::save(ctx, *pipeline, bytes);
        if (status == N4M_OK) *out = bytes.size();
        return status;
    });
}

N4M_API n4m_status_t n4m_multimodal_pipeline_export_to_buffer(n4m_context_t* ctx,
    const n4m_multimodal_pipeline_t* pipeline, void* buffer, size_t size, size_t* written) {
    if (written == nullptr) return N4M_ERR_NULL_POINTER;
    *written = 0;
    return guarded(ctx, [&]() {
        if (pipeline == nullptr || buffer == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<unsigned char> bytes;
        const auto status = n4m::multimodal::save(ctx, *pipeline, bytes);
        if (status != N4M_OK) return status;
        if (size < bytes.size()) { ctx->set_error("multimodal state output buffer is too small"); return N4M_ERR_INVALID_ARGUMENT; }
        std::memcpy(buffer, bytes.data(), bytes.size()); *written = bytes.size(); return N4M_OK;
    });
}

N4M_API n4m_status_t n4m_multimodal_pipeline_import_from_buffer(n4m_context_t* ctx,
    const n4m_multimodal_recipe_v1_t* recipe, const void* buffer, size_t size,
    n4m_multimodal_pipeline_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    return guarded(ctx, [&]() {
        std::unique_ptr<n4m_multimodal_pipeline_s> pipeline;
        const auto status = n4m::multimodal::load(ctx, recipe, buffer, size, pipeline);
        if (status == N4M_OK) *out = pipeline.release(); return status;
    });
}

namespace {
enum class ClassifierOperation { Transform, Decision, Probability };

n4m_status_t classifier_output_bound(n4m_context_t* ctx, std::int64_t rows, std::int64_t cols) {
    constexpr std::int64_t limit = 16 * 1024 * 1024;
    if (rows < 0 || cols <= 0 || cols > limit || rows > limit / cols) {
        ctx->set_error("classifier output exceeds element limit");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

n4m_status_t classifier_matrix(n4m_context_t* ctx, const n4m_multimodal_classifier_t* pipeline,
                               int32_t count, const n4m_multimodal_source_view_v1_t* sources,
                               n4m_matrix_view_t* out, ClassifierOperation operation) {
    return guarded(ctx, [&]() {
        if (pipeline == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<double> features;
        std::int64_t rows = 0, cols = 0;
        auto status = n4m::multimodal::features(ctx, pipeline->encoded, count, sources, features, rows, cols);
        if (status != N4M_OK) return status;
        std::int64_t outputs = cols;
        if (operation != ClassifierOperation::Transform)
            status = n4m_estimator_n_outputs(pipeline->encoded.model.get(), &outputs);
        if (status == N4M_OK) status = classifier_output_bound(ctx, rows, outputs);
        if (status == N4M_OK) status = output(ctx, out, rows, outputs);
        if (status != N4M_OK || rows == 0) return status;
        if (operation == ClassifierOperation::Transform) {
            auto* values = static_cast<double*>(out->data);
            for (std::int64_t row = 0; row < rows; ++row)
                for (std::int64_t col = 0; col < cols; ++col)
                    values[row * out->row_stride + col * out->col_stride] = features[static_cast<std::size_t>(row * cols + col)];
            return N4M_OK;
        }
        n4m_matrix_view_t x{};
        n4m_matrix_view_init_rowmajor(&x, features.data(), rows, cols, N4M_DTYPE_F64);
        return operation == ClassifierOperation::Probability
            ? n4m_estimator_predict_proba(ctx, pipeline->encoded.model.get(), &x, out)
            : n4m_estimator_decision_function(ctx, pipeline->encoded.model.get(), &x, out);
    });
}
}

N4M_API n4m_status_t n4m_multimodal_classifier_create(n4m_context_t* ctx,
    const n4m_multimodal_classifier_recipe_v1_t* recipe, n4m_multimodal_classifier_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    return guarded(ctx, [&]() {
        std::unique_ptr<n4m_multimodal_classifier_s> pipeline;
        const auto status = n4m::multimodal::classifier_create(ctx, recipe, pipeline);
        if (status == N4M_OK) *out = pipeline.release();
        return status;
    });
}
N4M_API void n4m_multimodal_classifier_destroy(n4m_multimodal_classifier_t* pipeline) { delete pipeline; }
N4M_API n4m_status_t n4m_multimodal_classifier_fit(n4m_context_t* ctx,
    n4m_multimodal_classifier_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, const int64_t* labels, int64_t n_labels) {
    return guarded(ctx, [&]() {
        return pipeline == nullptr ? N4M_ERR_NULL_POINTER
            : n4m::multimodal::classifier_fit(ctx, *pipeline, count, sources, labels, n_labels);
    });
}
N4M_API n4m_status_t n4m_multimodal_classifier_predict_labels(n4m_context_t* ctx,
    const n4m_multimodal_classifier_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, int64_t* out, int64_t n) {
    return guarded(ctx, [&]() {
        if (pipeline == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<double> features;
        std::int64_t rows = 0, cols = 0;
        auto status = n4m::multimodal::features(ctx, pipeline->encoded, count, sources, features, rows, cols);
        if (status != N4M_OK) return status;
        if (n != rows) return N4M_ERR_SHAPE_MISMATCH;
        std::int64_t classes = 0;
        status = n4m_estimator_n_outputs(pipeline->encoded.model.get(), &classes);
        if (status == N4M_OK) status = classifier_output_bound(ctx, rows, classes);
        if (status != N4M_OK) return status;
        if (rows == 0) return N4M_OK;
        if (out == nullptr) return N4M_ERR_NULL_POINTER;
        n4m_matrix_view_t x{};
        n4m_matrix_view_init_rowmajor(&x, features.data(), rows, cols, N4M_DTYPE_F64);
        return n4m_estimator_predict_labels(ctx, pipeline->encoded.model.get(), &x, out, rows);
    });
}
N4M_API n4m_status_t n4m_multimodal_classifier_decision_function(n4m_context_t* ctx,
    const n4m_multimodal_classifier_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, n4m_matrix_view_t* out) {
    return classifier_matrix(ctx, pipeline, count, sources, out, ClassifierOperation::Decision);
}
N4M_API n4m_status_t n4m_multimodal_classifier_predict_proba(n4m_context_t* ctx,
    const n4m_multimodal_classifier_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, n4m_matrix_view_t* out) {
    return classifier_matrix(ctx, pipeline, count, sources, out, ClassifierOperation::Probability);
}
N4M_API n4m_status_t n4m_multimodal_classifier_classes(const n4m_multimodal_classifier_t* pipeline,
    int64_t* out, int64_t capacity, int64_t* count) {
    if (pipeline == nullptr || count == nullptr) return N4M_ERR_NULL_POINTER;
    *count = 0;
    if (!pipeline->encoded.fitted) return N4M_ERR_NOT_FITTED;
    return n4m_estimator_classes(pipeline->encoded.model.get(), out, capacity, count);
}
N4M_API n4m_status_t n4m_multimodal_classifier_n_outputs(const n4m_multimodal_classifier_t* pipeline,
    int64_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = 0;
    if (!pipeline->encoded.fitted) return N4M_ERR_NOT_FITTED;
    return n4m_estimator_n_outputs(pipeline->encoded.model.get(), out);
}
N4M_API n4m_status_t n4m_multimodal_classifier_transform_cols(const n4m_multimodal_classifier_t* pipeline,
    int64_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = 0;
    if (!pipeline->encoded.fitted) return N4M_ERR_NOT_FITTED;
    *out = pipeline->encoded.encoded_cols;
    return N4M_OK;
}
N4M_API n4m_status_t n4m_multimodal_classifier_transform(n4m_context_t* ctx,
    const n4m_multimodal_classifier_t* pipeline, int32_t count,
    const n4m_multimodal_source_view_v1_t* sources, n4m_matrix_view_t* out) {
    return classifier_matrix(ctx, pipeline, count, sources, out, ClassifierOperation::Transform);
}
N4M_API n4m_status_t n4m_multimodal_classifier_export_size(n4m_context_t* ctx,
    const n4m_multimodal_classifier_t* pipeline, size_t* out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = 0;
    return guarded(ctx, [&]() {
        if (pipeline == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<unsigned char> bytes;
        const auto status = n4m::multimodal::classifier_save(ctx, *pipeline, bytes);
        if (status == N4M_OK) *out = bytes.size();
        return status;
    });
}
N4M_API n4m_status_t n4m_multimodal_classifier_export_to_buffer(n4m_context_t* ctx,
    const n4m_multimodal_classifier_t* pipeline, void* buffer, size_t size, size_t* written) {
    if (written == nullptr) return N4M_ERR_NULL_POINTER;
    *written = 0;
    return guarded(ctx, [&]() {
        if (pipeline == nullptr || buffer == nullptr) return N4M_ERR_NULL_POINTER;
        std::vector<unsigned char> bytes;
        const auto status = n4m::multimodal::classifier_save(ctx, *pipeline, bytes);
        if (status != N4M_OK) return status;
        if (size < bytes.size()) return N4M_ERR_INVALID_ARGUMENT;
        std::memcpy(buffer, bytes.data(), bytes.size());
        *written = bytes.size();
        return N4M_OK;
    });
}
N4M_API n4m_status_t n4m_multimodal_classifier_import_from_buffer(n4m_context_t* ctx,
    const n4m_multimodal_classifier_recipe_v1_t* expected, const void* buffer, size_t size,
    n4m_multimodal_classifier_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    return guarded(ctx, [&]() {
        std::unique_ptr<n4m_multimodal_classifier_s> pipeline;
        const auto status = n4m::multimodal::classifier_load(ctx, expected, buffer, size, pipeline);
        if (status == N4M_OK) *out = pipeline.release();
        return status;
    });
}
