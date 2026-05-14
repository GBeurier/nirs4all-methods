// SPDX-License-Identifier: CeCILL-2.1
//
// Phase 0 model-stub tests. PR 8 verifies every model-shaped function
// returns P4A_ERR_NOT_IMPLEMENTED and leaves out-params zeroed. Phase 1
// inverts these checks once NIPALS lands.

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(model_stubs, fit_returns_not_implemented) {
    p4a_context_t* ctx = nullptr;
    p4a_config_t*  cfg = nullptr;
    p4a_context_create(&ctx);
    p4a_config_create(&cfg);
    double buf[20];
    p4a_matrix_view_t X, Y;
    p4a_matrix_view_init_rowmajor(&X, buf, 4, 5, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Y, buf, 4, 1, P4A_DTYPE_F64);
    p4a_model_t* model = reinterpret_cast<p4a_model_t*>(0x1);  // pre-set garbage
    CHECK_EQ(p4a_model_fit(ctx, cfg, &X, &Y, &model), P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(model, nullptr);  // wrapper zeroed it
    p4a_config_destroy(cfg);
    p4a_context_destroy(ctx);
}

TEST(model_stubs, predict_returns_not_implemented) {
    p4a_context_t* ctx = nullptr;
    p4a_context_create(&ctx);
    p4a_matrix_view_t X;
    double buf[20];
    p4a_matrix_view_init_rowmajor(&X, buf, 4, 5, P4A_DTYPE_F64);
    p4a_matrix_view_t out;
    CHECK_EQ(p4a_model_predict(ctx, nullptr, &X, &out), P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(p4a_model_transform(ctx, nullptr, &X, &out), P4A_ERR_NOT_IMPLEMENTED);
    p4a_array_t* arr = reinterpret_cast<p4a_array_t*>(0x1);
    CHECK_EQ(p4a_model_predict_alloc(ctx, nullptr, &X, &arr), P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(arr, nullptr);
    p4a_context_destroy(ctx);
}

TEST(model_stubs, get_array_returns_not_implemented_and_zeros_out) {
    p4a_context_t* ctx = nullptr;
    p4a_context_create(&ctx);
    p4a_array_t* arr = reinterpret_cast<p4a_array_t*>(0x1);
    CHECK_EQ(p4a_model_get_array(ctx, nullptr, P4A_MODEL_COEFFICIENTS, &arr),
              P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(arr, nullptr);
    p4a_context_destroy(ctx);
}

TEST(model_stubs, serialization_returns_not_implemented) {
    size_t sz = 42;
    CHECK_EQ(p4a_model_export_size(nullptr, &sz), P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(sz, 0u);

    size_t written = 42;
    char buffer[16];
    CHECK_EQ(p4a_model_export_to_buffer(nullptr, buffer, sizeof(buffer), &written),
              P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(written, 0u);

    p4a_model_t* m = reinterpret_cast<p4a_model_t*>(0x1);
    CHECK_EQ(p4a_model_import_from_buffer(nullptr, buffer, sizeof(buffer), &m),
              P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(m, nullptr);

    uint32_t fmt = 99, maj = 99, min = 99, pat = 99;
    CHECK_EQ(p4a_serialization_inspect(buffer, sizeof(buffer), &fmt, &maj, &min, &pat),
              P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(fmt, 0u); CHECK_EQ(maj, 0u); CHECK_EQ(min, 0u); CHECK_EQ(pat, 0u);
}

TEST(model_stubs, array_accessors_return_not_implemented) {
    p4a_dtype_t dt = P4A_DTYPE_F64;
    int64_t r = -1, c = -1;
    p4a_matrix_view_t mv;
    CHECK_EQ(p4a_array_dtype(nullptr, &dt), P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(dt, P4A_DTYPE_UNKNOWN);
    CHECK_EQ(p4a_array_shape(nullptr, &r, &c), P4A_ERR_NOT_IMPLEMENTED);
    CHECK_EQ(r, 0); CHECK_EQ(c, 0);
    CHECK_EQ(p4a_array_view(nullptr, &mv), P4A_ERR_NOT_IMPLEMENTED);
    p4a_array_free(nullptr);  // NULL-safe
}
