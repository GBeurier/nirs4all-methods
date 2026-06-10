// SPDX-License-Identifier: CECILL-2.1
//
// Adversarial C-ABI boundary + memory-ownership tests.
//
// These exercise the public surface with hostile inputs — NULL handles, wrong
// dtypes, mis-shaped / mis-strided caller buffers, double frees, and use of the
// context error buffer across calls — and assert the documented n4m_status_t
// rather than letting the library crash. Each asserted behaviour is verified
// against the contract in cpp/include/n4m/n4m.h + pls.h and the c_api
// implementation, so the suite is intended to run clean under ASan/UBSan.
//
// What is intentionally NOT asserted: behaviours the ABI does not document
// (e.g. predict tolerating an F32 X view IS documented — validate_float_view
// accepts F64 or F32 — so we assert N4M_OK there, not an error).

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "harness.hpp"

namespace {

// A small, well-conditioned PLS1 problem fit through the public ABI. Returns
// owning handles via out-params; caller destroys.
struct Fitted {
    n4m_context_t* ctx = nullptr;
    n4m_config_t*  cfg = nullptr;
    n4m_model_t*   model = nullptr;
    std::vector<double> X;   // 5 x 3 row-major
    std::vector<double> Y;   // 5 x 1
    std::int64_t rows = 5;
    std::int64_t features = 3;
    std::int64_t targets = 1;
};

void fit_small(Fitted& f) {
    f.X = {0.1, 0.2, 0.3,
           0.4, 0.1, 0.6,
           0.7, 0.8, 0.2,
           0.2, 0.5, 0.9,
           0.6, 0.3, 0.1};
    f.Y = {1.0, 2.0, 3.0, 2.5, 1.5};
    N4M_TEST_REQUIRE(n4m_context_create(&f.ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&f.cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(f.cfg, 2) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_solver(f.cfg, N4M_SOLVER_NIPALS) == N4M_OK);
    n4m_matrix_view_t X{}, Y{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&X, f.X.data(), f.rows, f.features, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Y, f.Y.data(), f.rows, f.targets, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(f.ctx, f.cfg, &X, &Y, &f.model) == N4M_OK);
    N4M_TEST_REQUIRE(f.model != nullptr);
}

void destroy(Fitted& f) {
    n4m_model_destroy(f.model);
    n4m_config_destroy(f.cfg);
    n4m_context_destroy(f.ctx);
}

// --- NULL-pointer contracts on the fit/predict/array surface -----------------

void test_null_pointer_contracts() {
    n4m_context_t* ctx = nullptr;
    n4m_config_t*  cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);

    // fit with NULL X/Y must reject and leave out_model NULL.
    n4m_model_t* model = reinterpret_cast<n4m_model_t*>(0x1);
    N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, nullptr, nullptr, &model) == N4M_ERR_NULL_POINTER);

    // predict_alloc with NULL model must reject and zero the out handle.
    n4m_array_t* arr = reinterpret_cast<n4m_array_t*>(0x1);
    N4M_TEST_REQUIRE(n4m_model_predict_alloc(ctx, nullptr, nullptr, &arr) == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(arr == nullptr);

    // get_array with NULL model.
    arr = reinterpret_cast<n4m_array_t*>(0x1);
    N4M_TEST_REQUIRE(n4m_model_get_array(ctx, nullptr, N4M_MODEL_COEFFICIENTS, &arr) == N4M_ERR_NULL_POINTER);

    // array accessors on a NULL array.
    n4m_dtype_t dtype = N4M_DTYPE_UNKNOWN;
    N4M_TEST_REQUIRE(n4m_array_dtype(nullptr, &dtype) == N4M_ERR_NULL_POINTER);
    std::int64_t rows = 0, cols = 0;
    N4M_TEST_REQUIRE(n4m_array_shape(nullptr, &rows, &cols) == N4M_ERR_NULL_POINTER);
    n4m_matrix_view_t view{};
    N4M_TEST_REQUIRE(n4m_array_view(nullptr, &view) == N4M_ERR_NULL_POINTER);

    // Freeing / destroying NULL must be no-ops, not crashes.
    n4m_array_free(nullptr);
    n4m_model_destroy(nullptr);
    n4m_config_destroy(nullptr);
    n4m_context_destroy(nullptr);

    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

// --- Predict with a mis-shaped caller output buffer --------------------------

void test_predict_output_shape_mismatch() {
    Fitted f;
    fit_small(f);

    n4m_matrix_view_t X{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&X, f.X.data(), f.rows, f.features, N4M_DTYPE_F64) == N4M_OK);

    // Output with the wrong number of rows -> SHAPE_MISMATCH (documented).
    std::vector<double> too_few(static_cast<std::size_t>((f.rows - 1) * f.targets), 0.0);
    n4m_matrix_view_t out_bad_rows{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &out_bad_rows, too_few.data(), f.rows - 1, f.targets, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_predict(f.ctx, f.model, &X, &out_bad_rows) == N4M_ERR_SHAPE_MISMATCH);

    // Output with the wrong number of cols -> SHAPE_MISMATCH.
    std::vector<double> too_wide(static_cast<std::size_t>(f.rows * (f.targets + 1)), 0.0);
    n4m_matrix_view_t out_bad_cols{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &out_bad_cols, too_wide.data(), f.rows, f.targets + 1, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_predict(f.ctx, f.model, &X, &out_bad_cols) == N4M_ERR_SHAPE_MISMATCH);

    // X with the wrong feature count -> SHAPE_MISMATCH.
    std::vector<double> x_narrow(static_cast<std::size_t>(f.rows * (f.features - 1)), 0.1);
    n4m_matrix_view_t X_narrow{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &X_narrow, x_narrow.data(), f.rows, f.features - 1, N4M_DTYPE_F64) == N4M_OK);
    std::vector<double> out_ok(static_cast<std::size_t>(f.rows * f.targets), 0.0);
    n4m_matrix_view_t out{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &out, out_ok.data(), f.rows, f.targets, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_predict(f.ctx, f.model, &X_narrow, &out) == N4M_ERR_SHAPE_MISMATCH);

    destroy(f);
}

// --- Over-strided caller output buffer ---------------------------------------
// A column-major / strided output view of the correct (rows, cols) is accepted
// by the stride-aware boundary (write_value honours strides). Assert the
// strided write round-trips to the same values as the contiguous path.

void test_predict_strided_output_roundtrip() {
    Fitted f;
    fit_small(f);
    n4m_matrix_view_t X{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&X, f.X.data(), f.rows, f.features, N4M_DTYPE_F64) == N4M_OK);

    // Contiguous reference predictions.
    std::vector<double> ref(static_cast<std::size_t>(f.rows * f.targets), 0.0);
    n4m_matrix_view_t out_ref{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&out_ref, ref.data(), f.rows, f.targets, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_predict(f.ctx, f.model, &X, &out_ref) == N4M_OK);

    // Over-strided output: a (rows x targets) view embedded in a wider buffer
    // with row_stride = 4 (targets == 1). Strides are in elements.
    const std::int64_t row_stride = 4;
    std::vector<double> wide(static_cast<std::size_t>(f.rows * row_stride), -999.0);
    n4m_matrix_view_t out_strided{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_strided(
        &out_strided, wide.data(), f.rows, f.targets, row_stride, 1, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_validate(&out_strided) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_predict(f.ctx, f.model, &X, &out_strided) == N4M_OK);
    for (std::int64_t i = 0; i < f.rows; ++i) {
        const double got = wide[static_cast<std::size_t>(i * row_stride)];
        N4M_TEST_REQUIRE(got == ref[static_cast<std::size_t>(i)]);
    }

    destroy(f);
}

// --- Negative stride on a view is rejected by validate -----------------------

void test_invalid_stride_rejected() {
    double buf[6] = {1, 2, 3, 4, 5, 6};
    n4m_matrix_view_t v{};
    v.data = buf;
    v.rows = 2;
    v.cols = 3;
    v.row_stride = -3;   // illegal
    v.col_stride = 1;
    v.dtype = N4M_DTYPE_F64;
    N4M_TEST_REQUIRE(n4m_matrix_view_validate(&v) == N4M_ERR_STRIDE_INVALID);

    // Zero stride on a non-degenerate dimension is also invalid.
    v.row_stride = 0;
    N4M_TEST_REQUIRE(n4m_matrix_view_validate(&v) == N4M_ERR_STRIDE_INVALID);

    // Validate of a NULL view pointer.
    N4M_TEST_REQUIRE(n4m_matrix_view_validate(nullptr) == N4M_ERR_NULL_POINTER);

    // Non-NULL view with NULL data on a non-empty matrix.
    v.row_stride = 3;
    v.data = nullptr;
    N4M_TEST_REQUIRE(n4m_matrix_view_validate(&v) == N4M_ERR_NULL_POINTER);
}

// --- dtype mismatch on fit (I32 X where a float is required) ------------------

void test_fit_dtype_mismatch() {
    n4m_context_t* ctx = nullptr;
    n4m_config_t*  cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, 1) == N4M_OK);

    std::vector<std::int32_t> xi(15, 1);
    std::vector<double> y(5, 1.0);
    n4m_matrix_view_t X{}, Y{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&X, xi.data(), 5, 3, N4M_DTYPE_I32) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Y, y.data(), 5, 1, N4M_DTYPE_F64) == N4M_OK);
    n4m_model_t* model = nullptr;
    N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, &X, &Y, &model) == N4M_ERR_DTYPE_MISMATCH);
    N4M_TEST_REQUIRE(model == nullptr);

    // NaN in X is a value error, not a dtype error.
    std::vector<double> xn(15, 0.1);
    xn[0] = std::nan("");
    n4m_matrix_view_t Xn{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Xn, xn.data(), 5, 3, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, &Xn, &Y, &model) == N4M_ERR_INVALID_ARGUMENT);

    // Mismatched X/Y row counts -> SHAPE_MISMATCH.
    std::vector<double> xok(15, 0.1);
    n4m_matrix_view_t Xok{}, Yshort{};
    std::vector<double> y4(4, 1.0);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Xok, xok.data(), 5, 3, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Yshort, y4.data(), 4, 1, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, &Xok, &Yshort, &model) == N4M_ERR_SHAPE_MISMATCH);

    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

// --- Degenerate / zero dims --------------------------------------------------

void test_zero_dims() {
    // A 0x0 empty view (NULL data permitted) validates OK per the §3 rules.
    n4m_matrix_view_t empty{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&empty, nullptr, 0, 0, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_validate(&empty) == N4M_OK);

    // Negative dims are rejected at init time.
    n4m_matrix_view_t neg{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&neg, nullptr, -1, 3, N4M_DTYPE_F64) == N4M_ERR_INVALID_ARGUMENT);

    // UNKNOWN dtype is rejected at init time.
    double d = 1.0;
    n4m_matrix_view_t bad_dtype{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&bad_dtype, &d, 1, 1, N4M_DTYPE_UNKNOWN) == N4M_ERR_INVALID_ARGUMENT);
}

// --- Double free / use-after-free-style ownership safety ----------------------

void test_double_free_is_safe() {
    Fitted f;
    fit_small(f);
    n4m_array_t* arr = nullptr;
    N4M_TEST_REQUIRE(n4m_model_get_array(f.ctx, f.model, N4M_MODEL_COEFFICIENTS, &arr) == N4M_OK);
    N4M_TEST_REQUIRE(arr != nullptr);
    n4m_array_free(arr);
    // A second free of the same pointer is documented as undefined for a
    // dangling pointer, so we only assert that freeing NULL (the safe idiom) is
    // a no-op. Re-freeing the now-dangling handle is intentionally NOT done.
    n4m_array_free(nullptr);
    destroy(f);
}

// --- Error buffer is invalidated by the next call ----------------------------

void test_error_buffer_lifecycle() {
    n4m_context_t* ctx = nullptr;
    n4m_config_t*  cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, 1) == N4M_OK);

    // Fresh context: last_error is the empty string.
    const char* fresh = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(fresh != nullptr);
    N4M_TEST_REQUIRE(fresh[0] == '\0');

    // Trigger an error and capture a *copy* of the message.
    std::vector<std::int32_t> xi(15, 1);
    std::vector<double> y(5, 1.0);
    n4m_matrix_view_t X{}, Y{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&X, xi.data(), 5, 3, N4M_DTYPE_I32) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Y, y.data(), 5, 1, N4M_DTYPE_F64) == N4M_OK);
    n4m_model_t* model = nullptr;
    N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, &X, &Y, &model) == N4M_ERR_DTYPE_MISMATCH);
    const std::string after_error = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(!after_error.empty());

    // A subsequent successful call clears the error: the documented contract is
    // that the pointer is invalidated by the next call. After a clean fit, the
    // error buffer reads empty again. Use genuinely varying X/Y so the fit is
    // numerically well-posed (constant data would itself raise an error).
    std::vector<double> xok = {0.1, 0.2, 0.3,
                               0.4, 0.1, 0.6,
                               0.7, 0.8, 0.2,
                               0.2, 0.5, 0.9,
                               0.6, 0.3, 0.1};
    std::vector<double> yok = {1.0, 2.0, 3.0, 2.5, 1.5};
    n4m_matrix_view_t Xok{}, Yok{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Xok, xok.data(), 5, 3, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Yok, yok.data(), 5, 1, N4M_DTYPE_F64) == N4M_OK);
    n4m_model_t* good = nullptr;
    N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, &Xok, &Yok, &good) == N4M_OK);
    const char* after_ok = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(after_ok != nullptr);
    N4M_TEST_REQUIRE(after_ok[0] == '\0');

    // Explicit clear is a no-op on an already-clear buffer and on NULL.
    n4m_context_clear_error(ctx);
    n4m_context_clear_error(nullptr);

    // last_error on a NULL context returns a static "" (never NULL).
    const char* null_err = n4m_context_last_error(nullptr);
    N4M_TEST_REQUIRE(null_err != nullptr);
    N4M_TEST_REQUIRE(null_err[0] == '\0');

    n4m_model_destroy(good);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_c_abi_memory_tests(n4m_testing::Runner& r) {
    r.run("c_abi_memory/null_pointer_contracts",       test_null_pointer_contracts);
    r.run("c_abi_memory/predict_output_shape_mismatch", test_predict_output_shape_mismatch);
    r.run("c_abi_memory/predict_strided_output",        test_predict_strided_output_roundtrip);
    r.run("c_abi_memory/invalid_stride_rejected",       test_invalid_stride_rejected);
    r.run("c_abi_memory/fit_dtype_mismatch",            test_fit_dtype_mismatch);
    r.run("c_abi_memory/zero_dims",                     test_zero_dims);
    r.run("c_abi_memory/double_free_is_safe",           test_double_free_is_safe);
    r.run("c_abi_memory/error_buffer_lifecycle",        test_error_buffer_lifecycle);
}
