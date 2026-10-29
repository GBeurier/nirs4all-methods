// SPDX-License-Identifier: CECILL-2.1
// Generic procedure transport; parameter types come exclusively from the native
// manifest. Reuse the established MEX input/parameter/RAII translations.
#define N4M_ROLE_PIPELINE_HELPERS_ONLY
#include "n4m_role_pipeline_mex.cpp"

namespace {
struct ResultDeleter {
    void operator()(n4m_method_result_t* value) const {
        n4m_method_result_destroy(value);
    }
};
mxArray*
result_entry(n4m_method_result_t* result, const char* name, int32_t kind, n4m_context_t* ctx) {
    if (kind == N4M_RESULT_SCALAR) {
        double value = 0;
        check(n4m_method_result_get_scalar(result, name, &value), ctx, "result_scalar");
        return mxCreateDoubleScalar(value);
    }
    if (kind == N4M_RESULT_DOUBLE_MATRIX) {
        const double* value = nullptr;
        int64_t rows = 0, cols = 0;
        check(n4m_method_result_get_double_matrix(result, name, &value, &rows, &cols),
              ctx,
              "result_matrix");
        mxArray* output =
            mxCreateDoubleMatrix(static_cast<mwSize>(rows), static_cast<mwSize>(cols), mxREAL);
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j)
                mxGetPr(output)[static_cast<size_t>(j * rows + i)] = value[i * cols + j];
        return output;
    }
    if (kind == N4M_RESULT_INT_VECTOR) {
        const int32_t* value = nullptr;
        int32_t n = 0;
        check(n4m_method_result_get_int_vector(result, name, &value, &n), ctx, "result_int_vector");
        mxArray* output = mxCreateNumericMatrix(static_cast<mwSize>(n), 1, mxINT32_CLASS, mxREAL);
        if (n)
            std::memcpy(mxGetData(output), value, static_cast<size_t>(n) * sizeof(int32_t));
        return output;
    }
    if (kind == N4M_RESULT_INT64_VECTOR) {
        const int64_t* value = nullptr;
        int64_t n = 0;
        check(n4m_method_result_get_int64_vector(result, name, &value, &n),
              ctx,
              "result_int64_vector");
        mxArray* output = mxCreateNumericMatrix(static_cast<mwSize>(n), 1, mxINT64_CLASS, mxREAL);
        if (n)
            std::memcpy(mxGetData(output), value, static_cast<size_t>(n) * sizeof(int64_t));
        return output;
    }
    throw std::runtime_error("unknown native result entry type");
}
}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        require(nlhs == 1 && nrhs >= 2 && nrhs <= 4,
                "procedure(method_id, X, params, optional_X_target) returns one result struct");
        const auto method = text(prhs[0]);
        Entry entry;
        check(n4m_context_create(&entry.context), nullptr, "context_create");
        auto params =
            parameters(entry, method, nrhs >= 3 && !mxIsEmpty(prhs[2]) ? prhs[2] : nullptr);
        int32_t index = -1;
        check(n4m_method_find(method.c_str(), &index), entry.context, "method_find");
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        check(n4m_method_info_v1(index, &info), entry.context, "method_info");
        require(info.kind == N4M_METHOD_PROCEDURE && info.roles == N4M_ROLE_GENERIC,
                "method must have the native generic procedure role");
        auto X = matrix(prhs[1]);
        n4m_matrix_view_t reference{};
        n4m_fit_inputs_v1_t input{};
        input.struct_size = sizeof(input);
        input.X = &X;
        if (nrhs == 4 && !mxIsEmpty(prhs[3])) {
            reference = matrix(prhs[3]);
            input.X_target = &reference;
        }
        n4m_method_result_t* raw = nullptr;
        check(n4m_procedure_run(entry.context, index, params.get(), &input, &raw),
              entry.context,
              "procedure_run");
        std::unique_ptr<n4m_method_result_t, ResultDeleter> result(raw);
        MxOwner output(mxCreateStructMatrix(1, 1, 0, nullptr));
        int32_t count = 0;
        check(n4m_method_result_entry_count(raw, &count), entry.context, "result_count");
        for (int32_t i = 0; i < count; ++i) {
            const char* name = nullptr;
            int32_t kind = 0;
            check(n4m_method_result_entry(raw, i, &name, &kind), entry.context, "result_entry");
            require(mxAddField(output.get(), name) >= 0,
                    "native result name is not a valid struct field");
            mxSetField(output.get(), 0, name, result_entry(raw, name, kind, entry.context));
        }
        plhs[0] = output.release();
    } catch (const std::exception& error) {
        mexErrMsgIdAndTxt("n4m:procedure", "%s", error.what());
    }
}
