// SPDX-License-Identifier: CECILL-2.1
// MATLAB/Octave translation of the existing ABI 2.14 RolePipeline surface.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mex.h"
#include "n4m/estimator.h"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void check(n4m_status_t status, n4m_context_t* context, const char* action) {
    if (status == N4M_OK) return;
    std::string message = std::string(action) + ": " + n4m_status_to_string(status);
    const char* detail = context ? n4m_context_last_error(context) : nullptr;
    if (detail && *detail) message += std::string(" (") + detail + ")";
    throw std::runtime_error(message);
}
struct MxDeleter {
    void operator()(mxArray* value) const { if (value) mxDestroyArray(value); }
};
using MxOwner = std::unique_ptr<mxArray, MxDeleter>;
struct ParamsDeleter {
    void operator()(n4m_params_t* value) const { n4m_params_destroy(value); }
};
using ParamsOwner = std::unique_ptr<n4m_params_t, ParamsDeleter>;
struct PipelineDeleter {
    void operator()(n4m_role_pipeline_t* value) const { n4m_role_pipeline_destroy(value); }
};
using PipelineOwner = std::unique_ptr<n4m_role_pipeline_t, PipelineDeleter>;

struct Entry {
    n4m_context_t* context = nullptr;
    PipelineOwner pipeline;
    std::vector<std::string> methods;
    std::vector<ParamsOwner> params;
    std::vector<std::string> features;
    ~Entry() {
        pipeline.reset();
        params.clear();
        n4m_context_destroy(context);
    }
};
std::map<uint64_t, std::unique_ptr<Entry>> entries;
uint64_t next_id = 1;
bool registered = false;
void cleanup() { entries.clear(); }

bool vector_shape(const mxArray* value) {
    return mxGetNumberOfDimensions(value) <= 2 &&
           (mxIsEmpty(value) || mxGetM(value) == 1 || mxGetN(value) == 1);
}
void scalar_struct(const mxArray* value) {
    require(value && mxIsStruct(value) && mxGetNumberOfElements(value) == 1,
            "expected one scalar struct");
}
void closed_fields(const mxArray* value, const std::vector<std::string>& allowed) {
    scalar_struct(value);
    for (int i = 0; i < mxGetNumberOfFields(value); ++i) {
        const std::string key = mxGetFieldNameByNumber(value, i);
        bool found = false;
        for (const auto& name : allowed) if (name == key) found = true;
        require(found, "unknown recipe, option or auxiliary input field");
    }
}
std::string text(const mxArray* value) {
    require(value && mxIsChar(value) && mxGetNumberOfDimensions(value) <= 2 &&
                mxGetM(value) == 1, "expected a nonempty character row string");
    const mxChar* characters = mxGetChars(value);
    for (mwSize i = 0; i < mxGetNumberOfElements(value); ++i)
        require(characters[i] != 0, "strings must not contain NUL");
    char* raw = mxArrayToString(value);
    require(raw != nullptr, "could not decode a character string");
    std::string result(raw);
    mxFree(raw);
    require(!result.empty(), "strings must not be empty");
    return result;
}
std::vector<std::string> names(const mxArray* value) {
    require(value && mxIsCell(value) && vector_shape(value),
            "feature names must be a cell vector of character strings");
    std::vector<std::string> result;
    result.reserve(mxGetNumberOfElements(value));
    for (mwIndex i = 0; i < mxGetNumberOfElements(value); ++i)
        result.push_back(text(mxGetCell(value, i)));
    return result;
}
std::vector<const char*> pointers(const std::vector<std::string>& values) {
    std::vector<const char*> result;
    for (const auto& value : values) result.push_back(value.c_str());
    return result;
}
void numeric(const mxArray* value) {
    require(value && mxIsNumeric(value) && !mxIsComplex(value) && !mxIsSparse(value),
            "expected full real numeric data");
}
int64_t integer_at(const mxArray* value, mwIndex i) {
    numeric(value);
    switch (mxGetClassID(value)) {
    case mxINT8_CLASS: return static_cast<const int8_t*>(mxGetData(value))[i];
    case mxUINT8_CLASS: return static_cast<const uint8_t*>(mxGetData(value))[i];
    case mxINT16_CLASS: return static_cast<const int16_t*>(mxGetData(value))[i];
    case mxUINT16_CLASS: return static_cast<const uint16_t*>(mxGetData(value))[i];
    case mxINT32_CLASS: return static_cast<const int32_t*>(mxGetData(value))[i];
    case mxUINT32_CLASS: return static_cast<const uint32_t*>(mxGetData(value))[i];
    case mxINT64_CLASS: return static_cast<const int64_t*>(mxGetData(value))[i];
    case mxUINT64_CLASS: {
        const uint64_t n = static_cast<const uint64_t*>(mxGetData(value))[i];
        require(n <= static_cast<uint64_t>(INT64_MAX), "integer must fit int64");
        return static_cast<int64_t>(n);
    }
    case mxDOUBLE_CLASS: {
        const double n = mxGetPr(value)[i];
        require(std::isfinite(n) && n == std::floor(n) && std::fabs(n) <= 9007199254740991.0,
                "use int64 for integers outside the exact double range");
        return static_cast<int64_t>(n);
    }
    default: throw std::runtime_error("integers need an integer class or an exact double");
    }
}
int64_t integer(const mxArray* value) {
    require(value && mxGetNumberOfElements(value) == 1, "expected one integer");
    return integer_at(value, 0);
}
bool boolean(const mxArray* value) {
    require(value && mxGetNumberOfElements(value) == 1 && !mxIsSparse(value) &&
                !mxIsComplex(value), "expected one boolean");
    if (mxIsLogical(value)) return mxIsLogicalScalarTrue(value);
    const int64_t n = integer(value);
    require(n == 0 || n == 1, "boolean must be logical, zero or one");
    return n != 0;
}
std::vector<int64_t> integers(const mxArray* value) {
    numeric(value);
    require(vector_shape(value), "expected an integer vector");
    std::vector<int64_t> result;
    result.reserve(mxGetNumberOfElements(value));
    for (mwIndex i = 0; i < mxGetNumberOfElements(value); ++i)
        result.push_back(integer_at(value, i));
    return result;
}
const double* double_data(const mxArray* value) {
    require(value && mxIsDouble(value) && !mxIsComplex(value) && !mxIsSparse(value),
            "matrices and continuous inputs must be full real doubles");
    return mxGetPr(value);
}
n4m_matrix_view_t matrix(const mxArray* value) {
    const double* data = double_data(value);
    require(mxGetNumberOfDimensions(value) <= 2 && mxGetM(value) <= INT64_MAX &&
                mxGetN(value) <= INT64_MAX, "expected a two-dimensional matrix");
    n4m_matrix_view_t view{};
    check(n4m_matrix_view_init_colmajor(&view, const_cast<double*>(data),
          static_cast<int64_t>(mxGetM(value)), static_cast<int64_t>(mxGetN(value)), N4M_DTYPE_F64),
          nullptr, "matrix_view");
    return view;
}
std::vector<double> doubles(const mxArray* value) {
    const double* data = double_data(value);
    require(vector_shape(value), "expected a double vector");
    if (mxIsEmpty(value)) return {};
    return std::vector<double>(data, data + mxGetNumberOfElements(value));
}
mxArray* scalar_i64(int64_t n) {
    mxArray* out = mxCreateNumericMatrix(1, 1, mxINT64_CLASS, mxREAL);
    *static_cast<int64_t*>(mxGetData(out)) = n;
    return out;
}
mxArray* scalar_u64(uint64_t n) {
    mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(out)) = n;
    return out;
}
uint64_t token(const mxArray* value) {
    require(value && mxIsUint64(value) && !mxIsSparse(value) && !mxIsComplex(value) &&
                mxGetNumberOfElements(value) == 1, "expected a registry uint64 handle");
    const uint64_t id = *static_cast<const uint64_t*>(mxGetData(value));
    require(id != 0, "closed or unknown RolePipeline handle");
    return id;
}
Entry& lookup(const mxArray* value) {
    const auto found = entries.find(token(value));
    require(found != entries.end(), "closed or unknown RolePipeline handle");
    return *found->second;
}

ParamsOwner parameters(Entry& entry, const std::string& method, const mxArray* supplied) {
    int32_t index = -1;
    check(n4m_method_find(method.c_str(), &index), entry.context, "method_find");
    n4m_params_t* raw = nullptr;
    check(n4m_params_create(entry.context, index, &raw), entry.context, "params_create");
    ParamsOwner owner(raw);
    if (supplied) {
        scalar_struct(supplied);
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        check(n4m_method_info_v1(index, &info), entry.context, "method_info");
        for (int i = 0; i < mxGetNumberOfFields(supplied); ++i) {
            const char* name = mxGetFieldNameByNumber(supplied, i);
            const mxArray* value = mxGetFieldByNumber(supplied, 0, i);
            n4m_param_info_v1_t spec{};
            bool found = false;
            for (int32_t j = 0; j < info.n_params; ++j) {
                spec.struct_size = sizeof(spec);
                check(n4m_method_param_info_v1(index, j, &spec), entry.context, "param_info");
                if (std::strcmp(spec.name, name) == 0) { found = true; break; }
            }
            if (!found) throw std::runtime_error("unknown native parameter: " + std::string(name));
            n4m_status_t status = N4M_ERR_INVALID_ARGUMENT;
            switch (spec.type) {
            case N4M_METHOD_PARAM_INT:
                status = n4m_params_set_int(raw, name, integer(value)); break;
            case N4M_METHOD_PARAM_DOUBLE:
                require(value && mxGetNumberOfElements(value) == 1,
                        "double parameter must be scalar");
                status = n4m_params_set_double(raw, name, *double_data(value)); break;
            case N4M_METHOD_PARAM_BOOL:
                status = n4m_params_set_bool(raw, name, boolean(value) ? 1 : 0); break;
            case N4M_METHOD_PARAM_ENUM: {
                const std::string choice = text(value);
                status = n4m_params_set_enum(raw, name, choice.c_str()); break;
            }
            case N4M_METHOD_PARAM_INT_ARRAY: {
                const auto values = integers(value);
                status = n4m_params_set_int_array(raw, name, values.data(), values.size()); break;
            }
            case N4M_METHOD_PARAM_DOUBLE_ARRAY: {
                const auto values = doubles(value);
                status = n4m_params_set_double_array(raw, name, values.data(), values.size()); break;
            }
            default: throw std::runtime_error("unknown native parameter type");
            }
            check(status, entry.context, name);
        }
    }
    check(n4m_params_validate(entry.context, raw), entry.context, "params_validate");
    return owner;
}
PipelineOwner create_pipeline(Entry& entry) {
    const auto ids = pointers(entry.methods);
    std::vector<const n4m_params_t*> params;
    for (const auto& value : entry.params) params.push_back(value.get());
    n4m_role_pipeline_t* raw = nullptr;
    check(n4m_role_pipeline_create(entry.context, static_cast<int32_t>(ids.size()),
          ids.data(), params.data(), &raw), entry.context, "role_pipeline_create");
    PipelineOwner out(raw);
    const auto features = pointers(entry.features);
    check(n4m_role_pipeline_set_feature_names(entry.context, raw, features.data(), features.size()),
          entry.context, "set_feature_names");
    return out;
}
std::unique_ptr<Entry> create_entry(const mxArray* steps, const mxArray* features,
                                  const mxArray* options) {
    require(steps && (mxIsCell(steps) || mxIsStruct(steps)) && vector_shape(steps) &&
                mxGetNumberOfElements(steps) <= INT32_MAX,
            "steps must be a cell vector or struct vector of method_id and params");
    auto entry = std::unique_ptr<Entry>(new Entry);
    check(n4m_check_abi_compatibility(2, 14), nullptr, "RolePipeline ABI 2.14");
    check(n4m_context_create(&entry->context), nullptr, "context_create");
    if (options) {
        closed_fields(options, {"num_threads", "max_state_bytes"});
        if (const mxArray* n = mxGetField(options, 0, "num_threads")) {
            const int64_t count = integer(n);
            require(count >= 1 && count <= INT32_MAX, "num_threads must be positive int32");
            check(n4m_context_set_num_threads(entry->context, static_cast<int32_t>(count)),
                  entry->context, "set_num_threads");
        }
        if (const mxArray* n = mxGetField(options, 0, "max_state_bytes")) {
            const int64_t count = integer(n);
            require(count > 0, "max_state_bytes must be positive");
            check(n4m_context_set_max_state_bytes(entry->context, static_cast<uint64_t>(count)),
                  entry->context, "set_max_state_bytes");
        }
    }
    for (mwIndex i = 0; i < mxGetNumberOfElements(steps); ++i) {
        MxOwner copy;
        const mxArray* step = mxIsCell(steps) ? mxGetCell(steps, i) : nullptr;
        if (mxIsStruct(steps)) {
            const char* fields[] = {"method_id", "params"};
            copy.reset(mxCreateStructMatrix(1, 1, 2, fields));
            require(mxGetNumberOfFields(steps) >= 1 && mxGetNumberOfFields(steps) <= 2 &&
                        mxGetFieldNumber(steps, "method_id") >= 0 &&
                        (mxGetNumberOfFields(steps) == 1 || mxGetFieldNumber(steps, "params") >= 0),
                    "recipe steps allow only method_id and params");
            for (int j = 0; j < 2; ++j) {
                const mxArray* value = mxGetField(steps, i, fields[j]);
                if (value) mxSetField(copy.get(), 0, fields[j], mxDuplicateArray(value));
            }
            step = copy.get();
        }
        closed_fields(step, {"method_id", "params"});
        const std::string method = text(mxGetField(step, 0, "method_id"));
        entry->methods.push_back(method);
        entry->params.push_back(parameters(*entry, method, mxGetField(step, 0, "params")));
    }
    entry->features = names(features);
    entry->pipeline = create_pipeline(*entry);
    return entry;
}

struct FitStorage {
    n4m_fit_inputs_v1_t inputs{};
    n4m_matrix_view_t X{}, Y{}, target{};
    std::vector<int64_t> labels, groups, feature_groups, blocks, folds;
    FitStorage(const mxArray* x, const mxArray* y, const mxArray* aux) {
        closed_fields(aux, {"labels", "sample_weight", "groups", "feature_groups", "block_sizes",
                            "axis", "X_target", "fold_ids"});
        X = matrix(x);
        inputs.struct_size = sizeof(inputs);
        inputs.X = &X;
        if (!mxIsEmpty(y)) { Y = matrix(y); inputs.Y = &Y; }
        if (const mxArray* value = mxGetField(aux, 0, "labels")) {
            labels = integers(value); inputs.labels = labels.data(); inputs.n_labels = labels.size();
        }
        if (const mxArray* value = mxGetField(aux, 0, "groups")) {
            groups = integers(value); inputs.groups = groups.data(); inputs.n_groups = groups.size();
        }
        if (const mxArray* value = mxGetField(aux, 0, "feature_groups")) {
            feature_groups = integers(value); inputs.feature_groups = feature_groups.data();
            inputs.n_feature_groups = feature_groups.size();
        }
        if (const mxArray* value = mxGetField(aux, 0, "block_sizes")) {
            blocks = integers(value); inputs.block_sizes = blocks.data(); inputs.n_blocks = blocks.size();
        }
        if (const mxArray* value = mxGetField(aux, 0, "fold_ids")) {
            folds = integers(value); inputs.fold_ids = folds.data(); inputs.n_fold_ids = folds.size();
        }
        if (const mxArray* value = mxGetField(aux, 0, "sample_weight")) {
            inputs.sample_weight = double_data(value);
            require(vector_shape(value), "sample_weight must be a vector");
            inputs.n_sample_weight = mxGetNumberOfElements(value);
        }
        if (const mxArray* value = mxGetField(aux, 0, "axis")) {
            inputs.axis = double_data(value);
            require(vector_shape(value), "axis must be a vector");
            inputs.n_axis = mxGetNumberOfElements(value);
        }
        if (const mxArray* value = mxGetField(aux, 0, "X_target")) {
            target = matrix(value); inputs.X_target = &target;
        }
    }
};
void check_features(Entry& entry, const n4m_matrix_view_t& X, const mxArray* supplied) {
    const auto strings = names(supplied);
    require(strings.empty() || strings.size() == static_cast<uint64_t>(X.cols),
            "feature name count must match the input width");
    const auto values = pointers(strings);
    check(n4m_role_pipeline_check_features(entry.context, entry.pipeline.get(), X.cols,
          values.empty() ? nullptr : values.data()), entry.context, "check_features");
}
const char* role_name(uint32_t role) {
    switch (role) {
    case N4M_ROLE_TRANSFORMER: return "transformer";
    case N4M_ROLE_REGRESSOR: return "regressor";
    case N4M_ROLE_CLASSIFIER: return "classifier";
    case N4M_ROLE_SELECTOR: return "selector";
    case N4M_ROLE_SAMPLE_FILTER: return "sample_filter";
    default: throw std::runtime_error("unknown native pipeline role");
    }
}
n4m_role_pipeline_step_info_v1_t step_info(Entry& entry, int32_t index) {
    n4m_role_pipeline_step_info_v1_t info{};
    info.struct_size = sizeof(info);
    check(n4m_role_pipeline_step_info_v1(entry.pipeline.get(), index, &info),
          entry.context, "step_info");
    return info;
}
mxArray* steps_info(Entry& entry) {
    int32_t count = 0;
    check(n4m_role_pipeline_n_steps(entry.pipeline.get(), &count), entry.context, "n_steps");
    const char* fields[] = {"method_id", "role", "state_index", "contains_training_rows",
                            "n_features_in", "n_features_out"};
    MxOwner out(mxCreateStructMatrix(1, count, 6, fields));
    for (int32_t i = 0; i < count; ++i) {
        const auto info = step_info(entry, i);
        mxSetField(out.get(), i, "method_id", mxCreateString(info.method_id));
        mxSetField(out.get(), i, "role", mxCreateString(role_name(info.role)));
        mxSetField(out.get(), i, "state_index", scalar_i64(info.state_index));
        mxSetField(out.get(), i, "contains_training_rows",
                   mxCreateLogicalScalar(info.contains_training_rows != 0));
        mxSetField(out.get(), i, "n_features_in", scalar_i64(info.n_features_in));
        mxSetField(out.get(), i, "n_features_out", scalar_i64(info.n_features_out));
    }
    return out.release();
}
mxArray* feature_names(Entry& entry) {
    int64_t count = 0;
    check(n4m_role_pipeline_n_feature_names(entry.pipeline.get(), &count),
          entry.context, "n_feature_names");
    MxOwner out(mxCreateCellMatrix(1, static_cast<mwSize>(count)));
    for (int64_t i = 0; i < count; ++i) {
        const char* value = nullptr;
        check(n4m_role_pipeline_feature_name(entry.pipeline.get(), i, &value),
              entry.context, "feature_name");
        mxSetCell(out.get(), i, mxCreateString(value));
    }
    return out.release();
}
mxArray* export_states(Entry& entry, bool allow_rows) {
    int32_t count = 0;
    check(n4m_role_pipeline_n_states(entry.pipeline.get(), &count), entry.context, "n_states");
    const uint32_t flags = allow_rows ? N4M_EXPORT_ALLOW_TRAINING_ROWS : 0;
    MxOwner out(mxCreateCellMatrix(1, count));
    for (int32_t i = 0; i < count; ++i) {
        size_t size = 0, written = 0;
        check(n4m_role_pipeline_export_state_size(entry.context, entry.pipeline.get(), i, flags, &size),
              entry.context, "export_state_size");
        require(size > 0 && size <= std::numeric_limits<mwSize>::max(), "invalid native state size");
        MxOwner bytes(mxCreateNumericMatrix(1, static_cast<mwSize>(size), mxUINT8_CLASS, mxREAL));
        check(n4m_role_pipeline_export_state_to_buffer(entry.context, entry.pipeline.get(), i, flags,
              mxGetData(bytes.get()), size, &written), entry.context, "export_state_to_buffer");
        require(written == size, "native export wrote an inconsistent state size");
        mxSetCell(out.get(), i, bytes.release());
    }
    return out.release();
}
void import_states(Entry& entry, const mxArray* states) {
    require(states && mxIsCell(states) && vector_shape(states) &&
                mxGetNumberOfElements(states) <= INT32_MAX, "states must be a cell vector of uint8");
    std::vector<const void*> payloads;
    std::vector<size_t> sizes;
    for (mwIndex i = 0; i < mxGetNumberOfElements(states); ++i) {
        const mxArray* bytes = mxGetCell(states, i);
        require(bytes && mxIsUint8(bytes) && !mxIsComplex(bytes) && !mxIsSparse(bytes) &&
                    vector_shape(bytes) && !mxIsEmpty(bytes), "each N4ME state must be a nonempty uint8 vector");
        payloads.push_back(mxGetData(bytes)); sizes.push_back(mxGetNumberOfElements(bytes));
    }
    auto candidate = create_pipeline(entry);
    check(n4m_role_pipeline_import_states(entry.context, candidate.get(),
          static_cast<int32_t>(payloads.size()), payloads.data(), sizes.data()),
          entry.context, "import_states");
    entry.pipeline = std::move(candidate);
}
mxArray* classes(Entry& entry) {
    int64_t count = 0;
    check(n4m_role_pipeline_classes(entry.pipeline.get(), nullptr, 0, &count),
          entry.context, "classes");
    require(count >= 0 && static_cast<uint64_t>(count) <= std::numeric_limits<mwSize>::max(),
            "invalid native class count");
    MxOwner out(mxCreateNumericMatrix(1, static_cast<mwSize>(count), mxINT64_CLASS, mxREAL));
    check(n4m_role_pipeline_classes(entry.pipeline.get(), static_cast<int64_t*>(mxGetData(out.get())),
          count, &count), entry.context, "classes");
    return out.release();
}
mxArray* operate(Entry& entry, std::string command, const mxArray* data, const mxArray* features) {
    auto X = matrix(data);
    check_features(entry, X, features);
    if (command == "predict") {
        int32_t count = 0;
        check(n4m_role_pipeline_n_steps(entry.pipeline.get(), &count), entry.context, "n_steps");
        if (step_info(entry, count - 1).role == N4M_ROLE_CLASSIFIER) command = "predict_labels";
    }
    if (command == "predict_labels") {
        MxOwner out(mxCreateNumericMatrix(static_cast<mwSize>(X.rows), 1, mxINT64_CLASS, mxREAL));
        check(n4m_role_pipeline_predict_labels(entry.context, entry.pipeline.get(), &X,
              static_cast<int64_t*>(mxGetData(out.get())), X.rows), entry.context, "predict_labels");
        return out.release();
    }
    int64_t width = 0;
    check(command == "transform" ? n4m_role_pipeline_transform_cols(entry.pipeline.get(), &width) :
          n4m_role_pipeline_n_outputs(entry.pipeline.get(), &width), entry.context, "output_width");
    require(width >= 0 && static_cast<uint64_t>(width) <= std::numeric_limits<mwSize>::max(),
            "invalid native output width");
    MxOwner out(mxCreateDoubleMatrix(static_cast<mwSize>(X.rows), static_cast<mwSize>(width), mxREAL));
    auto view = matrix(out.get());
    n4m_status_t status = N4M_ERR_UNSUPPORTED;
    if (command == "predict")
        status = n4m_role_pipeline_predict(entry.context, entry.pipeline.get(), &X, &view);
    else if (command == "transform")
        status = n4m_role_pipeline_transform(entry.context, entry.pipeline.get(), &X, &view);
    else if (command == "decision_function")
        status = n4m_role_pipeline_decision_function(entry.context, entry.pipeline.get(), &X, &view);
    else if (command == "predict_proba")
        status = n4m_role_pipeline_predict_proba(entry.context, entry.pipeline.get(), &X, &view);
    check(status, entry.context, command.c_str());
    return out.release();
}
void dispatch(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs >= 1, "RolePipeline command is required");
    const std::string command = text(prhs[0]);
    if (command == "create") {
        require(nrhs == 4 && nlhs == 1, "create(steps, feature_names, options) requires one output");
        require(next_id != 0, "RolePipeline handle registry exhausted");
        auto entry = create_entry(prhs[1], prhs[2], prhs[3]);
        MxOwner id(scalar_u64(next_id));
        if (!registered) { mexAtExit(cleanup); registered = true; }
        entries.emplace(next_id++, std::move(entry));
        mexLock();
        plhs[0] = id.release();
        return;
    }
    require(nrhs >= 2, "RolePipeline registry handle is required");
    if (command == "close") {
        require(nrhs == 2 && nlhs == 0, "close(handle) takes no outputs");
        const uint64_t id = token(prhs[1]);
        auto found = entries.find(id);
        if (found == entries.end()) {
            require(id < next_id, "unknown RolePipeline handle");
            return;
        }
        entries.erase(found);
        mexUnlock();
        return;
    }
    Entry& entry = lookup(prhs[1]);
    if (command == "fit") {
        require(nrhs == 5 && nlhs == 0, "fit(handle, X, Y, aux) takes no outputs");
        FitStorage storage(prhs[2], prhs[3], prhs[4]);
        auto candidate = create_pipeline(entry);
        check(n4m_role_pipeline_fit(entry.context, candidate.get(), &storage.inputs),
              entry.context, "role_pipeline_fit");
        entry.pipeline = std::move(candidate);
    } else if (command == "import_states") {
        require(nrhs == 3 && nlhs == 0, "import_states(handle, states) takes no outputs");
        import_states(entry, prhs[2]);
    } else if (command == "set_feature_names") {
        require(nrhs == 3 && nlhs == 0, "set_feature_names(handle, names) takes no outputs");
        auto strings = names(prhs[2]); const auto values = pointers(strings);
        check(n4m_role_pipeline_set_feature_names(entry.context, entry.pipeline.get(),
              values.data(), values.size()), entry.context, "set_feature_names");
        entry.features = std::move(strings);
    } else if (command == "steps_info") {
        require(nrhs == 2 && nlhs == 1, "steps_info(handle) requires one output");
        plhs[0] = steps_info(entry);
    } else if (command == "feature_names") {
        require(nrhs == 2 && nlhs == 1, "feature_names(handle) requires one output");
        plhs[0] = feature_names(entry);
    } else if (command == "is_fitted") {
        require(nrhs == 2 && nlhs == 1, "is_fitted(handle) requires one output");
        int32_t value = 0;
        check(n4m_role_pipeline_is_fitted(entry.pipeline.get(), &value), entry.context, "is_fitted");
        plhs[0] = mxCreateLogicalScalar(value != 0);
    } else if (command == "export_states") {
        require(nrhs == 3 && nlhs == 1, "export_states(handle, allow_training_rows) requires one output");
        plhs[0] = export_states(entry, boolean(prhs[2]));
    } else if (command == "classes") {
        require(nrhs == 2 && nlhs == 1, "classes(handle) requires one output");
        plhs[0] = classes(entry);
    } else if (command == "predict" || command == "transform" ||
               command == "decision_function" || command == "predict_proba" ||
               command == "predict_labels") {
        require(nrhs == 4 && nlhs == 1, "operation(handle, X, feature_names) requires one output");
        plhs[0] = operate(entry, command, prhs[2], prhs[3]);
    } else if (command == "n_features_in" || command == "n_outputs" || command == "transform_cols") {
        require(nrhs == 2 && nlhs == 1, "width(handle) requires one output");
        int64_t value = 0;
        n4m_status_t status = command == "n_features_in" ?
            n4m_role_pipeline_n_features_in(entry.pipeline.get(), &value) : command == "n_outputs" ?
            n4m_role_pipeline_n_outputs(entry.pipeline.get(), &value) :
            n4m_role_pipeline_transform_cols(entry.pipeline.get(), &value);
        check(status, entry.context, command.c_str());
        plhs[0] = scalar_i64(value);
    } else throw std::runtime_error("unknown RolePipeline command: " + command);
}
}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try { dispatch(nlhs, plhs, nrhs, prhs); }
    catch (const std::exception& error) { mexErrMsgIdAndTxt("n4m:role_pipeline", "%s", error.what()); }
}
