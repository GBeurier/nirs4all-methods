// SPDX-License-Identifier: CECILL-2.1
// MATLAB/Octave handle bridge for the existing libn4m optimizer C ABI.
#include <algorithm>
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
#include "n4m/optimization.h"

namespace {
struct Entry {
    n4m_context_t* context = nullptr;
    n4m_optimizer_t* optimizer = nullptr;
    mxArray* space = nullptr;
    ~Entry() {
        n4m_optimizer_destroy(optimizer);
        n4m_context_destroy(context);
        if (space) mxDestroyArray(space);
    }
};
std::map<uint64_t, std::unique_ptr<Entry>> entries;
uint64_t next_id = 1;
bool registered = false;

void cleanup() { entries.clear(); }
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(n4m_status_t status, n4m_context_t* context, const char* action) {
    if (status == N4M_OK) return;
    std::string message = std::string(action) + ": " + n4m_status_to_string(status);
    if (context) {
        const char* detail = n4m_context_last_error(context);
        if (detail && detail[0]) message += std::string(" (") + detail + ")";
    }
    throw std::runtime_error(message);
}
const mxArray* field(const mxArray* item, mwIndex index, const char* name) {
    require(mxIsStruct(item), "expected a struct");
    const mxArray* value = mxGetField(item, index, name);
    require(value != nullptr, "missing optimizer field");
    return value;
}
const mxArray* optional(const mxArray* item, const char* name) {
    const mxArray* value = item && mxIsStruct(item) ? mxGetField(item, 0, name) : nullptr;
    return value && !mxIsEmpty(value) ? value : nullptr;
}
std::string text(const mxArray* value) {
    require(value && mxIsChar(value), "expected a character string");
    char* raw = mxArrayToString(value);
    require(raw != nullptr, "could not decode a string");
    std::string result(raw);
    mxFree(raw);
    return result;
}
double number(const mxArray* value) {
    require(value && (mxIsNumeric(value) || mxIsLogical(value)) &&
                !mxIsComplex(value) && !mxIsSparse(value) &&
                mxGetNumberOfElements(value) == 1, "expected one real number");
    const double result = mxGetScalar(value);
    require(std::isfinite(result), "number must be finite");
    return result;
}
int32_t i32(const mxArray* value) {
    const double n = number(value);
    require(n == std::floor(n) && n >= INT32_MIN && n <= INT32_MAX,
            "expected an int32 value");
    return static_cast<int32_t>(n);
}
int64_t i64(const mxArray* value) {
    if (value && mxIsInt64(value) && mxGetNumberOfElements(value) == 1)
        return *static_cast<const int64_t*>(mxGetData(value));
    const double n = number(value);
    require(n == std::floor(n) && std::fabs(n) <= 9007199254740992.0,
            "expected an exact int64 value");
    return static_cast<int64_t>(n);
}
uint64_t u64(const mxArray* value) {
    if (value && mxIsUint64(value) && mxGetNumberOfElements(value) == 1)
        return *static_cast<const uint64_t*>(mxGetData(value));
    const int64_t n = i64(value);
    require(n >= 0, "expected a nonnegative integer");
    return static_cast<uint64_t>(n);
}
mxArray* scalar_i64(int64_t n) {
    mxArray* value = mxCreateNumericMatrix(1, 1, mxINT64_CLASS, mxREAL);
    *static_cast<int64_t*>(mxGetData(value)) = n;
    return value;
}
mxArray* scalar_i32(int32_t n) {
    mxArray* value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *static_cast<int32_t*>(mxGetData(value)) = n;
    return value;
}
mxArray* scalar_u64(uint64_t n) {
    mxArray* value = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(value)) = n;
    return value;
}
int choice(const std::string& value, const std::vector<std::string>& names) {
    for (size_t i = 0; i < names.size(); ++i)
        if (value == names[i]) return static_cast<int>(i);
    throw std::runtime_error("unknown optimizer option: " + value);
}
std::vector<double> doubles(const mxArray* value) {
    require(mxIsDouble(value) && !mxIsComplex(value) && !mxIsSparse(value),
            "expected a real double vector");
    const mwSize n = mxGetNumberOfElements(value);
    if (n == 0) return {};
    const double* source = mxGetPr(value);
    std::vector<double> result(source, source + n);
    for (double x : result) require(std::isfinite(x), "vector values must be finite");
    return result;
}
void add_axis(n4m_search_space_t* space, const mxArray* declarations, mwIndex index) {
    const std::string name = text(field(declarations, index, "name"));
    const std::string kind = text(field(declarations, index, "kind"));
    n4m_status_t status = N4M_ERR_INVALID_ARGUMENT;
    if (kind == "int" || kind == "log_int") {
        const mxArray* step = mxGetField(declarations, index, "step");
        status = n4m_search_space_add_int(space, name.c_str(),
            i64(field(declarations, index, "low")), i64(field(declarations, index, "high")),
            step && !mxIsEmpty(step) ? i64(step) : 1, kind == "log_int");
    } else if (kind == "float" || kind == "log_float") {
        const mxArray* step = mxGetField(declarations, index, "step");
        status = n4m_search_space_add_float(space, name.c_str(),
            number(field(declarations, index, "low")),
            number(field(declarations, index, "high")),
            step && !mxIsEmpty(step) ? number(step) : 0,
            kind == "log_float");
    } else if (kind == "categorical") {
        const mxArray* values = field(declarations, index, "choices");
        const size_t count = mxGetNumberOfElements(values);
        require(count > 0 && count <= INT32_MAX, "invalid categorical choice count");
        require(!mxIsSparse(values), "categorical choices must be dense");
        if (mxIsCell(values)) {
            std::vector<std::string> strings;
            std::vector<const char*> pointers;
            strings.reserve(count);
            pointers.reserve(count);
            for (size_t i = 0; i < count; ++i) strings.push_back(text(mxGetCell(values, i)));
            for (const auto& item : strings) pointers.push_back(item.c_str());
            status = n4m_search_space_add_categorical(space, name.c_str(), N4M_CAT_STR,
                                                       pointers.data(), static_cast<int32_t>(count));
        } else if (mxIsLogical(values)) {
            const mxLogical* source = mxGetLogicals(values);
            std::vector<int32_t> bits(count);
            for (size_t i = 0; i < count; ++i) bits[i] = source[i] ? 1 : 0;
            status = n4m_search_space_add_categorical(space, name.c_str(), N4M_CAT_BOOL,
                                                       bits.data(), static_cast<int32_t>(count));
        } else if (mxIsInt64(values)) {
            status = n4m_search_space_add_categorical(space, name.c_str(), N4M_CAT_INT,
                                                       mxGetData(values), static_cast<int32_t>(count));
        } else {
            auto numeric = doubles(values);
            status = n4m_search_space_add_categorical(space, name.c_str(), N4M_CAT_FLOAT,
                                                       numeric.data(), static_cast<int32_t>(count));
        }
    } else if (kind == "ordinal") {
        auto values = doubles(field(declarations, index, "choices"));
        require(!values.empty() && values.size() <= INT32_MAX, "invalid ordinal choices");
        status = n4m_search_space_add_ordinal(space, name.c_str(), values.data(),
                                              static_cast<int32_t>(values.size()));
    } else if (kind == "sorted_tuple") {
        const mxArray* integer = mxGetField(declarations, index, "integer");
        status = n4m_search_space_add_sorted_tuple(space, name.c_str(),
            i32(field(declarations, index, "length")),
            number(field(declarations, index, "low")),
            number(field(declarations, index, "high")),
            integer && !mxIsEmpty(integer) ? static_cast<int32_t>(number(integer) != 0) : 0);
    } else throw std::runtime_error("unknown optimizer axis kind: " + kind);
    check(status, nullptr, "search_space_add");
}
void add_constraints(n4m_search_space_t* space, const mxArray* constraints) {
    if (!constraints || mxIsEmpty(constraints)) return;
    require(mxIsStruct(constraints), "constraints must be a struct array");
    const std::vector<std::string> kinds = {"mutex_group", "requires", "exclude",
                                             "condition_in", "condition_not_in"};
    for (size_t i = 0; i < mxGetNumberOfElements(constraints); ++i) {
        const int kind = choice(text(field(constraints, i, "kind")), kinds);
        const mxArray* refs = field(constraints, i, "refs");
        const mxArray* labels = mxGetField(constraints, i, "labels");
        require(mxIsCell(refs), "constraint refs must be a cell array");
        const size_t count = mxGetNumberOfElements(refs);
        require(count > 0 && count <= INT32_MAX, "invalid constraint reference count");
        require(!labels || mxIsEmpty(labels) ||
                (mxIsCell(labels) && mxGetNumberOfElements(labels) == count),
                "constraint labels must match refs");
        std::vector<std::string> names, label_values;
        std::vector<const char*> name_ptrs, label_ptrs;
        names.reserve(count);
        label_values.reserve(count);
        for (size_t j = 0; j < count; ++j) {
            names.push_back(text(mxGetCell(refs, j)));
            label_values.push_back(labels && !mxIsEmpty(labels) ? text(mxGetCell(labels, j)) : "");
        }
        for (size_t j = 0; j < count; ++j) {
            name_ptrs.push_back(names[j].c_str());
            label_ptrs.push_back(label_values[j].empty() ? nullptr : label_values[j].c_str());
        }
        check(n4m_search_space_add_constraint(space, static_cast<n4m_constraint_kind_t>(kind),
              name_ptrs.data(), label_ptrs.data(), static_cast<int32_t>(count)),
              nullptr, "search_space_add_constraint");
    }
}
std::unique_ptr<n4m_search_space_t, decltype(&n4m_search_space_destroy)>
build_space(const mxArray* declarations, const mxArray* constraints) {
    require(mxIsStruct(declarations) && mxGetNumberOfElements(declarations) > 0,
            "space must be a nonempty struct array");
    n4m_search_space_t* raw = nullptr;
    check(n4m_search_space_create(&raw), nullptr, "search_space_create");
    std::unique_ptr<n4m_search_space_t, decltype(&n4m_search_space_destroy)>
        space(raw, &n4m_search_space_destroy);
    for (size_t i = 0; i < mxGetNumberOfElements(declarations); ++i)
        add_axis(space.get(), declarations, i);
    add_constraints(space.get(), constraints);
    return space;
}
void set_options(n4m_optimizer_options_t& out, const mxArray* source) {
    n4m_optimizer_options_init(&out);
    if (!source || mxIsEmpty(source)) return;
    require(mxIsStruct(source) && mxGetNumberOfElements(source) == 1,
            "options must be one struct");
    const mxArray* value = nullptr;
    if ((value = optional(source, "sampler")))
        out.sampler = static_cast<n4m_sampler_kind_t>(choice(text(value),
            {"random", "sobol", "lhs", "ternary", "ga", "pso", "cmaes", "tpe", "gp_ei"}));
    if ((value = optional(source, "pruner")))
        out.pruner = static_cast<n4m_pruner_kind_t>(choice(text(value),
            {"none", "median", "asha", "hyperband", "racing"}));
    if ((value = optional(source, "direction")))
        out.direction = static_cast<n4m_opt_direction_t>(choice(text(value),
            {"auto", "minimize", "maximize"}));
    if ((value = optional(source, "eval_mode")))
        out.eval_mode = static_cast<n4m_eval_mode_t>(choice(text(value),
            {"best", "mean", "robust_best"}));
    if ((value = optional(source, "liar")))
        out.liar = static_cast<n4m_liar_kind_t>(choice(text(value),
            {"none", "min", "mean", "max"}));
    if ((value = optional(source, "metric"))) {
        const std::string metric = text(value);
        const std::vector<std::string> names = {"rmse", "mse", "mae", "r2", "accuracy",
            "balanced_accuracy", "f1", "logloss"};
        const int index = choice(metric, names);
        out.metric = static_cast<n4m_metric_t>(index < 4 ? index : index + 12);
    }
    if ((value = optional(source, "n_startup_trials"))) out.n_startup_trials = i32(value);
    if ((value = optional(source, "seed"))) out.seed = u64(value);
    if ((value = optional(source, "timeout_seconds"))) out.timeout_seconds = number(value);
    if ((value = optional(source, "max_resource"))) out.max_resource = i32(value);
    if ((value = optional(source, "reduction_factor"))) out.reduction_factor = i32(value);
}
uint64_t put(std::unique_ptr<Entry> entry, const mxArray* declarations) {
    require(next_id != 0, "optimizer handle registry exhausted");
    entry->space = mxDuplicateArray(declarations);
    require(entry->space != nullptr, "could not retain optimizer space");
    mexMakeArrayPersistent(entry->space);
    if (!registered) { mexAtExit(cleanup); registered = true; }
    const uint64_t id = next_id++;
    entries.emplace(id, std::move(entry));
    mexLock();
    return id;
}
Entry& lookup(const mxArray* handle) {
    require(handle && mxIsUint64(handle) && mxGetNumberOfElements(handle) == 1,
            "expected an optimizer uint64 handle");
    auto found = entries.find(u64(handle));
    require(found != entries.end(), "closed or unknown optimizer handle");
    return *found->second;
}
mxArray* category(const mxArray* choices, int32_t index) {
    require(index >= 0 && static_cast<size_t>(index) < mxGetNumberOfElements(choices),
            "native categorical index is invalid");
    if (mxIsCell(choices)) return mxDuplicateArray(mxGetCell(choices, index));
    if (mxIsLogical(choices))
        return mxCreateLogicalScalar(mxGetLogicals(choices)[index]);
    if (mxIsInt64(choices)) {
        mxArray* result = scalar_i64(static_cast<const int64_t*>(mxGetData(choices))[index]);
        return result;
    }
    return mxCreateDoubleScalar(mxGetPr(choices)[index]);
}
bool valid_field_name(const std::string& name) {
    if (name.empty() || name.size() > 63) return false;
    const auto letter = [](char c) { return (c >= 'A' && c <= 'Z') ||
                                           (c >= 'a' && c <= 'z'); };
    if (!letter(name[0])) return false;
    for (size_t i = 1; i < name.size(); ++i)
        if (!letter(name[i]) && !(name[i] >= '0' && name[i] <= '9') && name[i] != '_')
            return false;
    return true;
}
mxArray* trial_record(n4m_trial_t* trial, Entry& entry) {
    const char* fields[] = {"id", "parameters", "parameter_names", "parameter_values",
                            "status", "rung"};
    mxArray* result = mxCreateStructMatrix(1, 1, 6, fields);
    int64_t id = 0;
    int32_t rung = 0;
    n4m_trial_status_t status = N4M_TRIAL_RUNNING;
    check(n4m_trial_get_id(trial, &id), entry.context, "trial_get_id");
    check(n4m_trial_get_rung(trial, &rung), entry.context, "trial_get_rung");
    check(n4m_trial_get_status(trial, &status), entry.context, "trial_get_status");
    mxSetField(result, 0, "id", scalar_i64(id));
    mxSetField(result, 0, "status", scalar_i32(static_cast<int32_t>(status)));
    mxSetField(result, 0, "rung", scalar_i32(rung));
    const size_t count = mxGetNumberOfElements(entry.space);
    std::vector<std::string> names;
    std::vector<const char*> pointers;
    names.reserve(count);
    pointers.reserve(count);
    for (size_t i = 0; i < count; ++i)
        names.push_back(text(field(entry.space, i, "name")));
    for (const auto& name : names) pointers.push_back(name.c_str());
    const bool can_make_struct = count <= INT32_MAX &&
        std::all_of(names.begin(), names.end(), valid_field_name);
    mxArray* params = can_make_struct ?
        mxCreateStructMatrix(1, 1, static_cast<int>(count), pointers.data()) :
        mxCreateDoubleMatrix(0, 0, mxREAL);
    mxArray* names_out = mxCreateCellMatrix(1, static_cast<mwSize>(count));
    mxArray* values_out = mxCreateCellMatrix(1, static_cast<mwSize>(count));
    for (size_t i = 0; i < count; ++i) {
        const std::string kind = text(field(entry.space, i, "kind"));
        const std::string& name = names[i];
        const std::string active_name = kind == "sorted_tuple" ? name + "#0" : name;
        int32_t active = 0;
        check(n4m_trial_is_active(trial, active_name.c_str(), &active),
              entry.context, "trial_is_active");
        mxArray* value = nullptr;
        if (!active) value = mxCreateDoubleMatrix(0, 0, mxREAL);
        else if (kind == "int" || kind == "log_int") {
            int64_t n = 0;
            check(n4m_trial_get_int(trial, name.c_str(), &n), entry.context, "trial_get_int");
            value = scalar_i64(n);
        } else if (kind == "float" || kind == "log_float" || kind == "ordinal") {
            double n = 0;
            check(n4m_trial_get_float(trial, name.c_str(), &n), entry.context, "trial_get_float");
            value = mxCreateDoubleScalar(n);
        } else if (kind == "categorical") {
            int32_t index = -1;
            check(n4m_trial_get_category(trial, name.c_str(), &index, nullptr),
                  entry.context, "trial_get_category");
            value = category(field(entry.space, i, "choices"), index);
        } else if (kind == "sorted_tuple") {
            const int32_t length = i32(field(entry.space, i, "length"));
            require(length > 0, "invalid tuple length");
            value = mxCreateDoubleMatrix(1, static_cast<mwSize>(length), mxREAL);
            for (int32_t j = 0; j < length; ++j) {
                const std::string component = name + "#" + std::to_string(j);
                check(n4m_trial_get_float(trial, component.c_str(), &mxGetPr(value)[j]),
                      entry.context, "trial_get_tuple_component");
            }
        } else throw std::runtime_error("unknown saved optimizer axis kind");
        mxSetCell(names_out, i, mxCreateString(name.c_str()));
        if (can_make_struct)
            mxSetFieldByNumber(params, 0, static_cast<int>(i), mxDuplicateArray(value));
        mxSetCell(values_out, i, value);
    }
    mxSetField(result, 0, "parameters", params);
    mxSetField(result, 0, "parameter_names", names_out);
    mxSetField(result, 0, "parameter_values", values_out);
    return result;
}
mxArray* trace_entry(const n4m_method_result_t* trace, const char* name, int32_t kind,
                     n4m_context_t* context) {
    if (kind == N4M_RESULT_SCALAR) {
        double value = 0;
        check(n4m_method_result_get_scalar(trace, name, &value), context, name);
        return mxCreateDoubleScalar(value);
    }
    if (kind == N4M_RESULT_INT_VECTOR) {
        const int32_t* values = nullptr;
        int32_t size = 0;
        check(n4m_method_result_get_int_vector(trace, name, &values, &size), context, name);
        require(size >= 0, "invalid native trace vector size");
        mxArray* result = mxCreateNumericMatrix(1, static_cast<mwSize>(size), mxINT32_CLASS, mxREAL);
        if (size) std::memcpy(mxGetData(result), values, static_cast<size_t>(size) * sizeof(int32_t));
        return result;
    }
    if (kind == N4M_RESULT_INT64_VECTOR) {
        const int64_t* values = nullptr;
        int64_t size = 0;
        check(n4m_method_result_get_int64_vector(trace, name, &values, &size), context, name);
        require(size >= 0 && static_cast<uint64_t>(size) <= std::numeric_limits<mwSize>::max(),
                "invalid native trace int64 vector size");
        mxArray* result = mxCreateNumericMatrix(1, static_cast<mwSize>(size), mxINT64_CLASS, mxREAL);
        if (size) std::memcpy(mxGetData(result), values, static_cast<size_t>(size) * sizeof(int64_t));
        return result;
    }
    if (kind == N4M_RESULT_DOUBLE_MATRIX) {
        const double* values = nullptr;
        int64_t rows = 0, cols = 0;
        check(n4m_method_result_get_double_matrix(trace, name, &values, &rows, &cols),
              context, name);
        require(rows >= 0 && cols >= 0 &&
                static_cast<uint64_t>(rows) <= std::numeric_limits<mwSize>::max() &&
                static_cast<uint64_t>(cols) <= std::numeric_limits<mwSize>::max() &&
                (cols == 0 || static_cast<uint64_t>(rows) <=
                                  std::numeric_limits<mwSize>::max() /
                                      static_cast<uint64_t>(cols)),
                "invalid native trace matrix dimensions");
        mxArray* result = mxCreateDoubleMatrix(static_cast<mwSize>(rows),
                                               static_cast<mwSize>(cols), mxREAL);
        for (int64_t col = 0; col < cols; ++col)
            for (int64_t row = 0; row < rows; ++row)
                mxGetPr(result)[static_cast<size_t>(col * rows + row)] = values[row * cols + col];
        return result;
    }
    throw std::runtime_error("unknown native trace entry kind");
}
mxArray* trace_snapshot(Entry& entry, int64_t since_id) {
    n4m_method_result_t* raw = nullptr;
    check(n4m_optimizer_get_trials(entry.optimizer, since_id, &raw),
          entry.context, "optimizer_get_trials");
    std::unique_ptr<n4m_method_result_t, decltype(&n4m_method_result_destroy)>
        trace(raw, &n4m_method_result_destroy);
    int32_t count = 0;
    check(n4m_method_result_entry_count(trace.get(), &count), entry.context, "trace_entry_count");
    require(count >= 0, "invalid native trace entry count");
    std::vector<std::string> names;
    std::vector<int32_t> kinds;
    std::vector<const char*> pointers;
    names.reserve(count);
    kinds.reserve(count);
    pointers.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
        const char* name = nullptr;
        int32_t kind = -1;
        check(n4m_method_result_entry(trace.get(), i, &name, &kind), entry.context, "trace_entry");
        require(name != nullptr, "invalid native trace entry name");
        names.emplace_back(name);
        kinds.push_back(kind);
    }
    for (const auto& name : names) pointers.push_back(name.c_str());
    mxArray* result = mxCreateStructMatrix(1, 1, count, pointers.data());
    for (int32_t i = 0; i < count; ++i)
        mxSetFieldByNumber(result, 0, i, trace_entry(trace.get(), names[i].c_str(),
                                                    kinds[i], entry.context));
    return result;
}
mxArray* checkpoint(Entry& entry) {
    n4m_array_t* raw = nullptr;
    check(n4m_optimizer_save(entry.optimizer, &raw), entry.context, "optimizer_save");
    std::unique_ptr<n4m_array_t, decltype(&n4m_array_free)> blob(raw, &n4m_array_free);
    n4m_matrix_view_t view{};
    check(n4m_array_view(blob.get(), &view), entry.context, "optimizer_save_view");
    require(view.dtype == N4M_DTYPE_I64 && view.rows == 1 && view.cols > 0 && view.data &&
            static_cast<uint64_t>(view.cols) <= std::numeric_limits<mwSize>::max() / 8,
            "invalid native optimizer checkpoint array");
    const mwSize bytes = static_cast<mwSize>(view.cols) * 8;
    mxArray* result = mxCreateNumericMatrix(1, bytes, mxUINT8_CLASS, mxREAL);
    std::memcpy(mxGetData(result), view.data, bytes);
    return result;
}
void dispatch(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs >= 1, "optimizer command required");
    const std::string command = text(prhs[0]);
    if (command == "create") {
        require(nrhs >= 2 && nrhs <= 4 && nlhs == 1, "create(space, options, constraints)");
        const mxArray* constraints = nrhs >= 4 ? prhs[3] : nullptr;
        auto space = build_space(prhs[1], constraints);
        n4m_optimizer_options_t options{};
        set_options(options, nrhs >= 3 ? prhs[2] : nullptr);
        std::unique_ptr<Entry> entry(new Entry());
        check(n4m_context_create(&entry->context), nullptr, "context_create");
        check(n4m_optimizer_create(entry->context, space.get(), &options, &entry->optimizer),
              entry->context, "optimizer_create");
        plhs[0] = scalar_u64(put(std::move(entry), prhs[1]));
        return;
    }
    if (command == "load") {
        require(nrhs == 3 && nlhs == 1, "load(bytes, space)");
        require(mxIsUint8(prhs[1]) && mxGetNumberOfElements(prhs[1]) > 0,
                "checkpoint must be a nonempty uint8 vector");
        require(mxIsStruct(prhs[2]) && mxGetNumberOfElements(prhs[2]) > 0,
                "load needs the original search-space declarations");
        std::unique_ptr<Entry> entry(new Entry());
        check(n4m_context_create(&entry->context), nullptr, "context_create");
        check(n4m_optimizer_load(entry->context,
              static_cast<const uint8_t*>(mxGetData(prhs[1])),
              static_cast<uint64_t>(mxGetNumberOfElements(prhs[1])), &entry->optimizer),
              entry->context, "optimizer_load");
        plhs[0] = scalar_u64(put(std::move(entry), prhs[2]));
        return;
    }
    require(nrhs >= 2, "optimizer handle required");
    if (command == "close") {
        require(nrhs == 2 && nlhs == 0, "close(handle)");
        require(mxIsUint64(prhs[1]), "expected a uint64 handle");
        auto found = entries.find(u64(prhs[1]));
        if (found != entries.end()) { entries.erase(found); mexUnlock(); }
        return;
    }
    Entry& entry = lookup(prhs[1]);
    if (command == "ask") {
        require(nrhs == 2 && nlhs == 1, "ask(handle)");
        n4m_trial_t* trial = nullptr;
        check(n4m_optimizer_ask(entry.optimizer, &trial), entry.context, "optimizer_ask");
        plhs[0] = trial_record(trial, entry);
    } else if (command == "ask_batch") {
        require(nrhs == 3 && nlhs >= 1 && nlhs <= 2, "ask_batch(handle, n)");
        const int32_t n = i32(prhs[2]);
        require(n >= 0 && n <= 10000, "batch size must be between 0 and 10000");
        std::vector<n4m_trial_t*> trials(static_cast<size_t>(n));
        int32_t count = 0;
        const n4m_status_t status = n4m_optimizer_ask_batch(entry.optimizer, n,
            trials.empty() ? nullptr : trials.data(), &count);
        require(count >= 0 && count <= n, "invalid native committed batch count");
        if (status != N4M_OK && count == 0) check(status, entry.context, "optimizer_ask_batch");
        plhs[0] = mxCreateCellMatrix(1, static_cast<mwSize>(count));
        for (int32_t i = 0; i < count; ++i)
            mxSetCell(plhs[0], i, trial_record(trials[i], entry));
        if (nlhs == 2) plhs[1] = scalar_i32(status);
    } else if (command == "enqueue") {
        require(nrhs == 4 && nlhs == 0, "enqueue(handle, names, values)");
        require(mxIsCell(prhs[2]), "enqueue names must be a cell array");
        const auto values = doubles(prhs[3]);
        require(values.size() == mxGetNumberOfElements(prhs[2]) && !values.empty() &&
                values.size() <= INT32_MAX, "enqueue names and values must match");
        std::vector<std::string> names;
        std::vector<const char*> pointers;
        names.reserve(values.size());
        pointers.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i)
            names.push_back(text(mxGetCell(prhs[2], i)));
        for (const auto& name : names) pointers.push_back(name.c_str());
        check(n4m_optimizer_enqueue(entry.optimizer, pointers.data(), values.data(),
              static_cast<int32_t>(values.size())), entry.context, "optimizer_enqueue");
    } else if (command == "tell") {
        require(nrhs >= 5 && nrhs <= 6 && nlhs == 0,
                "tell(handle, id, status, score, error)");
        const int64_t id = i64(prhs[2]);
        const int status = choice(text(prhs[3]),
            {"running", "completed", "pruned", "failed", "cancelled"});
        require(status != 0, "running is not a terminal status");
        require(status == N4M_TRIAL_COMPLETED ? !mxIsEmpty(prhs[4]) : mxIsEmpty(prhs[4]),
                "completed trials need a score; other terminal states reject scores");
        const double score = mxIsEmpty(prhs[4]) ? 0 : number(prhs[4]);
        const std::string error = nrhs == 6 && !mxIsEmpty(prhs[5]) ? text(prhs[5]) : "";
        check(n4m_optimizer_tell_result(entry.optimizer, id,
              static_cast<n4m_trial_status_t>(status), score,
              error.empty() ? nullptr : error.c_str()), entry.context, "optimizer_tell_result");
    } else if (command == "intermediate") {
        require(nrhs == 5 && nlhs == 1, "intermediate(handle, id, step, score)");
        int32_t prune = 0;
        check(n4m_optimizer_tell_intermediate(entry.optimizer, i64(prhs[2]), i32(prhs[3]),
              number(prhs[4]), &prune), entry.context, "optimizer_tell_intermediate");
        require(prune == 0 || prune == 1, "invalid native pruning decision");
        plhs[0] = mxCreateLogicalScalar(prune != 0);
    } else if (command == "best") {
        require(nrhs == 2 && nlhs == 1, "best(handle)");
        n4m_trial_t* trial = nullptr;
        double score = 0;
        const n4m_status_t status = n4m_optimizer_best(entry.optimizer, &trial, &score);
        if (status == N4M_ERR_NOT_FITTED) { plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL); return; }
        check(status, entry.context, "optimizer_best");
        const char* fields[] = {"trial", "score"};
        plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
        mxSetField(plhs[0], 0, "trial", trial_record(trial, entry));
        mxSetField(plhs[0], 0, "score", mxCreateDoubleScalar(score));
    } else if (command == "trials") {
        require(nrhs == 3 && nlhs == 1, "trials(handle, since_id)");
        const int64_t since_id = i64(prhs[2]);
        require(since_id >= 0, "since_id must be nonnegative");
        plhs[0] = trace_snapshot(entry, since_id);
    } else if (command == "save") {
        require(nrhs == 2 && nlhs == 1, "save(handle)");
        plhs[0] = checkpoint(entry);
    } else throw std::runtime_error("unknown optimizer command: " + command);
}
}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try { dispatch(nlhs, plhs, nrhs, prhs); }
    catch (const std::exception& error) {
        mexErrMsgIdAndTxt("n4m:optimizer", "%s", error.what());
    }
}
