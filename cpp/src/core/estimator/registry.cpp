// SPDX-License-Identifier: CECILL-2.1
//
// Compiled method manifest and typed parameter storage.

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "core/common/context.hpp"
#include "core/estimator/generated_factories.hpp"
#include "core/estimator/spec.hpp"

namespace n4m::estimator {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// Namespace-scope const arrays have internal linkage; the factory
// declarations resolve to the adapter translation units.
#include "core/estimator/generated_manifest.inc"

namespace {
constexpr std::int32_t kMethodCount =
    static_cast<std::int32_t>(sizeof(kMethods) / sizeof(kMethods[0]));

bool is_int_type(n4m_method_param_type_t t) noexcept {
    return t == N4M_METHOD_PARAM_INT || t == N4M_METHOD_PARAM_BOOL || t == N4M_METHOD_PARAM_ENUM ||
           t == N4M_METHOD_PARAM_INT_ARRAY;
}
bool is_array_type(n4m_method_param_type_t t) noexcept {
    return t == N4M_METHOD_PARAM_INT_ARRAY || t == N4M_METHOD_PARAM_DOUBLE_ARRAY;
}
bool in_bounds(const ParamSpec& p, double v) noexcept {
    return std::isfinite(v) && (std::isnan(p.min_value) || v >= p.min_value) &&
           (std::isnan(p.max_value) || v <= p.max_value);
}
}  // namespace

std::int32_t method_count() noexcept { return kMethodCount; }

const MethodSpec* method_at(std::int32_t index) noexcept {
    return index >= 0 && index < kMethodCount ? &kMethods[index] : nullptr;
}

std::int32_t method_index(const char* method_id) noexcept {
    if (method_id == nullptr) return -1;
    for (std::int32_t i = 0; i < kMethodCount; ++i) {
        if (std::strcmp(kMethods[i].method_id, method_id) == 0) return i;
    }
    return -1;
}

std::int32_t param_index(const MethodSpec& spec, const char* name) noexcept {
    if (name == nullptr) return -1;
    for (std::int32_t i = 0; i < spec.n_params; ++i) {
        if (std::strcmp(spec.params[i].name, name) == 0) return i;
    }
    return -1;
}

Params::Params(const MethodSpec& spec) : spec_(&spec), values_(static_cast<std::size_t>(spec.n_params)) {}

const ParamSpec& Params::checked(const char* name, std::int32_t* index) const {
    const std::int32_t i = param_index(*spec_, name);
    if (i < 0) throw std::logic_error("unknown estimator parameter");
    *index = i;
    return spec_->params[i];
}

n4m_status_t Params::set_ints(const char* name, n4m_method_param_type_t type, const std::int64_t* v,
                              std::int64_t n) {
    const std::int32_t i = param_index(*spec_, name);
    if (i < 0 || v == nullptr) return i < 0 ? N4M_ERR_INVALID_ARGUMENT : N4M_ERR_NULL_POINTER;
    const ParamSpec& p = spec_->params[i];
    if (p.type != type || (is_array_type(type) ? n < 0 : n != 1)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (std::int64_t k = 0; k < n; ++k) {
        const double as_double = static_cast<double>(v[k]);
        if (type == N4M_METHOD_PARAM_BOOL && v[k] != 0 && v[k] != 1) return N4M_ERR_INVALID_ARGUMENT;
        if ((type == N4M_METHOD_PARAM_INT || type == N4M_METHOD_PARAM_INT_ARRAY) && !in_bounds(p, as_double)) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (type == N4M_METHOD_PARAM_ENUM && (v[k] < 0 || v[k] >= p.n_choices)) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    ParamValue& value = values_[static_cast<std::size_t>(i)];
    value.ints.assign(v, v + n);
    value.doubles.clear();
    value.set = true;
    return N4M_OK;
}

n4m_status_t Params::set_doubles(const char* name, n4m_method_param_type_t type, const double* v,
                                 std::int64_t n) {
    const std::int32_t i = param_index(*spec_, name);
    if (i < 0 || v == nullptr) return i < 0 ? N4M_ERR_INVALID_ARGUMENT : N4M_ERR_NULL_POINTER;
    const ParamSpec& p = spec_->params[i];
    if (p.type != type || (is_array_type(type) ? n < 0 : n != 1)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (std::int64_t k = 0; k < n; ++k) {
        if (!in_bounds(p, v[k])) return N4M_ERR_INVALID_ARGUMENT;
    }
    ParamValue& value = values_[static_cast<std::size_t>(i)];
    value.doubles.assign(v, v + n);
    value.ints.clear();
    value.set = true;
    return N4M_OK;
}

n4m_status_t Params::set_enum(const char* name, const char* choice) {
    const std::int32_t i = param_index(*spec_, name);
    if (i < 0 || choice == nullptr) return i < 0 ? N4M_ERR_INVALID_ARGUMENT : N4M_ERR_NULL_POINTER;
    const ParamSpec& p = spec_->params[i];
    if (p.type != N4M_METHOD_PARAM_ENUM) return N4M_ERR_INVALID_ARGUMENT;
    for (std::int64_t k = 0; k < p.n_choices; ++k) {
        if (std::strcmp(p.choices[k], choice) == 0) return set_ints(name, N4M_METHOD_PARAM_ENUM, &k, 1);
    }
    return N4M_ERR_INVALID_ARGUMENT;
}

const char* Params::missing_required() const noexcept {
    for (std::int32_t i = 0; i < spec_->n_params; ++i) {
        if (!values_[static_cast<std::size_t>(i)].set && !spec_->params[i].has_default) return spec_->params[i].name;
    }
    return nullptr;
}

bool Params::resolved(std::int32_t index, std::vector<std::int64_t>* ints,
                      std::vector<double>* doubles) const {
    const ParamSpec& p = spec_->params[index];
    const ParamValue& v = values_[static_cast<std::size_t>(index)];
    if (v.set) {
        *ints = v.ints;
        *doubles = v.doubles;
        return true;
    }
    if (!p.has_default) return false;
    if (is_int_type(p.type)) {
        ints->assign(p.default_int, p.default_int + p.default_length);
        doubles->clear();
    } else {
        doubles->assign(p.default_double, p.default_double + p.default_length);
        ints->clear();
    }
    return true;
}

const std::vector<std::int64_t>& Params::get_ints(const char* name) const {
    std::int32_t i = 0;
    const ParamSpec& p = checked(name, &i);
    if (!is_int_type(p.type) || !resolved(i, &scratch_ints_, &scratch_doubles_)) {
        throw std::logic_error("estimator parameter is not an integer or has no value");
    }
    return scratch_ints_;
}

const std::vector<double>& Params::get_doubles(const char* name) const {
    std::int32_t i = 0;
    const ParamSpec& p = checked(name, &i);
    if (is_int_type(p.type) || !resolved(i, &scratch_ints_, &scratch_doubles_)) {
        throw std::logic_error("estimator parameter is not a double or has no value");
    }
    return scratch_doubles_;
}

std::int64_t Params::get_int(const char* name) const { return get_ints(name).at(0); }
double Params::get_double(const char* name) const { return get_doubles(name).at(0); }

void set_error(n4m_context_t* ctx, const char* message) noexcept {
    if (ctx != nullptr) ctx->set_error(message);
}

void set_error_named(n4m_context_t* ctx, const char* what, const char* name) noexcept {
    if (ctx != nullptr) ctx->set_errorf("%s '%s'", what, name != nullptr ? name : "");
}

}  // namespace n4m::estimator
