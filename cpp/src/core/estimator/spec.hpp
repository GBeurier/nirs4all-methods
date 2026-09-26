// SPDX-License-Identifier: CECILL-2.1
//
// Internal types of the generic estimator surface (n4m/estimator.h):
// method specs compiled from the catalog, typed parameter values, the
// normalized fit inputs and the Adapter interface every method implements.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "n4m/n4m.h"
#include "n4m/estimator.h"

namespace n4m::estimator {

struct ParamSpec {
    const char* name;
    n4m_method_param_type_t type;
    bool has_default;
    const std::int64_t* default_int;  // INT/BOOL/ENUM/INT_ARRAY
    const double* default_double;     // DOUBLE/DOUBLE_ARRAY
    std::int64_t default_length;
    double min_value;  // NaN = unbounded
    double max_value;
    const char* const* choices;
    std::int32_t n_choices;
};

class Adapter;
struct MethodSpec;
using AdapterFactory = std::unique_ptr<Adapter> (*)(const MethodSpec&);

struct MethodSpec {
    const char* method_id;
    const char* fq_name;
    n4m_method_kind_t kind;
    std::uint32_t roles;
    const ParamSpec* params;
    std::int32_t n_params;
    n4m_input_requirement_t inputs[N4M_FIT_INPUT_COUNT];
    const char* state_format;
    AdapterFactory factory;
};

// Registry over the generated table.
std::int32_t method_count() noexcept;
const MethodSpec* method_at(std::int32_t index) noexcept;
std::int32_t method_index(const char* method_id) noexcept;  // -1 if unknown
std::int32_t param_index(const MethodSpec& spec, const char* name) noexcept;

// One resolved-or-unset value per spec parameter. Integer-like types store
// into `ints`, floating types into `doubles`.
struct ParamValue {
    bool set = false;
    std::vector<std::int64_t> ints;
    std::vector<double> doubles;
};

class Params {
  public:
    explicit Params(const MethodSpec& spec);
    const MethodSpec& spec() const noexcept { return *spec_; }

    n4m_status_t set_ints(const char* name, n4m_method_param_type_t type, const std::int64_t* v,
                          std::int64_t n);
    n4m_status_t set_doubles(const char* name, n4m_method_param_type_t type, const double* v,
                             std::int64_t n);
    n4m_status_t set_enum(const char* name, const char* choice);

    // First missing required parameter, or nullptr.
    const char* missing_required() const noexcept;

    // Resolved values (explicit or default). Callers use names from the spec;
    // an unknown name or a missing required value is a programming error
    // reported as an exception caught at the C boundary.
    std::int64_t get_int(const char* name) const;
    double get_double(const char* name) const;
    bool get_bool(const char* name) const { return get_int(name) != 0; }
    std::vector<std::int64_t> get_ints(const char* name) const;
    std::vector<double> get_doubles(const char* name) const;

    // Resolved value by index (for serialization and copies).
    bool resolved(std::int32_t index, std::vector<std::int64_t>* ints,
                  std::vector<double>* doubles) const;
    const ParamValue& raw(std::int32_t index) const { return values_[static_cast<std::size_t>(index)]; }
    ParamValue& raw(std::int32_t index) { return values_[static_cast<std::size_t>(index)]; }

  private:
    const ParamSpec& checked(const char* name, std::int32_t* index) const;
    const MethodSpec* spec_;
    std::vector<ParamValue> values_;
};

// Fit inputs with every optional field normalized (absent = null/0).
struct FitInputs {
    const n4m_matrix_view_t* X = nullptr;
    const n4m_matrix_view_t* Y = nullptr;
    const std::int64_t* labels = nullptr;
    std::int64_t n_labels = 0;
    const double* sample_weight = nullptr;
    std::int64_t n_sample_weight = 0;
    const std::int64_t* groups = nullptr;
    std::int64_t n_groups = 0;
    const std::int64_t* feature_groups = nullptr;
    std::int64_t n_feature_groups = 0;
    const std::int64_t* block_sizes = nullptr;
    std::int64_t n_blocks = 0;
    const double* axis = nullptr;
    std::int64_t n_axis = 0;
    const n4m_matrix_view_t* X_target = nullptr;
    const std::int64_t* fold_ids = nullptr;
    std::int64_t n_fold_ids = 0;
};

// One state block of an N4ME payload.
struct StateBlock {
    std::uint32_t tag = 0;
    std::vector<unsigned char> bytes;
};

class Adapter {
  public:
    virtual ~Adapter() = default;

    // Capabilities of the fitted state (N4M_CAP_*). Stable per adapter class
    // except for data-dependent flags such as RETAINS_TRAINING_ROWS.
    virtual std::uint64_t capabilities() const noexcept = 0;
    virtual n4m_status_t fit(n4m_context_t* ctx, const Params& params,
                             const FitInputs& inputs) = 0;
    virtual std::int64_t n_features_in() const noexcept = 0;
    virtual std::int64_t n_outputs() const noexcept = 0;
    virtual std::int64_t transform_cols() const noexcept { return 0; }

    virtual n4m_status_t transform(n4m_context_t*, const n4m_matrix_view_t&,
                                   n4m_matrix_view_t&) const {
        return N4M_ERR_UNSUPPORTED;
    }
    virtual n4m_status_t predict(n4m_context_t*, const n4m_matrix_view_t&,
                                 n4m_matrix_view_t&) const {
        return N4M_ERR_UNSUPPORTED;
    }
    virtual n4m_status_t decision_function(n4m_context_t*, const n4m_matrix_view_t&,
                                           n4m_matrix_view_t&) const {
        return N4M_ERR_UNSUPPORTED;
    }
    virtual n4m_status_t predict_proba(n4m_context_t*, const n4m_matrix_view_t&,
                                       n4m_matrix_view_t&) const {
        return N4M_ERR_UNSUPPORTED;
    }
    virtual n4m_status_t predict_labels(n4m_context_t*, const n4m_matrix_view_t&,
                                        std::int64_t*) const {
        return N4M_ERR_UNSUPPORTED;
    }
    virtual const std::vector<std::int64_t>* classes() const noexcept { return nullptr; }
    virtual const std::vector<std::int64_t>* selected_indices() const noexcept {
        return nullptr;
    }
    virtual n4m_status_t apply_mask(n4m_context_t*, const n4m_matrix_view_t&,
                                    const n4m_matrix_view_t*, std::uint8_t*) const {
        return N4M_ERR_UNSUPPORTED;
    }
    virtual const n4m_method_result_t* fit_result() const noexcept { return nullptr; }

    virtual n4m_status_t save_state(n4m_context_t* ctx, std::vector<StateBlock>& out) const = 0;
    virtual n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                                    const std::vector<StateBlock>& blocks) = 0;
};

// Shared helpers for adapters.
void set_error(n4m_context_t* ctx, const char* message) noexcept;
// Message "<what> '<name>'".
void set_error_named(n4m_context_t* ctx, const char* what, const char* name) noexcept;

}  // namespace n4m::estimator

struct n4m_params_s {
    explicit n4m_params_s(const ::n4m::estimator::MethodSpec& spec) : params(spec) {}
    ::n4m::estimator::Params params;
};

struct n4m_estimator_s {
    n4m_estimator_s(std::int32_t index, const ::n4m::estimator::Params& p,
                    std::unique_ptr<::n4m::estimator::Adapter> a)
        : method_index(index), params(p), adapter(std::move(a)) {}
    std::int32_t method_index;
    ::n4m::estimator::Params params;
    std::unique_ptr<::n4m::estimator::Adapter> adapter;
    bool fitted = false;
};

namespace n4m::estimator {

// N4ME wire format v1 (docs/abi/estimator_roles_design.md, D6).
n4m_status_t encode_state(n4m_context_t* ctx, const n4m_estimator_s& est,
                          std::vector<unsigned char>& out);
n4m_status_t decode_state(n4m_context_t* ctx, const unsigned char* bytes, std::size_t size,
                          std::uint64_t max_bytes, std::unique_ptr<n4m_estimator_s>& out);

}  // namespace n4m::estimator
