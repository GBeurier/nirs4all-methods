// SPDX-License-Identifier: CECILL-2.1
//
// Role pipeline: recipe validation, fit-input routing and row movement
// between steps. Every step is created, fitted, applied and serialized by
// the generic estimator surface (n4m_estimator_*); nothing numerical here.

#include "core/estimator/role_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "core/common/context.hpp"

namespace n4m::estimator {

namespace {

constexpr const char* kInputNames[N4M_FIT_INPUT_COUNT] = {
    "y", "labels", "sample_weight", "groups", "feature_groups",
    "blocks", "axis", "target_domain", "fold_ids"};

constexpr std::uint32_t kIntermediateRoles = N4M_ROLE_TRANSFORMER | N4M_ROLE_SELECTOR;
constexpr std::uint32_t kTerminalRoles = N4M_ROLE_REGRESSOR | N4M_ROLE_CLASSIFIER;

bool column_input(int k) noexcept {
    return k == N4M_FIT_INPUT_FEATURE_GROUPS || k == N4M_FIT_INPUT_BLOCKS ||
           k == N4M_FIT_INPUT_AXIS || k == N4M_FIT_INPUT_TARGET_DOMAIN;
}

const MethodSpec& spec_of(const n4m_role_pipeline_s::Step& step) {
    return *method_at(step.method_index);
}

// Clears the context message before a nested call, so a status without a
// message never picks up stale text.
template <typename Fn>
n4m_status_t nested(n4m_context_t* ctx, Fn&& fn) {
    ctx->clear_error();
    return fn();
}

n4m_matrix_view_t row_major(std::vector<double>& data, std::int64_t rows, std::int64_t cols) {
    n4m_matrix_view_t v{};
    v.data = data.data();
    v.rows = rows;
    v.cols = cols;
    v.row_stride = cols;
    v.col_stride = 1;
    v.dtype = N4M_DTYPE_F64;
    return v;
}

double at(const n4m_matrix_view_t& v, std::int64_t i, std::int64_t j) {
    return static_cast<const double*>(v.data)[i * v.row_stride + j * v.col_stride];
}

n4m_status_t check_view(n4m_context_t* ctx, const n4m_matrix_view_t* v, const char* name) {
    if (v == nullptr) {
        ctx->set_errorf("matrix '%s' is NULL", name);
        return N4M_ERR_NULL_POINTER;
    }
    const n4m_status_t st = n4m_matrix_view_validate(v);
    if (st != N4M_OK) {
        ctx->set_errorf("matrix '%s' is not a valid view", name);
        return st;
    }
    if (v->dtype != N4M_DTYPE_F64) {
        ctx->set_errorf("matrix '%s' must hold float64 values", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    return N4M_OK;
}

// Training rows moving through the sample filters; the caller's buffers
// until a filter drops rows, owned copies afterwards.
struct TrainingRows {
    n4m_fit_inputs_v1_t in{};
    n4m_matrix_view_t X{}, Y{}, label_column{};
    std::vector<double> x_store, y_store, weights_store, label_column_store;
    std::vector<std::int64_t> labels_store, groups_store, folds_store;

    std::int64_t rows() const noexcept { return X.rows; }

    // The class ids as one double column (targets of the non-terminal
    // steps of a classification recipe that require y).
    const n4m_matrix_view_t* labels_as_column() {
        label_column_store.resize(static_cast<std::size_t>(rows()));
        for (std::int64_t i = 0; i < rows(); ++i) {
            label_column_store[static_cast<std::size_t>(i)] = static_cast<double>(in.labels[i]);
        }
        label_column = row_major(label_column_store, rows(), 1);
        return &label_column;
    }

    void keep(const std::vector<std::int64_t>& kept) {
        auto gather_matrix = [&](const n4m_matrix_view_t& src, std::vector<double>& store,
                                 n4m_matrix_view_t& dst) {
            std::vector<double> out(kept.size() * static_cast<std::size_t>(src.cols));
            std::size_t k = 0;
            for (std::int64_t r : kept) {
                for (std::int64_t j = 0; j < src.cols; ++j) out[k++] = at(src, r, j);
            }
            store = std::move(out);
            dst = row_major(store, static_cast<std::int64_t>(kept.size()), src.cols);
        };
        auto gather = [&](const auto* src, auto& store) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(src)>>;
            std::vector<T> out;
            out.reserve(kept.size());
            for (std::int64_t r : kept) out.push_back(src[r]);
            store = std::move(out);
            return store.data();
        };
        const auto n = static_cast<std::int64_t>(kept.size());
        gather_matrix(X, x_store, X);
        if (in.Y != nullptr) gather_matrix(Y, y_store, Y);
        if (in.labels != nullptr) {
            in.labels = gather(in.labels, labels_store);
            in.n_labels = n;
        }
        if (in.sample_weight != nullptr) {
            in.sample_weight = gather(in.sample_weight, weights_store);
            in.n_sample_weight = n;
        }
        if (in.groups != nullptr) {
            in.groups = gather(in.groups, groups_store);
            in.n_groups = n;
        }
        if (in.fold_ids != nullptr) {
            in.fold_ids = gather(in.fold_ids, folds_store);
            in.n_fold_ids = n;
        }
    }
};

n4m_status_t read_inputs(n4m_context_t* ctx, const n4m_fit_inputs_v1_t* raw,
                         TrainingRows& rows) {
    if (raw == nullptr) {
        set_error(ctx, "fit inputs are NULL");
        return N4M_ERR_NULL_POINTER;
    }
    constexpr std::size_t kMin = offsetof(n4m_fit_inputs_v1_t, X) + sizeof(void*);
    if (raw->struct_size < kMin) {
        set_error(ctx, "n4m_fit_inputs_v1_t.struct_size is too small");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_fit_inputs_v1_t& in = rows.in;
    std::memcpy(&in, raw, std::min<std::size_t>(raw->struct_size, sizeof(in)));
    in.struct_size = sizeof(in);
    n4m_status_t st = check_view(ctx, in.X, "X");
    if (st != N4M_OK) return st;
    if (in.Y != nullptr && (st = check_view(ctx, in.Y, "y")) != N4M_OK) return st;
    if (in.X_target != nullptr && (st = check_view(ctx, in.X_target, "target_domain")) != N4M_OK) {
        return st;
    }
    rows.X = *in.X;
    if (in.Y != nullptr) rows.Y = *in.Y;
    const std::int64_t n = rows.rows();
    if (n <= 0 || rows.X.cols <= 0) {
        set_error(ctx, "X must have at least one row and one column");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    // Row-aligned inputs are subset here, so their lengths are checked here.
    const struct {
        const void* data;
        std::int64_t length;
        const char* name;
    } row_inputs[] = {{in.Y, in.Y != nullptr ? in.Y->rows : 0, "y"},
                      {in.labels, in.n_labels, "labels"},
                      {in.sample_weight, in.n_sample_weight, "sample_weight"},
                      {in.groups, in.n_groups, "groups"},
                      {in.fold_ids, in.n_fold_ids, "fold_ids"}};
    for (const auto& input : row_inputs) {
        if (input.data != nullptr && input.length != n) {
            set_error_named(ctx, "fit input has an incompatible length", input.name);
            return N4M_ERR_SHAPE_MISMATCH;
        }
    }
    return N4M_OK;
}

bool present(const n4m_fit_inputs_v1_t& in, int k) noexcept {
    const void* fields[N4M_FIT_INPUT_COUNT] = {in.Y,     in.labels,         in.sample_weight,
                                               in.groups, in.feature_groups, in.block_sizes,
                                               in.axis,   in.X_target,       in.fold_ids};
    return fields[k] != nullptr;
}

// Refuses every given input that no step receives.
n4m_status_t check_routing(n4m_context_t* ctx, const n4m_role_pipeline_s& p,
                           const n4m_fit_inputs_v1_t& in) {
    bool used[N4M_FIT_INPUT_COUNT] = {};
    bool original_columns = true;
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        const auto& step = p.steps[s];
        const MethodSpec& spec = spec_of(step);
        const bool terminal = s + 1 == p.steps.size();
        for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
            if (spec.inputs[k] == N4M_INPUT_NONE || !present(in, k)) continue;
            if (k == N4M_FIT_INPUT_Y && spec.inputs[k] != N4M_INPUT_REQUIRED) continue;
            if (column_input(k) && !original_columns) {
                if (spec.inputs[k] != N4M_INPUT_REQUIRED) continue;
                ctx->set_errorf(
                    "%s: fit input '%s' describes the pipeline input columns, which an "
                    "earlier transformer or selector changed",
                    step_label(p, s).c_str(), kInputNames[k]);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            used[k] = true;
        }
        if (!terminal && spec.inputs[N4M_FIT_INPUT_Y] == N4M_INPUT_REQUIRED &&
            in.Y == nullptr && in.labels != nullptr) {
            used[N4M_FIT_INPUT_LABELS] = true;
        }
        if ((step.role & kIntermediateRoles) != 0) original_columns = false;
    }
    for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
        if (present(in, k) && !used[k]) {
            set_error_named(ctx, "fit input is not used by any step of the pipeline",
                            kInputNames[k]);
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

bool valid_utf8(const char* s) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(s);
    while (*p != 0) {
        int extra = 0;
        std::uint32_t cp = 0;
        if (*p < 0x80) {
            ++p;
            continue;
        } else if ((*p & 0xE0) == 0xC0) {
            extra = 1;
            cp = *p & 0x1Fu;
        } else if ((*p & 0xF0) == 0xE0) {
            extra = 2;
            cp = *p & 0x0Fu;
        } else if ((*p & 0xF8) == 0xF0) {
            extra = 3;
            cp = *p & 0x07u;
        } else {
            return false;
        }
        ++p;
        for (int k = 0; k < extra; ++k, ++p) {
            if ((*p & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (*p & 0x3Fu);
        }
        const std::uint32_t min_cp[] = {0, 0x80, 0x800, 0x10000};
        if (cp < min_cp[extra] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
    }
    return true;
}

const char* role_name(std::uint32_t role) noexcept {
    switch (role) {
        case N4M_ROLE_SAMPLE_FILTER: return "sample filter";
        case N4M_ROLE_TRANSFORMER: return "transformer";
        case N4M_ROLE_SELECTOR: return "selector";
        case N4M_ROLE_REGRESSOR: return "regressor";
        default: return "classifier";
    }
}

// Resolved value of a parameter, for messages ("2", "[1, 2]", "true", "snv").
std::string format_param(const Params& params, std::int32_t index) {
    const ParamSpec& p = params.spec().params[index];
    std::vector<std::int64_t> ints;
    std::vector<double> doubles;
    params.resolved(index, &ints, &doubles);
    char buf[64];
    std::string out;
    const bool array =
        p.type == N4M_METHOD_PARAM_INT_ARRAY || p.type == N4M_METHOD_PARAM_DOUBLE_ARRAY;
    const std::size_t count = ints.empty() ? doubles.size() : ints.size();
    for (std::size_t k = 0; k < count; ++k) {
        if (p.type == N4M_METHOD_PARAM_BOOL) {
            std::snprintf(buf, sizeof(buf), "%s", ints[k] != 0 ? "true" : "false");
        } else if (p.type == N4M_METHOD_PARAM_ENUM) {
            std::snprintf(buf, sizeof(buf), "'%s'", p.choices[ints[k]]);
        } else if (!ints.empty()) {
            std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(ints[k]));
        } else {
            std::snprintf(buf, sizeof(buf), "%.17g", doubles[k]);
        }
        if (k > 0) out += ", ";
        out += buf;
    }
    return array ? "[" + out + "]" : out;
}

bool same_doubles(const std::vector<double>& a, const std::vector<double>& b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (!(a[k] == b[k] || (std::isnan(a[k]) && std::isnan(b[k])))) return false;
    }
    return true;
}

// First parameter whose resolved values differ, or -1.
std::int32_t first_param_mismatch(const Params& recipe, const Params& state) {
    std::vector<std::int64_t> ri, si;
    std::vector<double> rd, sd;
    for (std::int32_t i = 0; i < recipe.spec().n_params; ++i) {
        recipe.resolved(i, &ri, &rd);
        state.resolved(i, &si, &sd);
        if (ri != si || !same_doubles(rd, sd)) return i;
    }
    return -1;
}

// The capability a stateful step's role needs from its fitted state.
std::uint64_t needed_capability(std::uint32_t role) noexcept {
    switch (role) {
        case N4M_ROLE_REGRESSOR: return N4M_CAP_PREDICT;
        case N4M_ROLE_CLASSIFIER: return N4M_CAP_PREDICT_LABELS;
        default: return N4M_CAP_TRANSFORM;
    }
}

void reset(n4m_role_pipeline_s& p) noexcept {
    p.states.clear();
    p.n_features = 0;
    p.fitted = false;
}

}  // namespace

std::string state_label(const n4m_role_pipeline_s& p, std::size_t state) {
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        if (p.steps[s].state_index == static_cast<std::int32_t>(state)) {
            return "state " + std::to_string(state) + " (" + step_label(p, s) + ")";
        }
    }
    return "state " + std::to_string(state);
}

std::string step_label(const n4m_role_pipeline_s& pipeline, std::size_t step) {
    return "step " + std::to_string(step) + " (" + spec_of(pipeline.steps[step]).method_id + ")";
}

n4m_status_t pipeline_error(n4m_context_t* ctx, n4m_status_t st, const char* where) {
    const std::string message = ctx->last_error();
    ctx->set_errorf("%s: %s", where, message.empty() ? n4m_status_to_string(st) : message.c_str());
    return st;
}

n4m_status_t pipeline_create(n4m_context_t* ctx, std::int32_t n_steps,
                             const char* const* method_ids, const n4m_params_t* const* params,
                             std::unique_ptr<n4m_role_pipeline_s>& out) {
    if (n_steps < 1) {
        set_error(ctx, "a role pipeline needs at least one step (a regressor or classifier)");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    auto p = std::make_unique<n4m_role_pipeline_s>();
    bool after_filters = false;
    std::int32_t n_states = 0;
    for (std::int32_t s = 0; s < n_steps; ++s) {
        const char* id = method_ids[s];
        if (id == nullptr) {
            ctx->set_errorf("step %d: method id is NULL", s);
            return N4M_ERR_NULL_POINTER;
        }
        const std::int32_t index = method_index(id);
        if (index < 0) {
            ctx->set_errorf("step %d: unknown method '%s'", s, id);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const MethodSpec& spec = *method_at(index);
        if (spec.kind != N4M_METHOD_ESTIMATOR) {
            ctx->set_errorf("step %d (%s): a procedure cannot be a pipeline step", s, id);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const n4m_params_t* given = params != nullptr ? params[s] : nullptr;
        if (given != nullptr && &given->params.spec() != &spec) {
            ctx->set_errorf("step %d (%s): params were created for another method", s, id);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        Params values = given != nullptr ? given->params : Params(spec);
        if (const char* missing = values.missing_required()) {
            ctx->set_errorf("step %d (%s): missing required parameter '%s'", s, id, missing);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const bool last = s + 1 == n_steps;
        std::uint32_t role = 0;
        if (last) {
            if ((spec.roles & kTerminalRoles) == 0) {
                ctx->set_errorf(
                    "step %d (%s): a role pipeline ends with one regressor or classifier; "
                    "this step is not one",
                    s, id);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            role = (spec.roles & N4M_ROLE_REGRESSOR) != 0 ? N4M_ROLE_REGRESSOR
                                                          : N4M_ROLE_CLASSIFIER;
        } else if ((spec.roles & N4M_ROLE_SAMPLE_FILTER) != 0) {
            if (after_filters) {
                ctx->set_errorf(
                    "step %d (%s): sample filters come before every transformer or selector",
                    s, id);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            role = N4M_ROLE_SAMPLE_FILTER;
        } else if ((spec.roles & kIntermediateRoles) != 0) {
            role = (spec.roles & N4M_ROLE_SELECTOR) != 0 ? N4M_ROLE_SELECTOR
                                                         : N4M_ROLE_TRANSFORMER;
            after_filters = true;
        } else {
            ctx->set_errorf(
                "step %d (%s): only the last step may be a regressor or classifier; inner "
                "steps are sample filters, transformers or selectors",
                s, id);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const std::int32_t state_index = role == N4M_ROLE_SAMPLE_FILTER ? -1 : n_states++;
        p->steps.push_back({index, std::move(values), role, state_index});
    }
    out = std::move(p);
    return N4M_OK;
}

n4m_status_t pipeline_set_feature_names(n4m_context_t* ctx, n4m_role_pipeline_s& p,
                                        const char* const* names, std::int64_t n) {
    if (p.fitted) {
        set_error(ctx, "feature names are set before fit or import");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n < 0 || (n > 0 && names == nullptr)) {
        set_error(ctx, "feature names: n must be >= 0 and names non-NULL when n > 0");
        return n < 0 ? N4M_ERR_INVALID_ARGUMENT : N4M_ERR_NULL_POINTER;
    }
    std::vector<std::string> stored;
    stored.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        if (names[i] == nullptr) {
            ctx->set_errorf("feature name %lld is NULL", static_cast<long long>(i));
            return N4M_ERR_NULL_POINTER;
        }
        if (!valid_utf8(names[i])) {
            ctx->set_errorf("feature name %lld is not valid UTF-8", static_cast<long long>(i));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        stored.emplace_back(names[i]);
    }
    std::vector<std::string> sorted = stored;
    std::sort(sorted.begin(), sorted.end());
    const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
    if (dup != sorted.end()) {
        set_error_named(ctx, "duplicate feature name", dup->c_str());
        return N4M_ERR_INVALID_ARGUMENT;
    }
    p.feature_names = std::move(stored);
    return N4M_OK;
}

n4m_status_t pipeline_fit(n4m_context_t* ctx, n4m_role_pipeline_s& p,
                          const n4m_fit_inputs_v1_t* inputs) {
    reset(p);
    TrainingRows rows;
    n4m_status_t st = read_inputs(ctx, inputs, rows);
    if (st != N4M_OK) return st;
    const std::int64_t n_features = rows.X.cols;
    if (!p.feature_names.empty() &&
        static_cast<std::int64_t>(p.feature_names.size()) != n_features) {
        ctx->set_errorf("%zu feature names for %lld input columns", p.feature_names.size(),
                        static_cast<long long>(n_features));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    st = check_routing(ctx, p, rows.in);
    if (st != N4M_OK) return st;

    std::vector<std::unique_ptr<n4m_estimator_s>> states;
    bool original_columns = true;
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        const auto& step = p.steps[s];
        const MethodSpec& spec = spec_of(step);
        const std::string where = step_label(p, s);
        const bool terminal = s + 1 == p.steps.size();
        const n4m_fit_inputs_v1_t& all = rows.in;
        auto declared = [&](int k) { return spec.inputs[k] != N4M_INPUT_NONE && present(all, k); };

        n4m_fit_inputs_v1_t in{};
        in.struct_size = sizeof(in);
        in.X = &rows.X;
        if (spec.inputs[N4M_FIT_INPUT_Y] == N4M_INPUT_REQUIRED) {
            if (all.Y != nullptr) {
                in.Y = &rows.Y;
            } else if (!terminal && all.labels != nullptr) {
                in.Y = rows.labels_as_column();
            }
        }
        if (declared(N4M_FIT_INPUT_LABELS)) {
            in.labels = all.labels;
            in.n_labels = all.n_labels;
        }
        if (declared(N4M_FIT_INPUT_SAMPLE_WEIGHT)) {
            in.sample_weight = all.sample_weight;
            in.n_sample_weight = all.n_sample_weight;
        }
        if (declared(N4M_FIT_INPUT_GROUPS)) {
            in.groups = all.groups;
            in.n_groups = all.n_groups;
        }
        if (declared(N4M_FIT_INPUT_FOLD_IDS)) {
            in.fold_ids = all.fold_ids;
            in.n_fold_ids = all.n_fold_ids;
        }
        for (int k : {N4M_FIT_INPUT_FEATURE_GROUPS, N4M_FIT_INPUT_BLOCKS, N4M_FIT_INPUT_AXIS,
                      N4M_FIT_INPUT_TARGET_DOMAIN}) {
            if (!declared(k) || !original_columns) continue;
            switch (k) {
                case N4M_FIT_INPUT_FEATURE_GROUPS:
                    in.feature_groups = all.feature_groups;
                    in.n_feature_groups = all.n_feature_groups;
                    break;
                case N4M_FIT_INPUT_BLOCKS:
                    in.block_sizes = all.block_sizes;
                    in.n_blocks = all.n_blocks;
                    break;
                case N4M_FIT_INPUT_AXIS:
                    in.axis = all.axis;
                    in.n_axis = all.n_axis;
                    break;
                default:
                    in.X_target = all.X_target;
            }
        }

        auto est = std::make_unique<n4m_estimator_s>(step.method_index, step.params,
                                                     spec.factory(spec));
        st = nested(ctx, [&] { return n4m_estimator_fit(ctx, est.get(), &in); });
        if (st != N4M_OK) return pipeline_error(ctx, st, where.c_str());

        if (step.role == N4M_ROLE_SAMPLE_FILTER) {
            std::vector<std::uint8_t> mask(static_cast<std::size_t>(rows.rows()));
            st = nested(ctx, [&] {
                return n4m_estimator_apply_mask(ctx, est.get(), &rows.X, in.Y, mask.data(),
                                                rows.rows());
            });
            if (st != N4M_OK) return pipeline_error(ctx, st, where.c_str());
            std::vector<std::int64_t> kept;
            for (std::size_t i = 0; i < mask.size(); ++i) {
                if (mask[i] != 0) kept.push_back(static_cast<std::int64_t>(i));
            }
            if (kept.empty()) {
                ctx->set_errorf("%s: the filter removed every training row", where.c_str());
                return N4M_ERR_INVALID_ARGUMENT;
            }
            if (static_cast<std::int64_t>(kept.size()) != rows.rows()) rows.keep(kept);
            continue;
        }
        if (!terminal) {
            std::int64_t cols = 0;
            st = n4m_estimator_transform_cols(est.get(), &cols);
            if (st != N4M_OK) return pipeline_error(ctx, st, where.c_str());
            std::vector<double> out(static_cast<std::size_t>(rows.rows() * cols));
            n4m_matrix_view_t out_view = row_major(out, rows.rows(), cols);
            st = nested(ctx, [&] {
                return n4m_estimator_transform(ctx, est.get(), &rows.X, &out_view);
            });
            if (st != N4M_OK) return pipeline_error(ctx, st, where.c_str());
            rows.x_store = std::move(out);
            rows.X = row_major(rows.x_store, rows.rows(), cols);
            original_columns = false;
        }
        states.push_back(std::move(est));
    }
    p.states = std::move(states);
    p.n_features = n_features;
    p.fitted = true;
    return N4M_OK;
}

n4m_status_t pipeline_import(n4m_context_t* ctx, n4m_role_pipeline_s& p, std::int32_t n_states,
                             const void* const* buffers, const std::size_t* sizes) {
    reset(p);
    std::vector<const n4m_role_pipeline_s::Step*> stateful;
    for (const auto& step : p.steps) {
        if (step.state_index >= 0) stateful.push_back(&step);
    }
    if (n_states != static_cast<std::int32_t>(stateful.size())) {
        ctx->set_errorf("the recipe has %zu stateful steps but %d states were given",
                        stateful.size(), n_states);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (buffers == nullptr || sizes == nullptr) {
        set_error(ctx, "states or state_sizes is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    std::vector<std::unique_ptr<n4m_estimator_s>> states;
    std::int64_t n_features = 0;
    for (std::size_t i = 0; i < stateful.size(); ++i) {
        const auto& step = *stateful[i];
        const std::string where = state_label(p, i);
        if (buffers[i] == nullptr) {
            ctx->set_errorf("%s: state buffer is NULL", where.c_str());
            return N4M_ERR_NULL_POINTER;
        }
        n4m_estimator_t* raw = nullptr;
        n4m_status_t st = nested(
            ctx, [&] { return n4m_estimator_import_from_buffer(ctx, buffers[i], sizes[i], &raw); });
        std::unique_ptr<n4m_estimator_s> est(raw);
        if (st != N4M_OK) return pipeline_error(ctx, st, where.c_str());
        const MethodSpec& spec = *method_at(step.method_index);
        if (est->method_index != step.method_index) {
            ctx->set_errorf("%s: the state was fitted by '%s'", where.c_str(),
                            method_at(est->method_index)->method_id);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const std::int32_t mismatch = first_param_mismatch(step.params, est->params);
        if (mismatch >= 0) {
            ctx->set_errorf("%s: parameter '%s' is %s in the state but %s in the recipe",
                            where.c_str(), spec.params[mismatch].name,
                            format_param(est->params, mismatch).c_str(),
                            format_param(step.params, mismatch).c_str());
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if ((est->adapter->capabilities() & needed_capability(step.role)) == 0) {
            ctx->set_errorf("%s: the state cannot serve the step's %s role", where.c_str(),
                            role_name(step.role));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (i == 0) {
            n_features = est->adapter->n_features_in();
        } else {
            const std::int64_t previous = states.back()->adapter->transform_cols();
            if (est->adapter->n_features_in() != previous) {
                ctx->set_errorf("%s: the state reads %lld columns but the previous step outputs %lld",
                                where.c_str(), static_cast<long long>(est->adapter->n_features_in()),
                                static_cast<long long>(previous));
                return N4M_ERR_SHAPE_MISMATCH;
            }
        }
        states.push_back(std::move(est));
    }
    if (!p.feature_names.empty() &&
        static_cast<std::int64_t>(p.feature_names.size()) != n_features) {
        ctx->set_errorf("%zu feature names for %lld input columns", p.feature_names.size(),
                        static_cast<long long>(n_features));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    p.states = std::move(states);
    p.n_features = n_features;
    p.fitted = true;
    return N4M_OK;
}

n4m_status_t pipeline_check_features(n4m_context_t* ctx, const n4m_role_pipeline_s& p,
                                     std::int64_t n_columns, const char* const* names) {
    if (!p.fitted) {
        set_error(ctx, "role pipeline is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    if (n_columns != p.n_features) {
        ctx->set_errorf("the input has %lld columns; the pipeline was fitted on %lld",
                        static_cast<long long>(n_columns), static_cast<long long>(p.n_features));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (names == nullptr || p.feature_names.empty()) return N4M_OK;
    for (std::int64_t i = 0; i < n_columns; ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (names[i] == nullptr) {
            ctx->set_errorf("feature name %lld is NULL", static_cast<long long>(i));
            return N4M_ERR_NULL_POINTER;
        }
        if (p.feature_names[k] == names[i]) continue;
        const auto found = std::find(p.feature_names.begin(), p.feature_names.end(), names[i]);
        if (found == p.feature_names.end()) {
            ctx->set_errorf("input column %lld is '%s'; the pipeline was fitted with '%s' there",
                            static_cast<long long>(i), names[i], p.feature_names[k].c_str());
        } else {
            ctx->set_errorf(
                "input column %lld is '%s', fitted at column %lld: the columns are reordered",
                static_cast<long long>(i), names[i],
                static_cast<long long>(found - p.feature_names.begin()));
        }
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

std::int64_t pipeline_transform_cols(const n4m_role_pipeline_s& p) noexcept {
    return p.states.size() > 1 ? p.states[p.states.size() - 2]->adapter->transform_cols()
                               : p.n_features;
}

namespace {

// Runs X through the first `count` stateful steps (transformers/selectors).
// The last one writes into `final_out` when given, else into `store`.
n4m_status_t run_transformers(n4m_context_t* ctx, const n4m_role_pipeline_s& p,
                              const n4m_matrix_view_t* X, std::size_t count,
                              std::vector<double>& store, n4m_matrix_view_t& features,
                              n4m_matrix_view_t* final_out) {
    if (!p.fitted) {
        set_error(ctx, "role pipeline is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    n4m_status_t st = check_view(ctx, X, "X");
    if (st != N4M_OK) return st;
    if (X->cols != p.n_features) {
        ctx->set_errorf("X has %lld columns; the pipeline was fitted on %lld",
                        static_cast<long long>(X->cols), static_cast<long long>(p.n_features));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    features = *X;
    std::vector<double> next;
    for (std::size_t i = 0; i < count; ++i) {
        const n4m_estimator_s& est = *p.states[i];
        const bool last = i + 1 == count;
        n4m_matrix_view_t out_view{};
        if (last && final_out != nullptr) {
            out_view = *final_out;
        } else {
            const std::int64_t cols = est.adapter->transform_cols();
            next.assign(static_cast<std::size_t>(X->rows * cols), 0.0);
            out_view = row_major(next, X->rows, cols);
        }
        st = nested(ctx, [&] { return n4m_estimator_transform(ctx, &est, &features, &out_view); });
        if (st != N4M_OK) return pipeline_error(ctx, st, state_label(p, i).c_str());
        if (last && final_out != nullptr) break;
        store.swap(next);
        features = row_major(store, X->rows, out_view.cols);
    }
    return N4M_OK;
}

}  // namespace

n4m_status_t pipeline_features(n4m_context_t* ctx, const n4m_role_pipeline_s& p,
                               const n4m_matrix_view_t* X, std::vector<double>& store,
                               n4m_matrix_view_t& features) {
    const std::size_t count = p.fitted ? p.states.size() - 1 : 0;
    return run_transformers(ctx, p, X, count, store, features, nullptr);
}

n4m_status_t pipeline_transform(n4m_context_t* ctx, const n4m_role_pipeline_s& p,
                                const n4m_matrix_view_t* X, n4m_matrix_view_t* out) {
    if (out == nullptr) {
        set_error(ctx, "output view is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    std::vector<double> store;
    n4m_matrix_view_t features{};
    const std::size_t count = p.fitted ? p.states.size() - 1 : 0;
    n4m_status_t st = run_transformers(ctx, p, X, count, store, features, out);
    if (st != N4M_OK || count > 0) return st;
    // No transformer: the terminal step reads X itself.
    st = check_view(ctx, out, "out");
    if (st != N4M_OK) return st;
    if (out->rows != X->rows || out->cols != X->cols) {
        set_error(ctx, "output shape does not match the pipeline transform");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    auto* dst = static_cast<double*>(out->data);
    for (std::int64_t i = 0; i < X->rows; ++i) {
        for (std::int64_t j = 0; j < X->cols; ++j) {
            dst[i * out->row_stride + j * out->col_stride] = at(*X, i, j);
        }
    }
    return N4M_OK;
}

}  // namespace n4m::estimator
