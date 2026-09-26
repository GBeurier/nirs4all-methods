// SPDX-License-Identifier: CECILL-2.1
//
// Selector role adapters: a fitted selector keeps the input width and the
// selected column indices; transform returns those columns in ascending input
// order (the SelectorMixin convention every binding follows). The selection
// itself is the existing native kernel's; the adapters only build its inputs.

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/spec.hpp"

namespace n4m::estimator {

namespace {

constexpr std::uint32_t kTagSelection = 0x314C4553u;  // "SEL1"

struct PlanDeleter {
    void operator()(n4m_validation_plan_t* p) const noexcept { n4m_validation_plan_destroy(p); }
};
struct ConfigDeleter {
    void operator()(n4m_config_t* c) const noexcept { n4m_config_destroy(c); }
};
struct ResultDeleter {
    void operator()(n4m_method_result_t* r) const noexcept { n4m_method_result_destroy(r); }
};
using PlanPtr = std::unique_ptr<n4m_validation_plan_t, PlanDeleter>;
using ConfigPtr = std::unique_ptr<n4m_config_t, ConfigDeleter>;
using ResultPtr = std::unique_ptr<n4m_method_result_t, ResultDeleter>;

// Internal-CV plan: the caller's fold ids, or the canonical contiguous plan
// (fold size n / k, the last fold absorbs the remainder).
n4m_status_t make_plan(n4m_context_t* ctx, const FitInputs& in, std::int64_t n_folds,
                       PlanPtr& out) {
    const std::int64_t n = in.X->rows;
    std::vector<std::int64_t> fold(static_cast<std::size_t>(n));
    std::int64_t k = n_folds;
    if (in.fold_ids != nullptr) {
        k = 1 + *std::max_element(in.fold_ids, in.fold_ids + n);
        std::copy(in.fold_ids, in.fold_ids + n, fold.begin());
    } else {
        if (k < 2 || k > n) {
            set_error(ctx, "cv must be in [2, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const std::int64_t size = std::max<std::int64_t>(1, n / k);
        for (std::int64_t i = 0; i < n; ++i) {
            fold[static_cast<std::size_t>(i)] = std::min(i / size, k - 1);
        }
    }
    n4m_validation_plan_t* raw = nullptr;
    n4m_status_t st = n4m_validation_plan_create(&raw);
    if (st != N4M_OK) return st;
    out.reset(raw);
    st = n4m_validation_plan_set_n_samples(raw, n);
    for (std::int64_t f = 0; st == N4M_OK && f < k; ++f) {
        std::vector<std::int64_t> train, test;
        for (std::int64_t i = 0; i < n; ++i) {
            (fold[static_cast<std::size_t>(i)] == f ? test : train).push_back(i);
        }
        if (train.empty() || test.empty()) {
            set_error(ctx, "every fold must have test rows and leave train rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        st = n4m_validation_plan_add_fold(raw, train.data(), static_cast<std::int64_t>(train.size()),
                                          test.data(), static_cast<std::int64_t>(test.size()));
    }
    return st;
}

void put_u64(std::vector<unsigned char>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<unsigned char>(v >> (8 * i)));
}

std::uint64_t get_u64(const unsigned char* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

// Selectors whose kernel returns a MethodResult with int64 "selected_indices".
class SelectorAdapter final : public Adapter {
  public:
    using FitFn = std::function<n4m_status_t(n4m_context_t*, const Params&, const FitInputs&,
                                             n4m_method_result_t**)>;
    explicit SelectorAdapter(FitFn fit_fn) : fit_fn_(std::move(fit_fn)) {}

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SELECTED_INDICES | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override { return 0; }
    std::int64_t transform_cols() const noexcept override {
        return static_cast<std::int64_t>(selected_.size());
    }
    const std::vector<std::int64_t>* selected_indices() const noexcept override {
        return &selected_;
    }
    const n4m_method_result_t* fit_result() const noexcept override { return result_.get(); }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        result_.reset();
        selected_.clear();
        n_features_ = 0;
        n4m_method_result_t* raw = nullptr;
        n4m_status_t st = fit_fn_(ctx, params, in, &raw);
        ResultPtr result(raw);
        if (st != N4M_OK) return st;
        const std::int64_t* idx = nullptr;
        std::int64_t k = 0;
        st = n4m_method_result_get_int64_vector(result.get(), "selected_indices", &idx, &k);
        if (st != N4M_OK || k <= 0) {
            set_error(ctx, "selector returned no selected_indices");
            return st != N4M_OK ? st : N4M_ERR_NUMERICAL_FAILURE;
        }
        std::vector<std::int64_t> selected(idx, idx + k);
        st = validate(ctx, selected, in.X->cols);
        if (st != N4M_OK) return st;
        selected_ = std::move(selected);
        n_features_ = in.X->cols;
        result_ = std::move(result);
        return N4M_OK;
    }

    // Selected columns in ascending input order.
    n4m_status_t transform(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        if (X.dtype != N4M_DTYPE_F64 || out.dtype != N4M_DTYPE_F64 ||
            out.cols != transform_cols()) {
            set_error(ctx, "selector transform expects F64 views of the selected width");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        std::vector<std::int64_t> cols(selected_);
        std::sort(cols.begin(), cols.end());
        const auto* src = static_cast<const double*>(X.data);
        auto* dst = static_cast<double*>(out.data);
        for (std::int64_t i = 0; i < X.rows; ++i) {
            for (std::size_t j = 0; j < cols.size(); ++j) {
                dst[i * out.row_stride + static_cast<std::int64_t>(j) * out.col_stride] =
                    src[i * X.row_stride + cols[j] * X.col_stride];
            }
        }
        return N4M_OK;
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        StateBlock block;
        block.tag = kTagSelection;
        put_u64(block.bytes, static_cast<std::uint64_t>(n_features_));
        put_u64(block.bytes, selected_.size());
        for (std::int64_t v : selected_) put_u64(block.bytes, static_cast<std::uint64_t>(v));
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params&,
                            const std::vector<StateBlock>& blocks) override {
        if (blocks.size() != 1 || blocks[0].tag != kTagSelection || blocks[0].bytes.size() < 16) {
            set_error(ctx, "selector state must be one SEL1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        const auto& b = blocks[0].bytes;
        const std::uint64_t p = get_u64(b.data());
        const std::uint64_t k = get_u64(b.data() + 8);
        if (p == 0 || p > (std::uint64_t{1} << 31) || k == 0 || k > p || b.size() != 16 + 8 * k) {
            set_error(ctx, "selector state has inconsistent sizes");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        std::vector<std::int64_t> selected(k);
        for (std::uint64_t j = 0; j < k; ++j) {
            selected[j] = static_cast<std::int64_t>(get_u64(b.data() + 16 + 8 * j));
        }
        n4m_status_t st = validate(ctx, selected, static_cast<std::int64_t>(p));
        if (st != N4M_OK) return N4M_ERR_CORRUPT_BUFFER;
        selected_ = std::move(selected);
        n_features_ = static_cast<std::int64_t>(p);
        result_.reset();
        return N4M_OK;
    }

  private:
    static n4m_status_t validate(n4m_context_t* ctx, const std::vector<std::int64_t>& selected,
                                 std::int64_t p) {
        std::vector<std::int64_t> sorted(selected);
        std::sort(sorted.begin(), sorted.end());
        if (sorted.front() < 0 || sorted.back() >= p ||
            std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            set_error(ctx, "selected indices must be unique input columns");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        return N4M_OK;
    }

    FitFn fit_fn_;
    std::vector<std::int64_t> selected_;
    std::int64_t n_features_ = 0;
    ResultPtr result_;
};

}  // namespace

}  // namespace n4m::estimator
