// SPDX-License-Identifier: CECILL-2.1
//
// Sample-filter role adapters over the outlier kernels (core/filters/*):
// fit learns the keep rule on training rows, apply_mask writes 1 (keep) or 0
// (exclude) for any rows. The Y-outlier filter reads the one-column target at
// fit and apply; the others read X and ignore Y. Filters are train-only DAG
// nodes, so a kernel without state serialization (X-outlier: forests, LOF,
// MCD) is simply not SERIALIZABLE.

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/filters/high_leverage.h"
#include "core/filters/spectral_quality.h"
#include "core/filters/x_outlier.h"
#include "core/filters/y_outlier.h"

namespace n4m::estimator {

namespace {

constexpr std::uint32_t kTagFilter = 0x314C4653u;  // "SFL1"

// Rows the kernel reads: X, or the one-column Y for target filters.
struct Rows {
    const double* data;
    std::int64_t rows;
    std::int64_t cols;
};

template <typename S>
struct FilterKernel {
    std::function<S*(const Params&)> create;  // nullptr on invalid parameters
    void (*destroy)(S*);
    bool reads_y;
    n4m_status_t (*fit)(S*, const Rows&);
    n4m_status_t (*apply)(const S*, const Rows&, std::uint8_t*);
    n4m_status_t (*save)(const S*, n4m_state_writer_t*);  // nullptr: not serializable
    n4m_status_t (*load)(S*, n4m_state_reader_t*, std::int64_t n_features);
};

template <typename S>
class SampleFilterAdapter final : public Adapter {
  public:
    explicit SampleFilterAdapter(FilterKernel<S> kernel) : kernel_(std::move(kernel)) {}

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_APPLY_MASK | (kernel_.save != nullptr ? N4M_CAP_SERIALIZABLE : 0);
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override { return 0; }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        reset();
        n4m_status_t st = create(ctx, params);
        std::vector<double> storage;
        Rows rows{};
        if (st == N4M_OK) st = read(ctx, *in.X, in.Y, storage, rows);
        if (st == N4M_OK) st = kernel_.fit(state_.get(), rows);
        if (st != N4M_OK) {
            reset();
            return st;
        }
        n_features_ = in.X->cols;
        return N4M_OK;
    }

    n4m_status_t apply_mask(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                            const n4m_matrix_view_t* Y, std::uint8_t* mask) const override {
        std::vector<double> storage;
        Rows rows{};
        const n4m_status_t st = read(ctx, X, Y, storage, rows);
        return st != N4M_OK ? st : kernel_.apply(state_.get(), rows, mask);
    }

    n4m_status_t save_state(n4m_context_t* ctx, std::vector<StateBlock>& out) const override {
        if (kernel_.save == nullptr) {
            set_error(ctx, "this filter's fitted state is not serializable");
            return N4M_ERR_UNSUPPORTED;
        }
        n4m_state_writer_t writer;
        n4m_state_write_i64(&writer, n_features_);
        const n4m_status_t st = kernel_.save(state_.get(), &writer);
        if (st != N4M_OK) return st;
        StateBlock block;
        block.tag = kTagFilter;
        block.bytes = std::move(writer.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        reset();
        if (kernel_.load == nullptr || blocks.size() != 1 || blocks[0].tag != kTagFilter) {
            set_error(ctx, "sample filter state must be one SFL1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_state_reader_t reader(blocks[0].bytes.data(), blocks[0].bytes.size());
        std::int64_t width = 0;
        if (!n4m_state_read_i64(&reader, &width) || width <= 0 || width > (std::int64_t{1} << 31)) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_status_t st = create(ctx, params);
        if (st == N4M_OK) st = kernel_.load(state_.get(), &reader, width);
        if (st != N4M_OK || reader.remaining() != 0) {
            reset();
            set_error(ctx, "sample filter state does not match the method");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n_features_ = width;
        return N4M_OK;
    }

  private:
    n4m_status_t create(n4m_context_t* ctx, const Params& params) {
        state_.reset(kernel_.create(params));
        if (!state_) {
            set_error(ctx, "invalid filter parameters");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        return N4M_OK;
    }

    n4m_status_t read(n4m_context_t* ctx, const n4m_matrix_view_t& X, const n4m_matrix_view_t* Y,
                      std::vector<double>& storage, Rows& rows) const {
        const n4m_matrix_view_t* source = &X;
        if (kernel_.reads_y) {
            if (Y == nullptr || Y->cols != 1 || Y->rows != X.rows) {
                set_error(ctx, "this filter needs a one-column Y aligned with X");
                return Y == nullptr ? N4M_ERR_NULL_POINTER : N4M_ERR_SHAPE_MISMATCH;
            }
            source = Y;
        }
        const n4m_matrix_view_t v = contiguous_view(*source, storage);
        rows = {static_cast<const double*>(v.data), v.rows, v.cols};
        return N4M_OK;
    }

    void reset() noexcept {
        state_.reset();
        n_features_ = 0;
    }

    struct Destroy {
        void (*destroy)(S*);
        void operator()(S* s) const noexcept { destroy(s); }
    };
    FilterKernel<S> kernel_;
    std::unique_ptr<S, Destroy> state_{nullptr, Destroy{kernel_.destroy}};
    std::int64_t n_features_ = 0;
};

template <typename S>
std::unique_ptr<Adapter> sample_filter(FilterKernel<S> kernel) {
    return std::make_unique<SampleFilterAdapter<S>>(std::move(kernel));
}

// NaN marks an optional threshold as unused.
int is_set(double v) { return std::isnan(v) ? 0 : 1; }
double or_zero(double v) { return std::isnan(v) ? 0.0 : v; }

}  // namespace

std::unique_ptr<Adapter> make_filter_y_outlier(const MethodSpec&) {
    using S = n4m_filter_y_outlier_state_t;
    return sample_filter<S>(
        {[](const Params& p) {
             return n4m_filter_y_outlier_state_new(
                 to_i32(p.get_int("method")), p.get_double("threshold"),
                 p.get_double("lower_percentile"), p.get_double("upper_percentile"));
         },
         n4m_filter_y_outlier_state_free, true,
         [](S* s, const Rows& r) { return n4m_filter_y_outlier_state_fit(s, r.data, r.rows); },
         [](const S* s, const Rows& r, std::uint8_t* mask) {
             n4m_core_filter_stats_t stats{};
             return n4m_filter_y_outlier_state_apply(s, r.data, r.rows, mask, &stats);
         },
         n4m_filter_y_outlier_state_save,
         [](S* s, n4m_state_reader_t* r, std::int64_t) {
             return n4m_filter_y_outlier_state_load(s, r);
         }});
}

std::unique_ptr<Adapter> make_filter_x_outlier(const MethodSpec&) {
    using S = n4m_filter_x_outlier_state_t;
    return sample_filter<S>(
        {[](const Params& p) {
             const double threshold = p.get_double("threshold");
             return n4m_filter_x_outlier_state_new(
                 to_i32(p.get_int("method")), is_set(threshold), or_zero(threshold),
                 to_i32(p.get_int("n_components")), p.get_double("contamination"),
                 static_cast<std::uint64_t>(p.get_int("seed")), to_i32(p.get_int("n_estimators")),
                 p.get_int("max_samples"));
         },
         n4m_filter_x_outlier_state_free, false,
         [](S* s, const Rows& r) { return n4m_filter_x_outlier_state_fit(s, r.data, r.rows, r.cols); },
         [](const S* s, const Rows& r, std::uint8_t* mask) {
             return n4m_filter_x_outlier_state_apply(s, r.data, r.rows, r.cols, mask, nullptr);
         },
         nullptr, nullptr});
}

std::unique_ptr<Adapter> make_filter_high_leverage(const MethodSpec&) {
    using S = n4m_filter_leverage_state_t;
    return sample_filter<S>(
        {[](const Params& p) {
             const double absolute = p.get_double("absolute_threshold");
             return n4m_filter_leverage_state_new(
                 to_i32(p.get_int("method")), p.get_double("threshold_multiplier"),
                 is_set(absolute), or_zero(absolute), to_i32(p.get_int("n_components")),
                 p.get_bool("center") ? 1 : 0);
         },
         n4m_filter_leverage_state_free, false,
         [](S* s, const Rows& r) { return n4m_filter_leverage_state_fit(s, r.data, r.rows, r.cols); },
         [](const S* s, const Rows& r, std::uint8_t* mask) {
             return n4m_filter_leverage_state_apply(s, r.data, r.rows, r.cols, mask, nullptr,
                                                    nullptr, nullptr);
         },
         n4m_filter_leverage_state_save, n4m_filter_leverage_state_load});
}

// Stateless: the fitted state is the input width.
std::unique_ptr<Adapter> make_filter_spectral_quality(const MethodSpec&) {
    using S = n4m_filter_quality_state_t;
    return sample_filter<S>(
        {[](const Params& p) {
             const double max_value = p.get_double("max_value");
             const double min_value = p.get_double("min_value");
             return n4m_filter_quality_state_new(
                 p.get_double("max_nan_ratio"), p.get_double("max_zero_ratio"),
                 p.get_double("min_variance"), is_set(max_value), or_zero(max_value),
                 is_set(min_value), or_zero(min_value), p.get_bool("check_inf") ? 1 : 0);
         },
         n4m_filter_quality_state_free, false,
         [](S*, const Rows&) { return N4M_OK; },
         [](const S* s, const Rows& r, std::uint8_t* mask) {
             return n4m_filter_quality_state_apply(s, r.data, r.rows, r.cols, mask, nullptr,
                                                   nullptr, nullptr);
         },
         [](const S*, n4m_state_writer_t*) { return N4M_OK; },
         [](S*, n4m_state_reader_t*, std::int64_t) { return N4M_OK; }});
}

}  // namespace n4m::estimator
