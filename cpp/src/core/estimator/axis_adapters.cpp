// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Transformers whose fitted state is the source axis: AxisUnitConversion
// (nm, um <-> cm-1) and SpliceCorrection (detector jumps). The plan is rebuilt
// from the stored axis and the replayed parameters, so N4ME carries the axis
// only. AxisUnitConversion publishes the converted ascending axis as the fit
// result entry "axis" (with "source_columns"), also after an N4ME import.

#include <memory>
#include <utility>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/method_result.hpp"
#include "core/preprocessing/resampling/axis_unit_conversion.hpp"
#include "core/preprocessing/signal_conversion/splice_correction.hpp"

namespace n4m::estimator {
namespace {

constexpr std::uint32_t kTagAxis = 0x31535841u;  // "AXS1"

struct ResultDeleter {
    void operator()(n4m_method_result_t* r) const noexcept {
        n4m_method_result_destroy(r);
    }
};
using ResultPtr = std::unique_ptr<n4m_method_result_t, ResultDeleter>;

// Policy: Plan, build(params, axis, plan) -> status, apply(plan, X, out),
// result(plan) -> owned fit result or null, and the refusal message.
template<typename Policy>
class AxisPlanAdapter final : public Adapter {
public:
    explicit AxisPlanAdapter(std::uint64_t traits) {
        set_transform_traits(traits);
    }

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override {
        return static_cast<std::int64_t>(source_.size());
    }
    std::int64_t n_outputs() const noexcept override {
        return 0;
    }
    std::int64_t transform_cols() const noexcept override {
        return n_features_in();
    }
    const n4m_method_result_t* fit_result() const noexcept override {
        return result_.get();
    }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        if (in.axis == nullptr || in.X == nullptr || in.n_axis != in.X->cols) {
            set_error(ctx, "this transformer needs one axis value per input column");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        n4m_status_t st = build(ctx, params, std::vector<double>(in.axis, in.axis + in.n_axis));
        if (st != N4M_OK)
            return st;
        result_.reset(Policy::result(plan_));
        return N4M_OK;
    }

    n4m_status_t transform(n4m_context_t* ctx,
                           const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        if (source_.empty())
            return N4M_ERR_NOT_FITTED;
        if (X.dtype != N4M_DTYPE_F64 || out.dtype != N4M_DTYPE_F64 || X.cols != n_features_in()
            || out.cols != n_features_in() || out.rows != X.rows) {
            set_error(ctx, "transform expects F64 views of the fitted width");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return Policy::apply(plan_, X, out);
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        if (source_.empty())
            return N4M_ERR_NOT_FITTED;
        n4m_state_writer_t writer;
        n4m_state_write_i64(&writer, n_features_in());
        n4m_state_write_f64_array(&writer, source_.data(), n_features_in());
        out.push_back({kTagAxis, std::move(writer.bytes)});
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx,
                            const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        if (blocks.size() != 1 || blocks[0].tag != kTagAxis) {
            set_error(ctx, "state requires one AXS1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_state_reader_t reader(blocks[0].bytes.data(), blocks[0].bytes.size());
        std::int64_t width = 0;
        if (!n4m_state_read_i64(&reader, &width) || width <= 0
            || reader.remaining() / 8 < static_cast<std::size_t>(width)) {
            set_error(ctx, "invalid axis state");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        std::vector<double> source(static_cast<std::size_t>(width));
        if (!n4m_state_read_f64_array(&reader, source.data(), width) || reader.remaining() != 0) {
            set_error(ctx, "invalid axis state");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        // The converted axis gives meaning to the output columns: rebuild it.
        if (build(ctx, params, std::move(source)) != N4M_OK) return N4M_ERR_CORRUPT_BUFFER;
        result_.reset(Policy::result(plan_));
        return N4M_OK;
    }

    // Parameters are replayed; the state is the source axis.
    n4m_status_t check_params(n4m_context_t*, const Params&) const override {
        return N4M_OK;
    }

private:
    n4m_status_t build(n4m_context_t* ctx, const Params& params, std::vector<double> source) {
        source_.clear();
        typename Policy::Plan plan;
        const n4m_status_t st = Policy::build(params, source, plan);
        if (st != N4M_OK) {
            set_error(ctx, Policy::kRefusal);
            return st;
        }
        plan_ = std::move(plan);
        source_ = std::move(source);
        return N4M_OK;
    }

    std::vector<double> source_;
    typename Policy::Plan plan_;
    ResultPtr result_;
};

struct AxisUnitPolicy {
    using Plan = n4m::core::AxisUnitPlan;
    static constexpr const char* kRefusal =
        "axis values must be finite, positive and distinct in both units";
    static n4m_status_t build(const Params& p, const std::vector<double>& axis, Plan& plan) {
        return n4m::core::plan_axis_unit_conversion(axis.data(),
                                                    static_cast<std::int64_t>(axis.size()),
                                                    to_i32(p.get_int("source_unit")),
                                                    to_i32(p.get_int("target_unit")),
                                                    p.get_bool("spectral_density"),
                                                    plan);
    }
    static n4m_status_t
    apply(const Plan& plan, const n4m_matrix_view_t& X, n4m_matrix_view_t& out) {
        return n4m::core::apply_axis_unit_conversion(plan, X, out);
    }
    static n4m_method_result_t* result(const Plan& plan) {
        auto* result = new n4m_method_result_s();
        result->set_double_matrix(
            "axis", plan.axis, 1, static_cast<std::int64_t>(plan.axis.size()));
        result->set_int64_vector("source_columns", plan.order);
        return result;
    }
};

struct SplicePolicy {
    using Plan = n4m::core::SplicePlan;
    static constexpr const char* kRefusal =
        "splice correction needs a strictly increasing finite axis, increasing splices that leave "
        "every segment at least one band, window >= 0 and an existing reference segment";
    static n4m_status_t build(const Params& p, const std::vector<double>& axis, Plan& plan) {
        const std::vector<double> splices = p.get_doubles("splices");
        return n4m::core::plan_splice_correction(axis.data(),
                                                 static_cast<std::int64_t>(axis.size()),
                                                 splices.data(),
                                                 static_cast<std::int64_t>(splices.size()),
                                                 p.get_int("reference_segment"),
                                                 p.get_int("method") == 0,
                                                 p.get_double("window"),
                                                 p.get_bool("taper"),
                                                 plan);
    }
    static n4m_status_t
    apply(const Plan& plan, const n4m_matrix_view_t& X, n4m_matrix_view_t& out) {
        return n4m::core::apply_splice_correction(plan, X, out);
    }
    static n4m_method_result_t* result(const Plan&) {
        return nullptr;
    }
};

}  // namespace

std::unique_ptr<Adapter> make_tr_axis_unit_conversion(const MethodSpec&) {
    return std::make_unique<AxisPlanAdapter<AxisUnitPolicy>>(kAffine);
}

std::unique_ptr<Adapter> make_tr_splice_correction(const MethodSpec&) {
    return std::make_unique<AxisPlanAdapter<SplicePolicy>>(0);
}

}  // namespace n4m::estimator
