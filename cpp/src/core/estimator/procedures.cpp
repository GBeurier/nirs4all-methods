// SPDX-License-Identifier: CECILL-2.1
//
// Procedure runners (n4m_procedure_run): one-shot methods without reusable
// state. Splitters go through n4m_splitter_run fold by fold, the ABI 2.11
// augmentations through n4m_augmentation_run, the mixups through their
// kernels (one draw mixes X and y), and every other procedure through its own
// C function; the runners only map named parameters and normalized inputs
// onto those calls and pack a MethodResult. Augmenters whose kernels hold
// band positions or detector ranges take the axis in nanometres.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/augmentation/mixup/local_mixup.h"
#include "core/augmentation/mixup/mixup.h"
#include "core/common/rng_pcg64.h"
#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/method_result.hpp"

namespace n4m::estimator {

namespace {

struct ConfigDeleter {
    void operator()(n4m_config_t* c) const noexcept { n4m_config_destroy(c); }
};
struct ModelDeleter {
    void operator()(n4m_model_t* m) const noexcept { n4m_model_destroy(m); }
};
struct RngDeleter {
    void operator()(n4m_rng_pcg64_state_t* r) const noexcept { n4m_rng_pcg64_destroy(r); }
};
struct SplitGuard {
    n4m_split_result_t result{};
    ~SplitGuard() { n4m_split_result_destroy(&result); }
};
using ConfigPtr = std::unique_ptr<n4m_config_t, ConfigDeleter>;
using ModelPtr = std::unique_ptr<n4m_model_t, ModelDeleter>;
using ResultPtr = std::unique_ptr<n4m_method_result_s>;

// The contiguous row-major F64 layout most C kernels require.
n4m_status_t row_major_f64(n4m_context_t* ctx, const n4m_matrix_view_t& v, const char* name,
                           std::vector<double>& storage, n4m_matrix_view_t& out) {
    if (v.dtype != N4M_DTYPE_F64) {
        set_error_named(ctx, "input must be F64", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    out = contiguous_view(v, storage);
    return N4M_OK;
}

n4m_status_t failed(n4m_context_t* ctx, const Params& params, n4m_status_t st) {
    set_error_named(ctx, "procedure failed", params.spec().method_id);
    return st;
}

n4m_status_t finish(ResultPtr result, n4m_method_result_t** out) {
    *out = result.release();
    return N4M_OK;
}

template <typename Entry, std::size_t N>
const Entry& entry_for(const Entry (&table)[N], const Params& params) {
    for (const Entry& e : table) {
        if (std::strcmp(e.method_id, params.spec().method_id) == 0) return e;
    }
    throw std::logic_error("procedure runner has no entry for this method");
}

// Parameter value, or 0 when the method does not declare it.
std::int64_t int_or_zero(const Params& params, const char* name) {
    return param_index(params.spec(), name) >= 0 ? params.get_int(name) : 0;
}

n4m_status_t make_config(const Params& params, ConfigPtr& out) {
    n4m_config_t* raw = nullptr;
    const n4m_status_t st = n4m_config_create(&raw);
    if (st != N4M_OK) return st;
    out.reset(raw);
    return apply_config_params(params, raw);
}

// ---- Splitters ---------------------------------------------------------

struct SplitterEntry {
    const char* method_id;
    std::int32_t kind;
    bool uses_x;  // Y-only splitters split the rows of X without reading it
    bool folds;   // n_splits folds, otherwise one train/test split
};

constexpr SplitterEntry kSplitters[] = {
    {"splitters.kennard_stone", N4M_SPLITTER_KENNARD_STONE, true, false},
    {"splitters.spxy", N4M_SPLITTER_SPXY, true, false},
    {"splitters.spxy_fold", N4M_SPLITTER_SPXY_FOLD, true, true},
    {"splitters.spxy_g_fold", N4M_SPLITTER_SPXY_GROUP_FOLD, true, true},
    {"splitters.kmeans", N4M_SPLITTER_KMEANS, true, false},
    {"splitters.kbins_stratified", N4M_SPLITTER_KBINS_STRATIFIED, false, false},
    {"splitters.binned_strat_group_kfold", N4M_SPLITTER_BINNED_STRAT_GROUP_FOLD, false, true},
    {"splitters.systematic_circular", N4M_SPLITTER_SYSTEMATIC_CIRCULAR, false, false},
    {"splitters.split_splitter", N4M_SPLITTER_DATA_TWINNING, true, false},
};

// ---- Augmenters over n4m_augmentation_run --------------------------------

struct AugmentationEntry {
    const char* method_id;
    std::int32_t kind;
    const char* params[8];  // positional order of the dispatcher, then nullptr
};

constexpr AugmentationEntry kAugmentations[] = {
    {"augmentation.noise.gaussian_noise", N4M_AUG_GAUSSIAN_NOISE, {"sigma"}},
    {"augmentation.noise.multiplicative_noise", N4M_AUG_MULTIPLICATIVE_NOISE, {"sigma_gain"}},
    {"augmentation.noise.spike_noise", N4M_AUG_SPIKE_NOISE,
     {"n_spikes_min", "n_spikes_max", "amplitude_min", "amplitude_max"}},
    {"augmentation.noise.hetero_noise", N4M_AUG_HETERO_NOISE, {"noise_base", "noise_signal_dep"}},
    {"augmentation.drift.linear_drift", N4M_AUG_LINEAR_DRIFT,
     {"offset_min", "offset_max", "slope_min", "slope_max"}},
    {"augmentation.drift.path_length", N4M_AUG_PATH_LENGTH, {"path_length_std", "min_path_length"}},
    {"augmentation.spectral.band_perturb", N4M_AUG_BAND_PERTURB,
     {"n_bands", "bw_lo", "bw_hi", "gain_lo", "gain_hi", "offset_lo", "offset_hi"}},
    {"augmentation.spectral.band_mask", N4M_AUG_BAND_MASK,
     {"n_bands_lo", "n_bands_hi", "bw_lo", "bw_hi", "mode"}},
    {"augmentation.spectral.channel_dropout", N4M_AUG_CHANNEL_DROPOUT, {"dropout_prob", "mode"}},
    {"augmentation.spectral.gauss_jitter", N4M_AUG_GAUSS_JITTER,
     {"sigma_lo", "sigma_hi", "kernel_width"}},
    {"augmentation.spectral.unsharp_mask", N4M_AUG_UNSHARP_MASK,
     {"amount_lo", "amount_hi", "sigma", "kernel_width"}},
    {"augmentation.spectral.local_clip", N4M_AUG_LOCAL_CLIP, {"n_regions", "width_lo", "width_hi"}},
    {"augmentation.random.rotate_translate", N4M_AUG_ROTATE_TRANSLATE, {"p_range", "y_factor"}},
    {"augmentation.random.random_x_op", N4M_AUG_RANDOM_X_OP,
     {"op_kind", "operator_range_min", "operator_range_max"}},
    {"augmentation.scattering.scatter_sim_msc", N4M_AUG_SCATTER_SIM_MSC,
     {"a_low", "a_high", "b_low", "b_high"}},
    {"augmentation.scattering.dead_band", N4M_AUG_DEAD_BAND,
     {"n_bands", "width_low", "width_high", "noise_std", "probability", "variation_scope"}},
    {"augmentation.scattering.batch_effect", N4M_AUG_BATCH_EFFECT,
     {"offset_std", "slope_std", "gain_std", "variation_scope"}},
    {"augmentation.splines.spline_smoothing", N4M_AUG_SPLINE_SMOOTHING, {}},
    {"augmentation.splines.spline_x_perturbations", N4M_AUG_SPLINE_X_PERTURB,
     {"spline_degree", "perturbation_density", "perturbation_range_min",
      "perturbation_range_max"}},
    {"augmentation.splines.spline_y_perturbations", N4M_AUG_SPLINE_Y_PERTURB,
     {"spline_points", "perturbation_intensity"}},
    {"augmentation.splines.spline_x_simplification", N4M_AUG_SPLINE_X_SIMPLIFY,
     {"spline_points", "uniform"}},
    {"augmentation.splines.spline_curve_simplification", N4M_AUG_SPLINE_CURVE_SIMPLIFY,
     {"spline_points", "uniform"}},
};

// A named parameter as the dispatcher's positional double.
double positional(const Params& params, const char* name) {
    const std::int32_t i = param_index(params.spec(), name);
    if (i < 0) throw std::logic_error("augmentation parameter missing from the manifest");
    return params.spec().params[i].type == N4M_METHOD_PARAM_DOUBLE
               ? params.get_double(name)
               : static_cast<double>(params.get_int(name));
}

std::uint64_t seed_of(const Params& params) {
    return static_cast<std::uint64_t>(params.get_int("seed"));
}

using AugmentKernel = std::function<n4m_status_t(n4m_matrix_view_t, n4m_matrix_view_t)>;

// Runs an X -> X kernel on a row-major copy of the training rows and returns
// the augmented rows as "X".
n4m_status_t augment(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                     n4m_method_result_t** out, const AugmentKernel& kernel) {
    std::vector<double> x_rows;
    n4m_matrix_view_t X{};
    n4m_status_t st = row_major_f64(ctx, *in.X, "X", x_rows, X);
    if (st != N4M_OK) return st;
    std::vector<double> values(static_cast<std::size_t>(X.rows * X.cols));
    n4m_matrix_view_t augmented{};
    st = n4m_matrix_view_init_rowmajor(&augmented, values.data(), X.rows, X.cols, N4M_DTYPE_F64);
    if (st == N4M_OK) st = kernel(X, augmented);
    if (st != N4M_OK) return failed(ctx, params, st);
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_double_matrix("X", std::move(values), X.rows, X.cols);
    return finish(std::move(result), out);
}

// Seeded create / apply / destroy of one augmentation handle; the handle is
// destroyed before the RNG stream it borrows.
template <typename H, typename Create, typename Apply>
n4m_status_t seeded_apply(const Params& params, Create create, Apply apply, void (*destroy)(H*),
                          n4m_matrix_view_t X, n4m_matrix_view_t out) {
    n4m_rng_pcg64_state_t* raw_rng = nullptr;
    n4m_status_t st = n4m_rng_pcg64_create(seed_of(params), &raw_rng);
    if (st != N4M_OK) return st;
    const std::unique_ptr<n4m_rng_pcg64_state_t, RngDeleter> rng(raw_rng);
    H* raw = nullptr;
    st = create(&raw, rng.get());
    if (st != N4M_OK) return st;
    const std::unique_ptr<H, void (*)(H*)> handle(raw, destroy);
    return apply(handle.get(), X, out);
}

// The axis of the augmenters with nanometre constants: wavelengths in nm,
// finite and strictly increasing.
n4m_status_t nanometre_axis(n4m_context_t* ctx, const FitInputs& in) {
    for (std::int64_t j = 0; j < in.n_axis; ++j) {
        if (!std::isfinite(in.axis[j]) || (j > 0 && !(in.axis[j] > in.axis[j - 1]))) {
            set_error(ctx, "axis must hold finite, strictly increasing wavelengths in nm");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

// The axis as the (1 x n) view the edge-artifact apply functions take.
n4m_status_t axis_view(const FitInputs& in, n4m_matrix_view_t& axis) {
    return n4m_matrix_view_init_rowmajor(&axis, const_cast<double*>(in.axis), 1, in.n_axis,
                                         N4M_DTYPE_F64);
}

using MixKernel = std::function<n4m_status_t(n4m_rng_pcg64*, const n4m_matrix_view_t&,
                                             const n4m_matrix_view_t&, double*, double*)>;

// Runs a target-mixing kernel on row-major copies of the training rows and
// targets; returns the mixed rows as "X" and the mixed targets as "Y", both in
// the row order of the input.
n4m_status_t augment_with_targets(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                  n4m_method_result_t** out, const MixKernel& kernel) {
    std::vector<double> x_rows, y_rows;
    n4m_matrix_view_t X{}, Y{};
    n4m_status_t st = row_major_f64(ctx, *in.X, "X", x_rows, X);
    if (st == N4M_OK) st = row_major_f64(ctx, *in.Y, "y", y_rows, Y);
    if (st != N4M_OK) return st;
    std::vector<double> x_mixed(static_cast<std::size_t>(X.rows * X.cols));
    std::vector<double> y_mixed(static_cast<std::size_t>(Y.rows * Y.cols));
    n4m_rng_pcg64 rng{};
    n4m_pcg64_engine_seed(&rng, seed_of(params));
    st = kernel(&rng, X, Y, x_mixed.data(), y_mixed.data());
    if (st != N4M_OK) return failed(ctx, params, st);
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_double_matrix("X", std::move(x_mixed), X.rows, X.cols);
    result->set_double_matrix("Y", std::move(y_mixed), Y.rows, Y.cols);
    return finish(std::move(result), out);
}

const double* f64(const n4m_matrix_view_t& v) { return static_cast<const double*>(v.data); }

}  // namespace

n4m_status_t run_splitter(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                          n4m_method_result_t** out) {
    const SplitterEntry& e = entry_for(kSplitters, params);
    n4m_splitter_spec_t spec{};
    spec.kind = e.kind;
    spec.n_splits = to_i32(int_or_zero(params, "n_splits"));
    spec.y_metric = to_i32(int_or_zero(params, "y_metric"));
    spec.aggregation = to_i32(int_or_zero(params, "aggregation"));
    spec.n_bins = to_i32(int_or_zero(params, "n_bins"));
    spec.strategy = to_i32(int_or_zero(params, "strategy"));
    spec.shuffle = to_i32(int_or_zero(params, "shuffle"));
    spec.max_iter = to_i32(int_or_zero(params, "max_iter"));
    spec.test_size = param_index(params.spec(), "test_size") >= 0 ? params.get_double("test_size")
                                                                   : 0.0;
    spec.seed = static_cast<std::uint64_t>(int_or_zero(params, "seed"));

    std::vector<double> x_rows, y_rows;
    n4m_matrix_view_t X{}, Y{};
    n4m_status_t st = e.uses_x ? row_major_f64(ctx, *in.X, "X", x_rows, X) : N4M_OK;
    if (st == N4M_OK && in.Y != nullptr) st = row_major_f64(ctx, *in.Y, "y", y_rows, Y);
    if (st != N4M_OK) return st;

    const std::int32_t n_folds = e.folds ? spec.n_splits : 1;
    std::vector<std::int64_t> train, test, train_offsets{0}, test_offsets{0};
    for (std::int32_t fold = 0; fold < n_folds; ++fold) {
        SplitGuard split;
        st = n4m_splitter_run(&spec, e.uses_x ? &X : nullptr, in.Y != nullptr ? &Y : nullptr,
                              in.groups, in.n_groups, fold, &split.result);
        if (st != N4M_OK) return failed(ctx, params, st);
        const n4m_split_result_t& r = split.result;
        train.insert(train.end(), r.train_idx, r.train_idx + r.n_train);
        test.insert(test.end(), r.test_idx, r.test_idx + r.n_test);
        train_offsets.push_back(static_cast<std::int64_t>(train.size()));
        test_offsets.push_back(static_cast<std::int64_t>(test.size()));
    }
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_int64_vector(kFoldTrain, std::move(train));
    result->set_int64_vector(kFoldTest, std::move(test));
    result->set_int64_vector(kFoldTrainOffsets, std::move(train_offsets));
    result->set_int64_vector(kFoldTestOffsets, std::move(test_offsets));
    return finish(std::move(result), out);
}

n4m_status_t run_augmentation_dispatch(n4m_context_t* ctx, const Params& params,
                                       const FitInputs& in, n4m_method_result_t** out) {
    const AugmentationEntry& e = entry_for(kAugmentations, params);
    std::vector<double> values;
    for (const char* name : e.params) {
        if (name == nullptr) break;
        values.push_back(positional(params, name));
    }
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return n4m_augmentation_run(e.kind, values.data(), static_cast<std::int32_t>(values.size()),
                                    seed_of(params), X, o);
    });
}

n4m_status_t run_aug_poly_drift(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                n4m_method_result_t** out) {
    const std::vector<double> lo = params.get_doubles("coeff_min");
    const std::vector<double> hi = params.get_doubles("coeff_max");
    if (lo.empty() || lo.size() != hi.size()) {
        set_error(ctx, "coeff_min and coeff_max must be non-empty and of equal length");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::int32_t degree = to_i32(static_cast<std::int64_t>(lo.size()) - 1);
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_poly_drift_handle_t>(
            params,
            [&](n4m_aug_poly_drift_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_poly_drift_create(h, rng, degree, lo.data(), hi.data());
            },
            n4m_augmentation_poly_drift_apply, n4m_augmentation_poly_drift_destroy, X, o);
    });
}

n4m_status_t run_aug_wavelength_shift(n4m_context_t* ctx, const Params& params,
                                      const FitInputs& in, n4m_method_result_t** out) {
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_wavelength_shift_handle_t>(
            params,
            [&](n4m_aug_wavelength_shift_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_wavelength_shift_create(
                    h, rng, params.get_double("shift_lo"), params.get_double("shift_hi"), in.axis,
                    in.n_axis);
            },
            n4m_augmentation_wavelength_shift_apply, n4m_augmentation_wavelength_shift_destroy, X,
            o);
    });
}

n4m_status_t run_aug_wavelength_stretch(n4m_context_t* ctx, const Params& params,
                                        const FitInputs& in, n4m_method_result_t** out) {
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_wavelength_stretch_handle_t>(
            params,
            [&](n4m_aug_wavelength_stretch_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_wavelength_stretch_create(
                    h, rng, params.get_double("stretch_lo"), params.get_double("stretch_hi"),
                    in.axis, in.n_axis);
            },
            n4m_augmentation_wavelength_stretch_apply,
            n4m_augmentation_wavelength_stretch_destroy, X, o);
    });
}

n4m_status_t run_aug_local_warp(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                n4m_method_result_t** out) {
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_local_warp_handle_t>(
            params,
            [&](n4m_aug_local_warp_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_local_warp_create(
                    h, rng, to_i32(params.get_int("n_control_points")),
                    params.get_double("max_shift"), in.axis, in.n_axis);
            },
            n4m_augmentation_local_warp_apply, n4m_augmentation_local_warp_destroy, X, o);
    });
}

n4m_status_t run_aug_magnitude_warp(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                    n4m_method_result_t** out) {
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_magnitude_warp_handle_t>(
            params,
            [&](n4m_aug_magnitude_warp_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_magnitude_warp_create(
                    h, rng, to_i32(params.get_int("n_control_points")),
                    params.get_double("gain_lo"), params.get_double("gain_hi"), in.axis,
                    in.n_axis);
            },
            n4m_augmentation_magnitude_warp_apply, n4m_augmentation_magnitude_warp_destroy, X, o);
    });
}

n4m_status_t run_aug_emsc_distort(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                  n4m_method_result_t** out) {
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_emsc_distort_handle_t>(
            params,
            [&](n4m_aug_emsc_distort_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_emsc_distort_create(
                    h, rng, params.get_double("mult_low"), params.get_double("mult_high"),
                    params.get_double("add_low"), params.get_double("add_high"),
                    to_i32(params.get_int("polynomial_order")),
                    params.get_double("polynomial_strength"), params.get_double("correlation"),
                    in.axis, in.n_axis);
            },
            n4m_augmentation_emsc_distort_apply, n4m_augmentation_emsc_distort_destroy, X, o);
    });
}

n4m_status_t run_aug_edge_curvature(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                    n4m_method_result_t** out) {
    n4m_matrix_view_t axis{};
    const n4m_status_t st = axis_view(in, axis);
    if (st != N4M_OK) return st;
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_edge_curve_handle_t>(
            params,
            [&](n4m_aug_edge_curve_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_edge_curvature_create(
                    h, rng, params.get_double("curvature_strength"),
                    to_i32(params.get_int("curvature_type")), params.get_double("asymmetry"),
                    params.get_double("edge_focus"));
            },
            [&](const n4m_aug_edge_curve_handle_t* h, n4m_matrix_view_t x, n4m_matrix_view_t y) {
                return n4m_augmentation_edge_curvature_apply(h, x, axis, y);
            },
            n4m_augmentation_edge_curvature_destroy, X, o);
    });
}

n4m_status_t run_aug_instrument_broaden(n4m_context_t* ctx, const Params& params,
                                        const FitInputs& in, n4m_method_result_t** out) {
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_instrument_broaden_handle_t>(
            params,
            [&](n4m_aug_instrument_broaden_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_instrument_broaden_create(
                    h, rng, params.get_double("fwhm"), params.get_bool("use_fwhm_range") ? 1 : 0,
                    params.get_double("fwhm_low"), params.get_double("fwhm_high"),
                    to_i32(params.get_int("variation_scope")), in.axis, in.n_axis);
            },
            n4m_augmentation_instrument_broaden_apply, n4m_augmentation_instrument_broaden_destroy,
            X, o);
    });
}

n4m_status_t run_aug_truncated_peak(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                    n4m_method_result_t** out) {
    n4m_matrix_view_t axis{};
    const n4m_status_t st = axis_view(in, axis);
    if (st != N4M_OK) return st;
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_truncated_peak_handle_t>(
            params,
            [&](n4m_aug_truncated_peak_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_truncated_peak_create(
                    h, rng, params.get_double("peak_probability"),
                    params.get_double("amplitude_min"), params.get_double("amplitude_max"),
                    params.get_double("width_min"), params.get_double("width_max"),
                    params.get_bool("left_edge") ? 1 : 0, params.get_bool("right_edge") ? 1 : 0);
            },
            [&](const n4m_aug_truncated_peak_handle_t* h, n4m_matrix_view_t x,
                n4m_matrix_view_t y) { return n4m_augmentation_truncated_peak_apply(h, x, axis, y); },
            n4m_augmentation_truncated_peak_destroy, X, o);
    });
}

n4m_status_t run_aug_mixup(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                           n4m_method_result_t** out) {
    return augment_with_targets(
        ctx, params, in, out,
        [&](n4m_rng_pcg64* rng, const n4m_matrix_view_t& X, const n4m_matrix_view_t& Y,
            double* x_mixed, double* y_mixed) {
            const std::unique_ptr<n4m_aug_mixup_state_t, void (*)(n4m_aug_mixup_state_t*)> state(
                n4m_aug_mixup_state_new(params.get_double("alpha")), n4m_aug_mixup_state_free);
            if (!state) return N4M_ERR_INVALID_ARGUMENT;
            return n4m_aug_mixup_apply_impl(state.get(), rng, f64(X), X.rows, X.cols, f64(Y),
                                            Y.cols, x_mixed, y_mixed);
        });
}

n4m_status_t run_aug_local_mixup(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                 n4m_method_result_t** out) {
    return augment_with_targets(
        ctx, params, in, out,
        [&](n4m_rng_pcg64* rng, const n4m_matrix_view_t& X, const n4m_matrix_view_t& Y,
            double* x_mixed, double* y_mixed) {
            const std::unique_ptr<n4m_aug_local_mixup_state_t,
                                  void (*)(n4m_aug_local_mixup_state_t*)>
                state(n4m_aug_local_mixup_state_new(params.get_double("alpha"),
                                                    to_i32(params.get_int("k_neighbors"))),
                      n4m_aug_local_mixup_state_free);
            if (!state) return N4M_ERR_INVALID_ARGUMENT;
            return n4m_aug_local_mixup_apply_impl(state.get(), rng, f64(X), X.rows, X.cols,
                                                  f64(Y), Y.cols, x_mixed, y_mixed);
        });
}

n4m_status_t run_aug_temperature(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                 n4m_method_result_t** out) {
    const n4m_status_t st = nanometre_axis(ctx, in);
    if (st != N4M_OK) return st;
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_temperature_handle_t>(
            params,
            [&](n4m_aug_temperature_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_temperature_create(
                    h, rng, params.get_double("temperature_delta"),
                    params.get_bool("use_temp_range") ? 1 : 0, params.get_double("temp_low"),
                    params.get_double("temp_high"), params.get_bool("enable_shift") ? 1 : 0,
                    params.get_bool("enable_intensity") ? 1 : 0,
                    params.get_bool("enable_broadening") ? 1 : 0,
                    params.get_bool("region_specific") ? 1 : 0, in.axis, in.n_axis);
            },
            n4m_augmentation_temperature_apply, n4m_augmentation_temperature_destroy, X, o);
    });
}

n4m_status_t run_aug_moisture(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                              n4m_method_result_t** out) {
    const n4m_status_t st = nanometre_axis(ctx, in);
    if (st != N4M_OK) return st;
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_moisture_handle_t>(
            params,
            [&](n4m_aug_moisture_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_moisture_create(
                    h, rng, params.get_double("water_activity_delta"),
                    params.get_bool("use_aw_range") ? 1 : 0, params.get_double("aw_low"),
                    params.get_double("aw_high"), params.get_double("reference_water_activity"),
                    params.get_double("free_water_fraction"),
                    params.get_double("bound_water_shift"), params.get_double("moisture_content"),
                    params.get_bool("enable_shift") ? 1 : 0,
                    params.get_bool("enable_intensity") ? 1 : 0, in.axis, in.n_axis);
            },
            n4m_augmentation_moisture_apply, n4m_augmentation_moisture_destroy, X, o);
    });
}

n4m_status_t run_aug_particle_size(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                   n4m_method_result_t** out) {
    const n4m_status_t st = nanometre_axis(ctx, in);
    if (st != N4M_OK) return st;
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_particle_size_handle_t>(
            params,
            [&](n4m_aug_particle_size_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_particle_size_create(
                    h, rng, params.get_double("mean_size_um"),
                    params.get_double("size_variation_um"),
                    params.get_bool("use_size_range") ? 1 : 0,
                    params.get_double("size_range_low_um"),
                    params.get_double("size_range_high_um"),
                    params.get_double("reference_size_um"),
                    params.get_double("wavelength_exponent"),
                    params.get_double("size_effect_strength"),
                    params.get_bool("include_path_length") ? 1 : 0,
                    params.get_double("path_length_sensitivity"), in.axis, in.n_axis);
            },
            n4m_augmentation_particle_size_apply, n4m_augmentation_particle_size_destroy, X, o);
    });
}

n4m_status_t run_aug_detector_rolloff(n4m_context_t* ctx, const Params& params,
                                      const FitInputs& in, n4m_method_result_t** out) {
    n4m_matrix_view_t axis{};
    n4m_status_t st = nanometre_axis(ctx, in);
    if (st == N4M_OK) st = axis_view(in, axis);
    if (st != N4M_OK) return st;
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_detector_rolloff_handle_t>(
            params,
            [&](n4m_aug_detector_rolloff_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_detector_rolloff_create(
                    h, rng, to_i32(params.get_int("detector_model")),
                    params.get_double("effect_strength"), params.get_double("noise_amplification"),
                    params.get_bool("include_baseline_distortion") ? 1 : 0);
            },
            [&](const n4m_aug_detector_rolloff_handle_t* h, n4m_matrix_view_t x,
                n4m_matrix_view_t y) { return n4m_augmentation_detector_rolloff_apply(h, x, axis, y); },
            n4m_augmentation_detector_rolloff_destroy, X, o);
    });
}

n4m_status_t run_aug_stray_light(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                 n4m_method_result_t** out) {
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_stray_light_handle_t>(
            params,
            [&](n4m_aug_stray_light_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_stray_light_create(
                    h, rng, params.get_double("stray_light_fraction"),
                    params.get_double("edge_enhancement"), params.get_double("edge_width"),
                    params.get_bool("include_peak_truncation") ? 1 : 0);
            },
            [](const n4m_aug_stray_light_handle_t* h, n4m_matrix_view_t x,
               n4m_matrix_view_t y_out) {
                // The released entry point takes a wavelength view it only
                // checks for width; the profile uses the channel count.
                std::vector<double> channels(static_cast<std::size_t>(x.cols));
                for (std::size_t j = 0; j < channels.size(); ++j) channels[j] = static_cast<double>(j);
                n4m_matrix_view_t wl{};
                const n4m_status_t st = n4m_matrix_view_init_rowmajor(
                    &wl, channels.data(), 1, x.cols, N4M_DTYPE_F64);
                return st != N4M_OK ? st : n4m_augmentation_stray_light_apply(h, x, wl, y_out);
            },
            n4m_augmentation_stray_light_destroy, X, o);
    });
}

n4m_status_t run_aug_edge_artifacts(n4m_context_t* ctx, const Params& params,
                                    const FitInputs& in, n4m_method_result_t** out) {
    n4m_matrix_view_t axis{};
    n4m_status_t st = nanometre_axis(ctx, in);
    if (st == N4M_OK) st = axis_view(in, axis);
    if (st != N4M_OK) return st;
    const std::int32_t flags =
        (params.get_bool("detector_roll_off") ? N4M_AUG_EDGE_ARTIFACTS_DETECTOR_ROLL_OFF : 0) |
        (params.get_bool("stray_light") ? N4M_AUG_EDGE_ARTIFACTS_STRAY_LIGHT : 0) |
        (params.get_bool("edge_curvature") ? N4M_AUG_EDGE_ARTIFACTS_EDGE_CURVATURE : 0) |
        (params.get_bool("truncated_peaks") ? N4M_AUG_EDGE_ARTIFACTS_TRUNCATED_PEAKS : 0);
    return augment(ctx, params, in, out, [&](n4m_matrix_view_t X, n4m_matrix_view_t o) {
        return seeded_apply<n4m_aug_edge_artifacts_handle_t>(
            params,
            [&](n4m_aug_edge_artifacts_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_edge_artifacts_create(
                    h, rng, flags, params.get_double("overall_strength"),
                    to_i32(params.get_int("detector_model")));
            },
            [&](const n4m_aug_edge_artifacts_handle_t* h, n4m_matrix_view_t x,
                n4m_matrix_view_t y) { return n4m_augmentation_edge_artifacts_apply(h, x, axis, y); },
            n4m_augmentation_edge_artifacts_destroy, X, o);
    });
}

// ---- Generic procedures -------------------------------------------------

n4m_status_t run_approximate_press(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                   n4m_method_result_t** out) {
    ConfigPtr cfg;
    const n4m_status_t st = make_config(params, cfg);
    if (st != N4M_OK) return st;
    return n4m_metrics_approximate_press_compute(ctx, cfg.get(), in.X, in.Y,
                                                 to_i32(params.get_int("max_components")), out);
}

// X is the (max_components x n_folds) matrix of fold RMSE values.
n4m_status_t run_one_se_rule(n4m_context_t* ctx, const Params&, const FitInputs& in,
                             n4m_method_result_t** out) {
    std::vector<double> rows;
    n4m_matrix_view_t X{};
    const n4m_status_t st = row_major_f64(ctx, *in.X, "X", rows, X);
    if (st != N4M_OK) return st;
    return n4m_metrics_one_se_rule_compute(ctx, static_cast<const double*>(X.data),
                                           to_i32(X.rows), to_i32(X.cols), out);
}

namespace {
// PLS model with its training scores, which the monitoring thresholds use.
n4m_status_t fit_pls(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                     ModelPtr& model) {
    ConfigPtr cfg;
    n4m_status_t st = make_config(params, cfg);
    if (st == N4M_OK) st = n4m_config_set_store_scores(cfg.get(), 1);
    if (st != N4M_OK) return st;
    n4m_model_t* raw = nullptr;
    st = n4m_model_fit(ctx, cfg.get(), in.X, in.Y, &raw);
    model.reset(raw);
    return st;
}
}  // namespace

// T², Q and DModX of the training rows under the PLS model fitted on them.
n4m_status_t run_pls_diagnostics(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                 n4m_method_result_t** out) {
    ModelPtr model;
    const n4m_status_t st = fit_pls(ctx, params, in, model);
    if (st != N4M_OK) return st;
    return n4m_metrics_pls_diagnostics_compute(ctx, model.get(), in.X, in.X, out);
}

// Thresholds from the training rows (phase 1); the target-domain rows are
// the monitored samples (phase 2).
n4m_status_t run_pls_monitoring(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                n4m_method_result_t** out) {
    ModelPtr model;
    const n4m_status_t st = fit_pls(ctx, params, in, model);
    if (st != N4M_OK) return st;
    return n4m_metrics_pls_monitoring_run(ctx, model.get(), in.X, in.X_target,
                                          params.get_double("alpha"), out);
}

// X holds the predictions and y the reference values, one column each.
n4m_status_t run_regression_metrics(n4m_context_t* ctx, const Params& params,
                                    const FitInputs& in, n4m_method_result_t** out) {
    if (in.X->cols != 1 || in.Y->cols != 1) {
        set_error(ctx, "regression metrics take one prediction column (X) and one target (y)");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> pred_rows, true_rows;
    n4m_matrix_view_t pred{}, truth{};
    n4m_status_t st = row_major_f64(ctx, *in.X, "X", pred_rows, pred);
    if (st == N4M_OK) st = row_major_f64(ctx, *in.Y, "y", true_rows, truth);
    if (st != N4M_OK) return st;
    using Metric = n4m_status_t (*)(const double*, const double*, std::int64_t, double*);
    struct Named {
        const char* name;
        Metric fn;
    };
    const Named metrics[] = {{"rmse", n4m_metrics_regression_metrics_rmse},
                             {"mae", n4m_metrics_regression_metrics_mae},
                             {"bias", n4m_metrics_regression_metrics_bias},
                             {"sep", n4m_metrics_regression_metrics_sep},
                             {"rpd", n4m_metrics_regression_metrics_rpd},
                             {"rpiq", n4m_metrics_regression_metrics_rpiq},
                             {"r2", n4m_metrics_regression_metrics_r2},
                             {"nrmse", n4m_metrics_regression_metrics_nrmse}};
    auto result = std::make_unique<n4m_method_result_s>();
    for (const Named& m : metrics) {
        double value = 0.0;
        st = m.fn(static_cast<const double*>(truth.data), static_cast<const double*>(pred.data),
                  pred.rows, &value);
        if (st != N4M_OK) return failed(ctx, params, st);
        result->set_scalar(m.name, value);
    }
    return finish(std::move(result), out);
}

namespace {
using OutlierStatistic = n4m_status_t (*)(n4m_matrix_view_t, std::int32_t, double, double*,
                                          std::int64_t, double*);

// Per-row statistic as a (1 x n) matrix plus its upper control limit.
n4m_status_t outlier_statistic(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                               n4m_method_result_t** out, OutlierStatistic fn,
                               const char* name) {
    std::vector<double> rows;
    n4m_matrix_view_t X{};
    n4m_status_t st = row_major_f64(ctx, *in.X, "X", rows, X);
    if (st != N4M_OK) return st;
    std::vector<double> values(static_cast<std::size_t>(X.rows));
    double ucl = 0.0;
    st = fn(X, to_i32(params.get_int("n_components")), params.get_double("alpha"), values.data(),
            X.rows, &ucl);
    if (st != N4M_OK) return failed(ctx, params, st);
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_double_matrix(name, std::move(values), 1, X.rows);
    result->set_scalar("ucl", ucl);
    return finish(std::move(result), out);
}
}  // namespace

n4m_status_t run_hotelling_t2(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                              n4m_method_result_t** out) {
    return outlier_statistic(ctx, params, in, out, n4m_outlier_detection_hotelling_t2, "t2");
}

n4m_status_t run_q_residuals(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                             n4m_method_result_t** out) {
    return outlier_statistic(ctx, params, in, out, n4m_outlier_detection_q_residuals, "q");
}

// Explicit packed-series projection. Absence is never inferred from numeric zeros.
n4m_status_t run_ragged_summary(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                n4m_method_result_t** out) {
    std::vector<double> storage;
    n4m_matrix_view_t X{};
    const auto st = row_major_f64(ctx, *in.X, "X", storage, X);
    if (st != N4M_OK) return st;
    const auto offsets = params.get_ints("offsets");
    auto presence = params.get_ints("presence");
    const auto times = params.get_doubles("time_coordinates");
    auto invalid = [&](const char* message) {
        set_error_named(ctx, message, "utilities.ragged_summary");
        return N4M_ERR_INVALID_ARGUMENT;
    };
    if (offsets.size() < 2 || offsets.front() != 0 || offsets.back() != X.rows)
        return invalid("offsets must start at zero and cover all packed rows");
    const std::size_t samples = offsets.size() - 1;
    if (presence.empty()) presence.assign(samples, 1);
    if (presence.size() != samples || (!times.empty() && times.size() != static_cast<std::size_t>(X.rows)))
        return invalid("presence/time coordinate dimensions disagree with packed rows");
    const std::size_t width = static_cast<std::size_t>(4 * X.cols + 3);
    std::vector<double> features(samples * width, 0.0);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const auto start = offsets[sample], end = offsets[sample + 1];
        if (start < 0 || end < start || end > X.rows || (presence[sample] != 0 && presence[sample] != 1))
            return invalid("invalid offsets or presence values");
        if ((end == start) != (presence[sample] == 0))
            return invalid("empty sequences require explicit absence and absent sequences must be empty");
        if (!presence[sample]) {
            if (params.get_int("missing_policy") == 0) return invalid("absent modality requires explicit zero_with_indicator policy");
            continue;
        }
        if (!times.empty()) {
            for (auto row = start; row < end; ++row)
                if (!std::isfinite(times[static_cast<std::size_t>(row)]) || (row > start && times[static_cast<std::size_t>(row)] <= times[static_cast<std::size_t>(row - 1)]))
                    return invalid("time coordinates must be finite and strictly increasing within each sequence");
        }
        const auto count = end - start;
        auto* output = features.data() + sample * width;
        for (std::int64_t col = 0; col < X.cols; ++col) {
            long double mean = 0.0L, m2 = 0.0L;
            double lo = 0.0, hi = 0.0;
            for (auto row = start; row < end; ++row) {
                const double value = static_cast<const double*>(X.data)[row * X.row_stride + col * X.col_stride];
                if (!std::isfinite(value)) return invalid("present packed observations must be finite");
                if (row == start) lo = hi = value;
                else { lo = std::min(lo, value); hi = std::max(hi, value); }
                const long double delta = static_cast<long double>(value) - mean;
                mean += delta / static_cast<long double>(row - start + 1);
                m2 += delta * (static_cast<long double>(value) - mean);
            }
            output[static_cast<std::size_t>(4 * col)] = static_cast<double>(mean);
            output[static_cast<std::size_t>(4 * col + 1)] = static_cast<double>(std::sqrt(std::max(0.0L, m2 / static_cast<long double>(count))));
            output[static_cast<std::size_t>(4 * col + 2)] = lo;
            output[static_cast<std::size_t>(4 * col + 3)] = hi;
        }
        output[static_cast<std::size_t>(4 * X.cols)] = static_cast<double>(count);
        output[static_cast<std::size_t>(4 * X.cols + 1)] = times.empty() ? static_cast<double>(count - 1) : times[static_cast<std::size_t>(end - 1)] - times[static_cast<std::size_t>(start)];
        output[static_cast<std::size_t>(4 * X.cols + 2)] = 1.0;
        for (std::size_t col = 0; col < width; ++col)
            if (!std::isfinite(output[col])) return invalid("ragged summary output exceeds finite F64 range");
    }
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_double_matrix("features", features, static_cast<std::int64_t>(samples), static_cast<std::int64_t>(width));
    result->set_int64_vector("presence", presence);
    return finish(std::move(result), out);
}

n4m_status_t run_moments(n4m_context_t* ctx, const Params&, const FitInputs& in,
                         n4m_method_result_t** out) {
    return n4m_lowlevel_moments_compute(ctx, in.X, in.Y, out);
}

// The detector's free-text reason is not part of the result.
n4m_status_t run_signal_type(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                             n4m_method_result_t** out) {
    std::vector<double> rows;
    n4m_matrix_view_t X{};
    n4m_status_t st = row_major_f64(ctx, *in.X, "X", rows, X);
    if (st != N4M_OK) return st;
    n4m_signal_type_t type = N4M_SIGNAL_UNKNOWN;
    double confidence = 0.0;
    char reason[256] = {0};
    st = n4m_transform_signal_type_detector(X, in.axis, in.n_axis,
                                            params.get_double("confidence_threshold"), &type,
                                            &confidence, reason);
    if (st != N4M_OK) return failed(ctx, params, st);
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_scalar("signal_type", static_cast<double>(type));
    result->set_scalar("confidence", confidence);
    return finish(std::move(result), out);
}

// Explicit fold ids replace the contiguous `cv` plan.
n4m_status_t run_sweep(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                       n4m_method_result_t** out) {
    ConfigPtr cfg;
    const n4m_status_t st = make_config(params, cfg);
    if (st != N4M_OK) return st;
    const std::vector<double> lambdas = params.get_doubles("ridge_lambdas");
    std::vector<std::int32_t> components;
    for (std::int64_t k : params.get_ints("pls_components")) components.push_back(to_i32(k));
    std::vector<std::int32_t> folds;
    for (std::int64_t i = 0; i < in.n_fold_ids; ++i) folds.push_back(to_i32(in.fold_ids[i]));
    const std::int32_t cv = folds.empty() ? to_i32(params.get_int("cv")) : 0;
    const std::int32_t heads = to_i32(params.get_int("heads") + 1);  // ridge=1, pls=2, both=3
    return n4m_model_selection_sweep_run(
        ctx, cfg.get(), in.X, in.Y, cv, folds.empty() ? nullptr : folds.data(), in.n_fold_ids,
        lambdas.empty() ? nullptr : lambdas.data(), static_cast<std::int64_t>(lambdas.size()),
        components.empty() ? nullptr : components.data(),
        static_cast<std::int64_t>(components.size()), heads, out);
}

// Affine stack compression: X holds the base coefficients (features x base
// outputs); the base intercepts, the meta weights (base outputs x targets,
// row-major) and the meta intercept (one per target) are parameters.
n4m_status_t run_linear_stack_compress(n4m_context_t* ctx, const Params& params,
                                       const FitInputs& in, n4m_method_result_t** out) {
    std::vector<double> base_intercepts = params.get_doubles("base_intercepts");
    std::vector<double> meta_weights = params.get_doubles("meta_weights");
    std::vector<double> meta_intercept = params.get_doubles("meta_intercept");
    const std::int64_t m = in.X->cols;
    const auto q = static_cast<std::int64_t>(meta_intercept.size());
    if (static_cast<std::int64_t>(base_intercepts.size()) != m || q == 0 ||
        static_cast<std::int64_t>(meta_weights.size()) != m * q) {
        set_error(ctx, "base_intercepts needs one value per X column and meta_weights X.cols x "
                       "len(meta_intercept) values");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> coefficients(static_cast<std::size_t>(in.X->rows * q));
    std::vector<double> intercept(static_cast<std::size_t>(q));
    n4m_matrix_view_t bias{}, weights{}, meta{}, coef{}, icpt{};
    n4m_status_t st = n4m_matrix_view_init_rowmajor(&bias, base_intercepts.data(), 1, m,
                                                    N4M_DTYPE_F64);
    if (st == N4M_OK) {
        st = n4m_matrix_view_init_rowmajor(&weights, meta_weights.data(), m, q, N4M_DTYPE_F64);
    }
    if (st == N4M_OK) {
        st = n4m_matrix_view_init_rowmajor(&meta, meta_intercept.data(), 1, q, N4M_DTYPE_F64);
    }
    if (st == N4M_OK) {
        st = n4m_matrix_view_init_rowmajor(&coef, coefficients.data(), in.X->rows, q,
                                           N4M_DTYPE_F64);
    }
    if (st == N4M_OK) {
        st = n4m_matrix_view_init_rowmajor(&icpt, intercept.data(), 1, q, N4M_DTYPE_F64);
    }
    // One row-major copy: the result does not depend on the input layout.
    std::vector<double> storage;
    const n4m_matrix_view_t base = contiguous_view(*in.X, storage);
    if (st == N4M_OK) {
        st = n4m_ensemble_linear_stack_compress(&base, &bias, &weights, &meta, &coef, &icpt);
    }
    if (st != N4M_OK) return failed(ctx, params, st);
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_double_matrix("coefficients", std::move(coefficients), in.X->rows, q);
    result->set_double_matrix("intercept", std::move(intercept), 1, q);
    return finish(std::move(result), out);
}

n4m_status_t run_transfer_metrics(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                                  n4m_method_result_t** out) {
    std::vector<double> source_rows, target_rows;
    n4m_matrix_view_t source{}, target{};
    n4m_status_t st = row_major_f64(ctx, *in.X, "X", source_rows, source);
    if (st == N4M_OK) st = row_major_f64(ctx, *in.X_target, "target_domain", target_rows, target);
    if (st != N4M_OK) return st;
    n4m_transfer_metrics_t m{};
    st = n4m_domain_adaptation_transfer_metrics_compute(
        source, target, to_i32(params.get_int("n_components")),
        to_i32(params.get_int("k_neighbors")), seed_of(params), &m);
    if (st != N4M_OK) return failed(ctx, params, st);
    auto result = std::make_unique<n4m_method_result_s>();
    result->set_scalar("centroid_distance", m.centroid_distance);
    result->set_scalar("cka_similarity", m.cka_similarity);
    result->set_scalar("grassmann_distance", m.grassmann_distance);
    result->set_scalar("rv_coefficient", m.rv_coefficient);
    result->set_scalar("procrustes_disparity", m.procrustes_disparity);
    result->set_scalar("trustworthiness", m.trustworthiness);
    result->set_scalar("spread_distance", m.spread_distance);
    result->set_scalar("evr_source", m.evr_source);
    result->set_scalar("evr_target", m.evr_target);
    return finish(std::move(result), out);
}

}  // namespace n4m::estimator
