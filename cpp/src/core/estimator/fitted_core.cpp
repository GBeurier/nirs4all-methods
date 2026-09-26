// SPDX-License-Identifier: CECILL-2.1
//
// Transformer role adapters whose fitted state lives in the core kernels
// (cpp/src/core/preprocessing/**): OSC, flexible and wavelet PCA/SVD, the
// discretizers, baseline centring, log transform, derivative, resampler and
// the fitted Normalize / SimpleScale. The kernel state is serialized by the
// kernels' own <kernel>_state_save / _state_load, reached from the public
// handles through core_state.hpp. RangeDiscretizer and FCK static learn
// nothing: their state is the input width alone.

#include <cstdint>
#include <functional>
#include <vector>

#include "core/estimator/core_state.hpp"
#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/preprocessing/scaling/normalize.h"
#include "core/preprocessing/scaling/simple_scale.h"

namespace n4m::estimator {

namespace {

// Fit on a contiguous copy of X.
template <typename H>
std::function<n4m_status_t(n4m_context_t*, H*, const FitInputs&)> fit_on_x(
    n4m_status_t (*fit)(H*, n4m_matrix_view_t)) {
    return [fit](n4m_context_t*, H* h, const FitInputs& in) {
        std::vector<double> storage;
        return fit(h, contiguous_view(*in.X, storage));
    };
}

// Serialization through the public handle (core_state.hpp).
template <typename H>
FittedKernel<H> with_core_state(FittedKernel<H> kernel) {
    kernel.save = [](const H* h, n4m_state_writer_t* w) { return kernel_save(h, w); };
    kernel.load = [](H* h, n4m_state_reader_t* r, std::int64_t n) { return kernel_load(h, r, n); };
    return kernel;
}

// Kernels that learn nothing beyond the input width.
template <typename H>
FittedKernel<H> width_only(FittedKernel<H> kernel) {
    kernel.fit = [](n4m_context_t*, H*, const FitInputs&) { return N4M_OK; };
    kernel.save = [](const H*, n4m_state_writer_t*) { return N4M_OK; };
    kernel.load = [](H*, n4m_state_reader_t*, std::int64_t) { return N4M_OK; };
    return kernel;
}

// Runs an integer-coding kernel (int32 output) into the role's F64 output,
// which contiguous_call hands over row-major.
n4m_status_t int32_codes(n4m_matrix_view_t out,
                         const std::function<n4m_status_t(n4m_matrix_view_t)>& kernel) {
    std::vector<std::int32_t> codes(static_cast<std::size_t>(out.rows * out.cols));
    n4m_matrix_view_t view{};
    n4m_matrix_view_init_rowmajor(&view, codes.data(), out.rows, out.cols, N4M_DTYPE_I32);
    const n4m_status_t st = kernel(view);
    if (st != N4M_OK) return st;
    auto* dst = static_cast<double*>(out.data);
    for (std::size_t i = 0; i < codes.size(); ++i) dst[i] = static_cast<double>(codes[i]);
    return N4M_OK;
}

// Kernel state owned directly (no public handle): the fitted Normalize and
// SimpleScale variants.
template <typename S>
FittedKernel<S> owned_state(S* (*make)(const Params&), void (*destroy)(S*),
                            n4m_status_t (*fit)(S*, const double*, std::int64_t, std::int64_t),
                            n4m_status_t (*apply)(const S*, const double*, std::int64_t,
                                                  std::int64_t, double*),
                            n4m_status_t (*save)(const S*, n4m_state_writer_t*),
                            n4m_status_t (*load)(S*, n4m_state_reader_t*, std::int64_t)) {
    return {[make](S** s, const Params& p) {
                *s = make(p);
                return *s != nullptr ? N4M_OK : N4M_ERR_OUT_OF_MEMORY;
            },
            [fit](n4m_context_t*, S* s, const FitInputs& in) {
                std::vector<double> storage;
                const n4m_matrix_view_t x = contiguous_view(*in.X, storage);
                return fit(s, static_cast<const double*>(x.data), x.rows, x.cols);
            },
            [apply](const S* s, n4m_matrix_view_t x, n4m_matrix_view_t out) {
                return apply(s, static_cast<const double*>(x.data), x.rows, x.cols,
                             static_cast<double*>(out.data));
            },
            destroy,
            {},
            save,
            load};
}

}  // namespace

// ---- orthogonalization -----------------------------------------------------

std::unique_ptr<Adapter> make_tr_osc(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_osc_handle_t>(
        {[](n4m_pp_osc_handle_t** h, const Params& p) {
             return n4m_transform_osc_create(h, to_i32(p.get_int("n_components")),
                                             p.get_bool("scale") ? 1 : 0);
         },
         [](n4m_context_t* ctx, n4m_pp_osc_handle_t* h, const FitInputs& in) {
             if (in.Y->cols != 1) {
                 set_error(ctx, "OSC needs a univariate y");
                 return N4M_ERR_SHAPE_MISMATCH;
             }
             std::vector<double> x_storage, y_storage;
             const n4m_matrix_view_t y = contiguous_view(*in.Y, y_storage);
             return n4m_transform_osc_fit(h, contiguous_view(*in.X, x_storage),
                                          static_cast<const double*>(y.data), y.rows);
         },
         n4m_transform_osc_transform, n4m_transform_osc_destroy, {}, {}, {}}));
}

// ---- projections -----------------------------------------------------------

std::unique_ptr<Adapter> make_tr_flexible_pca(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_flex_pca_handle_t>(
        {[](n4m_pp_flex_pca_handle_t** h, const Params& p) {
             return n4m_decomposition_flexible_pca_create(h, p.get_double("n_components"));
         },
         fit_on_x(n4m_decomposition_flexible_pca_fit), n4m_decomposition_flexible_pca_transform,
         n4m_decomposition_flexible_pca_destroy,
         [](const n4m_pp_flex_pca_handle_t* h, std::int64_t, std::int64_t* out) {
             return n4m_decomposition_flexible_pca_output_cols(h, out);
         },
         {}, {}}));
}

std::unique_ptr<Adapter> make_tr_flexible_svd(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_flex_svd_handle_t>(
        {[](n4m_pp_flex_svd_handle_t** h, const Params& p) {
             return n4m_decomposition_flexible_svd_create(h, p.get_double("n_components"));
         },
         fit_on_x(n4m_decomposition_flexible_svd_fit), n4m_decomposition_flexible_svd_transform,
         n4m_decomposition_flexible_svd_destroy,
         [](const n4m_pp_flex_svd_handle_t* h, std::int64_t, std::int64_t* out) {
             return n4m_decomposition_flexible_svd_output_cols(h, out);
         },
         {}, {}}));
}

std::unique_ptr<Adapter> make_tr_wavelet_pca(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_wavelet_pca_handle_t>(
        {[](n4m_pp_wavelet_pca_handle_t** h, const Params& p) {
             return n4m_transform_wavelet_pca_create(
                 h, static_cast<n4m_pp_wavelet_family_t>(p.get_int("family")),
                 static_cast<n4m_pp_wavelet_boundary_t>(p.get_int("mode")),
                 to_i32(p.get_int("max_level")), p.get_double("n_components"));
         },
         fit_on_x(n4m_transform_wavelet_pca_fit), n4m_transform_wavelet_pca_transform,
         n4m_transform_wavelet_pca_destroy,
         [](const n4m_pp_wavelet_pca_handle_t* h, std::int64_t, std::int64_t* out) {
             return n4m_transform_wavelet_pca_output_cols(h, out);
         },
         {}, {}}));
}

std::unique_ptr<Adapter> make_tr_wavelet_svd(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_wavelet_svd_handle_t>(
        {[](n4m_pp_wavelet_svd_handle_t** h, const Params& p) {
             return n4m_transform_wavelet_svd_create(
                 h, static_cast<n4m_pp_wavelet_family_t>(p.get_int("family")),
                 static_cast<n4m_pp_wavelet_boundary_t>(p.get_int("mode")),
                 to_i32(p.get_int("max_level")), p.get_double("n_components"));
         },
         fit_on_x(n4m_transform_wavelet_svd_fit), n4m_transform_wavelet_svd_transform,
         n4m_transform_wavelet_svd_destroy,
         [](const n4m_pp_wavelet_svd_handle_t* h, std::int64_t, std::int64_t* out) {
             return n4m_transform_wavelet_svd_output_cols(h, out);
         },
         {}, {}}));
}

// ---- discretizers and resampling ------------------------------------------

std::unique_ptr<Adapter> make_tr_kbins_discretizer(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_kbins_disc_handle_t>(
        {[](n4m_pp_kbins_disc_handle_t** h, const Params& p) {
             return n4m_transform_kbins_discretizer_create(h, to_i32(p.get_int("n_bins")),
                                                           to_i32(p.get_int("strategy")));
         },
         fit_on_x(n4m_transform_kbins_discretizer_fit),
         [](const n4m_pp_kbins_disc_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
             return int32_codes(out, [&](n4m_matrix_view_t codes) {
                 return n4m_transform_kbins_discretizer_transform(h, X, codes);
             });
         },
         n4m_transform_kbins_discretizer_destroy, {}, {}, {}}));
}

std::unique_ptr<Adapter> make_tr_range_discretizer(const MethodSpec&) {
    return fitted(width_only<n4m_pp_range_disc_handle_t>(
        {[](n4m_pp_range_disc_handle_t** h, const Params& p) {
             const std::vector<double> edges = p.get_doubles("edges");
             return n4m_transform_range_discretizer_create(
                 h, edges.data(), static_cast<std::int64_t>(edges.size()));
         },
         {},
         [](const n4m_pp_range_disc_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
             return int32_codes(out, [&](n4m_matrix_view_t codes) {
                 return n4m_transform_range_discretizer_transform(h, X, codes);
             });
         },
         n4m_transform_range_discretizer_destroy, {}, {}, {}}));
}

// The target grid is `target_wavelengths`, or tgt_min + tgt_step * i for
// i < tgt_n; neither selects the identity grid (the post-crop source axis).
std::unique_ptr<Adapter> make_tr_resampler(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_resampler_handle_t>(
        {[](n4m_pp_resampler_handle_t** h, const Params& p) {
             std::vector<double> target = p.get_doubles("target_wavelengths");
             const std::int64_t tgt_n = p.get_int("tgt_n");
             if (tgt_n > 0) {
                 if (!target.empty()) return N4M_ERR_INVALID_ARGUMENT;
                 const double tgt_min = p.get_double("tgt_min");
                 const double tgt_step = p.get_double("tgt_step");
                 for (std::int64_t i = 0; i < tgt_n; ++i) {
                     target.push_back(tgt_min + tgt_step * static_cast<double>(i));
                 }
             }
             return n4m_transform_resampler_create(
                 h, target.empty() ? nullptr : target.data(),
                 target.empty() ? -1 : static_cast<std::int64_t>(target.size()),
                 to_i32(p.get_int("method")), p.get_double("crop_min"), p.get_double("crop_max"),
                 p.get_bool("use_crop") ? 1 : 0, p.get_double("fill_value"),
                 p.get_bool("bounds_error") ? 1 : 0, p.get_bool("extrapolate") ? 1 : 0);
         },
         [](n4m_context_t*, n4m_pp_resampler_handle_t* h, const FitInputs& in) {
             return n4m_transform_resampler_fit(h, in.axis, in.n_axis);
         },
         n4m_transform_resampler_transform, n4m_transform_resampler_destroy,
         [](const n4m_pp_resampler_handle_t* h, std::int64_t, std::int64_t* out) {
             *out = n4m_transform_resampler_output_cols(h);
             return N4M_OK;
         },
         {}, {}}));
}

// ---- scaling and derivative -----------------------------------------------

std::unique_ptr<Adapter> make_tr_baseline(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_baseline_handle_t>(
        {[](n4m_pp_baseline_handle_t** h, const Params&) {
             return n4m_transform_baseline_center_create(h);
         },
         fit_on_x(n4m_transform_baseline_center_fit), n4m_transform_baseline_center_transform,
         n4m_transform_baseline_center_destroy, {}, {}, {}}));
}

std::unique_ptr<Adapter> make_tr_log_transform(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_log_handle_t>(
        {[](n4m_pp_log_handle_t** h, const Params& p) {
             return n4m_transform_log_transform_create(h, p.get_double("base"),
                                                       p.get_double("offset"),
                                                       p.get_bool("auto_offset") ? 1 : 0,
                                                       p.get_double("min_value"));
         },
         fit_on_x(n4m_transform_log_transform_fit), n4m_transform_log_transform_transform,
         n4m_transform_log_transform_destroy, {}, {}, {}}));
}

std::unique_ptr<Adapter> make_tr_derivate(const MethodSpec&) {
    return fitted(with_core_state<n4m_pp_derivate_handle_t>(
        {[](n4m_pp_derivate_handle_t** h, const Params& p) {
             return n4m_transform_derivative_create(h, to_i32(p.get_int("order")),
                                                    p.get_double("delta"));
         },
         fit_on_x(n4m_transform_derivative_fit), n4m_transform_derivative_transform,
         n4m_transform_derivative_destroy,
         [](const n4m_pp_derivate_handle_t* h, std::int64_t in, std::int64_t* out) {
             *out = kernel_output_cols(h, in);
             return N4M_OK;
         },
         {}, {}}));
}

// Column statistics learned at fit and applied to new rows (the public
// n4m_transform_normalize / simple_scale recompute them per batch).
std::unique_ptr<Adapter> make_tr_normalize(const MethodSpec&) {
    return fitted(owned_state<n4m_pp_normalize_state_t>(
        [](const Params& p) {
            return n4m_pp_normalize_state_new(p.get_double("feature_min"),
                                              p.get_double("feature_max"));
        },
        n4m_pp_normalize_state_free, n4m_pp_normalize_state_fit, n4m_pp_normalize_apply,
        n4m_pp_normalize_state_save, n4m_pp_normalize_state_load));
}

std::unique_ptr<Adapter> make_tr_simple_scale(const MethodSpec&) {
    return fitted(owned_state<n4m_pp_simple_scale_fit_state_t>(
        [](const Params&) { return n4m_pp_simple_scale_fit_state_new(); },
        n4m_pp_simple_scale_fit_state_free, n4m_pp_simple_scale_fit_state_fit,
        n4m_pp_simple_scale_fit_state_apply, n4m_pp_simple_scale_fit_state_save,
        n4m_pp_simple_scale_fit_state_load));
}

// ---- specialized -----------------------------------------------------------

std::unique_ptr<Adapter> make_tr_fck_static(const MethodSpec&) {
    return fitted(width_only<n4m_pp_fck_static_handle_t>(
        {[](n4m_pp_fck_static_handle_t** h, const Params& p) {
             const std::vector<double> alphas = p.get_doubles("alphas");
             const std::vector<double> sigmas = p.get_doubles("sigmas");
             return n4m_transform_fck_static_create(
                 h, to_i32(p.get_int("kernel_size")), alphas.data(),
                 to_i32(static_cast<std::int64_t>(alphas.size())), sigmas.data(),
                 to_i32(static_cast<std::int64_t>(sigmas.size())));
         },
         {}, n4m_transform_fck_static_transform, n4m_transform_fck_static_destroy,
         [](const n4m_pp_fck_static_handle_t* h, std::int64_t in, std::int64_t* out) {
             *out = kernel_output_cols(h, in);
             return N4M_OK;
         },
         {}, {}}));
}

}  // namespace n4m::estimator
