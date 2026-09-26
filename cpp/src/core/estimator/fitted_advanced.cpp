// SPDX-License-Identifier: CECILL-2.1
//
// Fitted transformer adapters over the advanced kernels of c_api_advanced.cpp:
// alignment, paired source -> target standardization, and piecewise / localized
// / weighted scatter corrections. Their fitted state is written and restored by
// the kernel-side save / load functions (advanced_state.hpp).

#include <vector>

#include "core/estimator/advanced_state.hpp"
#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"

namespace n4m::estimator {

namespace {

template <typename H>
n4m_status_t fit_x(n4m_status_t (*fit)(H*, n4m_matrix_view_t), H* h, const FitInputs& in) {
    std::vector<double> storage;
    return fit(h, contiguous_view(*in.X, storage));
}

// Paired fit on the source rows (X) and the target-domain rows.
template <typename H>
n4m_status_t fit_pair(n4m_status_t (*fit)(H*, n4m_matrix_view_t, n4m_matrix_view_t), H* h,
                      const FitInputs& in) {
    std::vector<double> source, target;
    return fit(h, contiguous_view(*in.X, source), contiguous_view(*in.X_target, target));
}

// Alignment kernels share one create signature: an optional reference
// (empty = column mean of the fit X), the interval size and the maximum shift.
// A method without the interval / shift parameter passes 0.
template <typename H>
using AlignCreate = n4m_status_t (*)(H**, const double*, int64_t, int32_t, int32_t);

template <typename H>
std::unique_ptr<Adapter> align(AlignCreate<H> create, const char* interval_param,
                               const char* shift_param,
                               n4m_status_t (*fit)(H*, n4m_matrix_view_t),
                               n4m_status_t (*transform)(const H*, n4m_matrix_view_t,
                                                         n4m_matrix_view_t),
                               void (*destroy)(H*),
                               n4m_status_t (*save)(const H*, n4m_state_writer_t*),
                               n4m_status_t (*load)(H*, n4m_state_reader_t*, int64_t)) {
    return fitted<H>(
        {[=](H** h, const Params& p) {
             const std::vector<double> reference = p.get_doubles("reference");
             const std::int64_t size = interval_param ? p.get_int(interval_param) : 0;
             const std::int64_t shift = shift_param ? p.get_int(shift_param) : 0;
             return create(h, reference.data(), static_cast<int64_t>(reference.size()),
                           to_i32(size), to_i32(shift));
         },
         [fit](n4m_context_t*, H* h, const FitInputs& in) { return fit_x(fit, h, in); },
         transform, destroy, {}, save, load});
}

// Piecewise / localized MSC: window, optional reference, eps.
template <typename H>
std::unique_ptr<Adapter> msc(n4m_status_t (*create)(H**, const double*, int64_t, int32_t, double),
                             n4m_status_t (*fit)(H*, n4m_matrix_view_t),
                             n4m_status_t (*transform)(const H*, n4m_matrix_view_t,
                                                       n4m_matrix_view_t),
                             void (*destroy)(H*),
                             n4m_status_t (*save)(const H*, n4m_state_writer_t*),
                             n4m_status_t (*load)(H*, n4m_state_reader_t*, int64_t)) {
    return fitted<H>(
        {[create](H** h, const Params& p) {
             const std::vector<double> reference = p.get_doubles("reference");
             return create(h, reference.data(), static_cast<int64_t>(reference.size()),
                           to_i32(p.get_int("window_size")), p.get_double("eps"));
         },
         [fit](n4m_context_t*, H* h, const FitInputs& in) { return fit_x(fit, h, in); },
         transform, destroy, {}, save, load});
}

}  // namespace

std::unique_ptr<Adapter> make_tr_xcorr_align(const MethodSpec&) {
    return align<n4m_pp_xcorr_align_handle_t>(
        n4m_transform_xcorr_align_create, nullptr, "max_shift",
        n4m_transform_xcorr_align_fit, n4m_transform_xcorr_align_transform,
        n4m_transform_xcorr_align_destroy, xcorr_align_state_save, xcorr_align_state_load);
}

std::unique_ptr<Adapter> make_tr_icoshift_align(const MethodSpec&) {
    return align<n4m_pp_icoshift_align_handle_t>(
        n4m_transform_icoshift_align_create, "interval_size", "max_shift",
        n4m_transform_icoshift_align_fit, n4m_transform_icoshift_align_transform,
        n4m_transform_icoshift_align_destroy, icoshift_align_state_save,
        icoshift_align_state_load);
}

std::unique_ptr<Adapter> make_tr_dtw_align(const MethodSpec&) {
    return align<n4m_pp_dtw_align_handle_t>(
        n4m_transform_dtw_align_create, nullptr, nullptr, n4m_transform_dtw_align_fit,
        n4m_transform_dtw_align_transform, n4m_transform_dtw_align_destroy,
        dtw_align_state_save, dtw_align_state_load);
}

std::unique_ptr<Adapter> make_tr_cow_align(const MethodSpec&) {
    return align<n4m_pp_cow_align_handle_t>(
        n4m_transform_cow_align_create, "interval_size", "max_shift",
        n4m_transform_cow_align_fit, n4m_transform_cow_align_transform,
        n4m_transform_cow_align_destroy, cow_align_state_save, cow_align_state_load);
}

std::unique_ptr<Adapter> make_tr_direct_standardization(const MethodSpec&) {
    using H = n4m_pp_direct_standardization_handle_t;
    return fitted<H>(
        {[](H** h, const Params& p) {
             return n4m_domain_adaptation_direct_standardization_create(
                 h, p.get_bool("fit_intercept") ? 1 : 0, p.get_double("ridge"));
         },
         [](n4m_context_t*, H* h, const FitInputs& in) {
             return fit_pair(n4m_domain_adaptation_direct_standardization_fit, h, in);
         },
         n4m_domain_adaptation_direct_standardization_transform,
         n4m_domain_adaptation_direct_standardization_destroy,
         {},
         direct_standardization_state_save,
         direct_standardization_state_load});
}

std::unique_ptr<Adapter> make_tr_robust_direct_standardization(const MethodSpec&) {
    using H = n4m_pp_robust_direct_standardization_handle_t;
    return fitted<H>(
        {[](H** h, const Params& p) {
             return n4m_domain_adaptation_robust_direct_standardization_create(
                 h, p.get_bool("fit_intercept") ? 1 : 0, p.get_double("ridge"),
                 p.get_double("trim_quantile"), to_i32(p.get_int("max_iter")));
         },
         [](n4m_context_t*, H* h, const FitInputs& in) {
             return fit_pair(n4m_domain_adaptation_robust_direct_standardization_fit, h, in);
         },
         n4m_domain_adaptation_robust_direct_standardization_transform,
         n4m_domain_adaptation_robust_direct_standardization_destroy,
         {},
         robust_direct_standardization_state_save,
         robust_direct_standardization_state_load});
}

std::unique_ptr<Adapter> make_tr_piecewise_direct_standardization(const MethodSpec&) {
    using H = n4m_pp_piecewise_direct_standardization_handle_t;
    return fitted<H>(
        {[](H** h, const Params& p) {
             return n4m_domain_adaptation_piecewise_direct_standardization_create(
                 h, to_i32(p.get_int("window_size")), p.get_bool("fit_intercept") ? 1 : 0,
                 p.get_double("ridge"));
         },
         [](n4m_context_t*, H* h, const FitInputs& in) {
             return fit_pair(n4m_domain_adaptation_piecewise_direct_standardization_fit, h, in);
         },
         n4m_domain_adaptation_piecewise_direct_standardization_transform,
         n4m_domain_adaptation_piecewise_direct_standardization_destroy,
         {},
         piecewise_direct_standardization_state_save,
         piecewise_direct_standardization_state_load});
}

std::unique_ptr<Adapter> make_tr_saps(const MethodSpec&) {
    using H = n4m_pp_saps_handle_t;
    return fitted<H>(
        {[](H** h, const Params& p) {
             return n4m_transform_saps_create(h, to_i32(p.get_int("n_components")),
                                              p.get_double("score_weight"),
                                              p.get_bool("fit_intercept") ? 1 : 0,
                                              p.get_double("ridge"));
         },
         [](n4m_context_t*, H* h, const FitInputs& in) {
             return fit_pair(n4m_transform_saps_fit, h, in);
         },
         n4m_transform_saps_transform,
         n4m_transform_saps_destroy,
         {},
         saps_state_save,
         saps_state_load});
}

std::unique_ptr<Adapter> make_tr_local_centering(const MethodSpec&) {
    using H = n4m_pp_local_centering_handle_t;
    return fitted<H>({[](H** h, const Params&) { return n4m_transform_local_centering_create(h); },
                      [](n4m_context_t*, H* h, const FitInputs& in) {
                          return fit_pair(n4m_transform_local_centering_fit, h, in);
                      },
                      n4m_transform_local_centering_transform,
                      n4m_transform_local_centering_destroy,
                      {},
                      local_centering_state_save,
                      local_centering_state_load});
}

std::unique_ptr<Adapter> make_tr_piecewise_msc(const MethodSpec&) {
    return msc<n4m_pp_piecewise_msc_handle_t>(
        n4m_transform_piecewise_msc_create, n4m_transform_piecewise_msc_fit,
        n4m_transform_piecewise_msc_transform, n4m_transform_piecewise_msc_destroy,
        piecewise_msc_state_save, piecewise_msc_state_load);
}

std::unique_ptr<Adapter> make_tr_localized_msc(const MethodSpec&) {
    return msc<n4m_pp_localized_msc_handle_t>(
        n4m_transform_localized_msc_create, n4m_transform_localized_msc_fit,
        n4m_transform_localized_msc_transform, n4m_transform_localized_msc_destroy,
        localized_msc_state_save, localized_msc_state_load);
}

std::unique_ptr<Adapter> make_tr_piecewise_snv(const MethodSpec&) {
    using H = n4m_pp_piecewise_snv_handle_t;
    return fitted<H>(
        {[](H** h, const Params& p) {
             return n4m_transform_piecewise_snv_create(h, to_i32(p.get_int("window_size")),
                                                       to_i32(p.get_int("ddof")),
                                                       p.get_double("eps"));
         },
         [](n4m_context_t*, H* h, const FitInputs& in) {
             return fit_x(n4m_transform_piecewise_snv_fit, h, in);
         },
         n4m_transform_piecewise_snv_transform,
         n4m_transform_piecewise_snv_destroy,
         {},
         piecewise_snv_state_save,
         piecewise_snv_state_load});
}

std::unique_ptr<Adapter> make_tr_weighted_snv(const MethodSpec&) {
    using H = n4m_pp_weighted_snv_handle_t;
    return fitted<H>(
        {[](H** h, const Params& p) {
             const std::vector<double> weights = p.get_doubles("weights");
             return n4m_transform_weighted_snv_create(h, weights.data(),
                                                      static_cast<int64_t>(weights.size()),
                                                      to_i32(p.get_int("ddof")),
                                                      p.get_double("eps"));
         },
         [](n4m_context_t*, H* h, const FitInputs& in) {
             return fit_x(n4m_transform_weighted_snv_fit, h, in);
         },
         n4m_transform_weighted_snv_transform,
         n4m_transform_weighted_snv_destroy,
         {},
         weighted_snv_state_save,
         weighted_snv_state_load});
}

std::unique_ptr<Adapter> make_tr_vsn(const MethodSpec&) {
    using H = n4m_pp_vsn_handle_t;
    return fitted<H>(
        {[](H** h, const Params& p) { return n4m_transform_vsn_create(h, p.get_double("eps")); },
         [](n4m_context_t*, H* h, const FitInputs& in) {
             return fit_x(n4m_transform_vsn_fit, h, in);
         },
         n4m_transform_vsn_transform,
         n4m_transform_vsn_destroy,
         {},
         vsn_state_save,
         vsn_state_load});
}

}  // namespace n4m::estimator
