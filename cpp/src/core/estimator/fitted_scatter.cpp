// SPDX-License-Identifier: CECILL-2.1
//
// Fitted transformer adapters whose kernel state is reachable through the
// public accessors (MSC and EMSC: the learned reference spectrum; set_reference
// recomputes the derived quantities exactly as fit does).

#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"

namespace n4m::estimator {

namespace {

template <typename H>
n4m_status_t save_reference(const H* h, n4m_state_writer_t* w,
                            n4m_status_t (*size)(const H*, std::int64_t*),
                            n4m_status_t (*get)(const H*, double*, std::int64_t)) {
    std::int64_t cols = 0;
    n4m_status_t st = size(h, &cols);
    if (st != N4M_OK) return st;
    std::vector<double> reference(static_cast<std::size_t>(cols));
    st = get(h, reference.data(), cols);
    if (st == N4M_OK) n4m_state_write_f64_array(w, reference.data(), cols);
    return st;
}

template <typename H>
n4m_status_t load_reference(H* h, n4m_state_reader_t* r, std::int64_t width,
                            n4m_status_t (*set)(H*, const double*, std::int64_t)) {
    std::vector<double> reference(static_cast<std::size_t>(width));
    if (!n4m_state_read_f64_array(r, reference.data(), width)) return N4M_ERR_CORRUPT_BUFFER;
    return set(h, reference.data(), width);
}

n4m_status_t fit_x(n4m_status_t (*fit)(n4m_pp_msc_handle_t*, n4m_matrix_view_t),
                   n4m_pp_msc_handle_t* h, const FitInputs& in) {
    std::vector<double> storage;
    return fit(h, contiguous_view(*in.X, storage));
}

}  // namespace

// MSC stores per-row coefficients for its inverse during transform, so its
// transform takes a mutable handle; the estimator owns it exclusively.
std::unique_ptr<Adapter> make_tr_msc(const MethodSpec&) {
    return fitted<n4m_pp_msc_handle_t>(
        {[](n4m_pp_msc_handle_t** h, const Params&) { return n4m_transform_msc_create(h); },
         [](n4m_context_t*, n4m_pp_msc_handle_t* h, const FitInputs& in) {
             return fit_x(n4m_transform_msc_fit, h, in);
         },
         [](const n4m_pp_msc_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t out) {
             return n4m_transform_msc_transform(const_cast<n4m_pp_msc_handle_t*>(h), X, out);
         },
         n4m_transform_msc_destroy,
         {},
         [](const n4m_pp_msc_handle_t* h, n4m_state_writer_t* w) {
             return save_reference(h, w, n4m_transform_msc_reference_size,
                                   n4m_transform_msc_get_reference);
         },
         [](n4m_pp_msc_handle_t* h, n4m_state_reader_t* r, std::int64_t width) {
             return load_reference(h, r, width, n4m_transform_msc_set_reference);
         }});
}

std::unique_ptr<Adapter> make_tr_emsc(const MethodSpec&) {
    return fitted<n4m_pp_emsc_handle_t>(
        {[](n4m_pp_emsc_handle_t** h, const Params& p) {
             return n4m_transform_emsc_create(h, to_i32(p.get_int("degree")));
         },
         [](n4m_context_t*, n4m_pp_emsc_handle_t* h, const FitInputs& in) {
             std::vector<double> storage;
             return n4m_transform_emsc_fit(h, contiguous_view(*in.X, storage));
         },
         n4m_transform_emsc_transform,
         n4m_transform_emsc_destroy,
         {},
         [](const n4m_pp_emsc_handle_t* h, n4m_state_writer_t* w) {
             return save_reference(h, w, n4m_transform_emsc_reference_size,
                                   n4m_transform_emsc_get_reference);
         },
         [](n4m_pp_emsc_handle_t* h, n4m_state_reader_t* r, std::int64_t width) {
             return load_reference(h, r, width, n4m_transform_emsc_set_reference);
         }});
}

}  // namespace n4m::estimator
