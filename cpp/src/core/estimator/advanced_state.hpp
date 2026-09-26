// SPDX-License-Identifier: CECILL-2.1
//
// Fitted-state save / load of the advanced transformer kernels whose state
// lives in c_api/c_api_advanced.cpp (internal, not part of the public ABI;
// see state_io.h). Load restores the learned fields for an input width of
// `n_features` on a handle created with the fit-time parameters, recomputes
// the derived quantities as fit does and marks the handle fitted.

#pragma once

#include <cstdint>

#include "n4m/n4m.h"
#include "core/estimator/state_io.h"

// Alignment: the reference spectrum (learned column mean or the create-time
// reference).
n4m_status_t xcorr_align_state_save(const n4m_pp_xcorr_align_handle_t* h, n4m_state_writer_t* w);
n4m_status_t xcorr_align_state_load(n4m_pp_xcorr_align_handle_t* h, n4m_state_reader_t* r,
                                    int64_t n_features);
n4m_status_t icoshift_align_state_save(const n4m_pp_icoshift_align_handle_t* h,
                                       n4m_state_writer_t* w);
n4m_status_t icoshift_align_state_load(n4m_pp_icoshift_align_handle_t* h, n4m_state_reader_t* r,
                                       int64_t n_features);
n4m_status_t dtw_align_state_save(const n4m_pp_dtw_align_handle_t* h, n4m_state_writer_t* w);
n4m_status_t dtw_align_state_load(n4m_pp_dtw_align_handle_t* h, n4m_state_reader_t* r,
                                  int64_t n_features);
n4m_status_t cow_align_state_save(const n4m_pp_cow_align_handle_t* h, n4m_state_writer_t* w);
n4m_status_t cow_align_state_load(n4m_pp_cow_align_handle_t* h, n4m_state_reader_t* r,
                                  int64_t n_features);

// Paired source -> target maps: DS / robust DS coefficients, PDS per-window
// coefficients, SAPS mean + loadings + coefficients, local-centering offsets.
n4m_status_t direct_standardization_state_save(const n4m_pp_direct_standardization_handle_t* h,
                                               n4m_state_writer_t* w);
n4m_status_t direct_standardization_state_load(n4m_pp_direct_standardization_handle_t* h,
                                               n4m_state_reader_t* r, int64_t n_features);
n4m_status_t robust_direct_standardization_state_save(
    const n4m_pp_robust_direct_standardization_handle_t* h, n4m_state_writer_t* w);
n4m_status_t robust_direct_standardization_state_load(
    n4m_pp_robust_direct_standardization_handle_t* h, n4m_state_reader_t* r, int64_t n_features);
n4m_status_t piecewise_direct_standardization_state_save(
    const n4m_pp_piecewise_direct_standardization_handle_t* h, n4m_state_writer_t* w);
n4m_status_t piecewise_direct_standardization_state_load(
    n4m_pp_piecewise_direct_standardization_handle_t* h, n4m_state_reader_t* r,
    int64_t n_features);
n4m_status_t saps_state_save(const n4m_pp_saps_handle_t* h, n4m_state_writer_t* w);
n4m_status_t saps_state_load(n4m_pp_saps_handle_t* h, n4m_state_reader_t* r, int64_t n_features);
n4m_status_t local_centering_state_save(const n4m_pp_local_centering_handle_t* h,
                                        n4m_state_writer_t* w);
n4m_status_t local_centering_state_load(n4m_pp_local_centering_handle_t* h, n4m_state_reader_t* r,
                                        int64_t n_features);

// Scatter: piecewise / localized MSC reference, VSN weights; piecewise and
// weighted SNV store nothing beyond the width.
n4m_status_t piecewise_msc_state_save(const n4m_pp_piecewise_msc_handle_t* h,
                                      n4m_state_writer_t* w);
n4m_status_t piecewise_msc_state_load(n4m_pp_piecewise_msc_handle_t* h, n4m_state_reader_t* r,
                                      int64_t n_features);
n4m_status_t localized_msc_state_save(const n4m_pp_localized_msc_handle_t* h,
                                      n4m_state_writer_t* w);
n4m_status_t localized_msc_state_load(n4m_pp_localized_msc_handle_t* h, n4m_state_reader_t* r,
                                      int64_t n_features);
n4m_status_t vsn_state_save(const n4m_pp_vsn_handle_t* h, n4m_state_writer_t* w);
n4m_status_t vsn_state_load(n4m_pp_vsn_handle_t* h, n4m_state_reader_t* r, int64_t n_features);
n4m_status_t weighted_snv_state_save(const n4m_pp_weighted_snv_handle_t* h, n4m_state_writer_t* w);
n4m_status_t weighted_snv_state_load(n4m_pp_weighted_snv_handle_t* h, n4m_state_reader_t* r,
                                     int64_t n_features);
n4m_status_t piecewise_snv_state_save(const n4m_pp_piecewise_snv_handle_t* h,
                                      n4m_state_writer_t* w);
n4m_status_t piecewise_snv_state_load(n4m_pp_piecewise_snv_handle_t* h, n4m_state_reader_t* r,
                                      int64_t n_features);

// Slope / bias: the fitted pair. Interval generator: nothing beyond the
// width (the bands follow from the create-time width and step).
n4m_status_t slope_bias_state_save(const n4m_pp_slope_bias_handle_t* h, n4m_state_writer_t* w);
n4m_status_t slope_bias_state_load(n4m_pp_slope_bias_handle_t* h, n4m_state_reader_t* r,
                                   int64_t n_features);
n4m_status_t interval_generator_state_save(const n4m_interval_generator_handle_t* h,
                                           n4m_state_writer_t* w);
n4m_status_t interval_generator_state_load(n4m_interval_generator_handle_t* h,
                                           n4m_state_reader_t* r, int64_t n_features);
