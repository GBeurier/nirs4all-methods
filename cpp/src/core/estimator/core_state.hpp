// SPDX-License-Identifier: CECILL-2.1
//
// Internal access from the public preprocessing handles to their kernel
// fitted state, for the estimator adapters (fitted_core.cpp). The handles
// are defined in cpp/src/c_api/*.cpp, where these functions forward to the
// kernels' <kernel>_state_save / _state_load (core/estimator/state_io.h).
// Not part of the public ABI.

#pragma once

#include <cstdint>

#include "n4m/n4m.h"
#include "core/estimator/state_io.h"

namespace n4m::estimator {

#define N4M_CORE_STATE_IO(H)                                                   \
    n4m_status_t kernel_save(const H* h, n4m_state_writer_t* w);              \
    n4m_status_t kernel_load(H* h, n4m_state_reader_t* r, std::int64_t n_features);

N4M_CORE_STATE_IO(n4m_pp_osc_handle_t)
N4M_CORE_STATE_IO(n4m_pp_epo_handle_t)
N4M_CORE_STATE_IO(n4m_pp_flex_pca_handle_t)
N4M_CORE_STATE_IO(n4m_pp_flex_svd_handle_t)
N4M_CORE_STATE_IO(n4m_pp_wavelet_pca_handle_t)
N4M_CORE_STATE_IO(n4m_pp_wavelet_svd_handle_t)
N4M_CORE_STATE_IO(n4m_pp_kbins_disc_handle_t)
N4M_CORE_STATE_IO(n4m_pp_baseline_handle_t)
N4M_CORE_STATE_IO(n4m_pp_log_handle_t)
N4M_CORE_STATE_IO(n4m_pp_derivate_handle_t)
N4M_CORE_STATE_IO(n4m_pp_resampler_handle_t)

#undef N4M_CORE_STATE_IO

// Output width for an input width, from parameters the handle holds (the
// public helpers take the parameters, not the handle).
std::int64_t kernel_output_cols(const n4m_pp_derivate_handle_t* h, std::int64_t input_cols);
std::int64_t kernel_output_cols(const n4m_pp_fck_static_handle_t* h, std::int64_t input_cols);

}  // namespace n4m::estimator
