/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared filter banks and low-level DWT / IDWT helpers for the Phase 6
 * wavelets preprocessing operators (Wavelet, Haar, WaveletDenoise,
 * WaveletFeatures, WaveletPCA, WaveletSVD).
 *
 * The v1 surface supports four wavelet families:
 *
 *   - haar  : order-2 filter, the canonical Daubechies-1 wavelet.
 *   - db4   : Daubechies-4, order-8 orthogonal wavelet.
 *   - sym4  : Symlet-4, order-8 near-symmetric orthogonal wavelet.
 *   - coif1 : Coiflet-1, order-6 orthogonal wavelet.
 *
 * Filter bank coefficients are vendored from PyWavelets 1.6.0 to give
 * exact bit-equality with the frozen NumPy reference under
 * ``parity/python_generator/src/n4m_parity_wavelets_ref/``.
 *
 * Three boundary modes are supported in v1:
 *
 *   - periodization : signal treated as periodic, output length is
 *                     ceil(n / 2).  Matches pywt mode='periodization'.
 *   - symmetric     : "edge-repeat" reflection (PyWavelets ``symmetric``),
 *                     output length floor((n + L - 1) / 2).
 *   - zero          : pad with zeros, output length floor((n + L - 1) / 2).
 *
 * The longer ``reflect``, ``antisymmetric`` and ``smooth`` PyWavelets modes
 * are intentionally deferred to a later phase; they are not required by
 * any nirs4all-tier example.
 *
 * Pure C, no dependencies.  INTERNAL: never exposed by the public ABI.
 */
#ifndef N4M_CORE_COMMON_WAVELET_KERNELS_H
#define N4M_CORE_COMMON_WAVELET_KERNELS_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wavelet family identifiers — v1 supports the four families listed above. */
typedef enum n4m_wavelet_family_t {
    N4M_WAVELET_HAAR  = 0,
    N4M_WAVELET_DB4   = 1,
    N4M_WAVELET_SYM4  = 2,
    N4M_WAVELET_COIF1 = 3
} n4m_wavelet_family_t;

/* Boundary modes — v1 subset of the PyWavelets modes. */
typedef enum n4m_wavelet_mode_t {
    N4M_WAVELET_MODE_PERIODIZATION = 0,
    N4M_WAVELET_MODE_SYMMETRIC     = 1,
    N4M_WAVELET_MODE_ZERO          = 2
} n4m_wavelet_mode_t;

/* Maximum supported filter length across the v1 families. db4 and sym4 are
 * length 8, coif1 is length 6, haar is length 2.  16 leaves a comfortable
 * margin for stack-allocated working buffers. */
#define N4M_WAVELET_MAX_FILTER 16

/* Resolve a wavelet family name (e.g. "haar", "db4", "sym4", "coif1") to
 * the corresponding enum value.  Returns 0 on success and writes the
 * resolved family to ``*out``; returns -1 when the name is unknown. */
int n4m_wavelet_family_from_name(const char* name, n4m_wavelet_family_t* out);

/* Resolve a boundary mode name (e.g. "periodization", "symmetric", "zero")
 * to the corresponding enum value.  Returns 0 on success and writes the
 * resolved mode to ``*out``; returns -1 when the name is unknown. */
int n4m_wavelet_mode_from_name(const char* name, n4m_wavelet_mode_t* out);

/* Return the filter length L for a given wavelet family. */
int32_t n4m_wavelet_filter_length(n4m_wavelet_family_t family);

/* Fill ``out_lo`` (low-pass) and ``out_hi`` (high-pass) with the L-tap
 * decomposition filters for the given family.  ``out_lo`` and ``out_hi``
 * must have capacity >= filter_length(family). */
void n4m_wavelet_dec_filters(n4m_wavelet_family_t family,
                              double* out_lo, double* out_hi);

/* Fill the L-tap reconstruction filters. */
void n4m_wavelet_rec_filters(n4m_wavelet_family_t family,
                              double* out_lo, double* out_hi);

/* Compute the single-level DWT output length for a signal of length
 * ``n`` with the given family and mode.  Returns 0 for empty signals. */
int64_t n4m_wavelet_dwt_output_length(int64_t n,
                                       n4m_wavelet_family_t family,
                                       n4m_wavelet_mode_t mode);

/* Compute the maximum decomposition level for a signal of length ``n``
 * with the given family — matches ``pywt.dwt_max_level(n, wavelet)``,
 * which is ``floor(log2(n / (L - 1)))`` clamped to >= 0.  For haar
 * (L=2) this reduces to ``floor(log2(n))``. */
int32_t n4m_wavelet_dwt_max_level(int64_t n, n4m_wavelet_family_t family);

/* Single-level DWT of a 1-D signal.
 *
 * Inputs:
 *   - ``in``   : length-n signal.
 *   - ``n``    : signal length (>= 1).
 *   - ``family`` / ``mode`` : filter bank + boundary mode.
 *
 * Outputs:
 *   - ``out_cA`` : approximation coefficients, length
 *                  ``n4m_wavelet_dwt_output_length(n, family, mode)``.
 *   - ``out_cD`` : detail coefficients, same length as out_cA.
 *
 * Returns ``N4M_OK`` on success, ``N4M_ERR_INVALID_ARGUMENT`` for n < 1. */
n4m_status_t n4m_wavelet_dwt(const double* in, int64_t n,
                              n4m_wavelet_family_t family,
                              n4m_wavelet_mode_t mode,
                              double* out_cA, double* out_cD);

/* Single-level inverse DWT.  ``cA`` and ``cD`` both have length ``m``
 * (>= 1); ``n_out`` is the target signal length.  The reconstructed
 * signal is written to ``out`` (length ``n_out``).  Caller may pass
 * ``cD = NULL`` to denote an all-zero detail vector (used by some
 * downstream callers).  Returns ``N4M_OK`` on success.
 *
 * ``n_out`` is taken from the multi-level metadata, so the inverse can
 * trim the natural ``L + 2*(m - 1)`` reconstruction back to the
 * original signal length. */
n4m_status_t n4m_wavelet_idwt(const double* cA, const double* cD, int64_t m,
                               int64_t n_out,
                               n4m_wavelet_family_t family,
                               n4m_wavelet_mode_t mode,
                               double* out);

/* Multi-level DWT decomposition (PyWavelets ``wavedec`` semantics).
 *
 * Layout: ``coeffs`` is a packed buffer of length
 *   m_0 + m_0 + m_1 + ... + m_{level-1}
 * where m_k is the length of the detail coefficients at level k+1 and
 * also (by construction) the length of the approximation coefficient
 * at level k.  ``coef_lengths`` (length ``level + 1``) is filled with
 * ``[m_top_cA, m_top_cD, m_{top-1}_cD, ..., m_0_cD]``, matching the
 * Python output order ``[cA_n, cD_n, cD_{n-1}, ..., cD_1]``.
 *
 * ``offsets`` (length ``level + 1``) is filled with the start offset of
 * each chunk in ``coeffs``.  Both arrays are owned by the caller. */
n4m_status_t n4m_wavelet_wavedec(const double* in, int64_t n,
                                  n4m_wavelet_family_t family,
                                  n4m_wavelet_mode_t mode,
                                  int32_t level,
                                  double* coeffs,
                                  int64_t* coef_lengths,
                                  int64_t* offsets,
                                  int64_t coeffs_capacity);

/* Workspace-backed variant of ``n4m_wavelet_wavedec``.  The workspace is
 * caller-owned and may be reused across rows with identical shape/params. */
n4m_status_t n4m_wavelet_wavedec_workspace_length(
    int64_t n,
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t mode,
    int32_t level,
    int64_t* out_work_len);

n4m_status_t n4m_wavelet_wavedec_with_workspace(
    const double* in, int64_t n,
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t mode,
    int32_t level,
    double* coeffs,
    int64_t* coef_lengths,
    int64_t* offsets,
    int64_t coeffs_capacity,
    double* work,
    int64_t work_capacity);

/* Compute the total flattened coefficient length and per-level lengths
 * for a multi-level decomposition.  Used by callers to size buffers
 * before calling ``n4m_wavelet_wavedec``. */
n4m_status_t n4m_wavelet_wavedec_lengths(int64_t n,
                                          n4m_wavelet_family_t family,
                                          n4m_wavelet_mode_t mode,
                                          int32_t level,
                                          int64_t* out_total,
                                          int64_t* out_coef_lengths);

/* Multi-level reconstruction (PyWavelets ``waverec``) restoring a
 * length-n signal from packed coefficients with the same layout written
 * by ``n4m_wavelet_wavedec``.  Output is truncated to length ``n``. */
n4m_status_t n4m_wavelet_waverec(const double* coeffs,
                                  const int64_t* coef_lengths,
                                  const int64_t* offsets,
                                  int32_t level,
                                  n4m_wavelet_family_t family,
                                  n4m_wavelet_mode_t mode,
                                  int64_t n_out,
                                  double* out);

n4m_status_t n4m_wavelet_waverec_workspace_length(
    int64_t n_out,
    int32_t level,
    int64_t* out_work_len);

n4m_status_t n4m_wavelet_waverec_with_workspace(
    const double* coeffs,
    const int64_t* coef_lengths,
    const int64_t* offsets,
    int32_t level,
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t mode,
    int64_t n_out,
    double* out,
    double* work,
    int64_t work_capacity);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_WAVELET_KERNELS_H */
