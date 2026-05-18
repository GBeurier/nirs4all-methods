/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Heuristic spectroscopic signal type detection.
 *
 * Mirrors `nirs4all.data.signal_type.SignalTypeDetector.detect` so that the
 * Python and C engines agree on which signal class (absorbance, reflectance,
 * percent variants, etc.) a spectral matrix most likely represents. The
 * underlying heuristic is deterministic: value range, mean, std, and an
 * optional water-band cue at NIR canonical wavelengths.
 *
 * Internal-only; the public wrapper lives in c4a_api_signal_type.cpp.
 */
#ifndef CHEMOMETRICS4ALL_CORE_UTILITIES_SIGNAL_TYPE_DETECTOR_H
#define CHEMOMETRICS4ALL_CORE_UTILITIES_SIGNAL_TYPE_DETECTOR_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the detector's heuristic. `wavelengths` may be NULL; when non-NULL
 * it must have `cols` elements and the values are interpreted in nanometres.
 *
 *   X                    : row-major matrix of size rows*cols (rows or cols
 *                          may be 0 — empty matrices return UNKNOWN).
 *   rows, cols           : matrix shape; both >= 0.
 *   wavelengths          : NULL or length-cols axis in nanometres.
 *   confidence_threshold : if 0 < threshold <= 1 the detected enum is
 *                          downgraded to UNKNOWN when the computed confidence
 *                          falls below it. threshold <= 0 disables the check.
 *   type_out, conf_out   : required outputs; conf_out always in [0, 1].
 *   reason_buf           : 256-byte caller-owned buffer; receives a
 *                          human-readable summary. May be NULL.
 *
 * Returns C4A_OK on success, C4A_ERR_NULL_POINTER if any required output
 * pointer is NULL, C4A_ERR_INVALID_ARGUMENT for negative dimensions.
 */
c4a_status_t c4a_signal_detect_impl(const double* X,
                                     int64_t rows, int64_t cols,
                                     const double* wavelengths,
                                     int64_t wl_length,
                                     double confidence_threshold,
                                     int32_t* type_out,
                                     double* confidence_out,
                                     char* reason_buf);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_UTILITIES_SIGNAL_TYPE_DETECTOR_H */
