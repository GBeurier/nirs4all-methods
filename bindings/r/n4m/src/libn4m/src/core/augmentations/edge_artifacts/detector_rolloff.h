/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * DetectorRollOff edge-artifact augmenter — internal engine.
 *
 * Models a wavelength-dependent detector sensitivity curve and applies its
 * roll-off effects (noise amplification + slight baseline distortion) to a
 * matrix of spectra. Mirrors
 * `nirs4all.operators.augmentation.edge_artifacts.DetectorRollOffAugmenter`.
 *
 * The detector_model parameter selects one of five built-in literature-based
 * presets (see DetectorRollOff_Model enum below). Parameters of each preset
 * (optimal range, roll-off rate, min sensitivity) are encoded in the engine
 * and not exposed to the public ABI — only the integer selector is.
 *
 * Pure C, no external dependencies. INTERNAL.
 */
#ifndef N4M_CORE_AUGMENT_EDGE_DETECTOR_ROLLOFF_H
#define N4M_CORE_AUGMENT_EDGE_DETECTOR_ROLLOFF_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Detector model identifiers — selectable presets from the Python reference. */
typedef enum n4m_aug_detector_model_t {
    N4M_AUG_DETECTOR_INGAAS_STANDARD = 0,
    N4M_AUG_DETECTOR_INGAAS_EXTENDED = 1,
    N4M_AUG_DETECTOR_PBS              = 2,
    N4M_AUG_DETECTOR_SILICON_CCD      = 3,
    N4M_AUG_DETECTOR_GENERIC_NIR      = 4
} n4m_aug_detector_model_t;

/* Forward-declare the engine state struct. */
typedef struct n4m_aug_detector_rolloff_state_t
    n4m_aug_detector_rolloff_state_t;

n4m_aug_detector_rolloff_state_t* n4m_aug_detector_rolloff_state_new(
    int32_t detector_model,
    double effect_strength,
    double noise_amplification,
    int include_baseline_distortion);

void n4m_aug_detector_rolloff_state_free(
    n4m_aug_detector_rolloff_state_t* state);

/* Apply the roll-off to a (rows x cols) matrix of spectra in row-major
 * order. wavelengths is a (cols,) vector of wavelengths in nm.
 * `rng` is an internal PCG64 engine pointer (caller-owned).
 * Output `out` may alias `X`. */
n4m_status_t n4m_aug_detector_rolloff_state_apply(
    const n4m_aug_detector_rolloff_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_EDGE_DETECTOR_ROLLOFF_H */
