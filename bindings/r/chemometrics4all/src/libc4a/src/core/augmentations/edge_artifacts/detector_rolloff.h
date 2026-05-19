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
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_DETECTOR_ROLLOFF_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_DETECTOR_ROLLOFF_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Detector model identifiers — selectable presets from the Python reference. */
typedef enum c4a_aug_detector_model_t {
    C4A_AUG_DETECTOR_INGAAS_STANDARD = 0,
    C4A_AUG_DETECTOR_INGAAS_EXTENDED = 1,
    C4A_AUG_DETECTOR_PBS              = 2,
    C4A_AUG_DETECTOR_SILICON_CCD      = 3,
    C4A_AUG_DETECTOR_GENERIC_NIR      = 4
} c4a_aug_detector_model_t;

/* Forward-declare the engine state struct. */
typedef struct c4a_aug_detector_rolloff_state_t
    c4a_aug_detector_rolloff_state_t;

c4a_aug_detector_rolloff_state_t* c4a_aug_detector_rolloff_state_new(
    int32_t detector_model,
    double effect_strength,
    double noise_amplification,
    int include_baseline_distortion);

void c4a_aug_detector_rolloff_state_free(
    c4a_aug_detector_rolloff_state_t* state);

/* Apply the roll-off to a (rows x cols) matrix of spectra in row-major
 * order. wavelengths is a (cols,) vector of wavelengths in nm.
 * `rng` is an internal PCG64 engine pointer (caller-owned).
 * Output `out` may alias `X`. */
c4a_status_t c4a_aug_detector_rolloff_state_apply(
    const c4a_aug_detector_rolloff_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_DETECTOR_ROLLOFF_H */
