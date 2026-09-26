/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation.h — augmentation role header (ABI 2.0). */
#ifndef N4M_AUGMENTATION_H
#define N4M_AUGMENTATION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */
#include "n4m/augmentation/drift.h"
#include "n4m/augmentation/instrument.h"
#include "n4m/augmentation/mixup.h"
#include "n4m/augmentation/noise.h"
#include "n4m/augmentation/scattering.h"
#include "n4m/augmentation/spectral.h"
#include "n4m/augmentation/splines.h"
#include "n4m/augmentation/wavelength.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Closed, one-shot train-only X->X augmentation dispatch (ABI 2.11).
 * The selected existing augmenter is created with a fresh PCG64 stream seeded
 * by `seed`, applied once, and destroyed. X/out must be aligned row-major F64
 * matrices of identical shape. Y is deliberately absent: no paired-label
 * semantics or fitted-state portability is implied. `params` is positional;
 * see the enum comments and the corresponding existing create functions.
 * Mixup/LocalMixup and wavelength-dependent operators are excluded until a
 * safe paired-Y/axis contract exists. */
typedef enum n4m_augmentation_kind_t {
    N4M_AUG_GAUSSIAN_NOISE = 0,       /* sigma */
    N4M_AUG_MULTIPLICATIVE_NOISE = 1, /* sigma_gain */
    N4M_AUG_SPIKE_NOISE = 2,          /* n_min,n_max,amp_min,amp_max */
    N4M_AUG_HETERO_NOISE = 3,         /* noise_base,noise_signal_dep */
    N4M_AUG_LINEAR_DRIFT = 4,        /* offset_min,max,slope_min,max */
    N4M_AUG_PATH_LENGTH = 5,         /* std,min_path_length */
    N4M_AUG_BAND_PERTURB = 6,        /* n_bands,bw_lo,hi,gain_lo,hi,offset_lo,hi */
    N4M_AUG_BAND_MASK = 7,           /* n_bands_lo,hi,bw_lo,hi,mode */
    N4M_AUG_CHANNEL_DROPOUT = 8,     /* dropout_prob,mode */
    N4M_AUG_GAUSS_JITTER = 9,        /* sigma_lo,hi,kernel_width */
    N4M_AUG_UNSHARP_MASK = 10,       /* amount_lo,hi,sigma,kernel_width */
    N4M_AUG_LOCAL_CLIP = 11,         /* n_regions,width_lo,hi */
    N4M_AUG_ROTATE_TRANSLATE = 12,   /* p_range,y_factor */
    N4M_AUG_RANDOM_X_OP = 13,        /* op_kind,range_min,max */
    N4M_AUG_SCATTER_SIM_MSC = 14,    /* a_low,high,b_low,high */
    N4M_AUG_DEAD_BAND = 15,          /* n_bands,width_lo,hi,noise_std,p,scope */
    N4M_AUG_BATCH_EFFECT = 16,       /* offset_std,slope_std,gain_std,scope;
                                       index axis, no explicit wavelengths */
    N4M_AUG_SPLINE_SMOOTHING = 17,   /* no params */
    N4M_AUG_SPLINE_X_PERTURB = 18,   /* degree,density,range_min,max */
    N4M_AUG_SPLINE_Y_PERTURB = 19,   /* points,intensity */
    N4M_AUG_SPLINE_X_SIMPLIFY = 20,  /* points,uniform */
    N4M_AUG_SPLINE_CURVE_SIMPLIFY = 21 /* points,uniform */
} n4m_augmentation_kind_t;

N4M_API n4m_status_t n4m_augmentation_run(
    int32_t kind, const double* params, int32_t n_params, uint64_t seed,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_H */
