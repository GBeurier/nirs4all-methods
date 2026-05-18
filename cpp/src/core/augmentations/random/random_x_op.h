/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Random_X_Operation augmenter — internal engine.
 *
 * Applies a per-cell elementwise operator (multiplicative, additive, or
 * subtractive) with the operand drawn uniformly from
 * [operator_range_min, operator_range_max]. Mirrors
 * `nirs4all.operators.augmentation.random.Random_X_Operation`.
 *
 * Op encoding:
 *   0  multiply  (X * u)        — Python default operator.mul
 *   1  add       (X + u)        — operator.add
 *   2  subtract  (X - u)        — operator.sub
 *
 * Output is clipped to the float32 representable range to match the
 * reference's np.clip(..., -float32.max, float32.max) tail.
 *
 * Pure C. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_RANDOM_X_OP_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_RANDOM_X_OP_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum c4a_aug_random_x_op_kind_t {
    C4A_AUG_RANDOM_X_OP_MUL = 0,
    C4A_AUG_RANDOM_X_OP_ADD = 1,
    C4A_AUG_RANDOM_X_OP_SUB = 2
} c4a_aug_random_x_op_kind_t;

typedef struct c4a_aug_random_x_op_state_t c4a_aug_random_x_op_state_t;

c4a_aug_random_x_op_state_t* c4a_aug_random_x_op_state_new(
    int32_t op_kind,
    double  operator_range_min,
    double  operator_range_max);

void c4a_aug_random_x_op_state_free(c4a_aug_random_x_op_state_t* state);

c4a_status_t c4a_aug_random_x_op_state_apply(
    const c4a_aug_random_x_op_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_RANDOM_X_OP_H */
