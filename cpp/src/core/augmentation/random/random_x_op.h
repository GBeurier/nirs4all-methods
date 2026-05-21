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
#ifndef N4M_CORE_AUGMENT_RANDOM_X_OP_H
#define N4M_CORE_AUGMENT_RANDOM_X_OP_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_aug_random_x_op_kind_t {
    N4M_AUG_RANDOM_X_OP_MUL = 0,
    N4M_AUG_RANDOM_X_OP_ADD = 1,
    N4M_AUG_RANDOM_X_OP_SUB = 2
} n4m_aug_random_x_op_kind_t;

typedef struct n4m_aug_random_x_op_state_t n4m_aug_random_x_op_state_t;

n4m_aug_random_x_op_state_t* n4m_aug_random_x_op_state_new(
    int32_t op_kind,
    double  operator_range_min,
    double  operator_range_max);

void n4m_aug_random_x_op_state_free(n4m_aug_random_x_op_state_t* state);

n4m_status_t n4m_aug_random_x_op_state_apply(
    const n4m_aug_random_x_op_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_RANDOM_X_OP_H */
