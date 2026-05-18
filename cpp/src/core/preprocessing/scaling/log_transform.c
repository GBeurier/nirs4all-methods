/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LogTransform reference implementation.
 *
 * Three subtleties matter for byte-for-byte parity with nirs4all/numpy:
 *
 *   1. The fitted offset is computed against `X + offset` (call it X_temp)
 *      but is then applied to the raw X — never to X_temp. The nirs4all
 *      code reads:
 *          X_temp = X + self.offset
 *          if auto_offset and min(X_temp) <= 0:
 *              auto_computed_offset = min_value - min(X_temp)
 *          self._fitted_offset = self.offset + auto_computed_offset
 *          # transform uses raw X plus _fitted_offset
 *      The two paths differ by floating-point rounding when offset != 0.
 *
 *   2. The clamp `X' = where(X' <= 0, min_value, X')` is applied AFTER the
 *      additive offset, so any negatives that survive (e.g. when
 *      auto_offset is disabled) get replaced exactly by min_value.
 *
 *   3. `log(X) / log(base)` is the explicit form used by numpy when base is
 *      not exp(1). For natural log we call `log(X)` directly to avoid a
 *      spurious division by log(e) == 1.0 that would still be bit-exact but
 *      add a redundant rounding op.
 *
 * Phase 5b lifecycle: when ``auto_offset != 0`` the fitted offset is
 * captured by ``c4a_pp_log_state_fit`` and stored in the state struct.
 * ``c4a_pp_log_apply`` then reads the cached value instead of recomputing
 * it on every call (the old Phase 2 behaviour). When ``auto_offset == 0``
 * the operator stays stateless and ``_apply`` works without ``_fit``.
 */

#include "log_transform.h"

#include <math.h>
#include <stdlib.h>

struct c4a_pp_log_state_t {
    double base;
    double offset;
    int    auto_offset;
    double min_value;
    int    fitted;
    double fitted_offset;
};

c4a_pp_log_state_t* c4a_pp_log_state_new(double base, double offset,
                                          int auto_offset, double min_value) {
    c4a_pp_log_state_t* s = (c4a_pp_log_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->base          = base;
    s->offset        = offset;
    s->auto_offset   = auto_offset ? 1 : 0;
    s->min_value     = min_value;
    s->fitted        = 0;
    s->fitted_offset = 0.0;
    return s;
}

void c4a_pp_log_state_free(c4a_pp_log_state_t* state) {
    free(state);
}

int c4a_pp_log_state_is_fitted(const c4a_pp_log_state_t* state) {
    if (state == NULL) return 0;
    return state->fitted;
}

c4a_status_t c4a_pp_log_state_fit(c4a_pp_log_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols) {
    if (state == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    if (!state->auto_offset) {
        /* Stateless path: no scan needed. Mark fitted so consumers that
         * always call _fit + _transform see a well-defined state. */
        state->fitted_offset = state->offset;
        state->fitted        = 1;
        return C4A_OK;
    }

    if (rows == 0 || cols == 0) {
        /* Empty fit data — degenerate. Treat as "no negatives seen": the
         * fitted offset reduces to the user-supplied offset. */
        state->fitted_offset = state->offset;
        state->fitted        = 1;
        return C4A_OK;
    }
    if (X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }

    const size_t total = (size_t)rows * (size_t)cols;
    double x_temp_min = X[0] + state->offset;
    for (size_t i = 1; i < total; ++i) {
        const double v = X[i] + state->offset;
        if (v < x_temp_min) {
            x_temp_min = v;
        }
    }
    double fitted_offset = state->offset;
    if (x_temp_min <= 0.0) {
        fitted_offset = state->offset + (state->min_value - x_temp_min);
    }
    state->fitted_offset = fitted_offset;
    state->fitted        = 1;
    return C4A_OK;
}

c4a_status_t c4a_pp_log_apply(const c4a_pp_log_state_t* state,
                              const double* X, int64_t rows, int64_t cols,
                              double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }

    const size_t total = (size_t)rows * (size_t)cols;

    /* Determine the offset to apply. ``auto_offset == 0`` is the stateless
     * path: the user-supplied offset is used directly, no fit required.
     * ``auto_offset != 0`` requires _fit to have populated _fitted_offset. */
    double fitted_offset;
    if (state->auto_offset) {
        if (!state->fitted) {
            return C4A_ERR_NOT_FITTED;
        }
        fitted_offset = state->fitted_offset;
    } else {
        fitted_offset = state->offset;
    }

    /* Apply fitted_offset to RAW X, clamp non-positive results to
     * min_value, then take log (or log/log(base) for an arbitrary base).
     * nirs4all evaluates `np.log(X) / np.log(base)` as a true element-wise
     * division by log(base) — NOT a multiplication by a precomputed
     * reciprocal — so we replicate that exact arithmetic here.
     *
     * `base == 0.0` is the C-side sentinel for "natural log" so callers can
     * select it from bindings without needing the M_E macro (which is not
     * specified by ISO C). When base equals exp(1) bit-for-bit, log(base)
     * is 1.0 exactly and the division is a no-op — but we still take the
     * short path to skip the extra `log` call. */
    const int    natural_log = (state->base == 0.0) || (state->base == exp(1.0));
    const double log_base    = natural_log ? 1.0 : log(state->base);

    if (fitted_offset == 0.0) {
        if (natural_log) {
            for (size_t i = 0; i < total; ++i) {
                double v = X[i];
                if (v <= 0.0) v = state->min_value;
                out[i] = log(v);
            }
        } else {
            for (size_t i = 0; i < total; ++i) {
                double v = X[i];
                if (v <= 0.0) v = state->min_value;
                out[i] = log(v) / log_base;
            }
        }
    } else {
        if (natural_log) {
            for (size_t i = 0; i < total; ++i) {
                double v = X[i] + fitted_offset;
                if (v <= 0.0) v = state->min_value;
                out[i] = log(v);
            }
        } else {
            for (size_t i = 0; i < total; ++i) {
                double v = X[i] + fitted_offset;
                if (v <= 0.0) v = state->min_value;
                out[i] = log(v) / log_base;
            }
        }
    }

    return C4A_OK;
}
