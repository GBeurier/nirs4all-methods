/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal, C-callable serialization of kernel fitted state (not part of the
 * public ABI). Kernels that own an opaque state struct implement
 *
 *     n4m_status_t <kernel>_state_save(const <state>*, n4m_state_writer_t*);
 *     n4m_status_t <kernel>_state_load(<state>*, n4m_state_reader_t*,
 *                                      int64_t n_features);
 *
 * next to the struct definition, writing exactly the fields needed to
 * transform new rows (the input width is stored by the adapter and passed
 * back as `n_features`). Derived quantities are recomputed on load, as fit
 * computes them; sizes that do not match yield N4M_ERR_CORRUPT_BUFFER. The
 * estimator adapters embed these bytes in N4ME.
 *
 * Encoding: little-endian; an array is its length (i64) then its values.
 */
#ifndef N4M_CORE_ESTIMATOR_STATE_IO_H
#define N4M_CORE_ESTIMATOR_STATE_IO_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_state_writer_s n4m_state_writer_t;
typedef struct n4m_state_reader_s n4m_state_reader_t;

void n4m_state_write_i64(n4m_state_writer_t* w, int64_t v);
void n4m_state_write_f64(n4m_state_writer_t* w, double v);
void n4m_state_write_f64_array(n4m_state_writer_t* w, const double* v, int64_t n);
void n4m_state_write_i64_array(n4m_state_writer_t* w, const int64_t* v, int64_t n);

/* Each read returns 1 on success, 0 when the payload is truncated or does
 * not match (the adapter then reports N4M_ERR_CORRUPT_BUFFER). */
int n4m_state_read_i64(n4m_state_reader_t* r, int64_t* out);
int n4m_state_read_f64(n4m_state_reader_t* r, double* out);
/* Reads an array whose length must equal `expected` into caller storage. */
int n4m_state_read_f64_array(n4m_state_reader_t* r, double* out, int64_t expected);
int n4m_state_read_i64_array(n4m_state_reader_t* r, int64_t* out, int64_t expected);
/* Reads the length of the next array without consuming it (for ragged
 * states); fails when it exceeds `max_len`. */
int n4m_state_peek_array_length(n4m_state_reader_t* r, int64_t max_len, int64_t* out);
/* Reads an array of exactly `expected` (>= 1) values into a new malloc'd
 * buffer the caller owns (free()). The payload length is checked before
 * allocating. Returns N4M_OK, N4M_ERR_CORRUPT_BUFFER or
 * N4M_ERR_OUT_OF_MEMORY; `*out` is NULL on failure. */
n4m_status_t n4m_state_read_f64_array_new(n4m_state_reader_t* r, int64_t expected,
                                          double** out);
n4m_status_t n4m_state_read_i64_array_new(n4m_state_reader_t* r, int64_t expected,
                                          int64_t** out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_CORE_ESTIMATOR_STATE_IO_H */
