/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * R .Call gateway for the generic estimator roles and procedures
 * (n4m/estimator.h).
 *
 * Translation only: R matrices are passed as column-major views (no copy),
 * named R parameter lists are set through the typed n4m_params_* setters
 * using the native manifest's parameter types, and fitted state crosses
 * into R as N4ME bytes kept next to the external pointer so saveRDS/readRDS
 * rehydrate without refitting.
 */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "n4m/n4m.h"

static void r_est_finalize(SEXP ptr) {
    n4m_estimator_t* est = (n4m_estimator_t*)R_ExternalPtrAddr(ptr);
    if (est != NULL) {
        n4m_estimator_destroy(est);
        R_ClearExternalPtr(ptr);
    }
}

static SEXP r_est_wrap(n4m_estimator_t* est) {
    SEXP ptr = PROTECT(R_MakeExternalPtr(est, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, r_est_finalize, TRUE);
    UNPROTECT(1);
    return ptr;
}

static n4m_estimator_t* r_est_get(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrAddr(ptr) == NULL) {
        Rf_error("n4m estimator pointer is not alive; rehydrate it from its N4ME bytes");
    }
    return (n4m_estimator_t*)R_ExternalPtrAddr(ptr);
}

/* Copies the context message, releases what the caller registered, then
 * raises the R error (longjmp skips normal cleanup). */
static void r_est_fail(const char* where, n4m_status_t st, n4m_context_t* ctx,
                       n4m_params_t* params, n4m_estimator_t* est) {
    char buf[1024];
    buf[0] = '\0';
    if (ctx != NULL) {
        const char* msg = n4m_context_last_error(ctx);
        if (msg != NULL) {
            strncpy(buf, msg, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
        }
        n4m_context_destroy(ctx);
    }
    if (params != NULL) n4m_params_destroy(params);
    if (est != NULL) n4m_estimator_destroy(est);
    if (buf[0] != '\0') {
        Rf_error("%s: %s (%s)", where, n4m_status_to_string(st), buf);
    }
    Rf_error("%s: %s", where, n4m_status_to_string(st));
}

static n4m_context_t* r_est_context(void) {
    n4m_context_t* ctx = NULL;
    if (n4m_context_create(&ctx) != N4M_OK) Rf_error("n4m_context_create failed");
    return ctx;
}

static n4m_matrix_view_t r_est_view(SEXP m, const char* name) {
    if (!Rf_isReal(m) || !Rf_isMatrix(m)) Rf_error("%s must be a double matrix", name);
    n4m_matrix_view_t v;
    memset(&v, 0, sizeof(v));
    n4m_matrix_view_init_colmajor(&v, REAL(m), Rf_nrows(m), Rf_ncols(m), N4M_DTYPE_F64);
    return v;
}

/* A finite whole number representable as int64 (R integers arrive as doubles). */
/* A finite whole number within int64 (2^63 is exact in double). */
static int r_est_whole(double d) {
    return R_FINITE(d) && d >= -9223372036854775808.0 && d < 9223372036854775808.0 &&
           d == floor(d);
}

/* Sets each element of a named list with the manifest type of that name. */
static n4m_status_t r_est_set_params(n4m_params_t* params, int32_t index, SEXP values,
                                     const char** bad) {
    SEXP names = Rf_getAttrib(values, R_NamesSymbol);
    n4m_method_info_v1_t info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    n4m_status_t st = n4m_method_info_v1(index, &info);
    if (st != N4M_OK) return st;
    for (R_xlen_t i = 0; i < XLENGTH(values); ++i) {
        const char* name = CHAR(STRING_ELT(names, i));
        SEXP v = VECTOR_ELT(values, i);
        *bad = name;
        n4m_param_info_v1_t pi;
        int32_t p = 0;
        for (; p < info.n_params; ++p) {
            memset(&pi, 0, sizeof(pi));
            pi.struct_size = sizeof(pi);
            if (n4m_method_param_info_v1(index, p, &pi) == N4M_OK && strcmp(pi.name, name) == 0) {
                break;
            }
        }
        if (p == info.n_params) return N4M_ERR_INVALID_ARGUMENT;
        /* A scalar parameter takes exactly one value; nothing is recycled
         * or truncated. */
        const int array = pi.type == N4M_METHOD_PARAM_INT_ARRAY ||
                          pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY;
        if (!array && XLENGTH(v) != 1) return N4M_ERR_INVALID_ARGUMENT;
        switch (pi.type) {
            case N4M_METHOD_PARAM_INT: {
                const double d = Rf_asReal(v);
                if (!r_est_whole(d)) return N4M_ERR_INVALID_ARGUMENT;
                st = n4m_params_set_int(params, name, (int64_t)d);
                break;
            }
            case N4M_METHOD_PARAM_DOUBLE:
                st = n4m_params_set_double(params, name, Rf_asReal(v));
                break;
            case N4M_METHOD_PARAM_BOOL:
                if (Rf_asLogical(v) == NA_LOGICAL) return N4M_ERR_INVALID_ARGUMENT;
                st = n4m_params_set_bool(params, name, Rf_asLogical(v) == TRUE ? 1 : 0);
                break;
            case N4M_METHOD_PARAM_ENUM:
                if (!Rf_isString(v)) return N4M_ERR_INVALID_ARGUMENT;
                st = n4m_params_set_enum(params, name, CHAR(STRING_ELT(v, 0)));
                break;
            case N4M_METHOD_PARAM_INT_ARRAY: {
                SEXP d = PROTECT(Rf_coerceVector(v, REALSXP));
                int64_t* buf = (int64_t*)R_alloc((size_t)XLENGTH(d), sizeof(int64_t));
                for (R_xlen_t k = 0; k < XLENGTH(d); ++k) {
                    if (!r_est_whole(REAL(d)[k])) {
                        UNPROTECT(1);
                        return N4M_ERR_INVALID_ARGUMENT;
                    }
                    buf[k] = (int64_t)REAL(d)[k];
                }
                st = n4m_params_set_int_array(params, name, buf, (int64_t)XLENGTH(d));
                UNPROTECT(1);
                break;
            }
            case N4M_METHOD_PARAM_DOUBLE_ARRAY: {
                SEXP d = PROTECT(Rf_coerceVector(v, REALSXP));
                st = n4m_params_set_double_array(params, name, REAL(d), (int64_t)XLENGTH(d));
                UNPROTECT(1);
                break;
            }
            default:
                st = N4M_ERR_INVALID_ARGUMENT;
        }
        if (st != N4M_OK) return st;
    }
    return N4M_OK;
}

static int64_t* r_est_int64(SEXP v, const char* name) {
    if (Rf_isNull(v)) return NULL;
    SEXP d = PROTECT(Rf_coerceVector(v, REALSXP));
    int64_t* out = (int64_t*)R_alloc((size_t)XLENGTH(d), sizeof(int64_t));
    for (R_xlen_t k = 0; k < XLENGTH(d); ++k) {
        if (!r_est_whole(REAL(d)[k])) {
            Rf_error("%s must contain finite integers", name);
        }
        out[k] = (int64_t)REAL(d)[k];
    }
    UNPROTECT(1);
    return out;
}

static double* r_est_double(SEXP v) {
    SEXP d = PROTECT(Rf_coerceVector(v, REALSXP));
    double* out = (double*)R_alloc((size_t)XLENGTH(d), sizeof(double));
    memcpy(out, REAL(d), (size_t)XLENGTH(d) * sizeof(double));
    UNPROTECT(1);
    return out;
}

/* Fit / procedure inputs over R objects. `inputs` is a named list among
 * labels, sample_weight, groups, feature_groups, blocks, axis, X_target,
 * fold_ids (R 1-based ids are not used: class labels, groups and fold ids are
 * opaque integers, feature groups are labels, blocks are sizes). */
typedef struct {
    n4m_fit_inputs_v1_t in;
    n4m_matrix_view_t X, Y, T;
} r_est_inputs_t;

static void r_est_inputs(r_est_inputs_t* s, SEXP X, SEXP y, SEXP inputs) {
    memset(s, 0, sizeof(*s));
    s->in.struct_size = sizeof(s->in);
    s->X = r_est_view(X, "X");
    s->in.X = &s->X;
    if (!Rf_isNull(y)) {
        s->Y = r_est_view(y, "y");
        s->in.Y = &s->Y;
    }
    SEXP names = Rf_getAttrib(inputs, R_NamesSymbol);
    for (R_xlen_t i = 0; i < XLENGTH(inputs); ++i) {
        const char* name = CHAR(STRING_ELT(names, i));
        SEXP v = VECTOR_ELT(inputs, i);
        int64_t n = (int64_t)XLENGTH(v);
        if (strcmp(name, "labels") == 0) {
            s->in.labels = r_est_int64(v, name);
            s->in.n_labels = n;
        } else if (strcmp(name, "sample_weight") == 0) {
            s->in.sample_weight = r_est_double(v);
            s->in.n_sample_weight = n;
        } else if (strcmp(name, "axis") == 0) {
            s->in.axis = r_est_double(v);
            s->in.n_axis = n;
        } else if (strcmp(name, "groups") == 0) {
            s->in.groups = r_est_int64(v, name);
            s->in.n_groups = n;
        } else if (strcmp(name, "feature_groups") == 0) {
            s->in.feature_groups = r_est_int64(v, name);
            s->in.n_feature_groups = n;
        } else if (strcmp(name, "blocks") == 0) {
            s->in.block_sizes = r_est_int64(v, name);
            s->in.n_blocks = n;
        } else if (strcmp(name, "fold_ids") == 0) {
            s->in.fold_ids = r_est_int64(v, name);
            s->in.n_fold_ids = n;
        } else if (strcmp(name, "X_target") == 0) {
            s->T = r_est_view(v, "X_target");
            s->in.X_target = &s->T;
        } else {
            Rf_error("unknown fit input '%s'", name);
        }
    }
}

/* Validated native parameters of method `index` from a named R list. */
static n4m_params_t* r_est_params(n4m_context_t* ctx, int32_t index, SEXP values,
                                  const char* id) {
    n4m_params_t* params = NULL;
    n4m_status_t st = n4m_params_create(ctx, index, &params);
    if (st != N4M_OK) r_est_fail("n4m_params_create", st, ctx, NULL, NULL);
    const char* bad = "";
    st = r_est_set_params(params, index, values, &bad);
    if (st != N4M_OK) {
        n4m_context_destroy(ctx);
        n4m_params_destroy(params);
        Rf_error("invalid value for parameter '%s' of %s", bad, id);
    }
    return params;
}

static int32_t r_est_find(SEXP method_id, const char** id) {
    *id = CHAR(STRING_ELT(method_id, 0));
    int32_t index = -1;
    if (n4m_method_find(*id, &index) != N4M_OK) Rf_error("unknown n4m method '%s'", *id);
    return index;
}

/* fit(method_id, params, X, y, inputs) -> external pointer. */
SEXP r_n4m_estimator_fit(SEXP method_id, SEXP values, SEXP X, SEXP y, SEXP inputs) {
    const char* id = NULL;
    const int32_t index = r_est_find(method_id, &id);
    r_est_inputs_t s;
    r_est_inputs(&s, X, y, inputs);
    n4m_context_t* ctx = r_est_context();
    n4m_params_t* params = r_est_params(ctx, index, values, id);
    n4m_estimator_t* est = NULL;
    n4m_status_t st = n4m_estimator_create(ctx, id, params, &est);
    if (st != N4M_OK) r_est_fail("n4m_estimator_create", st, ctx, params, NULL);
    n4m_params_destroy(params);
    st = n4m_estimator_fit(ctx, est, &s.in);
    if (st != N4M_OK) r_est_fail("n4m_estimator_fit", st, ctx, NULL, est);
    n4m_context_destroy(ctx);
    return r_est_wrap(est);
}

/* One named result entry as an R value (matrices column-major). */
static SEXP r_est_entry(const n4m_method_result_t* res, const char* name, int32_t kind) {
    SEXP out = R_NilValue;
    if (kind == N4M_RESULT_DOUBLE_MATRIX) {
        const double* d = NULL;
        int64_t rows = 0, cols = 0;
        n4m_method_result_get_double_matrix(res, name, &d, &rows, &cols);
        out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) REAL(out)[j * rows + i] = d[i * cols + j];
        }
    } else if (kind == N4M_RESULT_INT_VECTOR) {
        const int32_t* d = NULL;
        int32_t n = 0;
        n4m_method_result_get_int_vector(res, name, &d, &n);
        out = PROTECT(Rf_allocVector(REALSXP, n));
        for (int32_t k = 0; k < n; ++k) REAL(out)[k] = (double)d[k];
    } else if (kind == N4M_RESULT_INT64_VECTOR) {
        const int64_t* d = NULL;
        int64_t n = 0;
        n4m_method_result_get_int64_vector(res, name, &d, &n);
        out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n));
        for (int64_t k = 0; k < n; ++k) REAL(out)[k] = (double)d[k];
    } else {
        double v = 0.0;
        n4m_method_result_get_scalar(res, name, &v);
        out = PROTECT(Rf_ScalarReal(v));
    }
    UNPROTECT(1);
    return out;
}

static SEXP r_est_int64_vector(const int64_t* d, int64_t n) {
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n));
    for (int64_t k = 0; k < n; ++k) REAL(out)[k] = (double)d[k];
    UNPROTECT(1);
    return out;
}

/* procedure(method_id, params, X, y, inputs) -> named list of the result
 * entries; a splitter result adds `.folds`, a list of (train, test) 0-based
 * row indices. */
SEXP r_n4m_procedure_run(SEXP method_id, SEXP values, SEXP X, SEXP y, SEXP inputs) {
    const char* id = NULL;
    const int32_t index = r_est_find(method_id, &id);
    r_est_inputs_t s;
    r_est_inputs(&s, X, y, inputs);
    n4m_context_t* ctx = r_est_context();
    n4m_params_t* params = r_est_params(ctx, index, values, id);
    n4m_method_result_t* res = NULL;
    n4m_status_t st = n4m_procedure_run(ctx, index, params, &s.in, &res);
    if (st != N4M_OK) r_est_fail("n4m_procedure_run", st, ctx, params, NULL);
    n4m_params_destroy(params);
    n4m_context_destroy(ctx);

    int32_t count = 0;
    n4m_method_result_entry_count(res, &count);
    int32_t n_folds = 0;
    const int has_folds = n4m_method_result_get_n_folds(res, &n_folds) == N4M_OK;
    const int32_t n = count + (has_folds ? 1 : 0);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n));
    for (int32_t i = 0; i < count; ++i) {
        const char* name = NULL;
        int32_t kind = 0;
        n4m_method_result_entry(res, i, &name, &kind);
        SET_STRING_ELT(names, i, Rf_mkChar(name));
        SET_VECTOR_ELT(out, i, r_est_entry(res, name, kind));
    }
    if (has_folds) {
        SEXP folds = PROTECT(Rf_allocVector(VECSXP, n_folds));
        for (int32_t f = 0; f < n_folds; ++f) {
            const int64_t *train = NULL, *test = NULL;
            int64_t n_train = 0, n_test = 0;
            n4m_method_result_get_fold(res, f, &train, &n_train, &test, &n_test);
            SEXP fold = PROTECT(Rf_allocVector(VECSXP, 2));
            SET_VECTOR_ELT(fold, 0, r_est_int64_vector(train, n_train));
            SET_VECTOR_ELT(fold, 1, r_est_int64_vector(test, n_test));
            SET_VECTOR_ELT(folds, f, fold);
            UNPROTECT(1);
        }
        SET_STRING_ELT(names, count, Rf_mkChar(".folds"));
        SET_VECTOR_ELT(out, count, folds);
        UNPROTECT(1);
    }
    n4m_method_result_destroy(res);
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(2);
    return out;
}

typedef n4m_status_t (*r_est_matrix_fn)(n4m_context_t*, const n4m_estimator_t*,
                                         const n4m_matrix_view_t*, n4m_matrix_view_t*);

/* Row-wise matrix operation; transform has its own width, the others one
 * column per output (classes for a classifier). */
static SEXP r_est_matrix_op(SEXP ptr, SEXP X, r_est_matrix_fn fn, int transform,
                            const char* where) {
    n4m_estimator_t* est = r_est_get(ptr);
    int64_t cols = 0;
    n4m_status_t st = transform ? n4m_estimator_transform_cols(est, &cols)
                                : n4m_estimator_n_outputs(est, &cols);
    if (st != N4M_OK) r_est_fail(where, st, NULL, NULL, NULL);
    n4m_matrix_view_t Xv = r_est_view(X, "X");
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(X), (int)cols));
    n4m_matrix_view_t Ov = r_est_view(out, "out");
    n4m_context_t* ctx = r_est_context();
    st = fn(ctx, est, &Xv, &Ov);
    if (st != N4M_OK) {
        UNPROTECT(1);
        r_est_fail(where, st, ctx, NULL, NULL);
    }
    n4m_context_destroy(ctx);
    UNPROTECT(1);
    return out;
}

SEXP r_n4m_estimator_predict(SEXP ptr, SEXP X) {
    return r_est_matrix_op(ptr, X, n4m_estimator_predict, 0, "n4m_estimator_predict");
}

SEXP r_n4m_estimator_transform(SEXP ptr, SEXP X) {
    return r_est_matrix_op(ptr, X, n4m_estimator_transform, 1, "n4m_estimator_transform");
}

SEXP r_n4m_estimator_decision_function(SEXP ptr, SEXP X) {
    return r_est_matrix_op(ptr, X, n4m_estimator_decision_function, 0,
                           "n4m_estimator_decision_function");
}

SEXP r_n4m_estimator_predict_proba(SEXP ptr, SEXP X) {
    return r_est_matrix_op(ptr, X, n4m_estimator_predict_proba, 0,
                           "n4m_estimator_predict_proba");
}

/* Class ids (as fitted) of each row. */
SEXP r_n4m_estimator_predict_labels(SEXP ptr, SEXP X) {
    n4m_estimator_t* est = r_est_get(ptr);
    n4m_matrix_view_t Xv = r_est_view(X, "X");
    const int64_t n = (int64_t)Rf_nrows(X);
    int64_t* buf = (int64_t*)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int64_t));
    n4m_context_t* ctx = r_est_context();
    n4m_status_t st = n4m_estimator_predict_labels(ctx, est, &Xv, buf, n);
    if (st != N4M_OK) r_est_fail("n4m_estimator_predict_labels", st, ctx, NULL, NULL);
    n4m_context_destroy(ctx);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n));
    for (int64_t k = 0; k < n; ++k) REAL(out)[k] = (double)buf[k];
    UNPROTECT(1);
    return out;
}

/* Keep mask (logical) of the rows of X; Y is the one-column target or NULL. */
SEXP r_n4m_estimator_apply_mask(SEXP ptr, SEXP X, SEXP Y) {
    n4m_estimator_t* est = r_est_get(ptr);
    n4m_matrix_view_t Xv = r_est_view(X, "X");
    n4m_matrix_view_t Yv;
    if (!Rf_isNull(Y)) Yv = r_est_view(Y, "y");
    const int64_t n = (int64_t)Rf_nrows(X);
    uint8_t* mask = (uint8_t*)R_alloc((size_t)(n > 0 ? n : 1), sizeof(uint8_t));
    n4m_context_t* ctx = r_est_context();
    n4m_status_t st =
        n4m_estimator_apply_mask(ctx, est, &Xv, Rf_isNull(Y) ? NULL : &Yv, mask, n);
    if (st != N4M_OK) r_est_fail("n4m_estimator_apply_mask", st, ctx, NULL, NULL);
    n4m_context_destroy(ctx);
    SEXP out = PROTECT(Rf_allocVector(LGLSXP, (R_xlen_t)n));
    for (int64_t k = 0; k < n; ++k) LOGICAL(out)[k] = mask[k] != 0;
    UNPROTECT(1);
    return out;
}

/* The native manifest as a JSON string. */
SEXP r_n4m_manifest_json(void) {
    size_t size = 0;
    if (n4m_method_manifest_json(NULL, 0, &size) != N4M_OK) Rf_error("n4m manifest unavailable");
    char* buf = R_alloc(size + 1, 1);
    if (n4m_method_manifest_json(buf, size, &size) != N4M_OK) Rf_error("n4m manifest unavailable");
    buf[size] = '\0';
    return Rf_mkString(buf);
}

/* Fitted class ids, sorted. */
SEXP r_n4m_estimator_classes(SEXP ptr) {
    n4m_estimator_t* est = r_est_get(ptr);
    int64_t count = 0;
    n4m_status_t st = n4m_estimator_classes(est, NULL, 0, &count);
    if (st != N4M_OK) r_est_fail("n4m_estimator_classes", st, NULL, NULL, NULL);
    int64_t* buf = (int64_t*)R_alloc((size_t)(count > 0 ? count : 1), sizeof(int64_t));
    st = n4m_estimator_classes(est, buf, count, &count);
    if (st != N4M_OK) r_est_fail("n4m_estimator_classes", st, NULL, NULL, NULL);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)count));
    for (int64_t k = 0; k < count; ++k) REAL(out)[k] = (double)buf[k];
    UNPROTECT(1);
    return out;
}

/* Selected input columns (0-based, native selection order). */
SEXP r_n4m_estimator_selected_indices(SEXP ptr) {
    n4m_estimator_t* est = r_est_get(ptr);
    int64_t count = 0;
    n4m_status_t st = n4m_estimator_selected_indices(est, NULL, 0, &count);
    if (st != N4M_OK) r_est_fail("n4m_estimator_selected_indices", st, NULL, NULL, NULL);
    int64_t* buf = (int64_t*)R_alloc((size_t)(count > 0 ? count : 1), sizeof(int64_t));
    st = n4m_estimator_selected_indices(est, buf, count, &count);
    if (st != N4M_OK) r_est_fail("n4m_estimator_selected_indices", st, NULL, NULL, NULL);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)count));
    for (int64_t k = 0; k < count; ++k) REAL(out)[k] = (double)buf[k];
    UNPROTECT(1);
    return out;
}

/* N4ME bytes; a state that embeds training rows exports only when
 * allow_training_rows is TRUE. */
SEXP r_n4m_estimator_export(SEXP ptr, SEXP allow_training_rows) {
    n4m_estimator_t* est = r_est_get(ptr);
    const uint32_t flags =
        Rf_asLogical(allow_training_rows) == TRUE ? N4M_EXPORT_ALLOW_TRAINING_ROWS : 0u;
    n4m_context_t* ctx = r_est_context();
    size_t size = 0;
    n4m_status_t st = n4m_estimator_export_size(ctx, est, flags, &size);
    if (st != N4M_OK) r_est_fail("n4m_estimator_export_size", st, ctx, NULL, NULL);
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)size));
    size_t written = 0;
    st = n4m_estimator_export_to_buffer(ctx, est, flags, RAW(out), size, &written);
    if (st != N4M_OK) {
        UNPROTECT(1);
        r_est_fail("n4m_estimator_export_to_buffer", st, ctx, NULL, NULL);
    }
    n4m_context_destroy(ctx);
    UNPROTECT(1);
    return out;
}

SEXP r_n4m_estimator_contains_training_rows(SEXP ptr) {
    int32_t out = 0;
    const n4m_status_t st = n4m_estimator_contains_training_rows(r_est_get(ptr), &out);
    if (st != N4M_OK) r_est_fail("n4m_estimator_contains_training_rows", st, NULL, NULL, NULL);
    return Rf_ScalarLogical(out != 0);
}

SEXP r_n4m_estimator_import(SEXP bytes) {
    if (TYPEOF(bytes) != RAWSXP) Rf_error("N4ME state must be a raw vector");
    n4m_context_t* ctx = r_est_context();
    n4m_estimator_t* est = NULL;
    n4m_status_t st = n4m_estimator_import_from_buffer(ctx, RAW(bytes), (size_t)XLENGTH(bytes), &est);
    if (st != N4M_OK) r_est_fail("n4m_estimator_import_from_buffer", st, ctx, NULL, NULL);
    n4m_context_destroy(ctx);
    return r_est_wrap(est);
}

SEXP r_n4m_estimator_alive(SEXP ptr) {
    return Rf_ScalarLogical(TYPEOF(ptr) == EXTPTRSXP && R_ExternalPtrAddr(ptr) != NULL);
}

/* list(method_id, capabilities, n_features_in, n_outputs, params) for a fitted
 * estimator; params are the resolved native values (names -> R values). */
SEXP r_n4m_estimator_info(SEXP ptr) {
    n4m_estimator_t* est = r_est_get(ptr);
    int32_t index = -1;
    uint64_t caps = 0;
    int64_t n_in = 0, n_out = 0;
    n4m_estimator_info(est, &index, &caps);
    n4m_estimator_n_features_in(est, &n_in);
    n4m_estimator_n_outputs(est, &n_out);
    n4m_method_info_v1_t info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    n4m_method_info_v1(index, &info);

    n4m_context_t* ctx = r_est_context();
    n4m_params_t* params = NULL;
    n4m_status_t st = n4m_estimator_get_params(ctx, est, &params);
    if (st != N4M_OK) r_est_fail("n4m_estimator_get_params", st, ctx, NULL, NULL);
    n4m_context_destroy(ctx);
    SEXP values = PROTECT(Rf_allocVector(VECSXP, info.n_params));
    SEXP value_names = PROTECT(Rf_allocVector(STRSXP, info.n_params));
    for (int32_t p = 0; p < info.n_params; ++p) {
        n4m_param_info_v1_t pi;
        memset(&pi, 0, sizeof(pi));
        pi.struct_size = sizeof(pi);
        n4m_method_param_info_v1(index, p, &pi);
        SET_STRING_ELT(value_names, p, Rf_mkChar(pi.name));
        int64_t count = 0;
        if (pi.type == N4M_METHOD_PARAM_DOUBLE || pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY) {
            n4m_params_get_double(params, pi.name, NULL, 0, &count);
            SEXP v = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)count));
            n4m_params_get_double(params, pi.name, REAL(v), count, &count);
            SET_VECTOR_ELT(values, p, v);
            UNPROTECT(1);
            continue;
        }
        n4m_params_get_int(params, pi.name, NULL, 0, &count);
        int64_t* buf = (int64_t*)R_alloc((size_t)(count > 0 ? count : 1), sizeof(int64_t));
        n4m_params_get_int(params, pi.name, buf, count, &count);
        SEXP v;
        if (pi.type == N4M_METHOD_PARAM_BOOL) {
            v = PROTECT(Rf_ScalarLogical(buf[0] != 0));
        } else if (pi.type == N4M_METHOD_PARAM_ENUM) {
            v = PROTECT(Rf_mkString(pi.choices[buf[0]]));
        } else {
            v = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)count));
            for (int64_t k = 0; k < count; ++k) REAL(v)[k] = (double)buf[k];
        }
        SET_VECTOR_ELT(values, p, v);
        UNPROTECT(1);
    }
    n4m_params_destroy(params);
    Rf_setAttrib(values, R_NamesSymbol, value_names);

    const char* fields[] = {"method_id", "capabilities", "n_features_in", "n_outputs", "params"};
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 5));
    SEXP out_names = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_VECTOR_ELT(out, 0, Rf_mkString(info.method_id));
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal((double)caps));
    SET_VECTOR_ELT(out, 2, Rf_ScalarReal((double)n_in));
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal((double)n_out));
    SET_VECTOR_ELT(out, 4, values);
    for (int k = 0; k < 5; ++k) SET_STRING_ELT(out_names, k, Rf_mkChar(fields[k]));
    Rf_setAttrib(out, R_NamesSymbol, out_names);
    UNPROTECT(4);
    return out;
}

/* ---- Role pipelines (ABI 2.14) ---------------------------------------- */

static void r_rp_finalize(SEXP ptr) {
    n4m_role_pipeline_t* p = (n4m_role_pipeline_t*)R_ExternalPtrAddr(ptr);
    if (p != NULL) {
        n4m_role_pipeline_destroy(p);
        R_ClearExternalPtr(ptr);
    }
}

static SEXP r_rp_wrap(n4m_role_pipeline_t* p) {
    SEXP ptr = PROTECT(R_MakeExternalPtr(p, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, r_rp_finalize, TRUE);
    UNPROTECT(1);
    return ptr;
}

static n4m_role_pipeline_t* r_rp_get(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrAddr(ptr) == NULL) {
        Rf_error("n4m role pipeline pointer is not alive; rehydrate it from its N4ME states");
    }
    return (n4m_role_pipeline_t*)R_ExternalPtrAddr(ptr);
}

/* Like r_est_fail, for a pipeline under construction. */
static void r_rp_fail(const char* where, n4m_status_t st, n4m_context_t* ctx,
                      n4m_role_pipeline_t* p) {
    if (p != NULL) n4m_role_pipeline_destroy(p);
    r_est_fail(where, st, ctx, NULL, NULL);
}

/* Native recipe from method ids and one named parameter list per step. */
static n4m_role_pipeline_t* r_rp_create(n4m_context_t* ctx, SEXP method_ids, SEXP values) {
    const R_xlen_t n = XLENGTH(method_ids);
    const char** ids = (const char**)R_alloc((size_t)(n > 0 ? n : 1), sizeof(char*));
    n4m_params_t** params = (n4m_params_t**)R_alloc((size_t)(n > 0 ? n : 1), sizeof(n4m_params_t*));
    memset(params, 0, (size_t)(n > 0 ? n : 1) * sizeof(n4m_params_t*));
    const char* bad = NULL;
    R_xlen_t failed = -1;
    for (R_xlen_t k = 0; k < n && failed < 0; ++k) {
        ids[k] = CHAR(STRING_ELT(method_ids, k));
        int32_t index = -1;
        /* An unknown id is reported by the native create, step included. */
        if (n4m_method_find(ids[k], &index) != N4M_OK) continue;
        if (n4m_params_create(ctx, index, &params[k]) != N4M_OK ||
            r_est_set_params(params[k], index, VECTOR_ELT(values, k), &bad) != N4M_OK) {
            failed = k;
        }
    }
    n4m_role_pipeline_t* p = NULL;
    n4m_status_t st = N4M_OK;
    if (failed < 0) {
        st = n4m_role_pipeline_create(ctx, (int32_t)n, ids, (const n4m_params_t* const*)params, &p);
    }
    for (R_xlen_t k = 0; k < n; ++k) {
        if (params[k] != NULL) n4m_params_destroy(params[k]);
    }
    if (failed >= 0) {
        n4m_context_destroy(ctx);
        Rf_error("invalid value for parameter '%s' of %s (step %d)", bad != NULL ? bad : "",
                 ids[failed], (int)failed);
    }
    if (st != N4M_OK) r_est_fail("n4m_role_pipeline_create", st, ctx, NULL, NULL);
    return p;
}

/* Refuses an invalid recipe (role order, final step, parameters). */
SEXP r_n4m_role_pipeline_validate(SEXP method_ids, SEXP values) {
    n4m_context_t* ctx = r_est_context();
    n4m_role_pipeline_destroy(r_rp_create(ctx, method_ids, values));
    n4m_context_destroy(ctx);
    return R_NilValue;
}

/* Column names as a C array, NULL for positional data. */
static const char** r_rp_names(SEXP names) {
    if (Rf_isNull(names)) return NULL;
    const R_xlen_t n = XLENGTH(names);
    const char** out = (const char**)R_alloc((size_t)(n > 0 ? n : 1), sizeof(char*));
    for (R_xlen_t k = 0; k < n; ++k) out[k] = Rf_translateCharUTF8(STRING_ELT(names, k));
    return out;
}

static void r_rp_set_names(n4m_context_t* ctx, n4m_role_pipeline_t* p, SEXP names) {
    if (Rf_isNull(names)) return;
    n4m_status_t st = n4m_role_pipeline_set_feature_names(ctx, p, r_rp_names(names),
                                                          (int64_t)XLENGTH(names));
    if (st != N4M_OK) r_rp_fail("n4m_role_pipeline_set_feature_names", st, ctx, p);
}

/* fit(method_ids, params, X, y, inputs, feature_names) -> external pointer. */
SEXP r_n4m_role_pipeline_fit(SEXP method_ids, SEXP values, SEXP X, SEXP y, SEXP inputs,
                             SEXP names) {
    r_est_inputs_t s;
    r_est_inputs(&s, X, y, inputs);
    n4m_context_t* ctx = r_est_context();
    n4m_role_pipeline_t* p = r_rp_create(ctx, method_ids, values);
    r_rp_set_names(ctx, p, names);
    n4m_status_t st = n4m_role_pipeline_fit(ctx, p, &s.in);
    if (st != N4M_OK) r_rp_fail("n4m_role_pipeline_fit", st, ctx, p);
    n4m_context_destroy(ctx);
    return r_rp_wrap(p);
}

/* import(method_ids, params, states, feature_names) -> external pointer. */
SEXP r_n4m_role_pipeline_import(SEXP method_ids, SEXP values, SEXP states, SEXP names) {
    const R_xlen_t n = XLENGTH(states);
    const void** buffers = (const void**)R_alloc((size_t)(n > 0 ? n : 1), sizeof(void*));
    size_t* sizes = (size_t*)R_alloc((size_t)(n > 0 ? n : 1), sizeof(size_t));
    for (R_xlen_t k = 0; k < n; ++k) {
        SEXP bytes = VECTOR_ELT(states, k);
        if (TYPEOF(bytes) != RAWSXP) Rf_error("N4ME state %d must be a raw vector", (int)k);
        buffers[k] = RAW(bytes);
        sizes[k] = (size_t)XLENGTH(bytes);
    }
    n4m_context_t* ctx = r_est_context();
    n4m_role_pipeline_t* p = r_rp_create(ctx, method_ids, values);
    r_rp_set_names(ctx, p, names);
    n4m_status_t st = n4m_role_pipeline_import_states(ctx, p, (int32_t)n, buffers, sizes);
    if (st != N4M_OK) r_rp_fail("n4m_role_pipeline_import_states", st, ctx, p);
    n4m_context_destroy(ctx);
    return r_rp_wrap(p);
}

typedef n4m_status_t (*r_rp_matrix_fn)(n4m_context_t*, const n4m_role_pipeline_t*,
                                        const n4m_matrix_view_t*, n4m_matrix_view_t*);

/* op(ptr, X, names, kind): kind 0 predict, 1 transform, 2 decision function,
 * 3 probabilities, 4 class ids. The input columns are checked first. */
SEXP r_n4m_role_pipeline_op(SEXP ptr, SEXP X, SEXP names, SEXP kind) {
    n4m_role_pipeline_t* p = r_rp_get(ptr);
    const int op = Rf_asInteger(kind);
    n4m_matrix_view_t Xv = r_est_view(X, "X");
    n4m_context_t* ctx = r_est_context();
    n4m_status_t st = n4m_role_pipeline_check_features(ctx, p, (int64_t)Rf_ncols(X),
                                                       r_rp_names(names));
    if (st != N4M_OK) r_est_fail("n4m_role_pipeline_check_features", st, ctx, NULL, NULL);
    const int64_t n = (int64_t)Rf_nrows(X);
    if (op == 4) {
        int64_t* buf = (int64_t*)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int64_t));
        st = n4m_role_pipeline_predict_labels(ctx, p, &Xv, buf, n);
        if (st != N4M_OK) r_est_fail("n4m_role_pipeline_predict_labels", st, ctx, NULL, NULL);
        n4m_context_destroy(ctx);
        return r_est_int64_vector(buf, n);
    }
    int64_t cols = 0;
    st = op == 1 ? n4m_role_pipeline_transform_cols(p, &cols) : n4m_role_pipeline_n_outputs(p, &cols);
    if (st != N4M_OK) r_est_fail("n4m_role_pipeline output width", st, ctx, NULL, NULL);
    const r_rp_matrix_fn fns[] = {n4m_role_pipeline_predict, n4m_role_pipeline_transform,
                                  n4m_role_pipeline_decision_function,
                                  n4m_role_pipeline_predict_proba};
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)n, (int)cols));
    n4m_matrix_view_t Ov = r_est_view(out, "out");
    st = fns[op](ctx, p, &Xv, &Ov);
    if (st != N4M_OK) {
        UNPROTECT(1);
        r_est_fail("n4m_role_pipeline", st, ctx, NULL, NULL);
    }
    n4m_context_destroy(ctx);
    UNPROTECT(1);
    return out;
}

/* One N4ME raw vector per stateful step. */
SEXP r_n4m_role_pipeline_export(SEXP ptr, SEXP allow_training_rows) {
    n4m_role_pipeline_t* p = r_rp_get(ptr);
    const uint32_t flags = Rf_asLogical(allow_training_rows) == TRUE
                               ? N4M_EXPORT_ALLOW_TRAINING_ROWS : 0u;
    int32_t n = 0;
    n4m_role_pipeline_n_states(p, &n);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    n4m_context_t* ctx = r_est_context();
    for (int32_t k = 0; k < n; ++k) {
        size_t size = 0;
        n4m_status_t st = n4m_role_pipeline_export_state_size(ctx, p, k, flags, &size);
        if (st != N4M_OK) {
            UNPROTECT(1);
            r_est_fail("n4m_role_pipeline_export_state_size", st, ctx, NULL, NULL);
        }
        SEXP bytes = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)size));
        size_t written = 0;
        st = n4m_role_pipeline_export_state_to_buffer(ctx, p, k, flags, RAW(bytes), size, &written);
        if (st != N4M_OK) {
            UNPROTECT(2);
            r_est_fail("n4m_role_pipeline_export_state_to_buffer", st, ctx, NULL, NULL);
        }
        SET_VECTOR_ELT(out, k, bytes);
        UNPROTECT(1);
    }
    n4m_context_destroy(ctx);
    UNPROTECT(1);
    return out;
}

/* list(steps = list(method_id, role, state_index, contains_training_rows,
 * n_features_in, n_features_out), n_features, feature_names, classes). */
SEXP r_n4m_role_pipeline_info(SEXP ptr) {
    n4m_role_pipeline_t* p = r_rp_get(ptr);
    int32_t n = 0;
    n4m_role_pipeline_n_steps(p, &n);
    SEXP ids = PROTECT(Rf_allocVector(STRSXP, n));
    SEXP role = PROTECT(Rf_allocVector(INTSXP, n));
    SEXP state = PROTECT(Rf_allocVector(INTSXP, n));
    SEXP rows = PROTECT(Rf_allocVector(LGLSXP, n));
    SEXP n_in = PROTECT(Rf_allocVector(REALSXP, n));
    SEXP n_out = PROTECT(Rf_allocVector(REALSXP, n));
    for (int32_t k = 0; k < n; ++k) {
        n4m_role_pipeline_step_info_v1_t info;
        memset(&info, 0, sizeof(info));
        info.struct_size = sizeof(info);
        n4m_role_pipeline_step_info_v1(p, k, &info);
        SET_STRING_ELT(ids, k, Rf_mkChar(info.method_id));
        INTEGER(role)[k] = (int)info.role;
        INTEGER(state)[k] = info.state_index;
        LOGICAL(rows)[k] = info.contains_training_rows != 0;
        REAL(n_in)[k] = (double)info.n_features_in;
        REAL(n_out)[k] = (double)info.n_features_out;
    }
    const char* step_fields[] = {"method_id", "role", "state_index", "contains_training_rows",
                                 "n_features_in", "n_features_out"};
    SEXP steps = PROTECT(Rf_allocVector(VECSXP, 6));
    SEXP step_names = PROTECT(Rf_allocVector(STRSXP, 6));
    SEXP columns[] = {ids, role, state, rows, n_in, n_out};
    for (int k = 0; k < 6; ++k) {
        SET_VECTOR_ELT(steps, k, columns[k]);
        SET_STRING_ELT(step_names, k, Rf_mkChar(step_fields[k]));
    }
    Rf_setAttrib(steps, R_NamesSymbol, step_names);

    int64_t n_features = 0, n_names = 0;
    n4m_role_pipeline_n_features_in(p, &n_features);
    n4m_role_pipeline_n_feature_names(p, &n_names);
    SEXP feature_names = PROTECT(n_names > 0 ? Rf_allocVector(STRSXP, (R_xlen_t)n_names) : R_NilValue);
    for (int64_t k = 0; k < n_names; ++k) {
        const char* name = NULL;
        n4m_role_pipeline_feature_name(p, k, &name);
        SET_STRING_ELT(feature_names, (R_xlen_t)k, Rf_mkCharCE(name, CE_UTF8));
    }
    SEXP classes = R_NilValue;
    int64_t count = 0;
    if (n4m_role_pipeline_classes(p, NULL, 0, &count) == N4M_OK) {
        int64_t* buf = (int64_t*)R_alloc((size_t)(count > 0 ? count : 1), sizeof(int64_t));
        n4m_role_pipeline_classes(p, buf, count, &count);
        classes = r_est_int64_vector(buf, count);
    }
    PROTECT(classes);
    const char* fields[] = {"steps", "n_features", "feature_names", "classes"};
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
    SEXP out_names = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_VECTOR_ELT(out, 0, steps);
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal((double)n_features));
    SET_VECTOR_ELT(out, 2, feature_names);
    SET_VECTOR_ELT(out, 3, classes);
    for (int k = 0; k < 4; ++k) SET_STRING_ELT(out_names, k, Rf_mkChar(fields[k]));
    Rf_setAttrib(out, R_NamesSymbol, out_names);
    UNPROTECT(12);
    return out;
}
