/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * R .Call gateway for the generic estimator roles (n4m/estimator.h).
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
        if (XLENGTH(v) < 1) return N4M_ERR_INVALID_ARGUMENT;
        switch (pi.type) {
            case N4M_METHOD_PARAM_INT:
                st = n4m_params_set_int(params, name, (int64_t)Rf_asReal(v));
                break;
            case N4M_METHOD_PARAM_DOUBLE:
                st = n4m_params_set_double(params, name, Rf_asReal(v));
                break;
            case N4M_METHOD_PARAM_BOOL:
                st = n4m_params_set_bool(params, name, Rf_asLogical(v) == TRUE ? 1 : 0);
                break;
            case N4M_METHOD_PARAM_ENUM:
                if (!Rf_isString(v)) return N4M_ERR_INVALID_ARGUMENT;
                st = n4m_params_set_enum(params, name, CHAR(STRING_ELT(v, 0)));
                break;
            case N4M_METHOD_PARAM_INT_ARRAY: {
                SEXP d = PROTECT(Rf_coerceVector(v, REALSXP));
                int64_t* buf = (int64_t*)R_alloc((size_t)XLENGTH(d), sizeof(int64_t));
                for (R_xlen_t k = 0; k < XLENGTH(d); ++k) buf[k] = (int64_t)REAL(d)[k];
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
        if (!R_FINITE(REAL(d)[k])) Rf_error("%s must contain finite integers", name);
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

/* fit(method_id, params, X, y, inputs) -> external pointer.
 * `inputs` is a named list among sample_weight, groups, feature_groups,
 * blocks, axis, X_target, fold_ids (R 1-based ids are not used: groups and
 * fold ids are opaque labels, feature groups are labels, blocks are sizes). */
SEXP r_n4m_estimator_fit(SEXP method_id, SEXP values, SEXP X, SEXP y, SEXP inputs) {
    const char* id = CHAR(STRING_ELT(method_id, 0));
    int32_t index = -1;
    if (n4m_method_find(id, &index) != N4M_OK) Rf_error("unknown n4m method '%s'", id);

    n4m_fit_inputs_v1_t in;
    memset(&in, 0, sizeof(in));
    in.struct_size = sizeof(in);
    n4m_matrix_view_t Xv = r_est_view(X, "X");
    in.X = &Xv;
    n4m_matrix_view_t Yv;
    if (!Rf_isNull(y)) {
        Yv = r_est_view(y, "y");
        in.Y = &Yv;
    }
    n4m_matrix_view_t Tv;
    SEXP names = Rf_getAttrib(inputs, R_NamesSymbol);
    for (R_xlen_t i = 0; i < XLENGTH(inputs); ++i) {
        const char* name = CHAR(STRING_ELT(names, i));
        SEXP v = VECTOR_ELT(inputs, i);
        int64_t n = (int64_t)XLENGTH(v);
        if (strcmp(name, "sample_weight") == 0) {
            in.sample_weight = r_est_double(v);
            in.n_sample_weight = n;
        } else if (strcmp(name, "axis") == 0) {
            in.axis = r_est_double(v);
            in.n_axis = n;
        } else if (strcmp(name, "groups") == 0) {
            in.groups = r_est_int64(v, name);
            in.n_groups = n;
        } else if (strcmp(name, "feature_groups") == 0) {
            in.feature_groups = r_est_int64(v, name);
            in.n_feature_groups = n;
        } else if (strcmp(name, "blocks") == 0) {
            in.block_sizes = r_est_int64(v, name);
            in.n_blocks = n;
        } else if (strcmp(name, "fold_ids") == 0) {
            in.fold_ids = r_est_int64(v, name);
            in.n_fold_ids = n;
        } else if (strcmp(name, "X_target") == 0) {
            Tv = r_est_view(v, "X_target");
            in.X_target = &Tv;
        } else {
            Rf_error("unknown fit input '%s'", name);
        }
    }

    n4m_context_t* ctx = r_est_context();
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
    n4m_estimator_t* est = NULL;
    st = n4m_estimator_create(ctx, id, params, &est);
    if (st != N4M_OK) r_est_fail("n4m_estimator_create", st, ctx, params, NULL);
    n4m_params_destroy(params);
    st = n4m_estimator_fit(ctx, est, &in);
    if (st != N4M_OK) r_est_fail("n4m_estimator_fit", st, ctx, NULL, est);
    n4m_context_destroy(ctx);
    return r_est_wrap(est);
}

static SEXP r_est_matrix_op(SEXP ptr, SEXP X, int transform) {
    n4m_estimator_t* est = r_est_get(ptr);
    int64_t cols = 0;
    n4m_status_t st = transform ? n4m_estimator_transform_cols(est, &cols)
                                : n4m_estimator_n_outputs(est, &cols);
    if (st != N4M_OK) r_est_fail(transform ? "transform" : "predict", st, NULL, NULL, NULL);
    n4m_matrix_view_t Xv = r_est_view(X, "X");
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(X), (int)cols));
    n4m_matrix_view_t Ov = r_est_view(out, "out");
    n4m_context_t* ctx = r_est_context();
    st = transform ? n4m_estimator_transform(ctx, est, &Xv, &Ov)
                   : n4m_estimator_predict(ctx, est, &Xv, &Ov);
    if (st != N4M_OK) {
        UNPROTECT(1);
        r_est_fail(transform ? "n4m_estimator_transform" : "n4m_estimator_predict", st, ctx,
                   NULL, NULL);
    }
    n4m_context_destroy(ctx);
    UNPROTECT(1);
    return out;
}

SEXP r_n4m_estimator_predict(SEXP ptr, SEXP X) { return r_est_matrix_op(ptr, X, 0); }

SEXP r_n4m_estimator_transform(SEXP ptr, SEXP X) { return r_est_matrix_op(ptr, X, 1); }

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

SEXP r_n4m_estimator_export(SEXP ptr) {
    n4m_estimator_t* est = r_est_get(ptr);
    n4m_context_t* ctx = r_est_context();
    size_t size = 0;
    const uint32_t flags = N4M_EXPORT_ALLOW_TRAINING_ROWS;
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
