/* SPDX-License-Identifier: CECILL-2.1 */
/* Thin R lifetime and value bridge for the native N4M optimizer. */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <math.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "n4m/optimization.h"

typedef struct {
    n4m_context_t* context;
    n4m_optimizer_t* optimizer;
    n4m_search_space_t* pending_space;
} r_optimizer_t;

static SEXP field(SEXP object, const char* name) {
    SEXP names = Rf_getAttrib(object, R_NamesSymbol);
    if (TYPEOF(object) != VECSXP || TYPEOF(names) != STRSXP)
        Rf_error("optimizer input must be a named list");
    for (R_xlen_t i = 0; i < XLENGTH(object); ++i)
        if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0)
            return VECTOR_ELT(object, i);
    Rf_error("optimizer input lacks '%s'", name);
    return R_NilValue;
}

static const char* string_scalar(SEXP value, const char* what) {
    if (TYPEOF(value) != STRSXP || XLENGTH(value) != 1 || STRING_ELT(value, 0) == NA_STRING)
        Rf_error("%s must be one string", what);
    return CHAR(STRING_ELT(value, 0));
}

static double number_scalar(SEXP value, const char* what) {
    if ((TYPEOF(value) != REALSXP && TYPEOF(value) != INTSXP) || XLENGTH(value) != 1)
        Rf_error("%s must be one finite number", what);
    double out = TYPEOF(value) == REALSXP ? REAL(value)[0] : (double)INTEGER(value)[0];
    if (!R_FINITE(out)) Rf_error("%s must be finite", what);
    return out;
}

static int32_t int_scalar(SEXP value, const char* what) {
    double x = number_scalar(value, what);
    if (x != floor(x) || x < INT32_MIN || x > INT32_MAX)
        Rf_error("%s must fit int32", what);
    return (int32_t)x;
}

static int64_t int64_scalar(SEXP value, const char* what) {
    double x = number_scalar(value, what);
    if (x != floor(x) || x < -9007199254740992.0 || x > 9007199254740992.0)
        Rf_error("%s must be an exactly representable integer", what);
    return (int64_t)x;
}

static int flag_scalar(SEXP value, const char* what) {
    if (TYPEOF(value) != LGLSXP || XLENGTH(value) != 1 || LOGICAL(value)[0] == NA_LOGICAL)
        Rf_error("%s must be TRUE or FALSE", what);
    return LOGICAL(value)[0] != 0;
}

static void optimizer_finalizer(SEXP ptr) {
    r_optimizer_t* handle = (r_optimizer_t*)R_ExternalPtrAddr(ptr);
    if (!handle) return;
    n4m_optimizer_destroy(handle->optimizer);
    n4m_search_space_destroy(handle->pending_space);
    n4m_context_destroy(handle->context);
    free(handle);
    R_ClearExternalPtr(ptr);
}

static r_optimizer_t* optimizer_handle(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("n4m_optimizer"))
        Rf_error("expected an n4m optimizer handle");
    r_optimizer_t* handle = (r_optimizer_t*)R_ExternalPtrAddr(ptr);
    if (!handle || !handle->optimizer) Rf_error("optimizer handle is closed");
    return handle;
}

SEXP r_n4m_optimizer_close(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP)
        Rf_error("expected an n4m optimizer handle");
    if (!R_ExternalPtrAddr(ptr)) return Rf_ScalarLogical(1);
    if (R_ExternalPtrTag(ptr) != Rf_install("n4m_optimizer"))
        Rf_error("expected an n4m optimizer handle");
    optimizer_finalizer(ptr);
    return Rf_ScalarLogical(1);
}

static void native_error(n4m_status_t status, n4m_context_t* context,
                         n4m_search_space_t* space, const char* operation) {
    /* During create, pending_space belongs to the external pointer finalizer.
     * Rf_error longjmps, so that finalizer also covers validation errors. */
    (void)space;
    char detail[1024] = {0};
    if (context) {
        const char* message = n4m_context_last_error(context);
        if (message) {
            strncpy(detail, message, sizeof(detail) - 1);
            detail[sizeof(detail) - 1] = '\0';
        }
    }
    if (detail[0]) Rf_error("%s: %s (%s)", operation, n4m_status_to_string(status), detail);
    Rf_error("%s: %s", operation, n4m_status_to_string(status));
}

static void check(n4m_status_t status, n4m_context_t* context,
                  n4m_search_space_t* space, const char* operation) {
    if (status != N4M_OK) native_error(status, context, space, operation);
}

static void add_parameter(n4m_search_space_t* native, const char* name, SEXP spec,
                          n4m_context_t* context) {
    int kind = int_scalar(field(spec, "kind"), "parameter kind");
    n4m_status_t status = N4M_ERR_INVALID_ARGUMENT;
    if (kind == N4M_PARAM_INT || kind == N4M_PARAM_LOG_INT) {
        status = n4m_search_space_add_int(native, name,
            int64_scalar(field(spec, "low"), "low"),
            int64_scalar(field(spec, "high"), "high"),
            int64_scalar(field(spec, "step"), "step"), kind == N4M_PARAM_LOG_INT);
    } else if (kind == N4M_PARAM_FLOAT || kind == N4M_PARAM_LOG_FLOAT) {
        status = n4m_search_space_add_float(native, name,
            number_scalar(field(spec, "low"), "low"),
            number_scalar(field(spec, "high"), "high"),
            number_scalar(field(spec, "step"), "step"), kind == N4M_PARAM_LOG_FLOAT);
    } else if (kind == N4M_PARAM_CATEGORICAL) {
        SEXP values = field(spec, "choices");
        int cat_type = int_scalar(field(spec, "category_type"), "category_type");
        if (XLENGTH(values) < 1 || XLENGTH(values) > INT32_MAX)
            Rf_error("categorical choices must have 1..INT32_MAX values");
        int32_t n = (int32_t)XLENGTH(values);
        if (cat_type == N4M_CAT_STR && TYPEOF(values) == STRSXP) {
            const char** items = (const char**)R_alloc((size_t)n, sizeof(char*));
            for (int32_t i = 0; i < n; ++i) {
                if (STRING_ELT(values, i) == NA_STRING) Rf_error("choices cannot contain NA");
                items[i] = CHAR(STRING_ELT(values, i));
            }
            status = n4m_search_space_add_categorical(native, name, N4M_CAT_STR, items, n);
        } else if (cat_type == N4M_CAT_FLOAT && TYPEOF(values) == REALSXP) {
            for (int32_t i = 0; i < n; ++i)
                if (!R_FINITE(REAL(values)[i])) Rf_error("choices must be finite");
            status = n4m_search_space_add_categorical(native, name, N4M_CAT_FLOAT, REAL(values), n);
        } else if (cat_type == N4M_CAT_INT &&
                   (TYPEOF(values) == INTSXP || TYPEOF(values) == REALSXP)) {
            int64_t* items = (int64_t*)R_alloc((size_t)n, sizeof(int64_t));
            for (int32_t i = 0; i < n; ++i) {
                if (TYPEOF(values) == INTSXP && INTEGER(values)[i] == NA_INTEGER)
                    Rf_error("choices cannot contain NA");
                double value = TYPEOF(values) == INTSXP
                    ? (double)INTEGER(values)[i] : REAL(values)[i];
                if (!R_FINITE(value) || value != floor(value) ||
                    value < -9007199254740992.0 || value > 9007199254740992.0)
                    Rf_error("integer categorical choices must be binary64-exact integers");
                items[i] = (int64_t)value;
            }
            status = n4m_search_space_add_categorical(native, name, N4M_CAT_INT, items, n);
        } else if (cat_type == N4M_CAT_BOOL && TYPEOF(values) == LGLSXP) {
            for (int32_t i = 0; i < n; ++i)
                if (LOGICAL(values)[i] == NA_LOGICAL) Rf_error("choices cannot contain NA");
            status = n4m_search_space_add_categorical(native, name, N4M_CAT_BOOL, LOGICAL(values), n);
        } else Rf_error("choices and category_type disagree");
    } else if (kind == N4M_PARAM_ORDINAL) {
        SEXP values = field(spec, "choices");
        if (TYPEOF(values) != REALSXP || XLENGTH(values) < 1 || XLENGTH(values) > INT32_MAX)
            Rf_error("ordinal choices must be a nonempty double vector");
        for (R_xlen_t i = 0; i < XLENGTH(values); ++i)
            if (!R_FINITE(REAL(values)[i])) Rf_error("ordinal choices must be finite");
        status = n4m_search_space_add_ordinal(native, name, REAL(values), (int32_t)XLENGTH(values));
    } else if (kind == N4M_PARAM_SORTED_TUPLE) {
        status = n4m_search_space_add_sorted_tuple(native, name,
            int_scalar(field(spec, "length"), "length"),
            number_scalar(field(spec, "low"), "low"),
            number_scalar(field(spec, "high"), "high"),
            flag_scalar(field(spec, "integer"), "integer"));
    } else Rf_error("unknown parameter kind");
    check(status, context, native, name);
}

static void add_constraints(n4m_search_space_t* native, SEXP constraints,
                            n4m_context_t* context) {
    if (TYPEOF(constraints) != VECSXP) Rf_error("constraints must be a list");
    for (R_xlen_t i = 0; i < XLENGTH(constraints); ++i) {
        SEXP item = VECTOR_ELT(constraints, i);
        int kind = int_scalar(field(item, "kind"), "constraint kind");
        SEXP refs = field(item, "refs");
        SEXP labels = field(item, "labels");
        if (TYPEOF(refs) != STRSXP || TYPEOF(labels) != STRSXP ||
            XLENGTH(refs) != XLENGTH(labels) || XLENGTH(refs) < 1 || XLENGTH(refs) > INT32_MAX)
            Rf_error("constraint refs and labels must be equal nonempty character vectors");
        int32_t n = (int32_t)XLENGTH(refs);
        const char** ref_ptrs = (const char**)R_alloc((size_t)n, sizeof(char*));
        const char** label_ptrs = (const char**)R_alloc((size_t)n, sizeof(char*));
        for (int32_t j = 0; j < n; ++j) {
            if (STRING_ELT(refs, j) == NA_STRING || STRING_ELT(labels, j) == NA_STRING)
                Rf_error("constraint refs and labels cannot contain NA");
            ref_ptrs[j] = CHAR(STRING_ELT(refs, j));
            label_ptrs[j] = CHAR(STRING_ELT(labels, j));
        }
        check(n4m_search_space_add_constraint(native, (n4m_constraint_kind_t)kind,
              ref_ptrs, label_ptrs, n), context, native, "add_constraint");
    }
}

static SEXP new_handle(void) {
    r_optimizer_t* handle = (r_optimizer_t*)calloc(1, sizeof(r_optimizer_t));
    if (!handle) Rf_error("optimizer handle allocation failed");
    SEXP ptr = PROTECT(R_MakeExternalPtr(handle, Rf_install("n4m_optimizer"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, optimizer_finalizer, TRUE);
    n4m_status_t status = n4m_context_create(&handle->context);
    if (status != N4M_OK) native_error(status, NULL, NULL, "context_create");
    UNPROTECT(1);
    return ptr;
}

SEXP r_n4m_optimizer_create(SEXP parameters, SEXP constraints, SEXP options) {
    if (TYPEOF(parameters) != VECSXP || XLENGTH(parameters) < 1 ||
        TYPEOF(Rf_getAttrib(parameters, R_NamesSymbol)) != STRSXP)
        Rf_error("space must be a nonempty named list of parameters");
    if (TYPEOF(options) != VECSXP) Rf_error("options must be a named list");
    SEXP ptr = PROTECT(new_handle());
    r_optimizer_t* handle = (r_optimizer_t*)R_ExternalPtrAddr(ptr);
    n4m_search_space_t* native = NULL;
    check(n4m_search_space_create(&native), handle->context, NULL, "search_space_create");
    handle->pending_space = native;
    SEXP names = Rf_getAttrib(parameters, R_NamesSymbol);
    for (R_xlen_t i = 0; i < XLENGTH(parameters); ++i) {
        if (STRING_ELT(names, i) == NA_STRING) Rf_error("parameter name cannot be NA");
        add_parameter(native, CHAR(STRING_ELT(names, i)), VECTOR_ELT(parameters, i), handle->context);
    }
    add_constraints(native, constraints, handle->context);
    n4m_optimizer_options_t native_options;
    n4m_optimizer_options_init(&native_options);
    native_options.sampler = (n4m_sampler_kind_t)int_scalar(field(options, "sampler"), "sampler");
    native_options.pruner = (n4m_pruner_kind_t)int_scalar(field(options, "pruner"), "pruner");
    native_options.direction = (n4m_opt_direction_t)int_scalar(field(options, "direction"), "direction");
    native_options.eval_mode = (n4m_eval_mode_t)int_scalar(field(options, "eval_mode"), "eval_mode");
    native_options.metric = (n4m_metric_t)int_scalar(field(options, "metric"), "metric");
    native_options.liar = (n4m_liar_kind_t)int_scalar(field(options, "liar"), "liar");
    native_options.n_startup_trials = int_scalar(field(options, "n_startup_trials"), "n_startup_trials");
    native_options.seed = (uint64_t)int64_scalar(field(options, "seed"), "seed");
    native_options.timeout_seconds = number_scalar(field(options, "timeout_seconds"), "timeout_seconds");
    native_options.max_resource = int_scalar(field(options, "max_resource"), "max_resource");
    native_options.reduction_factor = int_scalar(field(options, "reduction_factor"), "reduction_factor");
    n4m_status_t status = n4m_optimizer_create(handle->context, native, &native_options,
                                                 &handle->optimizer);
    check(status, handle->context, native, "optimizer_create");
    n4m_search_space_destroy(native);
    handle->pending_space = NULL;
    UNPROTECT(1);
    return ptr;
}

SEXP r_n4m_optimizer_load(SEXP blob) {
    if (TYPEOF(blob) != RAWSXP || XLENGTH(blob) == 0)
        Rf_error("checkpoint must be a nonempty raw vector");
    SEXP ptr = PROTECT(new_handle());
    r_optimizer_t* handle = (r_optimizer_t*)R_ExternalPtrAddr(ptr);
    check(n4m_optimizer_load(handle->context, RAW(blob), (uint64_t)XLENGTH(blob),
          &handle->optimizer), handle->context, NULL, "optimizer_load");
    UNPROTECT(1);
    return ptr;
}

static SEXP trial_value(const n4m_trial_t* trial, const char* name, SEXP spec,
                        n4m_context_t* context) {
    int kind = int_scalar(field(spec, "kind"), "parameter kind");
    int32_t active = 0;
    /* Tuple roots are declarations, while trial parameters are name#0, name#1... */
    char* first_component = NULL;
    if (kind == N4M_PARAM_SORTED_TUPLE) {
        first_component = (char*)R_alloc(strlen(name) + 32, 1);
        sprintf(first_component, "%s#0", name);
    }
    check(n4m_trial_is_active(trial, first_component ? first_component : name, &active),
          context, NULL, "trial_is_active");
    if (!active) return R_NilValue;
    if (kind == N4M_PARAM_INT || kind == N4M_PARAM_LOG_INT) {
        int64_t value = 0;
        check(n4m_trial_get_int(trial, name, &value), context, NULL, "trial_get_int");
        return Rf_ScalarReal((double)value);
    }
    if (kind == N4M_PARAM_FLOAT || kind == N4M_PARAM_LOG_FLOAT || kind == N4M_PARAM_ORDINAL) {
        double value = 0;
        check(n4m_trial_get_float(trial, name, &value), context, NULL, "trial_get_float");
        return Rf_ScalarReal(value);
    }
    if (kind == N4M_PARAM_CATEGORICAL) {
        int32_t index = -1;
        check(n4m_trial_get_category(trial, name, &index, NULL), context, NULL, "trial_get_category");
        SEXP choices = field(spec, "choices");
        if (index < 0 || (R_xlen_t)index >= XLENGTH(choices))
            Rf_error("native categorical index is out of range");
        if (TYPEOF(choices) == STRSXP) return Rf_ScalarString(STRING_ELT(choices, index));
        if (TYPEOF(choices) == REALSXP) return Rf_ScalarReal(REAL(choices)[index]);
        if (TYPEOF(choices) == INTSXP) return Rf_ScalarInteger(INTEGER(choices)[index]);
        return Rf_ScalarLogical(LOGICAL(choices)[index]);
    }
    if (kind == N4M_PARAM_SORTED_TUPLE) {
        int32_t n = int_scalar(field(spec, "length"), "length");
        SEXP values = PROTECT(Rf_allocVector(REALSXP, n));
        for (int32_t i = 0; i < n; ++i) {
            char* component = (char*)R_alloc(strlen(name) + 32, 1);
            sprintf(component, "%s#%d", name, i);
            check(n4m_trial_get_float(trial, component, &REAL(values)[i]), context, NULL,
                  "trial_get_tuple_component");
        }
        UNPROTECT(1);
        return values;
    }
    Rf_error("unknown parameter kind");
    return R_NilValue;
}

static SEXP trial_record(const n4m_trial_t* trial, SEXP parameters, n4m_context_t* context) {
    int64_t id = 0;
    int32_t rung = 0;
    n4m_trial_status_t status = N4M_TRIAL_RUNNING;
    check(n4m_trial_get_id(trial, &id), context, NULL, "trial_get_id");
    check(n4m_trial_get_rung(trial, &rung), context, NULL, "trial_get_rung");
    check(n4m_trial_get_status(trial, &status), context, NULL, "trial_get_status");
    SEXP values = PROTECT(Rf_allocVector(VECSXP, XLENGTH(parameters)));
    SEXP names = Rf_getAttrib(parameters, R_NamesSymbol);
    Rf_setAttrib(values, R_NamesSymbol, names);
    for (R_xlen_t i = 0; i < XLENGTH(parameters); ++i) {
        const char* name = CHAR(STRING_ELT(names, i));
        SET_VECTOR_ELT(values, i, trial_value(trial, name, VECTOR_ELT(parameters, i), context));
    }
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(out, 0, Rf_ScalarReal((double)id));
    SET_VECTOR_ELT(out, 1, values);
    SET_VECTOR_ELT(out, 2, Rf_ScalarInteger((int)status));
    SET_VECTOR_ELT(out, 3, Rf_ScalarInteger(rung));
    SEXP out_names = PROTECT(Rf_allocVector(STRSXP, 4));
    const char* labels[] = {"id", "parameters", "status", "rung"};
    for (int i = 0; i < 4; ++i) SET_STRING_ELT(out_names, i, Rf_mkChar(labels[i]));
    Rf_setAttrib(out, R_NamesSymbol, out_names);
    UNPROTECT(3);
    return out;
}

SEXP r_n4m_optimizer_ask(SEXP ptr, SEXP parameters) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    n4m_trial_t* trial = NULL;
    check(n4m_optimizer_ask(handle->optimizer, &trial), handle->context, NULL, "optimizer_ask");
    return trial_record(trial, parameters, handle->context);
}

SEXP r_n4m_optimizer_ask_batch(SEXP ptr, SEXP parameters, SEXP count) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    int32_t requested = int_scalar(count, "batch size");
    if (requested < 0 || requested > 10000)
        Rf_error("batch size must be between 0 and 10000");
    n4m_trial_t** trials = requested
        ? (n4m_trial_t**)R_alloc((size_t)requested, sizeof(n4m_trial_t*)) : NULL;
    int32_t committed = 0;
    n4m_status_t status = n4m_optimizer_ask_batch(handle->optimizer, requested,
                                                    trials, &committed);
    if (committed < 0 || committed > requested)
        Rf_error("native ask_batch returned an invalid count");
    if (status != N4M_OK && committed == 0)
        native_error(status, handle->context, NULL, "optimizer_ask_batch");
    SEXP out = PROTECT(Rf_allocVector(VECSXP, committed));
    for (int32_t i = 0; i < committed; ++i)
        SET_VECTOR_ELT(out, i, trial_record(trials[i], parameters, handle->context));
    if (status != N4M_OK) {
        SEXP code = PROTECT(Rf_ScalarInteger((int)status));
        Rf_setAttrib(out, Rf_install("native_status"), code);
        UNPROTECT(1);
    }
    UNPROTECT(1);
    return out;
}

SEXP r_n4m_optimizer_enqueue(SEXP ptr, SEXP values) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    SEXP names = Rf_getAttrib(values, R_NamesSymbol);
    if (TYPEOF(values) != REALSXP || TYPEOF(names) != STRSXP ||
        XLENGTH(values) < 1 || XLENGTH(values) > INT32_MAX)
        Rf_error("enqueue expects a nonempty named double vector");
    int32_t n = (int32_t)XLENGTH(values);
    const char** keys = (const char**)R_alloc((size_t)n, sizeof(char*));
    for (int32_t i = 0; i < n; ++i) {
        if (STRING_ELT(names, i) == NA_STRING || !R_FINITE(REAL(values)[i]))
            Rf_error("enqueue names and values must be finite and non-missing");
        keys[i] = CHAR(STRING_ELT(names, i));
    }
    check(n4m_optimizer_enqueue(handle->optimizer, keys, REAL(values), n),
          handle->context, NULL, "optimizer_enqueue");
    return Rf_ScalarLogical(1);
}

SEXP r_n4m_optimizer_tell(SEXP ptr, SEXP id, SEXP status, SEXP score, SEXP error) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    int trial_status = int_scalar(status, "status");
    n4m_status_t native_status = n4m_optimizer_tell_result(handle->optimizer,
        int64_scalar(id, "trial id"), (n4m_trial_status_t)trial_status,
        Rf_isNull(score) ? 0.0 : number_scalar(score, "score"),
        Rf_isNull(error) ? NULL : string_scalar(error, "error"));
    check(native_status, handle->context, NULL, "optimizer_tell_result");
    return Rf_ScalarLogical(1);
}

SEXP r_n4m_optimizer_intermediate(SEXP ptr, SEXP id, SEXP step, SEXP score) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    int32_t prune = 0;
    check(n4m_optimizer_tell_intermediate(handle->optimizer,
          int64_scalar(id, "trial id"), int_scalar(step, "step"),
          number_scalar(score, "score"), &prune), handle->context, NULL,
          "optimizer_tell_intermediate");
    return Rf_ScalarLogical(prune != 0);
}

SEXP r_n4m_optimizer_best(SEXP ptr, SEXP parameters) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    n4m_trial_t* trial = NULL;
    double score = 0;
    check(n4m_optimizer_best(handle->optimizer, &trial, &score), handle->context, NULL,
          "optimizer_best");
    SEXP record = PROTECT(trial_record(trial, parameters, handle->context));
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, record);
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal(score));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("trial"));
    SET_STRING_ELT(names, 1, Rf_mkChar("score"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(3);
    return out;
}

SEXP r_n4m_optimizer_save(SEXP ptr) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    n4m_array_t* blob = NULL;
    check(n4m_optimizer_save(handle->optimizer, &blob), handle->context, NULL, "optimizer_save");
    n4m_matrix_view_t view = {0};
    n4m_status_t status = n4m_array_view(blob, &view);
    if (status != N4M_OK) {
        n4m_array_free(blob);
        native_error(status, handle->context, NULL, "optimizer_save_view");
    }
    if (view.dtype != N4M_DTYPE_I64 || view.rows != 1 || view.cols < 1 ||
        view.col_stride != 1 || !view.data ||
        (uint64_t)view.cols > (uint64_t)R_XLEN_T_MAX / 8) {
        n4m_array_free(blob);
        Rf_error("native checkpoint size is invalid");
    }
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)view.cols * 8));
    memcpy(RAW(out), view.data, (size_t)view.cols * 8);
    n4m_array_free(blob);
    UNPROTECT(1);
    return out;
}

/* Preserve native rich-trace fields without recreating the optimizer's state
 * machine in R. Matrices are converted from the C row-major layout to R's
 * column-major layout. Int64 vectors stay exact; large values become strings. */
static SEXP trace_entry(const n4m_method_result_t* result, const char* name, int kind,
                        n4m_context_t* context) {
    if (kind == N4M_RESULT_SCALAR) {
        double value = 0;
        check(n4m_method_result_get_scalar(result, name, &value), context, NULL, name);
        return Rf_ScalarReal(value);
    }
    if (kind == N4M_RESULT_INT_VECTOR) {
        const int32_t* values = NULL;
        int32_t size = 0;
        check(n4m_method_result_get_int_vector(result, name, &values, &size), context, NULL, name);
        if (size < 0) Rf_error("native trace vector size is invalid");
        SEXP out = PROTECT(Rf_allocVector(INTSXP, size));
        for (int32_t i = 0; i < size; ++i) INTEGER(out)[i] = values[i];
        UNPROTECT(1);
        return out;
    }
    if (kind == N4M_RESULT_INT64_VECTOR) {
        const int64_t* values = NULL;
        int64_t size = 0;
        check(n4m_method_result_get_int64_vector(result, name, &values, &size), context, NULL, name);
        if (size < 0 || (uint64_t)size > (uint64_t)R_XLEN_T_MAX)
            Rf_error("native trace vector size is invalid");
        int exact = 1;
        for (int64_t i = 0; i < size; ++i)
            if (values[i] < -9007199254740992LL || values[i] > 9007199254740992LL)
                exact = 0;
        SEXP out = PROTECT(Rf_allocVector(exact ? REALSXP : STRSXP, (R_xlen_t)size));
        for (int64_t i = 0; i < size; ++i) {
            if (exact) REAL(out)[i] = (double)values[i];
            else {
                char buffer[32];
                snprintf(buffer, sizeof(buffer), "%" PRId64, values[i]);
                SET_STRING_ELT(out, i, Rf_mkChar(buffer));
            }
        }
        UNPROTECT(1);
        return out;
    }
    if (kind == N4M_RESULT_DOUBLE_MATRIX) {
        const double* values = NULL;
        int64_t rows = 0, cols = 0;
        check(n4m_method_result_get_double_matrix(result, name, &values, &rows, &cols),
              context, NULL, name);
        if (rows < 0 || cols < 0 || rows > INT_MAX || cols > INT_MAX ||
            (cols && rows > R_XLEN_T_MAX / cols))
            Rf_error("native trace matrix size is invalid");
        SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
        for (int64_t col = 0; col < cols; ++col)
            for (int64_t row = 0; row < rows; ++row)
                REAL(out)[col * rows + row] = values[row * cols + col];
        UNPROTECT(1);
        return out;
    }
    Rf_error("unknown native trace entry kind");
    return R_NilValue;
}

SEXP r_n4m_optimizer_trials(SEXP ptr, SEXP since_id) {
    r_optimizer_t* handle = optimizer_handle(ptr);
    n4m_method_result_t* result = NULL;
    check(n4m_optimizer_get_trials(handle->optimizer, int64_scalar(since_id, "since_id"),
          &result), handle->context, NULL, "optimizer_get_trials");
    int32_t count = 0;
    n4m_status_t status = n4m_method_result_entry_count(result, &count);
    if (status != N4M_OK || count < 0) {
        n4m_method_result_destroy(result);
        native_error(status != N4M_OK ? status : N4M_ERR_CORRUPT_BUFFER,
                     handle->context, NULL, "optimizer_trace_entries");
    }
    SEXP out = PROTECT(Rf_allocVector(VECSXP, count));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, count));
    for (int32_t i = 0; i < count; ++i) {
        const char* name = NULL;
        int32_t kind = -1;
        status = n4m_method_result_entry(result, i, &name, &kind);
        if (status != N4M_OK || !name) {
            n4m_method_result_destroy(result);
            native_error(status != N4M_OK ? status : N4M_ERR_CORRUPT_BUFFER,
                         handle->context, NULL, "optimizer_trace_entry");
        }
        SET_STRING_ELT(names, i, Rf_mkCharCE(name, CE_UTF8));
        SET_VECTOR_ELT(out, i, trace_entry(result, name, kind, handle->context));
    }
    Rf_setAttrib(out, R_NamesSymbol, names);
    n4m_method_result_destroy(result);
    UNPROTECT(2);
    return out;
}
