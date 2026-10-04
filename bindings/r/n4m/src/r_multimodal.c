/* SPDX-License-Identifier: CECILL-2.1 */
/* Native raw tensor/UTF-8 marshalling; no learned arithmetic in R. */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "n4m/multimodal.h"

static SEXP field(SEXP value, const char* name) {
    SEXP names = Rf_getAttrib(value, R_NamesSymbol);
    if (TYPEOF(value) != VECSXP || TYPEOF(names) != STRSXP)
        Rf_error("expected a named list");
    for (R_xlen_t i = 0; i < XLENGTH(value); ++i)
        if (!strcmp(CHAR(STRING_ELT(names, i)), name)) return VECTOR_ELT(value, i);
    Rf_error("missing field %s", name);
    return R_NilValue;
}
static const char* text(SEXP value) {
    if (TYPEOF(value) != STRSXP || XLENGTH(value) != 1 || STRING_ELT(value, 0) == NA_STRING)
        Rf_error("expected one UTF-8 string");
    return Rf_translateCharUTF8(STRING_ELT(value, 0));
}
static double number(SEXP value) {
    if (TYPEOF(value) == VECSXP && XLENGTH(value) == 1) value = VECTOR_ELT(value, 0);
    if (!(TYPEOF(value) == REALSXP || TYPEOF(value) == INTSXP) || XLENGTH(value) != 1)
        Rf_error("expected one numeric value");
    double x = Rf_asReal(value);
    if (!R_FINITE(x)) Rf_error("numeric value must be finite");
    return x;
}
static int64_t integer(SEXP value) {
    double x = number(value);
    if (floor(x) != x || x < -9007199254740991.0 || x > 9007199254740991.0)
        Rf_error("expected an exactly representable integer");
    return (int64_t)x;
}
static int boolean(SEXP value) {
    if (TYPEOF(value) != LGLSXP || XLENGTH(value) != 1 || LOGICAL(value)[0] == NA_LOGICAL)
        Rf_error("expected one boolean");
    return LOGICAL(value)[0] != 0;
}
static n4m_multimodal_recipe_v1_t source_configuration(SEXP recipe, SEXP schemas) {
    SEXP order = field(recipe, "source_order");
    if (TYPEOF(order) != STRSXP || XLENGTH(order) < 1 || XLENGTH(order) > 4)
        Rf_error("source_order requires 1..4 selected modality names");
    const int count = (int)XLENGTH(order);
    n4m_multimodal_recipe_v1_t out;
    memset(&out, 0, sizeof(out)); out.struct_size = sizeof(out); out.n_sources = count;
    n4m_multimodal_source_spec_v1_t* sources =
        (n4m_multimodal_source_spec_v1_t*)R_alloc(count, sizeof(*sources));
    memset(sources, 0, count * sizeof(*sources)); out.sources = sources;
    SEXP encoders = field(recipe, "encoders"), weights = field(recipe, "source_weights");
    for (int i = 0; i < count; ++i) {
        if (STRING_ELT(order, i) == NA_STRING) Rf_error("source names must not be missing");
        const char* name = Rf_translateCharUTF8(STRING_ELT(order, i));
        n4m_multimodal_source_spec_v1_t* s = sources + i;
        SEXP schema = field(schemas, name), encoder = field(encoders, name);
        s->struct_size = sizeof(*s); s->name = name;
        s->representation_id = text(field(schema, "representation_id"));
        s->dtype = text(field(schema, "dtype")); s->identity_utf8 = text(field(schema, "identity"));
        s->identity_bytes = strlen((const char*)s->identity_utf8);
        SEXP shape = field(schema, "input_shape");
        if (!(TYPEOF(shape) == INTSXP || TYPEOF(shape) == REALSXP || TYPEOF(shape) == VECSXP) || XLENGTH(shape) > 7)
            Rf_error("shape must be a numeric vector of at most seven dimensions");
        s->ndim = (int32_t)XLENGTH(shape);
        int64_t* dims = (int64_t*)R_alloc(s->ndim, sizeof(*dims)); s->shape = dims;
        for (int j = 0; j < s->ndim; ++j) {
            double d = TYPEOF(shape) == VECSXP ? number(VECTOR_ELT(shape, j)) :
                TYPEOF(shape) == REALSXP ? REAL(shape)[j] : INTEGER(shape)[j];
            if (!R_FINITE(d) || d < 1 || floor(d) != d || d > 1048576) Rf_error("invalid shape dimension");
            dims[j] = (int64_t)d;
        }
        s->weight = number(field(weights, s->name)); s->numeric_column = s->categorical_column = -1;
        const char* kind = text(field(encoder, "kind"));
        if (!strcmp(kind, "standard_scaler")) s->encoder = N4M_MULTIMODAL_STANDARD_SCALER;
        else if (!strcmp(kind, "tensor_pca")) {
            s->encoder = N4M_MULTIMODAL_TENSOR_PCA;
            s->n_components = integer(field(encoder, "n_components"));
            s->random_state = integer(field(encoder, "random_state")); s->whiten = boolean(field(encoder, "whiten"));
        } else if (!strcmp(kind, "column_transformer")) {
            s->encoder = N4M_MULTIMODAL_COLUMN_TRANSFORMER;
            s->ignore_unknown = !strcmp(text(field(encoder, "handle_unknown")), "ignore");
            s->numeric_column = integer(field(encoder, "numeric_columns"));
            s->categorical_column = integer(field(encoder, "categorical_columns"));
        } else Rf_error("unknown native multimodal encoder");
        if (s->encoder != N4M_MULTIMODAL_TENSOR_PCA) {
            s->with_mean = boolean(field(encoder, "with_mean")); s->with_std = boolean(field(encoder, "with_std"));
        }
    }
    return out;
}
static n4m_multimodal_recipe_v1_t configuration(SEXP recipe, SEXP schemas) {
    n4m_multimodal_recipe_v1_t out = source_configuration(recipe, schemas);
    SEXP params = field(field(recipe, "model"), "params");
    out.alpha = number(field(params, "alpha")); out.center_x = boolean(field(params, "center_x"));
    out.center_y = boolean(field(params, "center_y")); out.scale_x = boolean(field(params, "scale_x"));
    return out;
}
static n4m_multimodal_source_view_v1_t* views(SEXP blocks, SEXP schemas, int64_t* rows) {
    SEXP names = Rf_getAttrib(schemas, R_NamesSymbol);
    if (TYPEOF(schemas) != VECSXP || TYPEOF(names) != STRSXP || XLENGTH(schemas) < 1 ||
        XLENGTH(schemas) > 4 || XLENGTH(names) != XLENGTH(schemas))
        Rf_error("expected 1..4 ordered source schemas");
    const int count = (int)XLENGTH(schemas);
    n4m_multimodal_source_view_v1_t* out =
        (n4m_multimodal_source_view_v1_t*)R_alloc(count, sizeof(*out));
    memset(out, 0, count * sizeof(*out)); *rows = -1;
    for (int i = 0; i < count; ++i) {
        if (STRING_ELT(names, i) == NA_STRING) Rf_error("source names must not be missing");
        const char* name = Rf_translateCharUTF8(STRING_ELT(names, i));
        SEXP block = field(blocks, name), schema = field(schemas, name);
        SEXP dims = Rf_getAttrib(block, R_DimSymbol);
        if (TYPEOF(dims) != INTSXP || XLENGTH(dims) < 2 || XLENGTH(dims) > 8)
            Rf_error("raw sources must have sample-first dimensions");
        n4m_multimodal_source_view_v1_t* v = out + i;
        v->struct_size = sizeof(*v); v->name = name;
        v->representation_id = text(field(schema, "representation_id")); v->dtype = text(field(schema, "dtype"));
        v->identity_utf8 = text(field(schema, "identity")); v->identity_bytes = strlen((const char*)v->identity_utf8);
        v->rank = (int32_t)XLENGTH(dims);
        int64_t* shape = (int64_t*)R_alloc(v->rank, sizeof(*shape));
        int64_t* strides = (int64_t*)R_alloc(v->rank, sizeof(*strides));
        v->shape = shape; v->strides = strides;
        for (int j = 0; j < v->rank; ++j) { shape[j] = INTEGER(dims)[j]; strides[j] = j ? strides[j - 1] * shape[j - 1] : 1; }
        SEXP declared = field(schema, "input_shape");
        if (XLENGTH(declared) != v->rank - 1) Rf_error("raw source rank differs from its schema");
        for (int j = 1; j < v->rank; ++j) {
            double d = TYPEOF(declared) == VECSXP ? number(VECTOR_ELT(declared, j - 1)) :
                TYPEOF(declared) == REALSXP ? REAL(declared)[j - 1] :
                TYPEOF(declared) == INTSXP ? INTEGER(declared)[j - 1] : -1;
            if (d != shape[j]) Rf_error("raw source shape differs from its schema");
        }
        if (*rows >= 0 && *rows != shape[0]) Rf_error("source row counts differ"); *rows = shape[0];
        v->numeric_dtype = N4M_DTYPE_F64;
        if (strcmp(name, "metadata")) {
            if (TYPEOF(block) != REALSXP) Rf_error("raw numeric arrays must be doubles");
            if (!strcmp(v->dtype, "float32")) {
                const R_xlen_t count = XLENGTH(block);
                if (count > 16777216) Rf_error("raw float32 source exceeds native element bound");
                float* numeric = (float*)R_alloc(count ? count : 1, sizeof(*numeric));
                for (R_xlen_t j = 0; j < count; ++j) {
                    const double value = REAL(block)[j];
                    numeric[j] = (float)value;
                    if (!R_FINITE(value) || !R_FINITE((double)numeric[j]) || (double)numeric[j] != value)
                        Rf_error("declared float32 values require finite, lossless transport");
                }
                v->numeric_data = numeric; v->numeric_dtype = N4M_DTYPE_F32;
            } else v->numeric_data = REAL(block);
        } else {
            if (TYPEOF(block) != STRSXP || v->rank != 2 || shape[1] != 2)
                Rf_error("metadata must be a two-column character matrix");
            double* numeric = (double*)R_alloc(*rows, sizeof(*numeric)); v->numeric_data = numeric; strides[0] = 1;
            uint64_t* offsets = (uint64_t*)R_alloc(*rows + 1, sizeof(*offsets)); offsets[0] = 0;
            v->categorical_offsets = offsets;
            for (int64_t j = 0; j < *rows; ++j) {
                SEXP cell = STRING_ELT(block, j), cat = STRING_ELT(block, *rows + j);
                if (cell == NA_STRING || cat == NA_STRING) Rf_error("metadata cells must not be missing");
                const char* numeric_text = Rf_translateCharUTF8(cell); char* end = NULL;
                numeric[j] = strtod(numeric_text, &end);
                if (end == numeric_text || !R_FINITE(numeric[j])) Rf_error("invalid declared numeric metadata cell");
                while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
                if (*end) Rf_error("invalid trailing numeric metadata text");
                const size_t length = strlen(Rf_translateCharUTF8(cat));
                if (length > 1048576 || offsets[j] + length > 67108864) Rf_error("categorical UTF-8 input exceeds native bound");
                offsets[j + 1] = offsets[j] + length;
            }
            char* bytes = (char*)R_alloc(offsets[*rows] ? offsets[*rows] : 1, 1);
            for (int64_t j = 0; j < *rows; ++j) memcpy(bytes + offsets[j], Rf_translateCharUTF8(STRING_ELT(block, *rows + j)), offsets[j + 1] - offsets[j]);
            v->categorical_utf8 = bytes; v->utf8_bytes = offsets[*rows];
        }
    }
    return out;
}
static void finalizer(SEXP ptr) {
    n4m_multimodal_pipeline_destroy((n4m_multimodal_pipeline_t*)R_ExternalPtrAddr(ptr)); R_ClearExternalPtr(ptr);
}
static n4m_multimodal_pipeline_t* handle(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("n4m_multimodal_pipeline") ||
        R_ExternalPtrAddr(ptr) == NULL) Rf_error("native multimodal pipeline is closed or has the wrong pointer type");
    return (n4m_multimodal_pipeline_t*)R_ExternalPtrAddr(ptr);
}
static void check(n4m_status_t status, n4m_context_t* ctx) {
    if (status == N4M_OK) return;
    char message[2048]; snprintf(message, sizeof(message), "%s: %s", n4m_status_to_string(status), ctx ? n4m_context_last_error(ctx) : "");
    n4m_context_destroy(ctx); Rf_error("%s", message);
}
static n4m_context_t* context(void) {
    n4m_context_t* ctx = NULL; check(n4m_check_abi_compatibility(2, 16), NULL); check(n4m_context_create(&ctx), ctx); return ctx;
}
SEXP r_n4m_multimodal_create(SEXP recipe, SEXP schemas, SEXP state) {
    n4m_multimodal_recipe_v1_t cfg = configuration(recipe, schemas);
    if (!Rf_isNull(state) && TYPEOF(state) != RAWSXP) Rf_error("state must be raw N4MF bytes");
    if (!Rf_isNull(state) && XLENGTH(state) > 67108864) Rf_error("N4MF bytes exceed native limit");
    SEXP ptr = PROTECT(R_MakeExternalPtr(NULL, Rf_install("n4m_multimodal_pipeline"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, finalizer, TRUE);
    n4m_context_t* ctx = context(); n4m_multimodal_pipeline_t* pipeline = NULL;
    n4m_status_t status = Rf_isNull(state) ? n4m_multimodal_pipeline_create(ctx, &cfg, &pipeline) :
        n4m_multimodal_pipeline_import_from_buffer(ctx, &cfg, RAW(state), XLENGTH(state), &pipeline);
    check(status, ctx); n4m_context_destroy(ctx); R_SetExternalPtrAddr(ptr, pipeline); UNPROTECT(1); return ptr;
}
SEXP r_n4m_multimodal_fit(SEXP ptr, SEXP blocks, SEXP schemas, SEXP y) {
    n4m_multimodal_pipeline_t* p = handle(ptr); int64_t rows;
    n4m_multimodal_source_view_v1_t* input = views(blocks, schemas, &rows);
    if (TYPEOF(y) != REALSXP || XLENGTH(y) != rows) Rf_error("expected one double target per row");
    n4m_matrix_view_t target; check(n4m_matrix_view_init_rowmajor(&target, REAL(y), rows, 1, N4M_DTYPE_F64), NULL);
    n4m_context_t* ctx = context(); check(n4m_multimodal_pipeline_fit(ctx, p, (int32_t)XLENGTH(schemas), input, &target), ctx); n4m_context_destroy(ctx); return ptr;
}
SEXP r_n4m_multimodal_op(SEXP ptr, SEXP blocks, SEXP schemas, SEXP transform) {
    n4m_multimodal_pipeline_t* p = handle(ptr); int64_t rows, width = 1;
    n4m_multimodal_source_view_v1_t* input = views(blocks, schemas, &rows); int tr = boolean(transform);
    if (tr) check(n4m_multimodal_pipeline_transform_cols(p, &width), NULL);
    if (rows > INT_MAX || width > INT_MAX || width * rows > 16777216) Rf_error("output exceeds native shape bounds");
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)width)); n4m_matrix_view_t view;
    check(n4m_matrix_view_init_colmajor(&view, REAL(out), rows, width, N4M_DTYPE_F64), NULL);
    n4m_context_t* ctx = context(); check(tr ? n4m_multimodal_pipeline_transform(ctx, p, (int32_t)XLENGTH(schemas), input, &view) : n4m_multimodal_pipeline_predict(ctx, p, (int32_t)XLENGTH(schemas), input, &view), ctx);
    n4m_context_destroy(ctx); UNPROTECT(1); return out;
}
SEXP r_n4m_multimodal_export(SEXP ptr) {
    n4m_multimodal_pipeline_t* p = handle(ptr); n4m_context_t* ctx = context(); size_t size = 0;
    check(n4m_multimodal_pipeline_export_size(ctx, p, &size), ctx);
    /* R allocation happens before a new context so allocation failures cannot leak it. */
    n4m_context_destroy(ctx); SEXP out = PROTECT(Rf_allocVector(RAWSXP, size)); ctx = context();
    check(n4m_multimodal_pipeline_export_to_buffer(ctx, p, RAW(out), size, &size), ctx);
    n4m_context_destroy(ctx); UNPROTECT(1); return out;
}
SEXP r_n4m_multimodal_close(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("n4m_multimodal_pipeline"))
        Rf_error("expected a native multimodal pointer");
    finalizer(ptr); return R_NilValue;
}

/* Classifier ABI is additive; the existing Ridge pointer/tag is unchanged. */
static void classifier_finalizer(SEXP ptr) {
    n4m_multimodal_classifier_destroy((n4m_multimodal_classifier_t*)R_ExternalPtrAddr(ptr));
    R_ClearExternalPtr(ptr);
}
static n4m_multimodal_classifier_t* classifier_handle(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("n4m_multimodal_classifier") ||
        R_ExternalPtrAddr(ptr) == NULL)
        Rf_error("native multimodal classifier is closed or has the wrong pointer type");
    return (n4m_multimodal_classifier_t*)R_ExternalPtrAddr(ptr);
}
static n4m_context_t* classifier_context(void) {
    n4m_context_t* ctx = NULL;
    check(n4m_check_abi_compatibility(2, 17), NULL);
    check(n4m_context_create(&ctx), ctx);
    return ctx;
}
SEXP r_n4m_multimodal_classifier_create(SEXP recipe, SEXP schemas, SEXP state) {
    n4m_multimodal_recipe_v1_t sources = source_configuration(recipe, schemas);
    SEXP model = field(recipe, "model"), values = field(model, "params");
    const char* method_id = text(field(model, "method_id"));
    int64_t n_components = integer(field(values, "n_components"));
    int64_t max_iter = integer(field(values, "max_iter"));
    if (!Rf_isNull(state) && (TYPEOF(state) != RAWSXP || XLENGTH(state) == 0 || XLENGTH(state) > 67108864))
        Rf_error("expected bounded raw N4MC state bytes");
    SEXP ptr = PROTECT(R_MakeExternalPtr(NULL, Rf_install("n4m_multimodal_classifier"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, classifier_finalizer, TRUE);
    n4m_context_t* ctx = classifier_context(); n4m_params_t* params = NULL;
    n4m_multimodal_classifier_t* pipeline = NULL; int32_t method_index = -1;
    n4m_status_t status = n4m_method_find(method_id, &method_index);
    if (status == N4M_OK) status = n4m_params_create(ctx, method_index, &params);
    if (status == N4M_OK) status = n4m_params_set_int(params, "n_components", n_components);
    if (status == N4M_OK) status = n4m_params_set_int(params, "max_iter", max_iter);
    if (status == N4M_OK) status = n4m_params_validate(ctx, params);
    n4m_multimodal_classifier_recipe_v1_t cfg;
    memset(&cfg, 0, sizeof(cfg)); cfg.struct_size = sizeof(cfg);
    cfg.n_sources = sources.n_sources; cfg.sources = sources.sources;
    cfg.method_id = method_id; cfg.params = params;
    if (status == N4M_OK) status = Rf_isNull(state) ?
        n4m_multimodal_classifier_create(ctx, &cfg, &pipeline) :
        n4m_multimodal_classifier_import_from_buffer(ctx, &cfg, RAW(state), XLENGTH(state), &pipeline);
    n4m_params_destroy(params);
    check(status, ctx); n4m_context_destroy(ctx);
    R_SetExternalPtrAddr(ptr, pipeline); UNPROTECT(1); return ptr;
}
SEXP r_n4m_multimodal_classifier_fit(SEXP ptr, SEXP blocks, SEXP schemas, SEXP y) {
    n4m_multimodal_classifier_t* p = classifier_handle(ptr); int64_t rows;
    n4m_multimodal_source_view_v1_t* input = views(blocks, schemas, &rows);
    if (TYPEOF(y) != REALSXP || XLENGTH(y) != rows) Rf_error("expected one native class ID per row");
    int64_t* labels = (int64_t*)R_alloc(rows ? rows : 1, sizeof(*labels));
    for (int64_t i = 0; i < rows; ++i) {
        double value = REAL(y)[i];
        if (!R_FINITE(value) || floor(value) != value || fabs(value) > 9007199254740991.0)
            Rf_error("class IDs require finite lossless integer transport");
        labels[i] = (int64_t)value;
    }
    n4m_context_t* ctx = classifier_context();
    check(n4m_multimodal_classifier_fit(ctx, p, (int32_t)XLENGTH(schemas), input, labels, rows), ctx);
    n4m_context_destroy(ctx); return ptr;
}
SEXP r_n4m_multimodal_classifier_classes(SEXP ptr) {
    n4m_multimodal_classifier_t* p = classifier_handle(ptr); int64_t count = 0;
    check(n4m_multimodal_classifier_classes(p, NULL, 0, &count), NULL);
    if (count < 2 || count > 65536) Rf_error("class count exceeds native bounds");
    int64_t* ids = (int64_t*)R_alloc(count, sizeof(*ids));
    check(n4m_multimodal_classifier_classes(p, ids, count, &count), NULL);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, count));
    for (int64_t i = 0; i < count; ++i) {
        if (ids[i] < -9007199254740991LL || ids[i] > 9007199254740991LL)
            Rf_error("native class ID exceeds lossless R numeric range");
        REAL(out)[i] = (double)ids[i];
    }
    UNPROTECT(1); return out;
}
SEXP r_n4m_multimodal_classifier_op(SEXP ptr, SEXP blocks, SEXP schemas, SEXP operation) {
    n4m_multimodal_classifier_t* p = classifier_handle(ptr);
    const char* op = text(operation);
    const int labels = !strcmp(op, "class"), transform = !strcmp(op, "transform");
    const int probabilities = !strcmp(op, "prob"), decision = !strcmp(op, "decision");
    if (!labels && !transform && !probabilities && !decision) Rf_error("unsupported classifier operation");
    int64_t rows, width = 1;
    n4m_multimodal_source_view_v1_t* input = views(blocks, schemas, &rows);
    if (rows < 1 || rows > INT_MAX) Rf_error("output rows exceed native bounds");
    if (labels) {
        int64_t class_count = 0;
        check(n4m_multimodal_classifier_n_outputs(p, &class_count), NULL);
        if (class_count < 2 || rows > 16777216 / class_count)
            Rf_error("classifier prediction matrix exceeds native bounds");
        int64_t* ids = (int64_t*)R_alloc(rows, sizeof(*ids));
        SEXP out = PROTECT(Rf_allocVector(REALSXP, rows));
        n4m_context_t* ctx = classifier_context();
        check(n4m_multimodal_classifier_predict_labels(ctx, p, (int32_t)XLENGTH(schemas), input, ids, rows), ctx);
        n4m_context_destroy(ctx);
        for (int64_t i = 0; i < rows; ++i) {
            if (ids[i] < -9007199254740991LL || ids[i] > 9007199254740991LL)
                Rf_error("native prediction exceeds lossless R numeric range");
            REAL(out)[i] = (double)ids[i];
        }
        UNPROTECT(1); return out;
    }
    check(transform ? n4m_multimodal_classifier_transform_cols(p, &width) :
        n4m_multimodal_classifier_n_outputs(p, &width), NULL);
    if (width < 1 || width > INT_MAX || rows > 16777216 / width)
        Rf_error("output exceeds native shape bounds");
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)width)); n4m_matrix_view_t view;
    check(n4m_matrix_view_init_colmajor(&view, REAL(out), rows, width, N4M_DTYPE_F64), NULL);
    n4m_context_t* ctx = classifier_context();
    n4m_status_t status = transform ?
        n4m_multimodal_classifier_transform(ctx, p, (int32_t)XLENGTH(schemas), input, &view) : probabilities ?
        n4m_multimodal_classifier_predict_proba(ctx, p, (int32_t)XLENGTH(schemas), input, &view) :
        n4m_multimodal_classifier_decision_function(ctx, p, (int32_t)XLENGTH(schemas), input, &view);
    check(status, ctx); n4m_context_destroy(ctx); UNPROTECT(1); return out;
}
SEXP r_n4m_multimodal_classifier_export(SEXP ptr) {
    n4m_multimodal_classifier_t* p = classifier_handle(ptr);
    n4m_context_t* ctx = classifier_context(); size_t size = 0;
    check(n4m_multimodal_classifier_export_size(ctx, p, &size), ctx); n4m_context_destroy(ctx);
    if (size < 1 || size > 67108864) Rf_error("N4MC state exceeds native limit");
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, size)); ctx = classifier_context();
    check(n4m_multimodal_classifier_export_to_buffer(ctx, p, RAW(out), size, &size), ctx);
    n4m_context_destroy(ctx); UNPROTECT(1); return out;
}
SEXP r_n4m_multimodal_classifier_close(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("n4m_multimodal_classifier"))
        Rf_error("expected a native multimodal classifier pointer");
    classifier_finalizer(ptr); return R_NilValue;
}
