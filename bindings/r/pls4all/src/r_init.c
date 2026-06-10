/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Register the .Call entry points exposed by r_gateway.c and r_methods.c.
 */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* r_gateway.c */
SEXP r_n4m_version(void);
SEXP r_n4m_abi_version(void);
SEXP r_n4m_fit(SEXP X, SEXP Y, SEXP algo, SEXP n_components, SEXP store_scores,
                    SEXP center_x, SEXP scale_x, SEXP center_y, SEXP scale_y);
SEXP r_n4m_predict(SEXP model_ptr, SEXP X);
SEXP r_n4m_model_get_array(SEXP model_ptr, SEXP which);

/* r_methods.c — MethodResult fits */
SEXP r_n4m_sparse_simpls_fit(SEXP X, SEXP Y, SEXP n_components, SEXP sparsity_lambda);
SEXP r_n4m_cppls_fit(SEXP X, SEXP Y, SEXP n_components, SEXP gamma);
SEXP r_n4m_weighted_pls_fit(SEXP X, SEXP Y, SEXP n_components, SEXP sample_weights);
SEXP r_n4m_mb_pls_fit(SEXP X, SEXP Y, SEXP n_components, SEXP block_sizes);
SEXP r_n4m_pls_glm_fit(SEXP X, SEXP Y, SEXP n_components, SEXP poisson);
SEXP r_n4m_mir_pls_fit(SEXP X, SEXP Y, SEXP n_components);

/* r_methods.c — selectors */
SEXP r_n4m_variable_select_rank(SEXP model_ptr, SEXP X, SEXP method, SEXP top_k);
SEXP r_n4m_spa_select(SEXP X, SEXP Y, SEXP n_components, SEXP top_k);
SEXP r_n4m_cars_select(SEXP X, SEXP Y, SEXP n_components, SEXP n_iterations, SEXP min_features);

/* r_methods.c — diagnostics */
SEXP r_n4m_pls_diagnostics_compute(SEXP model_ptr, SEXP X);
SEXP r_n4m_approximate_press_compute(SEXP X, SEXP Y, SEXP max_components);
SEXP r_n4m_pls_monitoring_run(SEXP model_ptr, SEXP X_ref, SEXP X_mon, SEXP alpha);

/* r_dispatch.c — unified dispatcher (covers all method-result fits +
 * selectors + diagnostics by algo name + params list) */
SEXP r_n4m_dispatch_fit(SEXP algo, SEXP X, SEXP Y, SEXP n_components,
                         SEXP params,
                         SEXP center_x, SEXP scale_x,
                         SEXP center_y, SEXP scale_y);

static const R_CallMethodDef callMethods[] = {
    /* core */
    {"r_n4m_version",     (DL_FUNC)&r_n4m_version,     0},
    {"r_n4m_abi_version", (DL_FUNC)&r_n4m_abi_version, 0},
    {"r_n4m_fit",         (DL_FUNC)&r_n4m_fit,         9},
    {"r_n4m_predict",     (DL_FUNC)&r_n4m_predict,     2},
    {"r_n4m_model_get_array", (DL_FUNC)&r_n4m_model_get_array, 2},

    /* MethodResult fits */
    {"r_n4m_sparse_simpls_fit", (DL_FUNC)&r_n4m_sparse_simpls_fit, 4},
    {"r_n4m_cppls_fit",         (DL_FUNC)&r_n4m_cppls_fit,         4},
    {"r_n4m_weighted_pls_fit",  (DL_FUNC)&r_n4m_weighted_pls_fit,  4},
    {"r_n4m_mb_pls_fit",        (DL_FUNC)&r_n4m_mb_pls_fit,        4},
    {"r_n4m_pls_glm_fit",       (DL_FUNC)&r_n4m_pls_glm_fit,       4},
    {"r_n4m_mir_pls_fit",       (DL_FUNC)&r_n4m_mir_pls_fit,       3},

    /* selectors */
    {"r_n4m_variable_select_rank", (DL_FUNC)&r_n4m_variable_select_rank, 4},
    {"r_n4m_spa_select",           (DL_FUNC)&r_n4m_spa_select,           4},
    {"r_n4m_cars_select",          (DL_FUNC)&r_n4m_cars_select,          5},

    /* diagnostics */
    {"r_n4m_pls_diagnostics_compute",   (DL_FUNC)&r_n4m_pls_diagnostics_compute,   2},
    {"r_n4m_approximate_press_compute", (DL_FUNC)&r_n4m_approximate_press_compute, 3},
    {"r_n4m_pls_monitoring_run",        (DL_FUNC)&r_n4m_pls_monitoring_run,        4},

    /* unified dispatcher */
    {"r_n4m_dispatch_fit", (DL_FUNC)&r_n4m_dispatch_fit, 9},

    {NULL, NULL, 0},
};

void R_init_pls4all(DllInfo* info) {
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
