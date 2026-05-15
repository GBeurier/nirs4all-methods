/* SPDX-License-Identifier: CeCILL-2.1 */
/*
 * Register the .Call entry points exposed by r_gateway.c.
 */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP r_pls4all_version(void);
SEXP r_pls4all_abi_version(void);
SEXP r_pls4all_fit(SEXP X, SEXP Y, SEXP algo, SEXP n_components);
SEXP r_pls4all_predict(SEXP model_ptr, SEXP X);

static const R_CallMethodDef callMethods[] = {
    {"r_pls4all_version",     (DL_FUNC)&r_pls4all_version,     0},
    {"r_pls4all_abi_version", (DL_FUNC)&r_pls4all_abi_version, 0},
    {"r_pls4all_fit",         (DL_FUNC)&r_pls4all_fit,         4},
    {"r_pls4all_predict",     (DL_FUNC)&r_pls4all_predict,     2},
    {NULL, NULL, 0},
};

void R_init_pls4all(DllInfo* info) {
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
