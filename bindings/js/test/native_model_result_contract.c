/* SPDX-License-Identifier: CECILL-2.1
 * Deterministic malformed-MethodResult regression for the WASM model shim.
 * Build after cmake --preset dev-release && cmake --build --preset dev-release:
 * gcc -std=c11 -Wall -Wextra -Werror -ffunction-sections -fdata-sections \
 *   -Icpp/include -Ibuild/dev-release/generated \
 *   bindings/js/test/native_model_result_contract.c \
 *   -Wl,--gc-sections -Lbuild/dev-release/cpp/src \
 *   -Wl,-rpath,build/dev-release/cpp/src -ln4m -lm \
 *   -o /tmp/n4m-model-result-contract
 * /tmp/n4m-model-result-contract
 *
 * Only one fit and its MethodResult accessors are substituted. The real
 * libn4m still supplies context/config management and every other symbol.
 */
#include <stdio.h>
#include <stdint.h>

#define n4m_estimators_ridge_pls_fit stub_ridge_pls_fit
#define n4m_method_result_get_double_matrix stub_result_matrix
#define n4m_method_result_destroy stub_result_destroy
#include "../src/wasm_entry.c"
#undef n4m_estimators_ridge_pls_fit
#undef n4m_method_result_get_double_matrix
#undef n4m_method_result_destroy

static int scenario = 0;
static int destroyed = 0;
static const double coefficients[] = {1.0, 2.0};
static const double x_mean[] = {0.1, 0.2};
static const double y_mean[] = {0.3};

n4m_status_t stub_ridge_pls_fit(n4m_context_t* ctx, const n4m_config_t* cfg,
                                const n4m_matrix_view_t* X,
                                const n4m_matrix_view_t* Y,
                                double ridge_lambda,
                                n4m_method_result_t** out_result) {
    (void)ctx; (void)cfg; (void)X; (void)Y; (void)ridge_lambda;
    *out_result = (n4m_method_result_t*)(uintptr_t)1;
    return N4M_OK;
}

n4m_status_t stub_result_matrix(const n4m_method_result_t* result,
                                const char* name, const double** out_data,
                                int64_t* out_rows, int64_t* out_cols) {
    (void)result;
    if (strcmp(name, "coefficients") == 0) {
        if (scenario == 1) return N4M_ERR_INVALID_ARGUMENT;
        *out_data = coefficients;
        *out_rows = scenario == 2 ? 1 : 2;
        *out_cols = scenario == 2 ? 2 : 1;
        return N4M_OK;
    }
    if (strcmp(name, "x_mean") == 0) {
        if (scenario == 3) return N4M_ERR_INVALID_ARGUMENT;
        *out_data = x_mean;
        *out_rows = scenario == 4 ? 2 : 1;
        *out_cols = scenario == 4 ? 1 : 2;
        return N4M_OK;
    }
    if (strcmp(name, "y_mean") == 0) {
        if (scenario == 5) return N4M_ERR_INVALID_ARGUMENT;
        *out_data = y_mean;
        *out_rows = 1;
        *out_cols = scenario == 6 ? 2 : 1;
        return N4M_OK;
    }
    return N4M_ERR_INVALID_ARGUMENT; /* optional intercept is absent */
}

void stub_result_destroy(n4m_method_result_t* result) {
    (void)result;
    ++destroyed;
}

int main(void) {
    const double X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double Y[] = {0.2, 0.4, 0.6};
    for (scenario = 0; scenario <= 6; ++scenario) {
        double coef[2] = {91.0, 91.0};
        double xm[2] = {92.0, 92.0};
        double ym[1] = {93.0};
        double intercept[1] = {94.0};
        int has_intercept = -1;
        int prior_destroyed = destroyed;
        int status = n4m_wasm_model_fit("RidgePLS", NULL, 0, X, Y, 3, 2, 1, 1,
                                        coef, xm, ym, intercept, &has_intercept,
                                        NULL);
        if (destroyed != prior_destroyed + 1 || has_intercept != 0) {
            fprintf(stderr, "scenario %d: lifecycle or intercept flag\n", scenario);
            return 1;
        }
        if (scenario == 0) {
            if (status != N4M_OK || coef[0] != 1.0 || coef[1] != 2.0 ||
                xm[0] != 0.1 || xm[1] != 0.2 || ym[0] != 0.3 ||
                intercept[0] != 0.0) {
                fprintf(stderr, "complete result rejected or copied incorrectly\n");
                return 1;
            }
        } else if (status != N4M_ERR_INTERNAL || coef[0] != 91.0 ||
                   coef[1] != 91.0 || xm[0] != 92.0 || xm[1] != 92.0 ||
                   ym[0] != 93.0 || intercept[0] != 94.0) {
            fprintf(stderr, "scenario %d: accepted incomplete result or wrote partial model\n",
                    scenario);
            return 1;
        }
    }
    puts("missing and wrong-shaped fitted matrices fail closed");
    return 0;
}
