/* SPDX-License-Identifier: CECILL-2.1 */

#include "mex.h"
#include "pls4all/p4a.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    (void)nrhs;
    (void)prhs;
    const char *v = p4a_get_version_string();
    plhs[0] = mxCreateString(v == NULL ? "" : v);
    (void)nlhs;
}
