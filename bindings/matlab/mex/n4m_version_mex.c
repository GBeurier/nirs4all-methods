/* SPDX-License-Identifier: CECILL-2.1 */

#include "mex.h"
#include "n4m/n4m.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    (void)nrhs;
    (void)prhs;
    const char *v = n4m_get_version_string();
    plhs[0] = mxCreateString(v == NULL ? "" : v);
    (void)nlhs;
}
