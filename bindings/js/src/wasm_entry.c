/* SPDX-License-Identifier: CeCILL-2.1
 *
 * Emscripten entry point. The Emscripten linker needs at least one C
 * translation unit in the executable target — this file pulls in the
 * full public ABI header so all p4a_* symbols stay reachable when
 * Emscripten dead-strips. The EXPORTED_FUNCTIONS link flag (driven by
 * cpp/abi/expected_symbols_linux.txt at configure time) keeps them
 * exposed to JavaScript.
 */

#include "pls4all/p4a.h"

#include <stddef.h>

/* Touch every accessor on the matrix-view struct so Emscripten preserves
 * the in-line `p4a_matrix_view_init_*` helpers from the header. */
__attribute__((used)) static int p4a_wasm_keep_alive(void) {
    p4a_matrix_view_t v = {0};
    p4a_matrix_view_init_rowmajor(&v, NULL, 0, 0, P4A_DTYPE_F64);
    p4a_matrix_view_init_colmajor(&v, NULL, 0, 0, P4A_DTYPE_F64);
    return (int)(v.dtype + v.rows + v.cols);
}
