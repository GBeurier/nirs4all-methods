// SPDX-License-Identifier: CECILL-2.1

import { checkStatus, getModule, hasExportedFunction } from "./ffi.js";
import { Algorithm, Deflation, Matrix, Pls4allError, Solver, Status } from "./types.js";

function allocF64(values: Float64Array): { ptr: number; free: () => void } {
    const m = getModule();
    const ptr = m._malloc(values.length * 8);
    if (values.length > 0) m.HEAPF64.set(values, ptr / 8);
    return { ptr, free: () => m._free(ptr) };
}

export interface CoreFitOptions {
    algorithm?: Algorithm;
    solver?: Solver;
    deflation?: Deflation;
    nComponents: number;
    centerX?: boolean;
    scaleX?: boolean;
    centerY?: boolean;
    scaleY?: boolean;
}

export function fitPredictCore(X: Matrix, Y: Matrix,
                               options: CoreFitOptions): Matrix {
    if (!hasExportedFunction("p4a_wasm_model_fit_predict")) {
        throw new Pls4allError(
            Status.ERR_NOT_IMPLEMENTED,
            "p4a_wasm_model_fit_predict is not exported by the loaded " +
            "WASM module",
        );
    }
    if (X.rows !== Y.rows) {
        throw new Error(`X.rows (${X.rows}) must equal Y.rows (${Y.rows})`);
    }
    const m = getModule();
    const xBuf = allocF64(X.data);
    const yBuf = allocF64(Y.data);
    const out = m._malloc(X.rows * Y.cols * 8);
    try {
        const status = m.ccall(
            "p4a_wasm_model_fit_predict",
            "number",
            ["number", "number", "number", "number", "number",
             "number", "number", "number", "number",
             "number", "number", "number", "number", "number"],
            [
                xBuf.ptr,
                yBuf.ptr,
                X.rows,
                X.cols,
                Y.cols,
                options.algorithm ?? Algorithm.PLS_REGRESSION,
                options.solver ?? Solver.SIMPLS,
                options.deflation ?? Deflation.REGRESSION,
                options.nComponents,
                (options.centerX ?? true) ? 1 : 0,
                (options.scaleX ?? false) ? 1 : 0,
                (options.centerY ?? true) ? 1 : 0,
                (options.scaleY ?? false) ? 1 : 0,
                out,
            ],
        ) as number;
        checkStatus(status);
        return {
            rows: X.rows,
            cols: Y.cols,
            data: new Float64Array(m.HEAPU8.buffer, out, X.rows * Y.cols)
                .slice(),
        };
    } finally {
        m._free(out);
        yBuf.free();
        xBuf.free();
    }
}
