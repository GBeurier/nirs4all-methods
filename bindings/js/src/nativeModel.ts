// SPDX-License-Identifier: CECILL-2.1
//
// Owning wrapper for a native p4a_model_t* when method shims need model input.

import {
    checkStatus,
    getModule,
    hasExportedFunction,
} from "./ffi.js";
import { Matrix, Pls4allError, Status } from "./types.js";

function allocF64(values: Float64Array): { ptr: number; free: () => void } {
    const m = getModule();
    const ptr = m._malloc(values.length * 8);
    if (values.length > 0) m.HEAPF64.set(values, ptr / 8);
    return { ptr, free: () => m._free(ptr) };
}

export class NativeModel {
    private _ptr: number;

    private constructor(ptr: number) {
        this._ptr = ptr;
    }

    get handle(): number {
        return this._ptr;
    }

    static fitPls(
        X: Matrix,
        Y: Matrix,
        nComponents: number,
        storeScores: boolean = false,
    ): NativeModel {
        if (!hasExportedFunction("p4a_wasm_pls_model_fit")) {
            throw new Pls4allError(
                Status.ERR_NOT_IMPLEMENTED,
                "p4a_wasm_pls_model_fit is not exported by the loaded " +
                "WASM module",
            );
        }
        if (X.rows !== Y.rows) {
            throw new Error(`X.rows (${X.rows}) must equal Y.rows (${Y.rows})`);
        }
        const m = getModule();
        const xBuf = allocF64(X.data);
        const yBuf = allocF64(Y.data);
        const out = m._malloc(4);
        try {
            const status = m.ccall(
                "p4a_wasm_pls_model_fit",
                "number",
                ["number", "number", "number", "number", "number",
                 "number", "number", "number"],
                [
                    xBuf.ptr,
                    yBuf.ptr,
                    X.rows,
                    X.cols,
                    Y.cols,
                    nComponents,
                    storeScores ? 1 : 0,
                    out,
                ],
            ) as number;
            checkStatus(status);
            const ptr = Number(m.getValue(out, "i32"));
            if (ptr === 0) {
                throw new Pls4allError(
                    Status.ERR_INTERNAL,
                    "p4a_wasm_pls_model_fit returned a NULL handle",
                );
            }
            return new NativeModel(ptr);
        } finally {
            m._free(out);
            yBuf.free();
            xBuf.free();
        }
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall("p4a_model_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
    }
}
