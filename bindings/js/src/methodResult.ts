// SPDX-License-Identifier: CECILL-2.1
//
// Owning wrapper around p4a_method_result_t — the universal output
// container used by every method shipped in Batches 1-12 of the C ABI.

import { checkStatus, getModule } from "./ffi.js";
import { Matrix } from "./types.js";

export class MethodResult {
    private _ptr: number;

    constructor(ptr: number) {
        this._ptr = ptr;
    }

    get handle(): number {
        return this._ptr;
    }

    /** Read a named double matrix by name. Returns a copy in JS-owned memory. */
    matrix(name: string): Matrix {
        const m = getModule();
        const nameBytes = m.lengthBytesUTF8(name);
        const namePtr = m._malloc(nameBytes + 1);
        const dataPtrPtr = m._malloc(4);
        const rowsPtr = m._malloc(8);
        const colsPtr = m._malloc(8);
        try {
            m.stringToUTF8(name, namePtr, nameBytes + 1);
            const status = m.ccall(
                "p4a_method_result_get_double_matrix", "number",
                ["number", "number", "number", "number", "number"],
                [this._ptr, namePtr, dataPtrPtr, rowsPtr, colsPtr]) as number;
            checkStatus(status);
            const dataPtr = m.getValue(dataPtrPtr, "i32");
            // i64 lo/hi pair — WASM_BIGINT=1 returns BigInt; use HEAP32 instead.
            const rows = m.getValue(rowsPtr, "i64") as unknown as number;
            const cols = m.getValue(colsPtr, "i64") as unknown as number;
            const n = Number(rows) * Number(cols);
            const data = new Float64Array(n);
            if (n > 0) {
                data.set(m.HEAPF64.subarray(dataPtr / 8, dataPtr / 8 + n));
            }
            return { data, rows: Number(rows), cols: Number(cols) };
        } finally {
            m._free(namePtr);
            m._free(dataPtrPtr);
            m._free(rowsPtr);
            m._free(colsPtr);
        }
    }

    /** Read a named int32 vector. */
    vectorInt(name: string): Int32Array {
        const m = getModule();
        const nameBytes = m.lengthBytesUTF8(name);
        const namePtr = m._malloc(nameBytes + 1);
        const dataPtrPtr = m._malloc(4);
        const sizePtr = m._malloc(4);
        try {
            m.stringToUTF8(name, namePtr, nameBytes + 1);
            const status = m.ccall(
                "p4a_method_result_get_int_vector", "number",
                ["number", "number", "number", "number"],
                [this._ptr, namePtr, dataPtrPtr, sizePtr]) as number;
            checkStatus(status);
            const dataPtr = m.getValue(dataPtrPtr, "i32");
            const size = m.getValue(sizePtr, "i32");
            const out = new Int32Array(size);
            if (size > 0) {
                out.set(m.HEAP32.subarray(dataPtr / 4, dataPtr / 4 + size));
            }
            return out;
        } finally {
            m._free(namePtr);
            m._free(dataPtrPtr);
            m._free(sizePtr);
        }
    }

    /** Read a named scalar (returns NaN if not present). */
    scalar(name: string): number {
        const m = getModule();
        const nameBytes = m.lengthBytesUTF8(name);
        const namePtr = m._malloc(nameBytes + 1);
        const outPtr = m._malloc(8);
        try {
            m.stringToUTF8(name, namePtr, nameBytes + 1);
            const status = m.ccall(
                "p4a_method_result_get_scalar", "number",
                ["number", "number", "number"],
                [this._ptr, namePtr, outPtr]) as number;
            if (status !== 0) return NaN;
            return m.getValue(outPtr, "double");
        } finally {
            m._free(namePtr);
            m._free(outPtr);
        }
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall(
            "p4a_method_result_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
    }
}
