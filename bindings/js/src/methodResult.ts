// SPDX-License-Identifier: CECILL-2.1
//
// Owning wrapper around p4a_method_result_t — the universal output
// container used by every method shipped in Batches 1-12 of the C ABI.

import {
    checkStatus,
    getModule,
    hasExportedFunction,
    readI64,
} from "./ffi.js";
import { Matrix, Pls4allError, Status } from "./types.js";

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
            const dataPtr = Number(m.getValue(dataPtrPtr, "i32"));
            const rows = readI64(rowsPtr);
            const cols = readI64(colsPtr);
            const n = rows * cols;
            const data = new Float64Array(n);
            if (n > 0) {
                data.set(m.HEAPF64.subarray(dataPtr / 8, dataPtr / 8 + n));
            }
            return { data, rows, cols };
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
            const dataPtr = Number(m.getValue(dataPtrPtr, "i32"));
            const size = Number(m.getValue(sizePtr, "i32"));
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

    /** Read a named int64 vector without losing precision. */
    vectorInt64(name: string): BigInt64Array {
        if (!hasExportedFunction("p4a_method_result_get_int64_vector")) {
            throw new Pls4allError(
                Status.ERR_NOT_IMPLEMENTED,
                "p4a_method_result_get_int64_vector is not exported " +
                "by the loaded WASM module",
            );
        }
        const m = getModule();
        const nameBytes = m.lengthBytesUTF8(name);
        const namePtr = m._malloc(nameBytes + 1);
        const dataPtrPtr = m._malloc(4);
        const sizePtr = m._malloc(8);
        try {
            m.stringToUTF8(name, namePtr, nameBytes + 1);
            const status = m.ccall(
                "p4a_method_result_get_int64_vector", "number",
                ["number", "number", "number", "number"],
                [this._ptr, namePtr, dataPtrPtr, sizePtr]) as number;
            checkStatus(status);
            const dataPtr = Number(m.getValue(dataPtrPtr, "i32"));
            const size = readI64(sizePtr);
            if (size === 0) return new BigInt64Array(0);
            return new BigInt64Array(m.HEAPU8.buffer, dataPtr, size).slice();
        } finally {
            m._free(namePtr);
            m._free(dataPtrPtr);
            m._free(sizePtr);
        }
    }

    /** Read a named int64 vector and require every value to fit in JS number. */
    vectorInt64AsNumber(name: string): number[] {
        return Array.from(this.vectorInt64(name), (value) => {
            const n = Number(value);
            if (!Number.isSafeInteger(n)) {
                throw new Pls4allError(
                    Status.ERR_INVALID_ARGUMENT,
                    `int64 value ${value} cannot be represented safely as number`,
                );
            }
            return n;
        });
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
            return Number(m.getValue(outPtr, "double"));
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
