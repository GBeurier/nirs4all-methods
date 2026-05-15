// SPDX-License-Identifier: CeCILL-2.1
//
// Thin ccall/cwrap wrappers around the Emscripten module.

import { Status, Pls4allError } from "./types.js";

/** Loose typing of the Emscripten module factory's runtime instance. */
export interface EmModule {
    HEAPU8: Uint8Array;
    HEAP32: Int32Array;
    HEAPF64: Float64Array;
    _malloc(size: number): number;
    _free(ptr: number): void;
    ccall<T>(name: string, returnType: string | null,
             argTypes: string[], args: unknown[]): T;
    cwrap<T extends Function>(name: string, returnType: string | null,
                                argTypes: string[]): T;
    UTF8ToString(ptr: number): string;
    stringToUTF8(s: string, ptr: number, max: number): void;
    lengthBytesUTF8(s: string): number;
    getValue(ptr: number, type: string): number;
    setValue(ptr: number, value: number, type: string): void;
}

let _module: EmModule | null = null;

/** Load and cache the Emscripten module. Call once at app startup. */
export async function loadModule(): Promise<EmModule> {
    if (_module !== null) return _module;
    const factory = (await import("../p4a.js")).default as
        (opts?: object) => Promise<EmModule>;
    _module = await factory({});
    return _module;
}

export function getModule(): EmModule {
    if (_module === null) {
        throw new Error(
            "pls4all WASM module not loaded — call await loadModule() first."
        );
    }
    return _module;
}

export function checkStatus(status: number, ctxPtr: number = 0): void {
    if (status === Status.OK) return;
    let msg = "";
    if (ctxPtr !== 0) {
        const m = getModule();
        const cstr = m.ccall("p4a_context_last_error", "number",
                              ["number"], [ctxPtr]) as number;
        if (cstr !== 0) msg = m.UTF8ToString(cstr);
    }
    if (msg === "") {
        const m = getModule();
        const cstr = m.ccall("p4a_status_to_string", "number",
                              ["number"], [status]) as number;
        if (cstr !== 0) msg = m.UTF8ToString(cstr);
    }
    throw new Pls4allError(status, msg);
}

// ---- p4a_matrix_view_t layout (mirrors cpp/include/pls4all/p4a.h) -------
//
// struct {
//     void*    data;       // 4 bytes (WASM is 32-bit; offset 0)
//     int64_t  rows;       // 8 bytes (offset 8, aligned to 8)
//     int64_t  cols;       // 8 bytes (offset 16)
//     int64_t  row_stride; // 8 bytes (offset 24)
//     int64_t  col_stride; // 8 bytes (offset 32)
//     int32_t  dtype;      // 4 bytes (offset 40)
//     int32_t  reserved0;  // 4 bytes (offset 44)
// }; total = 48 bytes on WASM32 with the 8-byte alignment of int64_t.

export const MATRIX_VIEW_SIZE = 48;

/** Allocate a matrix-view struct and copy `data` into the WASM heap.
 *  Returns the view pointer; caller must `freeMatrixView` it. */
export function makeMatrixView(data: Float64Array, rows: number,
                                  cols: number, dtype: number = 2): {
    viewPtr: number;
    dataPtr: number;
    free: () => void;
} {
    const m = getModule();
    const expected = rows * cols;
    if (data.length !== expected) {
        throw new Error(
            `matrix data length ${data.length} != rows*cols (${expected})`
        );
    }
    const bytes = data.length * 8;
    const dataPtr = m._malloc(bytes);
    if (bytes > 0) {
        m.HEAPF64.set(data, dataPtr / 8);
    }
    const viewPtr = m._malloc(MATRIX_VIEW_SIZE);
    const status = m.ccall(
        "p4a_matrix_view_init_rowmajor",
        "number",
        ["number", "number", "number", "number", "number"],
        [viewPtr, dataPtr, rows, cols, dtype],
    ) as number;
    if (status !== Status.OK) {
        m._free(dataPtr);
        m._free(viewPtr);
        checkStatus(status);
    }
    return {
        viewPtr,
        dataPtr,
        free: () => {
            m._free(viewPtr);
            m._free(dataPtr);
        },
    };
}
