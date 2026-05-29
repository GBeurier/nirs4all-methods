// SPDX-License-Identifier: CECILL-2.1
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
    // n4m.js (the Emscripten loader) is staged into dist/ next to this module,
    // so the package is self-contained (files: ["dist/"]). See package.json
    // `stage:wasm`. The import path must stay relative to dist/ so the shipped
    // package resolves the loader it bundles.
    const factory = (await import("./n4m.js")).default as
        (opts?: object) => Promise<EmModule>;
    _module = await factory({});
    return _module;
}

export function getModule(): EmModule {
    if (_module === null) {
        throw new Error(
            "@nirs4all/methods-wasm not loaded — call await loadModule() first."
        );
    }
    return _module;
}

export function checkStatus(status: number, ctxPtr: number = 0): void {
    if (status === Status.OK) return;
    let msg = "";
    if (ctxPtr !== 0) {
        const m = getModule();
        const cstr = m.ccall("n4m_context_last_error", "number",
                              ["number"], [ctxPtr]) as number;
        if (cstr !== 0) msg = m.UTF8ToString(cstr);
    }
    if (msg === "") {
        const m = getModule();
        const cstr = m.ccall("n4m_status_to_string", "number",
                              ["number"], [status]) as number;
        if (cstr !== 0) msg = m.UTF8ToString(cstr);
    }
    throw new Pls4allError(status, msg);
}

// ---- n4m_matrix_view_t layout (mirrors cpp/include/pls4all/p4a.h) -------
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
 *  Returns the view pointer; caller must `free()` it.
 *
 *  NOTE: n4m_matrix_view_init_rowmajor takes `int64_t rows, int64_t cols`.
 *  The WASM module is built with `-s WASM_BIGINT=1`, so ccall marshals i64
 *  args as BigInt — passing a plain JS number for these slots silently
 *  corrupts the struct fields (rows/cols/strides become garbage, producing
 *  ~1e32 numerics downstream) and on emsdk >= 5.0.7 throws
 *  "TypeError: Cannot convert N to a BigInt". We therefore declare the two
 *  dimension args as 'i64' and pass BigInt(...). The data/out/dtype slots
 *  stay 32-bit. dtype defaults to N4M_DTYPE_F64 (= 1; see types.ts Dtype). */
export function makeMatrixView(data: Float64Array, rows: number,
                                  cols: number,
                                  dtype: number = 1 /* N4M_DTYPE_F64 */): {
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
    const bytes = Math.max(1, data.length) * 8;
    const dataPtr = m._malloc(bytes);
    if (data.length > 0) {
        m.HEAPF64.set(data, dataPtr / 8);
    }
    const viewPtr = m._malloc(MATRIX_VIEW_SIZE);
    const status = m.ccall(
        "n4m_matrix_view_init_rowmajor",
        "number",
        ["number", "number", "i64", "i64", "number"],
        [viewPtr, dataPtr, BigInt(rows), BigInt(cols), dtype],
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

/** Read a core-owned `n4m_array_t*` (e.g. from n4m_model_get_array) into a
 *  JS-owned Float64Array. Does NOT free the array — the caller must call
 *  `n4m_array_free` afterwards. i64 view fields (rows @8, cols @16) are read
 *  as BigInt under WASM_BIGINT and narrowed with Number(). */
export function readArrayView(arrPtr: number): {
    data: Float64Array; rows: number; cols: number;
} {
    const m = getModule();
    const vp = m._malloc(MATRIX_VIEW_SIZE);
    try {
        const st = m.ccall("n4m_array_view", "number",
                           ["number", "number"], [arrPtr, vp]) as number;
        checkStatus(st);
        const dataPtr = m.getValue(vp, "i32");
        const rows = Number(m.getValue(vp + 8, "i64"));
        const cols = Number(m.getValue(vp + 16, "i64"));
        const n = rows * cols;
        const data = new Float64Array(n);
        if (n > 0) {
            data.set(m.HEAPF64.subarray(dataPtr / 8, dataPtr / 8 + n));
        }
        return { data, rows, cols };
    } finally {
        m._free(vp);
    }
}
