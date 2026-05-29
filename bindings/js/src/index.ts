// SPDX-License-Identifier: CECILL-2.1
//
// Public TypeScript API for the @nirs4all/methods-wasm binding — a
// non-idiomatic function library over libn4m (raw typed arrays in/out). See
// INPUT_CONTRACT.md and examples/consume.mjs.
//
// Example:
//   import * as n4m from "@nirs4all/methods-wasm";
//   await n4m.loadModule();
//   const model = n4m.fitPls({ data: X, rows, cols }, { data: y, rows, cols: 1 }, 3);
//   const preds = n4m.predictPls(model, { data: X, rows, cols });

import { getModule } from "./ffi.js";

export { loadModule, getModule, makeMatrixView, readArrayView } from "./ffi.js";
export { Context } from "./context.js";
export { Config } from "./config.js";
export { Model, fitPls, predictPls, type PlsModel } from "./model.js";
export { MethodResult } from "./methodResult.js";
export {
    Status,
    Dtype,
    Algorithm,
    Solver,
    Deflation,
    Pls4allError,
    type Matrix,
} from "./types.js";

/** ABI / project version reported by the loaded WASM module. */
export function version(): string {
    const m = getModule();
    const ptr = m.ccall("n4m_get_version_string", "number",
                         [], []) as number;
    return ptr === 0 ? "" : m.UTF8ToString(ptr);
}

/** ABI MAJOR.MINOR.PATCH triple. */
export function abiVersion(): readonly [number, number, number] {
    const m = getModule();
    return [
        m.ccall("n4m_get_abi_version_major", "number", [], []) as number,
        m.ccall("n4m_get_abi_version_minor", "number", [], []) as number,
        m.ccall("n4m_get_abi_version_patch", "number", [], []) as number,
    ];
}

