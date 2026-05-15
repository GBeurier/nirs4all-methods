// SPDX-License-Identifier: CeCILL-2.1
//
// Public TypeScript API for the pls4all WebAssembly binding.
//
// Example:
//   import * as pls4all from "@pls4all/wasm";
//   await pls4all.loadModule();
//   const ctx = pls4all.Context.create();
//   const cfg = pls4all.Config.create();
//   cfg.setAlgorithm(pls4all.Algorithm.PLS_REGRESSION);
//   cfg.setSolver(pls4all.Solver.SIMPLS);
//   cfg.setNComponents(3);
//   cfg.setCenterX(true);
//   cfg.setCenterY(true);
//   const model = pls4all.Model.fit(ctx, cfg, X, Y);
//   const preds = model.predict(ctx, X_new);
//   model.destroy(); cfg.destroy(); ctx.destroy();

import { getModule } from "./ffi.js";

export { loadModule, getModule } from "./ffi.js";
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
    const ptr = m.ccall("p4a_get_version_string", "number",
                         [], []) as number;
    return ptr === 0 ? "" : m.UTF8ToString(ptr);
}

/** ABI MAJOR.MINOR.PATCH triple. */
export function abiVersion(): readonly [number, number, number] {
    const m = getModule();
    return [
        m.ccall("p4a_get_abi_version_major", "number", [], []) as number,
        m.ccall("p4a_get_abi_version_minor", "number", [], []) as number,
        m.ccall("p4a_get_abi_version_patch", "number", [], []) as number,
    ];
}

