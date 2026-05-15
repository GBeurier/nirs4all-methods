// SPDX-License-Identifier: CeCILL-2.1
//
// Node smoke test for the WASM build.
//
// What this smoke verifies:
// - The Emscripten module loads in Node.
// - Version + ABI introspection strings come back correctly.
// - Context / Config lifecycle is callable, including setters and
//   getters round-tripping through the C ABI.
//
// Numerical fit is currently a known-issue when invoked from outside
// the WASM module — see README.md "Status" section. A pure-C
// translation unit linking against the same `libp4a_static.a` runs the
// fit correctly (see `cmake --build --preset emscripten --target
// pls4all_wasm` and the C debug in `test/wasm_fit_debug.c`), but the
// equivalent calls invoked from a JS host return inconsistent
// `x_mean` / coefficients. Likely a packing / marshalling subtlety in
// the `WASM_BIGINT=1` ABI for `p4a_model_fit`; under active
// investigation.

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = resolve(here, "..", "..", "..", "build", "emscripten",
                        "bindings", "js");
const factory = (await import(resolve(wasmDir, "p4a.js"))).default;
const M = await factory({
    locateFile: (path) => resolve(wasmDir, path),
});

const versionPtr = M._p4a_get_version_string();
const version = M.UTF8ToString(versionPtr);
console.log("pls4all version:", version);
if (!version.startsWith("0.")) {
    throw new Error(`unexpected version string: ${version}`);
}

const abi = [
    M._p4a_get_abi_version_major(),
    M._p4a_get_abi_version_minor(),
    M._p4a_get_abi_version_patch(),
];
console.log("ABI:", abi.join("."));

const ctxOut = M._malloc(4);
let s = M._p4a_context_create(ctxOut);
if (s !== 0) throw new Error(`context_create: ${s}`);
const ctx = M.getValue(ctxOut, "i32");
M._free(ctxOut);

const cfgOut = M._malloc(4);
s = M._p4a_config_create(cfgOut);
if (s !== 0) throw new Error(`config_create: ${s}`);
const cfg = M.getValue(cfgOut, "i32");
M._free(cfgOut);

s = M._p4a_config_set_algorithm(cfg, 0);
if (s !== 0) throw new Error(`set_algorithm: ${s}`);
s = M._p4a_config_set_solver(cfg, 1);
if (s !== 0) throw new Error(`set_solver: ${s}`);
s = M._p4a_config_set_n_components(cfg, 3);
if (s !== 0) throw new Error(`set_n_components: ${s}`);

const buf = M._malloc(4);
M._p4a_config_get_n_components(cfg, buf);
const ncomp = M.getValue(buf, "i32");
if (ncomp !== 3) throw new Error(`n_components readback ${ncomp} != 3`);
M._free(buf);

M._p4a_config_destroy(cfg);
M._p4a_context_destroy(ctx);

console.log("WASM smoke OK");
