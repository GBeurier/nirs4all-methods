// SPDX-License-Identifier: CECILL-2.1
//
// Stage the Emscripten artefacts next to the compiled TypeScript wrapper so
// `npm pack` and `jsr publish` contain a self-contained WASM package.

import { copyFile, mkdir, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "..", "..");
const wasmDir = process.env.PLS4ALL_WASM_DIR ??
    resolve(repoRoot, "build", "emscripten", "bindings", "js");
const distDir = resolve(pkgRoot, "dist");

async function requireFile(path) {
    try {
        const info = await stat(path);
        if (!info.isFile()) throw new Error(`${path} is not a file`);
    } catch (err) {
        throw new Error(
            `Missing WASM artefact: ${path}\n` +
            `Build it first with:\n` +
            `  npm run build:wasm\n` +
            `or point PLS4ALL_WASM_DIR at a directory containing p4a.js ` +
            `and p4a.wasm.\n` +
            `Original error: ${err instanceof Error ? err.message : err}`);
    }
}

const jsSrc = resolve(wasmDir, "p4a.js");
const wasmSrc = resolve(wasmDir, "p4a.wasm");

await requireFile(jsSrc);
await requireFile(wasmSrc);
await mkdir(distDir, { recursive: true });
await copyFile(jsSrc, resolve(distDir, "p4a.js"));
await copyFile(wasmSrc, resolve(distDir, "p4a.wasm"));

console.log(`staged WASM artefacts from ${wasmDir}`);
