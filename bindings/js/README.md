# pls4all — WebAssembly binding

Browser + Node.js bindings for libp4a, compiled via Emscripten. The
package ships:

- `p4a.wasm` — pls4all C ABI compiled to WebAssembly.
- `p4a.js` — Emscripten MODULARIZE/EXPORT_ES6 loader.
- `dist/` — TypeScript wrappers (`Context`, `Config`, `Model`,
  `MethodResult`) emitted from `src/`.

## Build

```bash
# 1. Activate the Emscripten SDK.
source /path/to/emsdk/emsdk_env.sh
# 2. Configure + build the WASM preset (zero deps beyond Emscripten).
cmake --preset emscripten
cmake --build --preset emscripten --target pls4all_wasm
# Artifacts land in build/emscripten/bindings/js/{p4a.js,p4a.wasm}.

# 3. Build the TypeScript wrapper for distribution:
cd bindings/js && npm run build
```

## Smoke test (Node)

```bash
node bindings/js/test/run_smoke.mjs
```

The smoke test loads the WASM module, prints the version / ABI triple
and round-trips a `Context` + `Config` lifecycle. It exercises the C
ABI symbol exports and Emscripten heap I/O paths.

## API surface

```typescript
import * as pls4all from "@pls4all/wasm";

await pls4all.loadModule();
console.log(pls4all.version());     // "0.82.0+abi.1.12.0"
console.log(pls4all.abiVersion());  // [1, 12, 0]

const ctx = pls4all.Context.create();
const cfg = pls4all.Config.create();
cfg.setAlgorithm(pls4all.Algorithm.PLS_REGRESSION);
cfg.setSolver(pls4all.Solver.SIMPLS);
cfg.setNComponents(3);
cfg.setCenterX(true);
cfg.setCenterY(true);
// ... see test/run_smoke.mjs for the full lifecycle.
cfg.destroy();
ctx.destroy();
```

## Build options

The CMake `emscripten` preset sets:

- `WASM_BIGINT=1` — int64 ABI params are exchanged as `BigInt`. Pass
  `BigInt(n)` when calling `_p4a_matrix_view_init_rowmajor` and similar.
- `MODULARIZE=1`, `EXPORT_ES6=1` — module factory is the default export.
- `ALLOW_MEMORY_GROWTH=1`, `INITIAL_MEMORY=64MB`, `MAXIMUM_MEMORY=2GB`.
- `EXPORTED_FUNCTIONS` — driven by `cpp/abi/expected_symbols_linux.txt`;
  every `p4a_*` symbol that ships in `libp4a.so` is exported here as
  `_p4a_*`. Plus `_malloc` / `_free` for matrix-view allocation.

## Status

- ✅ ABI surface exposed (159 `_p4a_*` symbols).
- ✅ Version + ABI introspection.
- ✅ Context / Config / Model / MethodResult lifecycle.
- ⚠️ **Known issue:** numerical PLS fit produces incorrect
  `x_mean` / `coefficients` under Emscripten 5.0.7 with the current
  build flags. The fit path is reachable but should not be relied on
  for accurate predictions until this is investigated. Suspected
  cause: int64 / matrix-view interaction in the Emscripten clang
  toolchain. The native Linux build (and Python / R bindings) is
  unaffected — this is specific to the WASM target.

## Layout

```
bindings/js/
├── CMakeLists.txt        # Emscripten target wired into the project preset
├── package.json
├── tsconfig.json
├── src/
│   ├── wasm_entry.c      # Pulls in the full ABI header for Emscripten
│   ├── ffi.ts            # Module loader + matrix-view helpers
│   ├── types.ts          # Mirrored C enums + error class
│   ├── context.ts        # Context wrapper
│   ├── config.ts         # Config wrapper
│   ├── model.ts          # Model fit / predict
│   ├── methodResult.ts   # Universal p4a_method_result_t wrapper
│   └── index.ts          # Public barrel
└── test/
    └── run_smoke.mjs     # Node smoke (version + lifecycle)
```
