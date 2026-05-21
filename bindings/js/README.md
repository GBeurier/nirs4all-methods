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

# 3. Build the TypeScript wrapper and stage p4a.js / p4a.wasm into dist:
cd bindings/js && npm run build
```

If the WASM artifacts were built elsewhere, point `PLS4ALL_WASM_DIR` at
the directory containing `p4a.js` and `p4a.wasm` before running
`npm run build`.

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
console.log(pls4all.version());     // "0.97.3+abi.1.16.0"
console.log(pls4all.abiVersion());  // [1, 16, 0]

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

The sklearn-style wrapper is available from both the top-level barrel and
the `./sklearn` subpath:

```typescript
import { loadModule } from "@pls4all/wasm";
import { PLSRegression } from "@pls4all/wasm/sklearn";

await loadModule();
const reg = new PLSRegression({ nComponents: 3 });
reg.fit(X, y);
const yhat = reg.predict(Xnew);
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

- ✅ ABI surface exposed through the generated `_p4a_*` export list.
- ✅ Version + ABI introspection.
- ✅ Context / Config / Model / MethodResult / AOM lifecycle.
- ✅ Full `p4a_method_result_t` method surface in `src/methods.ts`,
  including selectors, diagnostics, multi-block methods, AOM/POP, and
  int64 vector reads.
- ✅ **PLS fit / predict parity with native Python pls4all at
  numerical floor (1e-16 RMSE-rel)** — see
  `test/run_smoke.mjs` and `test/parity_fixture.json`.
- ✅ Method binding parity gate: `npm run test:methods:strict` requires
  every method wrapper to be exported by the loaded WASM and callable.

PLS fit is exposed through the public `p4a_pls_fit_simple`
raw-pointer ABI plus the TypeScript `fitPls` / `predictPls` helpers.
The lower-level
`_p4a_model_fit` is still exported but should NOT be invoked directly
from JS with `p4a_matrix_view_t*` arguments — Emscripten 5.0.7 mis-
compiles that call path. The TS `Model` class internally funnels
through the raw-pointer helper.

Benchmark integration lives in
`benchmarks/cross_binding/scripts/bench_js_wasm.mjs`; select it with
`--only js_wasm` in the cross-binding orchestrator.

## Publication preflight

```bash
cd bindings/js
npm ci
npm run build:wasm
npm run build
npm pack --dry-run
```

The npm package publishes `dist/index.js`, `dist/sklearn.js`, their
declarations, and `dist/p4a.{js,wasm}`. `jsr.json` mirrors the same
entrypoints for JSR publication. Because this package publishes generated
JavaScript plus `.d.ts` declarations rather than TypeScript sources, the
JSR command is `npx jsr publish --allow-dirty --allow-slow-types`.

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
│   ├── aom.ts            # OperatorBank / ValidationPlan / AOM results
│   ├── core.ts           # Core model fit/predict helper
│   ├── model.ts          # PLS fit / predict
│   ├── methodResult.ts   # Universal p4a_method_result_t wrapper
│   ├── methods.ts        # Full extra method/selector binding surface
│   ├── nativeModel.ts    # Native p4a_model_t handle for diagnostics
│   └── index.ts          # Public barrel
└── test/
    ├── run_smoke.mjs          # Node smoke (version + parity)
    └── run_methods_smoke.mjs  # Full method-surface smoke/strict gate
```
