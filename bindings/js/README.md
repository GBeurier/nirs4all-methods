# @nirs4all/methods-wasm — WebAssembly binding

Browser + Node.js binding for **libn4m** (the `nirs4all-methods` portable
PLS/NIRS engine), compiled via Emscripten. It is a **non-idiomatic function
library**: raw typed arrays in, typed arrays out. Estimator ergonomics and the
multi-component aggregation live in the separate `nirs4all-lite` repo — not
here. See [`INPUT_CONTRACT.md`](INPUT_CONTRACT.md) and
[`examples/consume.mjs`](examples/consume.mjs).

The package ships:

- `n4m.wasm` — the libn4m C ABI compiled to WebAssembly (full engine).
- `n4m.js` — Emscripten MODULARIZE/EXPORT_ES6 loader.
- `dist/` — TypeScript wrappers (`Context`, `Config`, `Model`, `MethodResult`)
  emitted from `src/`.

## Build

```bash
# 1. Activate the Emscripten SDK.
source /path/to/emsdk/emsdk_env.sh
# 2. Configure + build the WASM preset (zero deps beyond Emscripten).
cmake --preset emscripten
cmake --build --preset emscripten --target pls4all_wasm
# Artifacts land in build/emscripten/bindings/js/{n4m.js,n4m.wasm}.

# 3. Build the TypeScript wrapper for distribution:
cd bindings/js && npm run build && npm run stage:wasm
```

## Smoke test (Node)

```bash
node bindings/js/test/run_smoke.mjs   # fits PLS, checks parity vs native Python
node bindings/js/examples/consume.mjs # the downstream-consumption example
```

The smoke test fits a SIMPLS PLS regression through the raw-pointer entrypoint,
predicts in-sample, and compares coefficients + predictions to a frozen native
fixture (`test/parity_fixture.json`) at a 1e-9 isolated band (achieved ~1e-16).

## API surface

```typescript
import * as n4m from "@nirs4all/methods-wasm";

await n4m.loadModule();
console.log(n4m.version());     // "0.98.0+abi.1.9.0"
console.log(n4m.abiVersion());  // [1, 9, 0]

const rows = 40, cols = 6;
const X = new Float64Array(rows * cols);   // row-major
const y = new Float64Array(rows);
// ... fill X, y ...

const model = n4m.fitPls({ data: X, rows, cols },
                         { data: y, rows, cols: 1 }, 3);
const preds = n4m.predictPls(model, { data: X, rows, cols });
```

`Context` / `Config` / `MethodResult` are also exported for the lower-level
path. There is no idiomatic (sklearn-style) layer — that is intentional.

## Build options

The CMake `emscripten` preset sets:

- `WASM_BIGINT=1` — int64 ABI params are exchanged as `BigInt`.
- `MODULARIZE=1`, `EXPORT_ES6=1` — the module factory is the default export.
- `ALLOW_MEMORY_GROWTH=1`, `INITIAL_MEMORY=64MB`, `MAXIMUM_MEMORY=2GB`.
- `EXPORTED_FUNCTIONS` — driven by `cpp/abi/expected_symbols_linux.txt`; every
  `n4m_*` symbol that ships in `libn4m` is exported here as `_n4m_*` (the full
  engine surface), plus `_malloc`/`_free` and the raw-pointer PLS shims.

## Generic method path (status)

`fitPls` / `predictPls` (and the other raw-pointer shims in
`src/wasm_entry.c`, e.g. `n4m_wasm_pls_fit_legacy`) are callable end-to-end from
JS today and are at the numerical floor.

The **generic "call any of the 188 methods by id"** path is **not yet enabled**.
Emscripten miscompiles JS-built `n4m_matrix_view_t*` *parameters* to the deep
numeric entrypoints (`n4m_model_fit`, the selectors, ...) — they return garbage
even with a byte-correct struct, while the raw-double-pointer shims are
bit-exact. So methods that take a matrix-view argument must be reached through a
C raw-pointer shim built inside the WASM boundary (the `n4m_wasm_pls_fit_legacy`
pattern). Closing the gap to the full 188-method surface is a tracked follow-up
with two viable routes:

1. **Fix the Emscripten codegen** (try a newer emsdk, `-O0` vs `-O2`, dropping
   `WASM_BIGINT`). If the view-pointer path becomes correct, the JS
   `MethodResult` path reaches every `method_result`-producing method with no
   per-method C glue.
2. **Per-family raw-pointer shims** in `wasm_entry.c` (one shim per output
   shape), routing JS callers through the proven raw-pointer boundary.

Note: the JS `makeMatrixView` helper in `ffi.ts` also passes the `int64`
`rows`/`cols` to `n4m_matrix_view_init_rowmajor` via `ccall("number")`, which is
itself wrong under `WASM_BIGINT` — but it is moot until the deeper view-pointer
parameter codegen issue above is resolved, since that helper feeds exactly that
broken path.

## Layout

```
bindings/js/
├── CMakeLists.txt        # Emscripten target wired into the project preset
├── package.json          # @nirs4all/methods-wasm
├── tsconfig.json
├── INPUT_CONTRACT.md      # the raw-array input contract (copied by nirs4all-lite)
├── src/
│   ├── wasm_entry.c      # ABI header pull-in + raw-pointer PLS shims
│   ├── ffi.ts            # Module loader + matrix-view helpers
│   ├── types.ts          # Mirrored C enums + error class
│   ├── context.ts        # Context wrapper
│   ├── config.ts         # Config wrapper
│   ├── model.ts          # Model fit / predict (raw-pointer path)
│   ├── methodResult.ts   # Universal n4m_method_result_t wrapper
│   └── index.ts          # Public barrel
├── examples/
│   └── consume.mjs       # downstream-consumption example
└── test/
    └── run_smoke.mjs     # Node smoke + parity vs native Python
```
