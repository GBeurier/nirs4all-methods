# Phase 32 — WebAssembly binding scaffold

**Status:** shipped — local tag `phase-32-wasm-binding`
(`0.83.0+abi.1.12.0`).

First language binding beyond Python and R: a WebAssembly build of
`libp4a` via Emscripten 5.0.7, plus a TypeScript wrapper. The phase
exists so the webapp track can call into pls4all directly from the
browser without a Python backend round-trip.

## Layout

```
bindings/js/
├── CMakeLists.txt           # Emscripten target wired into the `emscripten` preset
├── src/
│   ├── wasm_entry.c         # Force-keeps the public ABI alive against dead-stripping
│   ├── ffi.ts               # Module loader + heap helpers
│   ├── types.ts             # Mirrored C enums + Pls4allError
│   ├── context.ts           # Context wrapper
│   ├── config.ts            # Config wrapper
│   ├── model.ts             # Model fit / predict
│   ├── methodResult.ts      # Universal p4a_method_result_t wrapper
│   └── index.ts             # Public barrel
├── test/
│   ├── run_smoke.mjs        # Node smoke (version + lifecycle)
│   └── wasm_fit_debug.c     # Pure-C reference diagnostic
├── package.json
└── tsconfig.json
```

## Build

```bash
source $EMSDK/emsdk_env.sh
cmake --preset emscripten
cmake --build --preset emscripten   # produces p4a.js + p4a.wasm
node bindings/js/test/run_smoke.mjs   # smoke
```

The Emscripten preset sets:

- `WASM_BIGINT=1` for native int64 ABI exchange.
- `MODULARIZE=1`, `EXPORT_ES6=1` — module factory is the default export.
- `ALLOW_MEMORY_GROWTH=1`, `INITIAL_MEMORY=64MB`, `MAXIMUM_MEMORY=2GB`.
- `EXPORTED_FUNCTIONS` is generated at configure time from
  `cpp/abi/expected_symbols_linux.txt` — every `p4a_*` export plus
  `_malloc` / `_free`.
- `-O2`, `ASSERTIONS=1`.

## Known issue

`p4a_model_fit` invoked from a JavaScript host returns incorrect
`x_mean` and coefficients. The same library — `libp4a_static.a` from
the very same Emscripten build — fits correctly when called from a C
translation unit compiled into the same WASM module (see
`bindings/js/test/wasm_fit_debug.c`). The bug is therefore in the
JS↔WASM marshalling path, not in the C++ numerics.

Suspected cause: a packing or ABI subtlety in how int64 fields on
`p4a_matrix_view_t` interact with `WASM_BIGINT=1` when the struct is
populated from JS and read back from C++. Lifecycle paths
(Context / Config create/destroy, version queries, getters/setters)
work correctly; fit / predict / method-result calls produce garbage.

The README and smoke test document this; the fit path stays
disabled in the smoke until the root cause is fixed.

## ABI delta

- 0 new symbols (the WASM binding is consumer-side, no new C ABI
  exports). 159 symbols total, unchanged.
- Project version 0.82 → 0.83.

## Local gate

- 256 native C++ tests pass (dev-release, asan-ubsan, ubsan).
- WASM build produces `p4a.js` (89 KB) + `p4a.wasm` (480 KB).
- Node smoke: version + ABI + Context/Config lifecycle PASS.
- `git diff --check`: clean.
