# JavaScript / WebAssembly binding (skeleton)

Phase 0 reserves this directory. Phase 2 ships a predict-first WebAssembly
build compiled via Emscripten, with a TypeScript wrapper.

Planned layout:

```
bindings/js/
├── package.json
├── tsconfig.json
├── emscripten_build.sh        # invokes the emscripten preset on libp4a_c_static
├── src/
│   ├── index.ts               # public TypeScript API
│   ├── ffi.ts                 # raw ccall / cwrap signatures
│   ├── context.ts
│   └── model.ts               # Phase 2 (predict-only initially)
└── test/
    └── version.test.ts
```

The npm package will ship `npls.wasm`, `npls.js`, and `npls.d.ts`. Browser
and Node.js are first-class targets; we use Emscripten's native `ccall` /
`cwrap` rather than `embind` so the binding stays free of additional
glue-code dependencies.
