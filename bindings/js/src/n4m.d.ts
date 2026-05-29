// Types for the sibling Emscripten loader `./n4m.js` (staged into dist/ at pack
// time via package.json `stage:wasm`; it ships no TS types of its own). tsc
// resolves the dynamic `import("./n4m.js")` in ffi.ts to this declaration; the
// real loader is dist/n4m.js at runtime.
declare const factory: (opts?: object) => Promise<unknown>;
export default factory;
