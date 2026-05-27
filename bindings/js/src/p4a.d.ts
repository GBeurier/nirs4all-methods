// Types for the sibling Emscripten loader `./p4a.js` (staged into dist/ at pack
// time via package.json `stage:wasm`; it ships no TS types of its own). tsc
// resolves the dynamic `import("./p4a.js")` in ffi.ts to this declaration; the
// real loader is dist/p4a.js at runtime.
declare const factory: (opts?: object) => Promise<unknown>;
export default factory;
