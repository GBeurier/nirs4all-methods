# JavaScript / WebAssembly binding

The current package is
[`@nirs4all/methods`](https://github.com/GBeurier/nirs4all-methods/tree/main/bindings/js),
version 1.0.19 in this checkout. It is a browser and Node.js WebAssembly
binding for the ABI-2 `libn4m` engine; it is not a Phase-0 placeholder.

```js
import * as n4m from "@nirs4all/methods";

await n4m.loadModule();
console.log(n4m.abiVersion());

const X = { data: new Float64Array(40 * 6), rows: 40, cols: 6 };
const Y = { data: new Float64Array(40), rows: 40, cols: 1 };
const model = n4m.fitPls(X, Y, 3);
const prediction = n4m.predictPls(model, X);
```

The public TypeScript barrel exports `Context`, `Config`, `MethodResult`,
matrix helpers, PLS/model functions, AOM/POP helpers, splitter functions, and
preprocessing operator lifecycle functions. For methods without a dedicated
high-level helper, `MethodResult.run` invokes the current catalogued C-ABI
method-result producer; see the
[binding README](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/js/README.md)
for its required context, configuration, and matrix-view setup.

Build and run the checked binding locally with `npm run build`, then stage the
WASM artifact and run `npm test` from `bindings/js`.
