# @nirs4all/methods — WebAssembly binding

Browser + Node.js binding for **libn4m** (the `nirs4all-methods` portable
PLS/NIRS engine), compiled via Emscripten. It is a **non-idiomatic function
library**: raw typed arrays in, typed arrays out. Estimator ergonomics and the
multi-component aggregation live in the separate `nirs4all-core` repo — not
here. See [`INPUT_CONTRACT.md`](INPUT_CONTRACT.md) and
[`examples/consume.mjs`](examples/consume.mjs).

The package ships:

- `n4m.wasm` — the libn4m C ABI compiled to WebAssembly (full engine).
- `n4m.js` — Emscripten MODULARIZE/EXPORT_ES6 loader.
- `dist/` — TypeScript wrappers (`Context`, `Config`, `Model`, `MethodResult`)
  emitted from `src/`.
- `Optimizer` — an owning typed facade over the native ask/tell HPO engine.

## Build

```bash
# 1. Activate the Emscripten SDK.
source /path/to/emsdk/emsdk_env.sh
# 2. Configure + build the WASM preset (zero deps beyond Emscripten).
cmake --preset emscripten
cmake --build --preset emscripten --target n4m_wasm
# Artifacts land in build/emscripten/bindings/js/{n4m.js,n4m.wasm}.

# 3. Build the TypeScript wrapper for distribution:
cd bindings/js && npm run build && npm run stage:wasm
```

## Smoke test (Node)

```bash
cd bindings/js
npm test                 # PLS parity + API/generic/AOM/new-pack smokes
node examples/consume.mjs # the downstream-consumption example
```

The smoke suite fits a SIMPLS PLS regression through the raw-pointer entrypoint,
checks the public API and generic method path, exercises POP/AOM helpers, and
gates the broad-model-pack additions (`ECR`, `O2PLS`, AOM Ridge/Stack,
DataTwinning, SystematicCircular). The PLS smoke compares coefficients +
predictions to a frozen native fixture (`test/parity_fixture.json`) at a 1e-9
isolated band (achieved ~1e-16).

The additive raw `MultimodalClassifierPipeline` requires ABI 2.17 and the
native PLS-logistic head. `fit(blocks, labels)` keeps string or exact integer
original labels; `predictProba(blocks)` returns columns in `classes()` order.
`exportState()` writes N4MC bytes without training rows.
`MultimodalClassifierPipeline.fromState(state, recipe, schemas, { classNames })`
restores the original typed label table; omitting it exposes native class IDs.

## API surface

```typescript
import * as n4m from "@nirs4all/methods";

await n4m.loadModule();
console.log(n4m.version());     // "1.2.1+abi.2.17.0"
console.log(n4m.abiVersion());  // [2, 17, 0]

const rows = 40, cols = 6;
const X = new Float64Array(rows * cols);   // row-major
const y = new Float64Array(rows);
// ... fill X, y ...

const model = n4m.fitPls({ data: X, rows, cols },
                         { data: y, rows, cols: 1 }, 3);
const preds = n4m.predictPls(model, { data: X, rows, cols });

const split = n4m.computeSplitIndices("KennardStone", { data: X, rows, cols }, null);
// `computeSplit()` remains available when a compact train/test mask is enough.

// One native route also covers all nine splitter kinds, including fold kinds.
const ordered = n4m.splitNative("SPXYFold", { data: X, rows, cols },
                                { data: y, rows, cols: 1 },
                                { nSplits: 3, foldIndex: 0 });
// ordered.trainIndices / ordered.testIndices are zero-based Int32Array values.

// Training-only X-to-X augmentation; no Y mixing or fitted-state export.
const noisyTrainX = n4m.augmentNative("GaussianNoise",
                                     { data: X, rows, cols }, [0.03], 42);
```

`Context` / `Config` / `MethodResult` are also exported for the lower-level
path. There is no idiomatic (sklearn-style) layer — that is intentional.

Fitted preprocessing uses the native pipeline handle. Its N4MP bytes contain
the fitted state and original ordered operator plan, not an N4MM model:

```typescript
const ctx = n4m.Context.create();
const recipe = [
  { kind: n4m.PipelineOperatorKind.SNV, params: [] },
  { kind: n4m.PipelineOperatorKind.MSC, params: [] },
];
const pipeline = n4m.NativePreprocessingPipeline.fit(ctx, recipe, trainX);
const payload = pipeline.toBytes();
const restored = n4m.NativePreprocessingPipeline.fromBytes(ctx, payload, recipe);
const transformed = restored.transform(heldOutX);
console.log(restored.nFeatures, restored.steps); // plan read from native handle
restored.destroy();
pipeline.destroy();
ctx.destroy();
```

`trainX` and `heldOutX` are row-major `{ data: Float64Array, rows, cols }`.
OSC/EPO additionally require a training `Y` matrix at `fit`. The optional
`fromBytes` recipe argument checks the imported native plan, including
positional parameters; applications should also bind the blob to their
feature schema. Native N4MP v1 supports operator kinds 0–14 only (15 kinds).

## Native optimizer

`Optimizer` uses the same C ABI and N4MOPT checkpoint as Python and R. It owns
its context; call `dispose()` after the study. Integer axes, trial ids and
integer categorical choices use `bigint`, so the browser does not silently
round the native int64 values. The host supplies the objective and score; no
search or pruning algorithm is implemented in TypeScript.

```typescript
const space = {
  components: { kind: "int", low: 1, high: 8 },
  alpha: { kind: "log_float", low: 1e-4, high: 1e-1 },
  branch: { kind: "categorical", type: "string", choices: ["nir", "fusion"] },
} as const;
const study = n4m.Optimizer.create(space, { sampler: "tpe", seed: 42n });
try {
  for (let i = 0; i < 30; i++) {
    const trial = study.ask();
    const score = evaluate(trial.parameters); // application-owned validation
    study.tell(trial, "completed", score);
  }
  console.log(study.best(), study.trialRecords());
  const checkpoint = study.save(); // portable N4MOPT bytes
  const resumed = n4m.Optimizer.load(checkpoint, space);
  resumed.dispose();
} finally {
  study.dispose();
}
```

`askBatch`, `enqueue`, `intermediate`, terminal outcomes and conditional
constraints also call native ABI functions. `trials()` returns an owning
`MethodResult` rich-trace snapshot that the caller destroys; `trialRecords()`
decodes and releases that snapshot. The search-space declaration passed to
`load()` is for typed JS decoding; the native checkpoint owns the optimizer
state. Keep that declaration identical to the original one. The binding does
not fit a model or tune a DAG by itself.

The 14 committed HPO golden traces run through JS/WASM with a fixed score tape
and exact proposed parameters, statuses, pruning decisions, errors and event
order. The JS closed-form objective is checked against that tape within a
relative `1e-12` band: Python `**2` can differ from JS multiplication by one
binary64 ULP. A separate gate resumes checkpoints Python → JS/WASM and
JS/WASM → Python exactly. The [cross-binding CI](https://github.com/GBeurier/nirs4all-methods/actions/runs/36548060107)
also reproduces all 45 sampler-pruner compositions from temporary native traces.

## Build options

The CMake `emscripten` preset sets:

- `WASM_BIGINT=1` — int64 ABI params are exchanged as `BigInt`.
- `MODULARIZE=1`, `EXPORT_ES6=1` — the module factory is the default export.
- `ALLOW_MEMORY_GROWTH=1`, `INITIAL_MEMORY=64MB`, `MAXIMUM_MEMORY=2GB`.
- `EXPORTED_FUNCTIONS` — driven by `cpp/abi/expected_symbols_linux.txt`; every
  `n4m_*` symbol that ships in `libn4m` is exported here as `_n4m_*` (the full
  engine surface), plus `_malloc`/`_free` and the raw-pointer PLS shims.

## Generic method path (enabled)

Two ways to reach the engine, both bit-exact vs native:

- **Raw-pointer shims** — `fitPls` / `predictPls` and the other shims in
  `src/wasm_entry.c` (e.g. `n4m_wasm_pls_fit_legacy`): typed arrays in, typed
  arrays out, no handle bookkeeping.
- **Generic `MethodResult.run`** — call **any** of the ~188 `method_result`
  producers by symbol, passing `n4m_matrix_view_t*` arguments built from JS.

The generic path **works** and is regression-tested (`test/run_generic_method.mjs`
fits `n4m_estimators_sparse_simpls_fit` and the generic `n4m_model_fit` and matches the raw
`n4m_estimators_pls_fit` oracle byte-for-byte). The previous "Emscripten miscompiles
matrix-view parameters" diagnosis was **wrong**: the bug was the TS `ccall` layer
passing a JS `number` for the `int64_t` `rows`/`cols` fields. Marshalling those
dims as `BigInt` under `WASM_BIGINT` (`ffi.ts` / `makeMatrixView`) made the deep
view-pointer path byte-correct — see the note at the top of `src/wasm_entry.c`.

```js
import { loadModule, Context, Config, MethodResult } from "@nirs4all/methods";
await loadModule();
const ctx = new Context(), cfg = new Config();
const X = makeMatrixView(rows, cols, dataF64);   // row-major Float64Array
const res = MethodResult.run("n4m_estimators_sparse_simpls_fit", ctx, cfg, [X /* , Y */]);
const coef = res.matrix("coefficients");          // typed array + shape
res.destroy(); cfg.destroy(); ctx.destroy();
```

## Layout

```
bindings/js/
├── CMakeLists.txt        # Emscripten target wired into the project preset
├── package.json          # @nirs4all/methods
├── tsconfig.json
├── INPUT_CONTRACT.md      # the raw-array input contract (consumed by nirs4all-core)
├── src/
│   ├── wasm_entry.c      # ABI header pull-in + raw-pointer PLS shims
│   ├── ffi.ts            # Module loader + matrix-view helpers
│   ├── types.ts          # Mirrored C enums + error class
│   ├── context.ts        # Context wrapper
│   ├── config.ts         # Config wrapper
│   ├── model.ts          # Model fit / predict (raw-pointer path)
│   ├── methodResult.ts   # Universal n4m_method_result_t wrapper
│   ├── optimization.ts   # Owning native HPO facade
│   └── index.ts          # Public barrel
├── examples/
│   └── consume.mjs       # downstream-consumption example
└── test/
    ├── run_smoke.mjs     # Node PLS smoke + parity vs native Python
    ├── run_api.mjs
    ├── run_generic_method.mjs
    ├── run_pop_aom.mjs
    └── run_new_pack.mjs  # broad-model-pack smoke
```
