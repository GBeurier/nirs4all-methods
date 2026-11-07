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
console.log(n4m.version());     // "1.3.14+abi.2.22.0"
console.log(n4m.abiVersion());  // [2, 22, 0]

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

### SNV parameter fidelity

`ppCreate("StandardNormalVariate")` supports only omitted params or an actual
empty Array. Every other value, including typed arrays and default-looking
nonempty vectors, refuses before module access. Use the existing named `SNV`
and `RolePipeline` APIs for `with_mean`, `with_std` and `ddof`; no positional
SNV codec exists. Named-role flags must be real booleans. ddof must be a finite
integer number in `[0, INT32_MAX]`; null, strings, boxed scalars and bigint do
not silently coerce. Admission precedes native lookup/allocation/setters.

The owner rejects an invalid variance denominator on nonempty scaled input,
while centering-only remains valid. Out-of-range raw WASM integer setters
can still reach the generic exception-disabled i32 adapter abort: that
separate raw-ABI issue is not qualified by the SNV host validator.


### Checked raw integer narrowing

The generic raw C ABI stores integer parameters as int64. A kernel that needs
int32 rejects an out-of-range value with `N4M_ERR_INVALID_ARGUMENT` and
`integer parameter exceeds int32`, rather than truncating it. This check occurs
when the adapter uses the value (for example, SNV fit), not necessarily at
parameter storage or estimator creation. Genuine wide parameters such as seeds
remain int64. Named SNV host admission and positional `ppCreate` are separate;
SNV positional options still require an empty array.

Emscripten C++ objects use JavaScript-based `-fexceptions`, and the owner CMake
static target propagates the matching final-link option. Embedders that discard
CMake usage requirements must preserve this profile. This does not select native
Wasm EH or change browser/Node runtime requirements. Always deploy a matching
WASM/glue pair; default assets are refreshed only as part of the agreed release.
This JavaScript-EH default applies to builds without the Rust mixed-effects
backend. Builds with `N4M_WITH_MIXEFF` (the `emscripten-analysis` package) link a
Rust panic-unwind archive, so CMake compiles and links every C++ object with
native Wasm exception handling instead, whatever `N4M_WASM_EH_MODEL` says, and that
package requires browser/Node Wasm exception-handling support. This is a build
requirement of the analysis package only; it does not promote the private
native-experimental candidate below.

`npm test` includes `test/run_raw_i32_status.mjs`. With no qualification variables,
it uses this package's `dist/n4m.js` and `dist/n4m.wasm`. Qualification supplies all
four of `N4M_WASM_PATH`, `N4M_WASM_GLUE_PATH`, `N4M_EXPECTED_WASM_SHA256` and
`N4M_EXPECTED_GLUE_SHA256`; partial configuration or a hash mismatch fails before
module execution. The script isolates raw cases in children and checks statuses,
output guards, failed-refit preservation and repeated-refusal memory stability.
`--parity` emits valid existing SNV fixture outputs and retained positive
robust-PLS/selector controls as exact bytes for comparison with the retained baseline. An abort is a failure, never a typed refusal.

## Private Native Wasm EH candidate (not a shipping profile)

`N4M_WASM_EH_MODEL` defaults to `javascript`. Only the explicit Emscripten
`-DN4M_WASM_EH_MODEL=native-experimental` selects the private experiment;
unknown values or a native-experimental request on a non-Emscripten toolchain
are refused. One matching compile/link mode covers all 143 linked owner C++
objects. C sources require no C++ exception option. No math, public ABI/status,
owner state format, scientific profile, or default package asset changes.

Experimental outputs live only in `native-eh-candidate/` in the existing JS
binary directory. The normal pair has a private capability sidecar and a
candidate-only pre-js gate. Before candidate instantiation, the gate requires
the legacy `try`, `catch`, `throw`, `catch_all`, `rethrow`, and `delegate`
instruction set, plus JS i32 `WebAssembly.Tag`/`Exception`, `is`, `getArg`, and
`traceStack`. Missing WebAssembly refuses with `RuntimeUnavailable`; a present
runtime lacking this feature refuses with `Unsupported`, both naming
`wasm-exception-handling`. Artifact corruption and unrelated instantiation
errors are not reclassified. There is no retry, alternate engine or JS-EH
fallback. The public `loadModule` assigns its cached module only after the
factory resolves; direct factory loading has the same pre-js admission gate.
The emitted `__cpp_exception` tag is checked after instantiation as a runtime
witness, not used as a host-refusal decision.

The explicit, `EXCLUDE_FROM_ALL` target `n4m_wasm_eh_boundary_tests` links the
same owner archive plus one private allocator fixture. Only this separate pair
has fault-injection exports. It checks real owner guards returning 9 for
`std::bad_alloc` and 255 for a non-allocating test exception, then same-module
valid recovery. It is never installed or staged into package/default assets.
`test/run_native_eh_candidate.mjs` requires exact pair, sidecar, archive,
public-entrypoint and fixture paths/hashes; its missing-feature simulations
are isolated and labelled. These source tests still require an independently
admitted real build and runtime qualification.

Earlier Node 22.21.1/24.16.0 synthetic feature observations do not qualify this
candidate, browsers, workers or other platforms. Browser adoption and frontend
pins/assets require a joint maintainer batch. EH execution requirements are
separate from `native-workflow@1`, `native-conformal@1` and fitted-state
readability, none of which grants execution permission. Cleanup while foreign
JS aborts unwind is not claimed equivalent across EH modes. The previous
JS-EH pair's binary/glue size and four of six fixed timing failures remain;
this experiment grants no cost waiver. Future candidate qualification retains
all C1–C10 conditions, the six causal mutants, byte anchors and unchanged 1.25
size/per-row timing limits. No candidate runtime or browser support is claimed
from source or configuration alone.

### Private qualification corrections (source only)

The private fixture now snapshots both live C++ `new` allocations and allocator
in-use bytes (`mallinfo().uordblks`) into caller-provided storage. Successful
ordinary/aligned `new` increments the first metric; matching non-null deletes
decrement it. The second metric also covers malloc-backed exception storage.
Both must increase with a live valid owner object and return **exactly** to the
same baseline after warmup, valid destroy, every injected fault and every
recovery. No leak allowance or unchecked exception warmup is accepted.
Linear-memory capacity is retained only as an additional bounded check; its
stability alone is not cleanup evidence. These assertions remain unexecuted.

One valid `create` in the same module/run determines the allocation count,
verified by a second valid create. `n4m_test_eh_arm(fault, skip)` then visits
every allocation index for both real exception types, including allocation
sites with resolved parameters and adapters already live. Each must return the
owner status/diagnostic with null output and exactly one injected throw, then
permit valid create/fit/transform with unchanged byte anchors. A bounded repeat
continues the same exact memory checks. Only the private boundary pair exposes
the single added memory-snapshot export; the normal pair does not.

Both Module and sidecar declare `encoding: wasm-legacy-eh`. This is a declared
requirement, not proof of emitted code. A future pinned installed disassembler
must decode the complete actual binary and compare its EH encoding/opcodes to
the preflight coverage; unexpected encoding/opcodes require a reviewed source
successor. The launcher additionally checks absence of `invoke_` imports and
isolated corrupt-artifact errors distinct from host-capability refusals.

The explicit `--tests-first-js-eh` route admits the exact old JS-EH pair using
an independently admitted notes-only sidecar. It loads the real factory and
checks actual context/metadata calls, then must fail at
`NATIVE_EH_REQUIRED_CONTRACT_ABSENT` before later C++ tag requirements. A missing
file, import error or another assertion is not that red. No red is claimed from
source alone. Full recomposed Python qualification must keep the accepted R
availability environment and three R selectors, isolated pytest plugins/cache
and fresh bounded basetemps. All prior cost failures and required gates remain.

## Ranking candidates

`rank(scores, present, options)` sends a row-major `Matrix` of scores and a
required row-major `Uint8Array` presence mask to the native ranking ABI. Each
row is a comparable block and each column a candidate. Mask entries are 0 or 1;
masked scores may be nonfinite. For example:

```js
import { loadModule, rank } from "@nirs4all/methods";
await loadModule();
const result = rank({ rows: 2, cols: 2, data: Float64Array.of(1, 2, 3, 2) },
                    Uint8Array.of(1, 1, 1, 1),
                    { method: "bradley_terry", higher_is_better: false });
```

Methods are `average_rank`, `elo`, `bradley_terry`, `trueskill` and `openskill`.
The result identifies the native variant and effective options. Unavailable
quantities are `null`, including ranks across disconnected comparison graphs.
Bayesian uncertainty is an approximate algorithm posterior standard deviation,
not a frequentist standard error. The Friedman statistic is descriptive and
uses complete blocks; the caller must assess independence.
