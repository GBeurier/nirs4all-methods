# Binding specification (n4m)

Normative contract every in-tree binding implements. Short by design; the
conformance gate (`benchmarks/cross_binding/tests/` + the inter-binding parity)
is what actually enforces it.

## Scope (current)

**Target languages:** R, Python, Octave/MATLAB, JS. Every other language is on
hold under `bindings/_archive/` and is out of scope until a `binding-request`
picks it up.

## Two layers — what is generated vs hand-written

This is the load-bearing architecture decision (maintainer + Codex review,
superseding the original §2.11 "generate everything from profiles" plan, which
was justified only by a 10-language / 20-profile target):

| Layer | What | How it's produced |
|-------|------|-------------------|
| **Raw** | Object classes that wrap the C ABI **faithfully** — one handle-wrapping class per method/family (`create` → `fit`/`transform`/`predict`/`destroy` + typed result accessors). NOT a bag of C-style free functions. | **Generation-assisted**: the raw surface is driven by the catalog **raw manifest** (`build/catalog/rendered_api/raw_manifest.json`, from `catalog/scripts/render_api.py`). Per-method FFI plumbing across ~160 methods is mechanical and error-prone to hand-write, so it is generated/registry-driven. |
| **Idiomatic** | Small adapters that map the raw classes onto the host ecosystem's "big types": Python sklearn `BaseEstimator`/`Transformer`/`Predictor`; R `formula` + S3. **1–2 profiles** for Python and R only. | **Hand-written**, shared, conformance-tested. A per-`(method × profile)` template/codegen engine is **explicitly NOT built** at 4-language scale — it is over-engineering. Idiomatic adaptation is a handful of base classes that dispatch by method, not generated per method. |

There is **no** per-method idiomatic codegen and **no** profile/template engine.
The old metric "0 hand-written idiomatic wrappers" is replaced by:

> **0 hand-written per-method *raw* FFI plumbing; idiomatic adapters are
> shared, hand-written, and conformance-tested.**

## Raw layer — required entry points (every binding)

1. **Library loading** — locate + load `libn4m.{so,dll,dylib}` (or the JS WASM
   Module). Respect an env override (`N4M_LIB_PATH` for Python).
2. **ABI version probe** — call `n4m_check_abi_compatibility(header_major,
   header_minor)` before any other call; fail loudly on skew.
3. **Matrix marshalling** — accept the host's native matrix (NumPy array, R
   matrix, MATLAB array, JS TypedArray) and build `n4m_matrix_view_t` with
   explicit `row_stride`/`col_stride`. **Never force a copy** (NumPy row-major,
   R/MATLAB column-major, transposes and slices all pass zero-copy).
4. **Status / error** — read `n4m_status_t` from every fallible call; on
   non-OK, copy the error string out of the context-owned buffer **before** any
   other call on the same context. Translate to a host exception/condition.
5. **Memory ownership** — never free a core-allocated buffer from the host;
   never expect the core to free a host buffer. Two output APIs: caller-
   allocated, and core-allocated + `n4m_array_free`.
6. **Object classes** — expose handle-wrapping classes (not free functions):
   lifecycle (`create`/`destroy` via ctor/`__del__`/finalizer), the operation
   methods, and typed result accessors. The class set + their ABI symbols come
   from the **raw manifest**.
7. **Symbol enumeration** — at startup, know which methods this `libn4m` build
   exposes (from the compiled-in manifest); emit a host-native error if a
   method is requested but the symbol is missing.

## Raw manifest (advisory) + ABI reconciliation

`render_api.py` projects `catalog/methods/*.yaml` into
`build/catalog/rendered_api/raw_manifest.json`. **It is a catalog projection,
not binding truth.** The catalog `abi_symbols` are largely auto-discovered
guesses, so the manifest also reconciles them against the authoritative ABI
snapshot `cpp/abi/expected_symbols_linux.txt` and reports the gap:

```jsonc
{ "abi": "1.9.0", "source": "catalog projection (advisory)",
  "reconciliation": {
    "real_abi_symbol_count": 669, "catalog_symbol_count": 427,
    "catalog_real": 8, "catalog_guessed": 419, "catalog_coverage_pct": 1.9,
    "unmapped_real_abi_symbols": [...], "unmapped_real_count": 662 },
  "methods": [ { "method_id": "...", "abi_symbols": [...],
                 "real_abi_symbols": [...], "guessed_abi_symbols": [...] }, ... ] }
```

`test_raw_manifest_reconciliation` runs this **non-strict** (exit 0) — it
quantifies the **Phase-B catalog-cleanup debt**, it does not gate. Strict
manifest-driven raw-class generation is **gated on this reconciliation reaching
~100 %**; until then, generating object classes from the catalog would bake bad
symbol names / signatures into a public-looking API.

## The real (behavioral) conformance gate

Because a ctypes/`.Call`/MEX binding can address *any* exported symbol,
symbol-level "does the binding expose X" is meaningless. Conformance is
**behavioral**, and these gates exist today:

- donor **raw ↔ idiomatic** parity (bit-exact) — `test_donor_binding_specs.py`
- donor **raw ≡ C++** (via `n4m_donor_bench --dump-output`)
- donor **n4m ↔ nirs4all** reference parity / allowlisted expected divergence
- (Phase C / C11) **inter-binding parity** `rmse_rel < 1e-12` vs C++ raw

`n4m.raw` (the object-class raw layer for all ~160 methods) is **deferred until
the catalog↔ABI reconciliation is done** — see the manifest. A small,
manually-curated pilot subset is the only acceptable earlier step.

## Idiomatic layer

- **Python** — one `sklearn`-style base per role (`Transformer` / `Regressor` /
  `Classifier`) in `n4m/sklearn/_base.py`, dispatching by method onto the raw
  classes. `check_estimator`-clean where applicable.
- **R** — `formula` + S3 adapters (`n4m/R/sklearn_methods.R`) over the raw
  dispatcher.
- Anti-drift: a generated smoke test for **every** manifest method through each
  declared adapter (constructor signature, `get_params`, basic fit/transform),
  plus the donor raw↔idiomatic + raw≡C++ gates already in
  `benchmarks/cross_binding/tests/test_donor_binding_specs.py`.

## Conformance

A binding is spec-compliant when its **behavioral** gates pass (see above):
inter-binding parity (`rmse_rel < 1e-12` vs C++ raw, Phase C / C11), the donor
raw↔idiomatic + raw≡C++ gates, and its idiomatic smoke tests. CI hard-fails on
those. The manifest reconciliation is a non-gating Phase-B debt report — not a
pass/fail conformance criterion (a ctypes-style binding can address any symbol,
so symbol-coverage proves nothing).
