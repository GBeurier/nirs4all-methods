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
not binding truth.** Every per-method `abi_symbols` array now comes from the
reviewed `catalog/abi_method_map.yaml` + `catalog/abi_missing_methods.yaml`
(the source of truth, applied by `reconcile_abi.py --write`), so the
reconciliation is complete:

```jsonc
{ "abi": "1.9.0", "source": "catalog projection (advisory)",
  "reconciliation": {
    "real_abi_symbol_count": 669, "catalog_symbol_count": 548,
    "catalog_real": 548, "catalog_guessed": 0, "catalog_coverage_pct": 100.0,
    "unmapped_real_abi_symbols": [...], "unmapped_real_count": 121 },
  "methods": [ { "method_id": "...", "abi_symbols": [...],
                 "real_abi_symbols": [...], "guessed_abi_symbols": [...] }, ... ] }
```

The 121 `unmapped_real_abi_symbols` are the **infra/support surface** —
context, config, array marshalling, status, version, RNG, matrix view, plus
the shared model / split-result handles — explicitly tracked in
`catalog/abi_infra.yaml`. They belong to no operator and so do not appear in
any method's `abi_symbols`. The actual gate is `union(catalog method symbols,
infra symbols) == snapshot` (`reconcile_abi.py --check`,
`validate.py --strict-abi`), both green: **548 + 121 = 669/669**.

The two gates are enforced in `catalog-validate.yml` (`--strict-abi`). Strict
manifest-driven raw-class generation is no longer gated on reconciliation
coverage — that debt is paid; the remaining gate is the **behavioral**
conformance gate below.

## The real (behavioral) conformance gate

Because a ctypes/`.Call`/MEX binding can address *any* exported symbol,
symbol-level "does the binding expose X" is meaningless. Conformance is
**behavioral**, and these gates exist today:

- donor **raw ↔ idiomatic** parity (bit-exact) — `test_donor_binding_specs.py`
- donor **raw ≡ C++** (via `n4m_donor_bench --dump-output`)
- donor **n4m ↔ nirs4all** reference parity / allowlisted expected divergence
- (Phase C / C11) **inter-binding parity** `rmse_rel < 1e-12` vs C++ raw

`n4m.raw` (the object-class raw layer for all ~160 methods) is **unblocked**:
the catalog↔ABI reconciliation is complete (548 method + 121 infra = 669/669
exported `n4m_*` symbols, gated in `catalog-validate.yml`). Bringing the raw
layer up is the next pilot.

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

A binding is spec-compliant when its **behavioral** gates pass: inter-binding
parity (`rmse_rel < 1e-12` vs C++ raw, Phase C / C11), the donor raw↔idiomatic +
raw≡C++ gates, and its idiomatic smoke tests. What hard-fails **in CI today**:
the donor gates (`test_donor_binding_specs.py`) and the Python leg of the
cross-binding gate. The R / Octave / JS legs of `cross-binding-parity.yml`
now also run on every push (`r-binding`, `octave-mex`, `js-wasm` jobs;
locally-verified `rmse_rel ≈ 9.4e-18` / `4e-16` / `1e-16` against the shared
Python-generated fixture). The manifest reconciliation is **not** a
conformance criterion (a ctypes-style binding can address any symbol, so
symbol-coverage proves nothing) — it is now a healthy state report rather
than a Phase-B debt report.

## F-prep scope (binding-scaling infra) — reduced + deferred

Phase F-prep in the refactor plan (`docs/REFACTOR_PLAN.md`, F-prep-1…7) was the
infrastructure to scale to 10+ languages: spec → conformance → **profiles** →
**render** → skeletons → unified CI/release matrices. The 4-language target
(R, Python, Octave/MATLAB, JS) + the hand-written-idiomatic decision above
**reduce** it sharply:

| F-prep task | Status |
|-------------|--------|
| F-prep-1 SPEC.md | **Done** (this file). |
| F-prep-2 conformance suite (≥3 methods/category) | **Live** — the donor raw↔idiomatic + raw≡C++ gates (`benchmarks/cross_binding/tests/test_donor_binding_specs.py`) hard-fail in CI; `cross-binding-parity.yml` runs all four bindings on every push (`harness-and-python-smoke`, `js-wasm`, `octave-mex`, `r-binding`; MATLAB stays manual — no licensed runner). |
| F-prep-3 framework-profile **schema** | **Obsolete** — idiomatic adapters are hand-written (see "Two layers"); no profile YAML. |
| F-prep-4 per-`(method × profile)` **template engine** | **Obsolete** — the explicitly-rejected over-engineering at 4-language scale. |
| F-prep-5 skeleton generator (`make new-binding LANG=`) | **Deferred, low value** — with a closed 4-language target set there is no new binding to scaffold. The `make new-binding` target is a stub that prints "not yet wired (Phase F-prep)". |
| F-prep-6 unified `bindings.yml` CI matrix | **Deferred** — would consolidate the per-language workflows into one matrix. A refactor of existing CI, not new capability. |
| F-prep-7 `release-bindings.yml` matrix | **Deferred** — single workflow looping the publish matrix; today publication is per-binding (PyPI/CRAN automated; JS/MATLAB/Octave manual — see `docs/dev/release_process.md`). |

**Net:** F-prep is no longer a prerequisite phase. What remained meaningful
(F-prep-6/-7, a CI/release **consolidation**) is optional housekeeping; the
generative parts (F-prep-3/-4) are cancelled. **Trigger to revisit:** a request
for a 5th+ target language — at which point the unified CI/release matrix and a
skeleton generator start to pay for themselves.

> Indexed in [`../DEFERRALS.md`](../DEFERRALS.md).
