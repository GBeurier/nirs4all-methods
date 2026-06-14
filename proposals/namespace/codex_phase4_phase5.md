**A. Phase 4: NO-GO**

Most Phase 4 checks pass, but there is one blocker: generated per-method docs still publish ABI-1 C symbols.

Passed:
- [docs/methods/index.md](/home/delete/nirs4all/nirs4all-methods/docs/methods/index.md:1) has 209 entries, 12 `n4m.<role>` groups, and 209/209 links resolve.
- `bench-data.json` and `proposals/dashboard_v2/bench-data.js` are remapped to namespace roles, with 183 benchmarked algos and no fabricated 209-method expansion.
- `proposals/dashboard_v2` is offline/no-CDN.
- [MIGRATION_ABI2.md](/home/delete/nirs4all/nirs4all-methods/docs/MIGRATION_ABI2.md:1), [abi/reference.md](/home/delete/nirs4all/nirs4all-methods/docs/abi/reference.md:1), and [bindings/python.md](/home/delete/nirs4all/nirs4all-methods/docs/bindings/python.md:1) correctly describe ABI-2 roles.

Blocker:
- 78 generated `docs/methods/*.md` pages still contain old symbols from `_rename_map.tsv`. Example: [boosting_pls.md](/home/delete/nirs4all/nirs4all-methods/docs/methods/boosting_pls.md:37) and [line 61](/home/delete/nirs4all/nirs4all-methods/docs/methods/boosting_pls.md:61) render `n4m_boosting_pls_fit`, while the catalog has `n4m_ensemble_boosting_pls_fit`.
- Root cause: [build_methods.py](/home/delete/nirs4all/nirs4all-methods/docs/_extras/build_methods.py:45) still points rich-page metadata at removed `bindings/_catalog/sklearn_tier2.yaml`; [usage_section](/home/delete/nirs4all/nirs4all-methods/docs/_extras/build_methods.py:2565) falls back to legacy `<method>_fit` names and hardcoded AOM symbols.

Required fix: wire `catalog/methods.yaml` / `_rename_map.tsv` into per-method C-symbol rendering, update stale bibliography implementation strings, regenerate `docs/methods`, and add a doc check that no old `_rename_map.tsv` symbol appears in generated method pages. Then run `python docs/_extras/build_methods.py --strict` and `sphinx-build -b html docs docs/_build/html`. I could not run Sphinx here because `sphinx` is not installed.

**B. Phase 5 Locked Plan**

1. `nirs4all-methods` first: fix Phase 4 docs, build ABI-2 native artifacts, regenerate JS/WASM dist, keep Python/JS package floor at `0.99.0`.

2. `nirs4all-web`:
- Edit [studio-lite/src/catalog/nodes.ts](/home/delete/nirs4all/nirs4all-web/studio-lite/src/catalog/nodes.ts:13). Its `n4m: { ... }` blocks declare 56 symbols; 53 are old ABI-1 names. Remap all 53 through `_rename_map.tsv`.
- Examples: `n4m_pp_snv_transform -> n4m_transform_snv_transform`, `n4m_pp_savgol_transform -> n4m_transform_savitzky_golay_transform`, `n4m_pls_fit_simple -> n4m_estimators_pls_fit`, `n4m_pls_lda_fit -> n4m_estimators_pls_lda_fit`, `n4m_aom_global_select -> n4m_model_selection_aom_pls_select`, splitter symbols to `n4m_model_selection_*`.
- No edit needed in [validate-catalog.mjs](/home/delete/nirs4all/nirs4all-web/studio-lite/scripts/validate-catalog.mjs:31); it already checks `nodes.ts` against ABI snapshots.
- Rerun [build-wasm.sh](/home/delete/nirs4all/nirs4all-web/studio-lite/scripts/build-wasm.sh:41) after methods JS dist is rebuilt; generated `src/engine/wasm/methods/*` currently contains old staged symbols and should not be hand-edited.
- Gates: `npm run validate:catalog`, `npm run typecheck`, `npm run test`, `npm run build`, `npm run build:single`, then browser smoke tests. WASM/Emscripten pieces are env-bound.

3. `nirs4all-lite`:
- [bindings/python/pyproject.toml](/home/delete/nirs4all/nirs4all-lite/bindings/python/pyproject.toml:34): bump both `nirs4all-methods>=0.98.0` entries to `>=0.99.0`.
- [bindings/python/src/nirs4all_lite/_execution.py](/home/delete/nirs4all/nirs4all-lite/bindings/python/src/nirs4all_lite/_execution.py:172): replace removed imports with:
  `from n4m.transform.scatter import SNV`
  `from n4m.transform.smoothing import SavitzkyGolay`
  `from n4m.model_selection.splitters import KennardStoneSplitter`
  `pls4all.sklearn.PLSRegression` remains valid.
- [bindings/rust/nirs4all/src/lib.rs](/home/delete/nirs4all/nirs4all-lite/bindings/rust/nirs4all/src/lib.rs:772): remap dynamic symbol loads:
  `n4m_split_kennard_stone_* -> n4m_model_selection_kennard_stone_*`,
  `n4m_pp_snv_* -> n4m_transform_snv_*`,
  `n4m_pp_savgol_* -> n4m_transform_savitzky_golay_*`.
  Shared infra symbols like `n4m_model_fit` stay unchanged.
- [compat/upstreams.toml](/home/delete/nirs4all/nirs4all-lite/compat/upstreams.toml:47): imports are still valid; despite the scout, there is no current version/commit pin there. Add/update one only if release policy requires it.
- WASM tests need no source edit; point `NIRS4ALL_METHODS_JS_DIST` at the regenerated ABI-2 methods dist.
- Gates: `make test`, Rust fmt/clippy/test, Python unittest with methods installed, WASM tests with ABI-2 dist, plus strict Python/Rust/WASM/R/Octave parity. R, Octave/MATLAB, and WASM toolchain checks are env-bound.

Execution order: methods artifacts → web catalog/WASM staging → lite dependency/import/symbol updates → optional doc/name sweeps.

**C. No Code Change Confirmed**

- `nirs4all-studio`: no methods ABI/import coupling found.
- `nirs4all-io`: only docs and its own `n4io` ABI references.
- `nirs4all-datasets`: only docs/catalog validation references and its own ABI files.
- `nirs4all` main: internal planning notes only.

Optional doc sweep: IO has a stale doc example mentioning `n4m/sklearn`; not runtime coupling.

**D. Risks**

- Published docs can look migrated via the index while method pages still advertise dead ABI-1 symbols.
- Web can pass TypeScript while failing at runtime if `src/engine/wasm/methods` is not restaged from ABI-2 dist.
- Lite Rust symbol errors appear at runtime, not compile time.
- Leaving dependency floors at `0.98.0` allows mixed ABI installs.
- Sphinx, Emscripten/WASM, R, and Octave/MATLAB gates need the proper CI/maintainer environment.