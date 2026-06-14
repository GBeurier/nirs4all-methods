**A. Verdict**

Overall: **NO-GO for the whole migration program** until the methods docs residual is fixed.

Web and lite pass the requested ABI-2 remap checks. Methods docs pass the exact `_rename_map.tsv` column-1 symbol lint, but the generated operator pages still publish ABI-1 wildcard C ABI prefixes, e.g. [split_kennard_stone.md](/home/delete/nirs4all/nirs4all-methods/docs/methods/split_kennard_stone.md:3) shows `n4m_split_kennard_stone_*`. I found **118 pages** with old wildcard prefixes such as `n4m_pp_*`, `n4m_aug_*`, `n4m_split_*`.

Required fix: update `build_methods.py` operator-page rendering to source ABI-2 C symbols from `catalog/methods.yaml` / `_rename_map.tsv`, regenerate/clean `docs/methods`, and extend doc-lint to catch legacy wildcard prefixes or require rendered public C ABI names to be ABI-2.

Other checks:
- Exact old `_rename_map.tsv` column-1 symbols in `docs/methods/*.md`: **0**.
- `boosting_pls.md` correctly renders `n4m_ensemble_boosting_pls_fit`.
- AOM/selector pages checked: AOM and CARS are ABI-2; `*_result_t` names like `n4m_aom_global_result_t` are legitimate type names, not false positives.
- Web: **53 unique old symbols remapped**, with unchanged shared shims `n4m_wasm_pls_predict_from_coeffs` and `n4m_wasm_model_predict_from_coeffs`.
- Lite: Python floors/imports and Rust dynamic loads/error strings are ABI-2; shared infra remains unchanged; `compat/upstreams.toml` unchanged.

**B. Must-Pass Gates**

Methods `feat/namespace-ml-abi2`:
- `python docs/_extras/build_methods.py --strict`
- `python -m pytest docs/_extras/test_methods_doc_lint.py -q` after strengthening lint for wildcard prefixes
- `sphinx-build -b html docs docs/_build/html --keep-going`
- `python catalog/scripts/selftest.py && python catalog/scripts/validate.py`
- ABI/export-symbol checks for Linux/macOS/Windows snapshots
- `make build PRESET=dev-debug && make test PRESET=dev-debug`
- Python/R/JS/Matlab binding gates relevant to the ABI-2 namespace branch

Web `feat/n4m-abi2-namespace`:
- `npm run validate:catalog`
- `npm run typecheck`
- `npm run test`
- `npm run build`
- `npm run build:single`
- Browser/WASM smoke against ABI-2 staged methods artifacts

Lite `feat/n4m-abi2-namespace`:
- `make test`
- Rust fmt/clippy/test
- Python unittest discovery with `nirs4all-methods>=0.99.0`
- WASM tests with ABI-2 methods dist
- Strict Rust parity against `libn4m.so.2.0.0`
- R CMD check and Octave/MATLAB smoke where CI has those toolchains

**C. Remaining Risk**

The main risk is docs drift: exact ABI-1 function names are blocked, but stale wildcard ABI-1 prefixes still survive and the operator-page generator points at the removed `n4m/sklearn` source. Web and lite remaining risks are runtime/environmental: staged WASM exports must match ABI-2, and lite Rust users can still point at an old lib unless CI and release packaging enforce `libn4m.so.2.0.0`.