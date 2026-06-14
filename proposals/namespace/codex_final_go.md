**(A) Overall**

GO for the whole migration program.

The previous sole NO-GO is resolved. I inspected `git diff main -- docs/_extras docs/methods` and verified:

- [build_methods.py](/home/delete/nirs4all/nirs4all-methods/docs/_extras/build_methods.py:3324) derives `load_prefix_old_to_new()` from `proposals/namespace/_rename_map.tsv`.
- [render_operator_page()](/home/delete/nirs4all/nirs4all-methods/docs/_extras/build_methods.py:3947) applies that map before publishing `<prefix>_*`.
- The post-render in-place pass at [build_methods.py:4254](/home/delete/nirs4all/nirs4all-methods/docs/_extras/build_methods.py:4254) rewrites wildcard prefixes and exact ABI-1 symbols across method pages.
- The lint at [build_methods.py:4026](/home/delete/nirs4all/nirs4all-methods/docs/_extras/build_methods.py:4026) and [test_methods_doc_lint.py](/home/delete/nirs4all/nirs4all-methods/docs/_extras/test_methods_doc_lint.py:40) catches exact old symbols, per-method old prefixes, and coarse `n4m_pp_`, `n4m_aug_`, `n4m_split_`, `n4m_aom_` wildcard families, while allowing `_t` types.
- Spot checks pass: `split_kennard_stone` now has `n4m_model_selection_kennard_stone_*`, `pp_snv` has `n4m_transform_snv_*`, and `aug_mixup` has `n4m_augmentation_mixup_*`.
- Direct scans found `0` exact old symbols from the 566-entry rename map and `0` forbidden legacy prefix/wildcard hits excluding `_t` types. `lint_generated_pages()` returns no findings.
- The generated operator wildcard pages count is `118`.

Non-blocking note: `docs/methods/filter_x_outlier.md` still mentions `n4m_filter_x_outlier_method_t`; that is a public `_t` type in the ABI-2 header, not an exported old function or wildcard prefix. It does not change the GO.

No files were modified.

**(B) Merge Gates**

For `nirs4all-methods feat/namespace-ml-abi2`:

- C++ CI matrix: all `.github/workflows/ci.yml` build/test presets across Linux GCC/Clang, macOS, Windows MSVC/MinGW, plus FITPACK.
- ABI surface: `.github/workflows/abi-check.yml` Linux/macOS/Windows symbol snapshots, all exports `n4m_`, Linux SONAME `libn4m.so.2`, no RPATH/RUNPATH, no forbidden runtime deps.
- Catalog: `python catalog/scripts/validate.py --strict-abi --check-references`, split catalog check, catalog self-test.
- Docs: `python3 docs/_extras/build_methods.py --strict`, `python -m pytest docs/_extras/test_methods_doc_lint.py -q`, Sphinx HTML build.
- Parity/cross-binding: parity contracts, fixture determinism, C++ parity tests, Python smoke, donor/reference lockfile, sklearn oracle gate, AOM/POP self-consistency, golden snapshots, JS/WASM, Octave MEX, R binding.
- Sanitizers: ASan, UBSan, TSan, ASan+UBSan.
- Version sync: `scripts/bump_version.sh --check`.
- Env-bound/manual before release or if touched: CUDA workflow, MATLAB licensed checks, release packaging workflows.

For `web/lite feat/n4m-abi2-namespace`:

- `nirs4all-lite`: CI rust fmt/clippy/test/package, Python unittest/build/twine on 3.11/3.12, npm WASM test/pack, strict methods parity, R CMD build/check, Octave/MATLAB smoke.
- `nirs4all-web/studio-lite`: `npm ci`, `npm run check:lite-shim`, `npm run typecheck`, `npm run test`, `npm run validate:catalog`, `npm run build`, `npm run build:single`.
- Browser/WASM smoke where applicable: preview server plus `SMOKE_URL=http://localhost:4173/ node tests/smoke.mjs`.
- Pages deploy build must pass: `.github/workflows/deploy-pages.yml` runs `npm ci && npm run build`.

Merge order: methods first, then web/lite pinned to the merged ABI-2 methods surface/artifacts.