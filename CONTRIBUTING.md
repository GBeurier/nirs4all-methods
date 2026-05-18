# Contributing to chemometrics4all

Thanks for considering a contribution. chemometrics4all is built **phase by phase** (see [ROADMAP.md](ROADMAP.md)) with a strict review cadence. The canonical design spec is [ARCHITECTURE.md](ARCHITECTURE.md). Read both before opening a PR.

## Prerequisites

- C++17 toolchain: GCC ≥ 11, Clang ≥ 14, or MSVC 2022
- CMake ≥ 3.21
- Ninja (recommended)
- Python ≥ 3.10 (for the parity generator and Python binding, from Phase 1 onward)
- R ≥ 4.3 (optional, only for parity cross-checks against prospectr / mdatools)

## Workflow

1. **Pick an open phase** in [ROADMAP.md](ROADMAP.md). The phase brief lives at `roadmap/phase-N-<name>.md`.
2. **Codex pre-review** of the brief: the brief is reviewed by Codex *before* any code is written. Transcript at `docs/reviews/phase-N/codex-pre.md`. If the brief needs changes, edit it first and re-review.
3. **Implementation** in small atomic commits, each keeping CI green. Where multiple operators in a phase are independent, parallel agent work is encouraged.
4. **Codex post-review** of the implementation diff: transcript at `docs/reviews/phase-N/codex-post.md`. Apply Codex's corrections.
5. **Opus independent post-review**: an opus-class model runs an adversarial review independent of Codex. Transcript at `docs/reviews/phase-N/opus-post.md`. If Codex and Opus disagree, the more conservative reviewer wins.
6. **Push + user update**: commit the final state, push to `main`, verify CI green on GitHub, post a 3-5 line user update.

## Ground rules

1. The canonical spec is [ARCHITECTURE.md](ARCHITECTURE.md). Any violation must be discussed in a Codex review before being merged.
2. **Parity gates are non-negotiable.** Any numerical change must regenerate the relevant `parity/fixtures/*.json` and keep `parity/tolerances.md` green. If a fixture diverges beyond its tolerance row, the PR fails.
3. **ABI surface is monitored.** Any change to the `c4a_*` symbol set must regenerate `cpp/abi/expected_symbols_{linux,macos,windows}.txt` and add an entry to `docs/abi/changes_log.md` with rationale.
4. **Versioning is dual.** Project semver lives in `C4A_PROJECT_VERSION_*` and ABI semver in `C4A_ABI_VERSION_*`, both in `cpp/include/chemometrics4all/c4a_version.h`. Run `scripts/bump_version.sh` to propagate to all manifests; never edit them by hand.

## Adding a new preprocessing, augmentation, splitter, or filter

Each operator follows the same recipe:

1. Implement in `cpp/src/core/<category>/<name>.{hpp,cpp}` with the canonical namespace `chemometrics4all::core::<category>`.
2. Add the C ABI wrappers in `cpp/src/c_api/c_api_<category>.cpp` and the declarations in `cpp/include/chemometrics4all/c4a.h`.
3. Add the operator's symbols to `cpp/abi/expected_symbols_*.txt` (re-run `nm -D --defined-only libc4a.so | awk '$2=="T" {print $3}' | sort -u`).
4. Write a parity-fixture generator in `parity/python_generator/src/chemometrics4all_parity/adapters/<name>.py` using the pinned Python reference (numpy 1.26.4, scipy 1.11.4, pywt 1.6.0, pybaselines 1.1.4, prospectr 0.2.7).
5. Commit the generated fixture to `parity/fixtures/<id>_v1.json` and add the tolerance row to `parity/tolerances.md`.
6. Write a doctest-style smoke + parity test in `cpp/tests/test_<name>.cpp` and append it to `cpp/tests/CMakeLists.txt`.
7. Add a Sphinx page at `docs/algorithms/<name>.md` describing the algorithm, references, and parity policy.
8. Add a sklearn-compatible Python wrapper in `bindings/python/src/chemometrics4all/sklearn/<category>.py` (added in Phase 22 onward; until then the operator is C-only).

## Commit messages

Conventional Commits style:

- `feat(snv): add stateless StandardNormalVariate operator`
- `fix(banded-solver): correct boundary handling in LDLT decomposition`
- `test(kennard-stone): add prospectr parity fixture`
- `docs(roadmap): update Phase 5 status`
- `chore(abi): regenerate expected_symbols after Phase 4`
- `refactor(core): extract pca_helper into common/`
- `build(cmake): enable BLAS backend preset`
- `ci(workflows): re-enable parity-gate.yml for Phase 2`

Each commit ends with the co-author trailer when authored by an agent:

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Code style

- `.clang-format` and `.clang-tidy` are the canonical sources. CI fails any drift.
- C++ docstrings: brief one-liner above the function; full Doxygen-style only on the public header `c4a.h`.
- Public C ABI symbols MUST be `extern "C"`, `C4A_API`-decorated, exception-translated at the boundary, and never allocate behind opaque handles without a matching `_destroy`.

## License

By contributing you agree that your contributions are licensed under CeCILL-2.1.
