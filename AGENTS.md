# Methods contributor instructions

## Ownership and architecture

- Methods owns numerical implementations, including existing duplicated production numerics, filters/augmentation and splitter algorithms.
- Bindings adapt generic native fit/transform/predict/export roles, never maintain per-binding algorithm implementations/lists.
- [DAG](../dag-ml/AGENTS.md) owns execution, variants, fold scheduling, selection and scoring context; [DMD](../dag-ml-data/AGENTS.md) owns data/identity constraints.
- Preserve the specifically accepted generic identity-aligned conformal exception in DAG; it is not blanket numerical ownership.
- C++17 is internal; public C11-compatible `n4m_*` headers contain no STL/Eigen/smart pointers or escaping exceptions. C API wrappers translate exceptions to status. Zero mandatory runtime deps; accelerators/tooling are optional/test-only.
- Never free across languages: caller buffers or core-owned arrays released by the owner helper. Context error strings are invalidated by the next call. Matrix views carry explicit row/column strides; do not impose a boundary copy.
- Closed-library model: methods/bindings/references/subsets enter via repository changes, not runtime plugins or bypasses of the public C ABI. `pls4all` is a packaging subset, not a second numerical engine.
- Python uses ctypes.CDLL, releases the GIL during calls, and needs one Context per thread; do not share model/context handles across threads without the ABI's explicit contract. Avoid workers-times-threads oversubscription.
- Start with [architecture](docs/ARCHITECTURE.md), [contributing](CONTRIBUTING.md), [testing](docs/dev/testing.md), [workflow](docs/dev/workflow.md) and the owned [operations note](docs/dev/agent_notes.md).
- Sibling links are optional navigation.

- `docs/ARCHITECTURE.md` overrides this file when present, under current user scope and safety limits.
- Methods supplies metric primitives; DAG owns metric definitions and scoring context. Shared fixtures qualify named profiles, not universal equivalence.

## Build and numerical checks

- Use CMake presets, never hand-rolled build flags/directories.
- `make bootstrap`/`make doctor` support setup; `make build PRESET=dev-debug` and `make test PRESET=dev-debug` wrap presets.
- Optional production BLAS/OpenMP, CUDA and cross-target variants remain in operations.
- Run CTest/doctest and relevant Python/R/Octave binding tests; numerical changes preserve/update justified parity fixtures/tolerances.
- Full registry sweeps, thread/reference settings and overnight commands remain in operations; binding parity is distinct from reference parity.
- Keep C++ clang-format/clang-tidy (4 spaces/100 columns/no tabs), typed Ruff-formatted Python, base R binding/tidyverse reference style and MATLAB/Octave intersection.
- Use matching issue/PR templates and scoped Conventional Commits with exact parity/ABI checks.

## Release / ABI gates

Three checks are **release blockers**; CI enforces them and any change that touches them needs the snapshots regenerated in the same PR:

```bash
# Version sync (header → all downstream manifests)
scripts/bump_version.sh --check

# ABI symbol surface — must match exactly. If a new public n4m_* symbol is
# intentional, regenerate the snapshot AND document in docs/abi/changes_log.md.
nm -D --defined-only build/dev-release/cpp/src/libn4m.so.<ver> \
  | awk '{print $3}' | sort -u \
  | diff -u cpp/abi/expected_symbols_linux.txt -

# SONAME / linkage sanity
readelf -d build/dev-release/cpp/src/libn4m.so.<ver> \
  | grep -E 'SONAME|NEEDED|RPATH|RUNPATH'
```

To regenerate the per-platform ABI snapshots after an intentional surface change, use `scripts/regen_abi_snapshots.sh`.

ABI versioning (`N4M_ABI_VERSION_*` in `cpp/include/n4m/n4m_version.h`) is **independent of project version** and bumps only on surface change. `n4m_check_abi_compatibility(header_major, header_minor)` is how bindings detect header/runtime skew.


## Working rules

- Read applicable local and deeper instructions before editing. Preserve dirty/concurrent work; make authorized owner fixes and propagate affected contracts/bindings/consumers. Do not reset another author's work.
- Use CodeGraph FIRST for implementation navigation only if this checkout's root already has `.codegraph/`; use its exact projectPath or run the CLI here. Never auto-index or silently substitute an ancestor index.
- Unindexed code and documentation use ordinary source navigation. Explicit source, tests and build tools are final authority. Owner links are navigation; this policy works without a parent workspace.
- All application/project content is English: UI text/subtitles/tooltips/errors, docs/comments/docstrings, test/fixture prose, reports/issues/PRs, commits and generated material. Preserve identifiers, quotations and historical/provenance meaning; converse with the maintainer in French.
- Do not commit credentials, private data, generated workspaces, caches or build outputs. Follow local licensing/fixture restrictions and report actual commands/exits/skips; a command reference grants no execution/publication permission.
