# Repository Guidelines

## Project Structure & Module Organization

`cpp/` contains the C++17 engine: public C ABI headers in `cpp/include/n4m/`,
internal implementation in `cpp/src/core/`, ABI wrappers in `cpp/src/c_api/`,
tests in `cpp/tests/`, and exported-symbol snapshots in `cpp/abi/`.
`bindings/` holds language bindings; Python sources live under
`bindings/python/src/{n4m,pls4all}` with tests in `bindings/python/tests/`.
`catalog/` is the method and subset metadata source, with schemas and scripts.
Use `docs/`, `parity/`, `benchmarks/`, and `.github/` for documentation,
fixtures, performance/parity tooling, and issue/PR templates.

## Build, Test, and Development Commands

- `make bootstrap`: enter the supported dev setup, preferring the devcontainer.
- `make doctor`: verify required tools and paths.
- `make build PRESET=dev-debug`: configure and build through CMake presets.
- `make test PRESET=dev-debug`: run CTest for that preset.
- `cmake --preset blas-omp && cmake --build --preset blas-omp -j`: production
  build with BLAS and OpenMP.
- `PYTHONPATH=bindings/python/src python -m pytest bindings/python/tests -q`:
  run Python binding tests.
- `python catalog/scripts/selftest.py && python catalog/scripts/validate.py`:
  run the Phase B catalog parser and structural checks.
- Full registry parity sweep with 8 native/sklearn jobs per cell:
  `BENCH_SKLEARN_N_JOBS=8 python benchmarks/cross_binding/orchestrator.py --algorithms all --registry-cells --threads 8 --workers 1 --libn4m-build blas-omp --n-runs 2 --canonical-pls4all-only --reference-backends registry --timeout 180 --out-csv /tmp/n4m_full_parity.csv --force --flush-each-cell`.

## Coding Style & Naming Conventions

Use CMake presets; do not hand-roll build directories or flags. C++ is C++17
with LLVM-derived `clang-format` settings: 4-space indentation, 100-column
limit, no tabs. Public C headers must remain C11-compatible, `extern "C"` clean,
and free of STL types or exceptions across the ABI. Exported C symbols use the
`n4m_*` prefix. Internal names follow `.clang-tidy`: lower_case functions and
variables, CamelCase types, UPPER_CASE macros. Python code should be type-hinted
and formatted/linted with `ruff format` and `ruff check`.

## Testing Guidelines

C++ tests use doctest behind CTest. Run `ctest --preset dev-release
--output-on-failure` before broad changes, and use
`./build/dev-release/cpp/tests/n4m_tests --test-case="*aom*"` for focused
filters. Keep test files named `test_<area>.cpp` or `test_<feature>.py`.
Numerical changes must update or preserve parity fixtures and tolerances.
Binding changes should include the relevant language test, for example
`Rscript bindings/r/test_parity.R` for R parity.

## Commit & Pull Request Guidelines

Recent history mixes Conventional Commits (`docs:`, `fix:`, `test:`) with
phase and fixup commits. Prefer Conventional Commits with scopes when useful,
such as `fix(abi): preserve symbol order`. PRs should use the matching template
under `.github/PULL_REQUEST_TEMPLATE/`, link the issue template request when
applicable, describe parity/ABI impact, and list the exact commands run.

## Agent-Specific Instructions

Respect the closed-library model from `CONTRIBUTING.md`: new methods, bindings,
references, and packaging subsets are added only through repository changes.
Do not add runtime plugin hooks or bypass the public C ABI for bindings.
During the Phase B catalog transition, edit `catalog/methods.yaml` and
regenerate `catalog/methods/<id>.yaml` with `split_legacy_methods.py` until the
legacy YAML is retired. Python bindings use `ctypes.CDLL` and release the GIL
during native calls, but `n4m.Context` / `pls4all.Context` are not thread-safe;
create one context per thread.
