# Development — Testing

This page is the operational checklist for the current pls4all test
surface. `CONTRIBUTING.md` describes policy; this page lists the commands
that maintainers should run before changing parity, bindings, dashboard
generation or release packaging.

## Core C++ gate

```bash
cmake --preset dev-release
cmake --build --preset dev-release --parallel
ctest --preset dev-release --output-on-failure
```

May 2026 audit result: `ctest --preset dev-release` passed locally
(`pls4all_tests`, 1/1).

## Fixture determinism

```bash
cd parity/python_generator
python -m pip install -r requirements-lock.txt -e .
generate-fixtures --check --out ../fixtures
```

Current blocker: AOM/POP fixture generation still searches for the
historical `AOM_v0` oracle under workstation paths. A clean clone without
that oracle fails before it can prove fixture determinism. The release gate
needs a pinned or vendored source for that oracle.

## Lockfile

```bash
python -m benchmarks.parity_timing.lockfile --check
```

May 2026 audit result: passed locally for 71 methods. This check is
structural; it does not prove the referenced libraries are importable or
that the live reference parity gate is green.

## Python binding tests

```bash
PYTHONPATH=bindings/python/src \
python -m pytest bindings/python/tests -q
```

May 2026 audit result: one failure in
`test_sklearn_selectors.py::test_selector_in_pipeline[UVESelector]`.
The selector can choose zero features on the synthetic pipeline smoke
dataset, which then makes sklearn `PLSRegression` receive an `(n, 0)`
matrix. Either the estimator needs an explicit `min_features`/fallback
policy or the smoke test must use UVE parameters known to select at least
one feature.

The narrower sklearn parity script currently passed:

```bash
bindings/python/scripts/sklearn_parity_gate.sh
```

That script does not cover the full selector pipeline smoke, so it cannot
replace the full Python test suite.

## Cross-binding samples

Small reference-parity sample:

```bash
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pls pcr --registry-cells --threads 1 \
  --libp4a-build blas-omp --n-runs 1 \
  --canonical-pls4all-only --reference-backends registry \
  --timeout 180 --out-csv /tmp/pls4all_audit_cross_binding.csv --force
```

May 2026 audit result: `pls`, `pcr`, sklearn and R references were mostly
green, but an external `ikpls` row was marked as binding-parity failure
while still passing reference parity. That confirms the current reporting
bug: external libraries must not be evaluated by binding parity.

Slow-method smoke:

```bash
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pcr iriv_select vissa_select bve_select pso_select \
  --registry-cells --threads 1 --libp4a-build blas-omp --n-runs 1 \
  --only-pls4all --timeout 240 \
  --out-csv /tmp/pls4all_audit_slow_methods.csv --force
```

Use this only as a timing smoke. Because `--only-pls4all` omits external
oracle rows, reference parity is not evaluated in that run.

## Release preflight

```bash
scripts/bump_version.sh --check

nm -D --defined-only build/dev-release/cpp/src/libp4a.so.1.16.0 \
  | awk '{print $3}' | sort -u \
  | diff -u cpp/abi/expected_symbols_linux.txt -

readelf -d build/dev-release/cpp/src/libp4a.so.1.16.0 \
  | grep -E 'SONAME|NEEDED|RPATH|RUNPATH'
```

The ABI symbol snapshot must match intentionally. If a new `p4a_*`
symbol is public, update `cpp/abi/expected_symbols_linux.txt` and explain
the ABI change in `docs/abi/changes_log.md`.

May 2026 audit result: version sync passed, SONAME was `libp4a.so.1`,
but the Linux symbol diff failed because the current shared library
exports additional public `p4a_*` symbols not present in
`cpp/abi/expected_symbols_linux.txt`.

## Docs and dashboard

```bash
python -m json.tool docs/_static/bench-data.json >/dev/null
python docs/_extras/build_landing.py \
  --results benchmarks/cross_binding/results \
  --out /tmp/bench-data.json
python -m sphinx -b html docs docs/_build/html --keep-going
```

The dashboard page embeds benchmark JSON at Sphinx build time. Updating
only `docs/_static/bench-data.json` does not refresh an already-built
`index.html`.
