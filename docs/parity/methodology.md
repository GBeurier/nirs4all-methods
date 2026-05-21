# Parity methodology

pls4all uses three related but distinct parity surfaces. They should not
be collapsed into a single "green/red" result.

## Fixture parity

Fixture parity is the C++ correctness gate. JSON fixtures under
`parity/fixtures` are generated from pinned Python/R/nirs4all references
and compiled into the C++ tests. `ctest` must pass before any numerical
change lands.

The schema lives at `parity/schema/fixture_schema_v1.md`; the generator
and lock files live under `parity/python_generator`.

Current caveat: AOM/POP fixture generation still depends on the historical
`AOM_v0` bench oracle path. Until that reference is vendored or resolved
from a pinned package, a clean clone can fail `generate-fixtures --check`
before reaching the comparison step.

## Reference parity

Reference parity compares pls4all and external implementations with the
canonical library selected for each method in
`benchmarks/parity_timing/registry.py`. It answers whether the method
implementation behaves like the literature or established package oracle.

External libraries are included in this comparison. If sklearn, R `pls`,
libPLS or another external implementation diverges from the canonical
oracle, the dashboard should show that divergence explicitly.

Reference parity has three documented contract classes:

- **strict**: numerical equivalence is expected and required by the release
  gate (`rmse_rel <= 1e-3`, with `<= 1e-6` displayed as exact).
- **variant**: the C++ method is useful but intentionally differs from the
  external package contract. The matching package-compatible behavior should
  be exposed as a named variant and tested separately.
- **non-release diagnostic**: the executable oracle only validates package
  availability, output shape, mask overlap, or algorithm-family behavior.
  These rows remain visible as divergent under the strict release gate unless
  the method is explicitly excluded from release parity.

Wide `MethodSpec.rmse_rel_tol` values are diagnostic smoke thresholds, not a
way to turn a strict dashboard divergence into a validated release parity.

## Binding parity

Binding parity compares pls4all language bindings against the native C++
row for the same method and data. It answers whether Python, R, MATLAB and
future bindings are thin and faithful wrappers over the C ABI.

External libraries are not bindings and should be marked not applicable
for this gate.

## Release rule

A release-quality run needs:

- fixture parity green in CI;
- binding parity green for every shipped pls4all binding row;
- reference parity green under the strict release gate, or an explicit
  non-release/variant note explaining why strict parity is not the current
  contract;
- missing external references classified as `paper_only` or
  `not_available`, not hidden as infrastructure errors;
- tolerances documented strongly enough that a wide selector mask
  threshold is visible as divergent diagnostic evidence, not exact agreement.
