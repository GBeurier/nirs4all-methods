<!-- .github/PULL_REQUEST_TEMPLATE/method-add.md -->

## New method: <method_id>

Closes #<issue_number> (mandatory if a `method-request.yml` issue exists)

> **How adding a method actually works.** A method's facts (numerical code, ABI
> symbols, parity reference + tolerance, binding wrappers, catalog metadata) are
> **hand-edited across ~6 surfaces** — there is no single-source catalog that
> generates the rest. Edit each surface below and keep them consistent. The
> catalog is a *manifest* (ABI symbols, TUs, publications), not the source of
> truth; the **registry** (`benchmarks/parity_timing/registry.py`) is what
> actually drives parity + timing. Full walkthrough:
> [`CONTRIBUTING.md` → "Adding a new method"](../../CONTRIBUTING.md).

### 1. C++ core + C ABI

- [ ] Numerical core under `cpp/src/core/<category>/<method_id>.{c,cpp,h}`
      (C++17 internal; no STL/Eigen/exceptions cross the boundary)
- [ ] `extern "C"` wrapper in `cpp/src/c_api/c_api_<category>.cpp`, or — for a
      `MethodResult`-style op — a case in the registry dispatcher
      `cpp/src/c_api/c_api_method_result.cpp`
- [ ] Public `N4M_API` declarations in the matching `cpp/include/n4m/<category>.h`
      (umbrella `n4m.h` already includes the per-category headers)
- [ ] Builds via the existing `CONFIGURE_DEPENDS` globs — no `n4m_targets.cmake`
      edit unless a new vendored/optional dependency is introduced

### 2. ABI surface (release blocker — CI fails closed)

- [ ] `cpp/abi/expected_symbols_{linux,macos,windows}.txt` regenerated for **all
      three** platforms (Linux locally with `nm -D --defined-only … | awk … | sort -u`;
      macOS/Windows via CI artifacts) and committed in this PR
- [ ] `docs/abi/changes_log.md` updated: each new `n4m_*` symbol + a one-line reason
- [ ] `since_abi` bumped only if the ABI surface changed (independent of project version)

### 3. Tests (C++ doctest + parity fixture)

- [ ] doctest cases in `cpp/tests/test_<category>.cpp` (cover create/apply,
      null/shape errors, **and that a non-trivial input actually changes** — not
      just determinism, which a no-op passes)
- [ ] If the method is deterministic-given-seed, freeze a self-fixture under
      `parity/fixtures/<id>_v1.json` (IEEE-754 big-endian hex) and replay it in
      the doctest with `assert_close` at 1e-12 abs / 1e-13 rel
- [ ] `ctest --preset dev-release --output-on-failure` green; single method:
      `./build/dev-release/cpp/tests/n4m_tests --test-case="*<method_id>*"`

### 4. Reference & parity (skip if genuinely paper-only)

- [ ] `MethodSpec` added to `benchmarks/parity_timing/registry.py` — this is where
      the external **reference**, `adapted_params`, and `rmse_rel_tol` live
- [ ] Reference is an installable, documented donor (see `parity/REFERENCES.md`);
      run `parity/scripts/check_references.py` to confirm it is resolvable
- [ ] Per-method parity verified: `python parity/scripts/per_method_parity.py
      <method_id> --reference <ref> --tol 1e-8` — paste the verdict below
- [ ] Tolerance is `1e-8 rmse_rel` (Gate B) unless a tighter/looser value is
      justified inline in the registry note
- [ ] Paper-only methods instead: self-consistency via
      `make parity-paper-only METHOD=<method_id>` (`parity/comparator/self_consistency.py`)

<details>
<summary>Per-method parity verdict</summary>

```
(paste `per_method_parity.py <method_id> …` output: Gate A max|Δ| per binding,
 Gate B rmse_rel + max_abs vs the reference, verdict)
```
</details>

### 5. Catalog manifest (metadata only)

- [ ] Method row added to `catalog/methods.yaml`, then split files regenerated:
      `python catalog/scripts/split_legacy_methods.py` → `catalog/methods/<id>.yaml`
- [ ] `abi_symbols`, `tu`, `headers`, `parity`, `publications` filled to match the code
- [ ] `python catalog/scripts/validate.py` green (`--strict-abi` reconciles symbols
      against `cpp/abi/expected_symbols_*.txt`)

### 6. Bindings

- [ ] Python: ABI-close function in `bindings/python/src/n4m/python.py` +
      argtypes in `bindings/python/src/n4m/_ffi_decls.py`; sklearn-style estimator
      in `bindings/python/src/n4m/sklearn/` if an idiomatic profile is wanted
- [ ] Wrappers are **hand-written** (`render_api.py` emits metadata only — it does
      not generate package APIs); other language bindings (R/JS/Octave/…) added only
      if the method is in scope for them
- [ ] Inter-binding parity green (`rmse_rel < 1e-12` vs the C++ core) — the
      cross-binding orchestrator (`benchmarks/cross_binding/orchestrator.py`) for
      dashboard methods, or the per-binding smoke for others

### 7. Docs & changelog

- [ ] `CHANGELOG.md` entry under the next unreleased section
- [ ] If on the benchmark dashboard: add the op to `benchmarks/cross_binding/donor_ops.py`
      so it is timed (Layer 2/3) and refresh `docs/_static/bench-data.json`

### Reviewer checklist

- [ ] All ~6 surfaces edited and mutually consistent (id / ABI symbol / params /
      tolerance / reference are re-typed in several places — verify they match)
- [ ] ABI snapshots regenerated for all three platforms + `changes_log.md` updated
- [ ] A non-trivial input is shown to change (no silent no-op locked as "passing")
- [ ] Parity verdict attached and within the declared tolerance
- [ ] No method facts duplicated into a *new* place that isn't one of the known surfaces
