# Phase 3 — Opus independent post-review transcript

**Date**: 2026-05-18
**Reviewer**: Claude Opus 4.7 (via `feature-dev:code-reviewer` agent)
**Scope**: Phase 3 — 4 stateful scatter & scaling preprocessings (MSC, EMSC, Baseline column centering, Derivate) + introduction of the stateful `_fit/_transform` ABI contract + cleanup of pls4all-inherited orphan declarations from `c4a.h`
**Verdict**: ACCEPT with 3 recommended follow-ups — all applied.

## Summary

Reviewer cross-checked algorithm semantics against nirs4all source (`/home/delete/nirs4all/nirs4all/nirs4all/operators/transforms/{nirs,signal,scalers}.py`) and verified:

- **MSC reference axis** is per-row mean (length n samples) — matches nirs4all `np.mean(tmp_x, axis=1)`. The Phase 3 brief incorrectly stated `axis=0` in one place; implementer correctly followed nirs4all.
- **EMSC basis order** is `[reference, w^1, ..., w^d]` with `c[0]` = reference coefficient — matches nirs4all exactly.
- **EMSC Householder QR** correctly implements norm, reflection-with-vk0-normalization, apply_qt, and back-substitution. Choice of QR over Cholesky-on-normal-equations is justified by ill-conditioned raw-integer wavelength basis.
- **Refit memory safety**: all four operators allocate new buffers first, free old buffers only on success-path commit. Failed allocs leave the prior fit intact.
- **`-fvisibility=hidden` + version script + `expected_symbols_*.txt`** triple-defense: 79-symbol count exact, no internal `_state_*` engine functions leaked.
- **c4a.h trim** (2208 → 680 lines): all orphan §7-§15 PLS declarations removed, no dangling references, single header guard, all 5 `#endif`s account for.

## High-confidence issues

**None.** Reviewer searched aggressively for axis bugs, basis-order bugs, memory leaks on refit, fitted-flag-before-success bugs, and ABI leakage.

## Medium-confidence issues — all addressed

### #1 No refit test exists — FIXED

The brief and `c4a.h` §5 mandate sklearn-style refit semantics ("calling `_fit` again replaces prior state"). The engines implement this correctly via source inspection, but no test exercised the path.

**Resolution**: Added 3 tests (`pp_msc_refit`, `pp_emsc_refit`, `pp_baseline_refit`) that fit on `X1`, transform `X3`, fit again on `X2`, transform `X3` again — assert outputs differ (different fit state) or match algebraically (Baseline subtraction with known means). All pass. Total: 48 → **51 tests**.

### #2 Derivate semantically diverges from nirs4all's `Derivate` class — DEFERRED + DOCUMENTED

Real divergence: nirs4all's `Derivate` uses `np.gradient(X, delta, axis=0)` (sample-axis, shape-preserving) while c4a uses `np.diff(X, n=order, axis=1) / delta^order` (wavelength-axis, column-shrinking). c4a's choice matches nirs4all's own `FirstDerivative`/`SecondDerivative` which use axis=1 (the spectroscopically correct convention).

**Resolution**: Added explicit entry to `docs/reviews/DEFERRALS.md` flagging this gap. Users porting pipelines from nirs4all should map `Derivate(order=1)` calls to `FirstDerivative`/`SecondDerivative` semantics. A future Phase 4 may ship `c4a_pp_gradient_*` for full nirs4all `Derivate` parity if needed.

### #3 EMSC uses `* inv_c0` instead of `/ c[0]` — FIXED

The `1.0 / c[0]` precomputation introduced an avoidable ~1-ULP per element vs nirs4all's true `/ coeffs[0]`. The Derivate engine explicitly chose true division for byte-exact match; EMSC should follow the same discipline.

**Resolution**: Switched `orow[j] = (xrow[j] - poly_val) * inv_c0` to `orow[j] = (xrow[j] - poly_val) / c0` in `emsc.c:307`. Comment added documenting the convention. Tests still pass within the 5e-10 abs / 5e-10 rel EMSC tolerance.

## Low-priority deferrals

- **#4** `scale=True` flavour of MSC/EMSC (nirs4all default, but rarely used in practice) — tracked in DEFERRALS as Phase TBD.
- **#5** EMSC exact-zero check on `c[0]` could miss near-singular cases (e.g., `1e-18`) — defer to Phase 5+ numerical robustness sweep.
- **#6** No `c4a_pp_emsc_inverse_transform` — intentional (polynomial subtraction is data-driven per row).
- **#7** No `c4a_pp_derivate_is_fitted` — intentional (stateless lifecycle).
- **#8** `PHASES.md` Phase 3 row has empty verdict columns — gets populated on commit.

## Things done well

- Per-row vs per-column mean axis correctly matched to each operator (MSC = per-row, EMSC = per-column) despite the brief's typo.
- Householder QR correctly implemented (norm, reflection, apply_qt, back-sub).
- Refit memory pattern: alloc-first-then-free-old is consistent across all 4 operators.
- EMSC basis order `[ref, w^1, ..., w^d]` matches nirs4all (`c[0]` = ref).
- `c4a.h` trim is clean: 2208 → 680 lines, no dangling references.
- 79-symbol ABI exact, 0 leakage of internal engines.
- DEFERRALS / PHASES cross-phase tracking artefacts in place.

## Post-fix verification

```
cmake --build --preset dev-debug                            → 7/7 incremental, zero warnings
./build/dev-debug/cpp/cli/chemometrics4all_cli --selfcheck  → OK
./build/dev-debug/cpp/tests/chemometrics4all_tests          → 51/51 passing (48 + 3 refit)
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.3.0 | awk '$2=="T" {print $3}' | wc -l → 79
wc -l cpp/include/chemometrics4all/c4a.h                    → 680
```
