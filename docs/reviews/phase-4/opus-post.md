# Phase 4 — Opus independent post-review transcript

**Date**: 2026-05-18
**Reviewer**: Claude Opus 4.7 (via `feature-dev:code-reviewer` agent)
**Scope**: Phase 4 — 5 stateless derivative / smoothing preprocessings (SavitzkyGolay, FirstDerivative, SecondDerivative, NorrisWilliams, Gaussian) + shared `padding.{c,h}` boundary index resolver
**Verdict**: **ACCEPT** — no high-confidence blocking issues.

## Summary

Reviewer cross-referenced scipy 1.11.4 source for `savgol_filter`, `_fit_edge_polyfit`, `gaussian_filter1d`, `_gaussian_kernel1d`, `np.gradient`, plus nirs4all's `norris_williams.py`. All algorithm semantics verified:

- **SG Vandermonde-QR pseudo-inverse** mathematically equivalent to scipy's min-norm SVD-lstsq (`c = V (V^T V)^{-1} y`). 2-ULP gap vs scipy is the structural QR-vs-SVD floor, absorbed by 1e-9 / 1e-10 tolerance.
- **SG `interp` mode** correctly replicates scipy's `_fit_edge` polynomial edge fit byte-for-byte.
- **SG `deriv > polyorder` short-circuit** matches scipy's early-return.
- **SG sign-flip** correctly handled via `correlate1d(x, weights[::-1])` semantics (kernel reversed before applying).
- **FirstDerivative `edge_order=2`** uses exact integer coefficients `(-3, 4, -1)` and `(3, -4, 1)` matching `np.gradient`.
- **SecondDerivative** composes via two FirstDerivative passes (NOT analytical formula), matching `np.gradient ∘ np.gradient`.
- **NorrisWilliams** correctly implements nirs4all's centred-MA + gap-diff (NOT SG-smoothing as the brief incorrectly stated). Implementer's deviation from brief was the right call.
- **Gaussian Hermite recursion** matches scipy's `_gaussian_kernel1d` exactly (D matrix on superdiagonal, P on subdiagonal, `order` iterations, polynomial evaluation, kernel reversal for correlation semantics).
- **Padding helper** correctly distinguishes scipy.signal's `mirror` (no edge repeat, period 2(n-1)) from scipy.ndimage's `reflect` (edge repeat, period 2n). Branch-free Euclidean modulus.

## High-confidence issues
**None.**

## Medium-confidence issues
1. **SG `cols < window_length` rejected for ALL modes** (scipy only rejects for `interp`) — minor parity divergence, documented contract is consistent. Add a docs note.
2. **`interp` mode duplicates Vandermonde-QR work per row** — performance, not correctness. Precompute edge QR factors at `_create` time when `mode == interp`. Defer to Phase 5+.

## Low-priority / Phase 5+ concerns
3. **Floating-point grouping in Gaussian kernel** — `exp(-0.5 / sigma2 * xi * xi)` vs scipy's `exp(-0.5 / sigma2 * x**2)` differ by ~1 ULP per entry. Absorbed by tolerance. Cosmetic fix.
4. **JSON parser duplication across 3 test files** — 270 LOC × 3. Extract to `cpp/tests/fixture_parser.hpp` at Phase 5 opener.
5. **SG / Gaussian fixture sweeps could include smaller windows / tighter sigmas** — current sweeps (22 SG, 26 Gaussian) are reasonable but could add `w=3`, `polyorder=0`, `sigma=0.3` corner cases.

## Things done well
- Engine layering uniform across all 5 ops (internal C state + extern "C" wrapper).
- Opaque-handle pattern + ABI naming convention preserved.
- Memory ownership: per-row scratch buffer pattern, in-place transform support (`X == out` aliasing works).
- Empty inputs short-circuit before allocation.
- 5 boundary modes implemented for both SG and Gaussian with scipy-exact semantics.
- ABI symbol count 79 → 94 exact, version-script + visibility-hidden + clean engines = 0 leak.
- Acknowledgement comments document deferred refactors (QR extraction, JSON parser) at the right scope.

## Post-implementation state
- Build: 44/44 targets clean, 0 warnings.
- Tests: 61/61 (7 phase-0 + 13 phase-1 + 14 phase-2 + 17 phase-3 + 10 phase-4).
- CLI: `chemometrics4all 0.1.0+abi.1.4.0 (ABI 1.4.0)`.
- ABI: 94 c4a_* symbols exported; 0 non-c4a leaks.
- c4a.h: 831 lines (after Phase 4's §10 banner).
