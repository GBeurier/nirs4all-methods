# Phase 2 — Opus independent post-review transcript

**Date**: 2026-05-18
**Reviewer**: Claude Opus 4.7 (via `feature-dev:code-reviewer` agent)
**Scope**: Phase 2 — 7 stateless scatter & scaling preprocessings (SNV, LocalSNV, RobustSNV, AreaNormalization, Normalize, SimpleScale, LogTransform)
**Verdict**: REVISE → high-confidence fixes applied; medium/low-priority items tracked.

## Summary

Reviewer praised the bit-exact NumPy parity discipline (true division vs reciprocal-multiplication, pair-sum trapz form, Lomuto+median-of-three quickselect with `hi-lo==1` base case) and the consistent 2-layer C engine + C ABI veneer pattern. Three high-confidence issues identified:

## High-confidence issues — all addressed

| # | Issue | Resolution |
|---|-------|------------|
| 1 | LogTransform misclassified as stateless; recomputes offset on every transform | **Documented** the per-call recomputation behaviour in c4a.h (`auto_offset` semantics now explicit). Proper `_fit/_transform` split deferred to Phase 3 when stateful-operator ABI lands. |
| 2 | Public ABI missing enum constants (`c4a_pp_lsnv_pad_mode_t`, `c4a_pp_area_method_t` only declared in internal headers) | **Hoisted** both enums into `cpp/include/chemometrics4all/c4a.h` with proper enum tags `C4A_PP_LSNV_PAD_{REFLECT,EDGE,CONSTANT}` and `C4A_PP_AREA_{SUM,ABS_SUM,TRAPZ}`. Internal headers now reference the public declaration. |
| 3 | SNV flat-row threshold `s >= 1e-15` diverges from nirs4all's `std[std==0] = 1.0` | **Fixed**: changed to `s != 0.0` to match nirs4all exactly. "Bit-exact" claim now genuine. |

## Medium-confidence — deferred with tracking

- **#4** Missing parity coverage for `(with_mean=False, with_std=False)` SNV identity case, flat-row case, `cols=1` LSNV case — tracked for Phase 3 opener (fixture sweep extension).
- **#5** AreaNormalization degenerate row lengths — covered by existing guard, parity-tested implicitly. Add explicit `trapz_single_col` fixture in Phase 3.
- **#6** LogTransform doc claimed "fit-time" semantics but engine is per-call — addressed via #1 above.
- **#7** `c4a_pp_log_min_value <= 0` validation lives only in C wrapper, not internal `_state_new` — symmetry fix deferred to Phase 3.

## Low-priority deferrals

- **#8** Symbol export filter — addressed via Codex R5 below (`c4a_linux.map` added).
- **#9** `c4a_array_t` orphan declarations — Codex recommends delete; will be handled in Phase 3 opener.
- **#10** Fixture generator uses absolute path `NIRS4ALL_ROOT = /home/delete/...` — addressed via Codex R2 (parity pyproject.toml).
- **#11** Smoke tests don't assert numerical output — tracked.
- **#12** LSNV reflect padding on `cols=1` — divergence documented; tracked for Phase 3.

## Things done well

- Bit-exact NumPy parity discipline (true division, pair-sum trapz).
- Lomuto+median-of-three quickselect correctness.
- Matrix-view validation pattern consistently applied across all 21 transforms.
- 2-layer C engine + C ABI veneer cleanly applied.
- Cumsum-based moving statistics in LSNV byte-exact vs NumPy.
