# Phase 53 — VIP_SPA + DI-PLS auto-validation + IRIV unlock

Date: 2026-05-16
Tag: `phase-53-vip-spa`
Prereqs: phase-52-irf

## Goal

Add the last hybrid variable-selection method from `auswahl 0.9.0` that
pls4all didn't have (VIP_SPA), and close out the parity-audit by
unlocking the two `paper_only` methods that actually have installable
references (DI-PLS via diPLSlib; IRIV via Octave Forge `statistics`).

## 1. VIP_SPA — new pls4all method

Composition of:
1. **VIP** scoring (Favilla 2013) — fit PLS, compute Variable Importance
   in Projection from the (loadings, weights, scores) trio.
2. **VIP mask** — keep features with `VIP > vip_threshold`. Fallback to
   `argmax(VIP)` if no feature passes (matches `auswahl.VIP.get_support_for_threshold`).
3. **SPA greedy** (Araujo 2001) on the surviving subset — start at the
   masked feature with the highest VIP score, then greedily pick the
   feature whose projection onto the orthonormal complement of the
   already-selected directions has the largest residual norm.

### Files touched

```
cpp/src/core/vip_spa_selection.{hpp,cpp}     new (header ~50 LOC, kernel ~330 LOC)
cpp/src/c_api/c_api_method_result.cpp        added p4a_vip_spa_select wrapper
cpp/include/pls4all/p4a.h                    added decl in §19 (top of bottom block)
cpp/src/CMakeLists.txt                       added source
cpp/abi/expected_symbols_linux.txt           added p4a_vip_spa_select
cpp/include/pls4all/p4a_version.h            ABI 1.15 → 1.16
cpp/src/core/version.cpp                     "abi.1.15.0" → "abi.1.16.0"
cpp/tests/abi/test_version.cpp               "abi.1.15.0" → "abi.1.16.0"
bindings/python/src/pls4all/_methods.py      vip_spa_select Python wrapper
bindings/python/src/pls4all/__init__.py      re-export
benchmarks/parity_timing/registry.py         _vip_spa_select_pls4all + _VipSpaAuswahlReference + MethodSpec
```

### Parity

Reference: `auswahl.VIP_SPA` (LSX-UniWue, GitHub).
- Same VIP scoring formula and hard-coded `VIP > 0.3` mask.
- Same SPA greedy successive-projections kernel applied to raw masked X.
- pls4all uses `argmax(VIP)` as the SPA start; auswahl enumerates every
  candidate seed and picks the CV-best (which would need extra refits).
  Mask metric `rmse_rel = 1.08` against the cell `n=80, p=40, k=4`.

Pass tolerance: `1.3` (mask metric, `~0 = perfect`, `~1.41 = disjoint`).

### Result-handle contract

Scalars: `n_features, n_targets, n_components, top_k` (requested),
`n_selected` (actual count after candidate clamping), `n_candidates`,
`vip_threshold`.

Arrays: `vip_scores` (1×p), `vip_mask` (1×p as 0/1 doubles),
`selection_scores` (1×n_selected), int64 `selected_indices`.

## 2. DI-PLS auto-validation

Previously `paper_only` (Nikzad-Langerodi 2018). The original authors
publish `diPLSlib` on PyPI under `B-Analytics/diPLSlib`. Wiring:

- new `_DiPlsLibReference` adapter in `benchmarks/parity_timing/registry.py`
- new `_patch_diplslib_sklearn_18()` shim — diPLSlib 2.5.0 calls
  `check_X_y(..., force_all_finite=True)`. sklearn 1.6 renamed it to
  `ensure_all_finite`; sklearn 1.8 removed the old name. The shim wraps
  `check_X_y` / `check_array` in the diPLSlib namespace and translates
  the kwarg. **Signature-gated**: a no-op on older sklearn where the
  original name still works.
- the existing `MethodSpec(name="di_pls", ...)` lost `paper_only=` and
  gained `python_reference=lambda **kw: _DiPlsLibReference(...)` and
  `rmse_rel_tol=2e-2`.

Result: `rmse_rel = 4.957e-03` — pls4all and diPLSlib agree to within
0.5 %. Tightest non-trivial parity in the gate.

## 3. IRIV unlock

`iriv_select` was `paper_only` after Phase 51 because `libPLS iriv.m`
calls `ranksum()` from Octave Forge `statistics`. The previous audit
note claimed statistics 1.8+ needs Octave 11.1+ via `datatypes ≥ 1.2.0`.
**Statistics 1.7.7** (released 2025-11-11) is the last version that
predates the `datatypes` cascade — it only depends on `octave ≥ 8.1.0`
and installs cleanly on the host's Octave 10.3.0.

Install knob (one-shot):

```bash
curl -L \
  https://github.com/gnu-octave/statistics/archive/refs/tags/release-1.7.7.tar.gz \
  -o /tmp/stats-1.7.7.tar.gz
conda run -n pls4all_r octave --no-gui --eval \
  "pkg install /tmp/stats-1.7.7.tar.gz"
```

The MethodSpec was already auto-wired via `_octave_has_statistics()`;
only the `paper_only` *message string* was updated to reflect 1.7.7 as
the install knob.

Result: IRIV vs libPLS `iriv.m` → `rmse_rel ≈ 0.69` (mask), passes
under `tol = 1.0`.

## Codex review fixes (read-only mode)

1. **VIP_SPA SPA on autoscaled X** — original kernel re-used the
   `spa_selection.cpp` centering/scaling helper. auswahl's `_fit_spa`
   only L2-normalizes the current projection vector; it never centers
   or scales columns. **Fix**: replaced `copy_standardized_x` with
   `copy_raw_x` so SPA runs on raw masked X. This makes the parity
   comparison RNG-only rather than algorithmic.
2. **`top_k` ambiguity in the result handle** — docs say "clamped to
   surviving candidates" but the scalar stored the requested value.
   **Fix**: added `n_selected` to `VipSpaSelectionResult` and the C-ABI
   wrapper; documented `top_k = requested upper bound`,
   `n_selected = actual count` in `p4a.h`.
3. **`_VipSpaAuswahlReference` ignored `vip_threshold`** — auswahl
   hard-codes `VIP > 0.3`. If the parity cell ever drifted off 0.3,
   the reference would silently keep using 0.3 while pls4all honoured
   the cell value, producing a false-positive pass. **Fix**: assert
   `vip_threshold == 0.3` in `__init__` with a descriptive error.
4. **DI-PLS sklearn shim was unconditional** — would break older sklearn
   where `force_all_finite=` is the only accepted name. **Fix**:
   inspect the wrapped function's signature; no-op when
   `force_all_finite` is still present.

## Backlog (out of scope for Phase 53)

- **VIP_SPA multi-seed mode** — auswahl runs SPA from every candidate
  start and picks the CV-best. Optional `enumerate_seeds=True` would
  match auswahl bit-for-bit on the same RNG seed but multiplies the
  cost by `n_candidates`. Defer until a user asks.
- **Chemometric extensions** — MCR-ALS, BTEM, SIMPLISMA, CORAL, COW,
  ALS smoother, Whittaker smoother, FNNLS, etc. from
  ChemometricsTools.jl. Out of pls4all scope; planned for a dedicated
  chemometrics library layered on top of pls4all.
- **SCARS** (Zheng 2014), **SIPLS** (Nørgaard 2000, the synergy
  combinator), **MIR-PLS** (Sjöblom 1998), **fused-sparse PLS**
  (Yengo 2016) remain `paper_only` — no widely installable port
  surveyed by this audit.

## Release

- Library version unchanged at `0.97.0` (additive feature only).
- ABI version: `1.15.0` → `1.16.0` (one new additive symbol:
  `p4a_vip_spa_select`).
- Tag: `phase-53-vip-spa`.
