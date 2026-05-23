# Parity audit — Closeout (May 2026)

Audit period: 2026-05-14 → 2026-05-16
Closed at: pls4all 0.97.0, ABI 1.16.0

## Headline numbers

| Metric                       | Audit start | Audit end |
|------------------------------|------------:|----------:|
| Methods in the parity gate   | 33          | **68**    |
| Methods with real `ok` rows  | 11 (33 %)   | **64 (94 %)** |
| `paper_only` methods         | 22          | **4**     |
| Reference rows in `parity.csv` | 56        | **132**   |
| Languages bridged            | 2 (Py, R)   | **4** (Py, R, MATLAB/Octave, in-tree refs) |

The 4 remaining `paper_only` methods (`fused_sparse_pls`, `mir_pls`,
`scars_select`, `sipls_select`) have no installable port in any of the
audited library ecosystems and are kept with their canonical citation
only.

## What we did (phase by phase)

### Phase 1 — Inventory and library catalogue
- `01_inventory.md`: 64 numerical methods scanned from
  `cpp/src/core/*.{cpp,hpp}` and the public ABI.
- `02_library_catalogue.md`: per-method survey of every R / Python /
  MATLAB / Julia / .NET library claiming to implement the same algorithm
  (with install probe + canonical citation).

### Phase 2 — Erode `paper_only` 24 → 8
Wired R packages `plsVarSel`, `chemometrics`, `enpls`, `mboost`,
`mdatools`, `multiblock`, `OmicsPLS`, `softImpute`, `JICO`, `kernlab`,
`plsRglm`, `plsRcox`, `sgPLS`, `spls`, and Bioc `mixOmics`/`ropls`/
`survcomp` as references. Built an `rpy2` in-process bridge to amortise
per-call Rscript startup (~1.5 s → ~0.05 s per gate cell). Mask metric
introduced for variable-selection parity (`mask ~0=perfect, ~1=half
disagree, ~1.41=disjoint`).

### Phase 3 — Erode `paper_only` 8 → 7
libPLS 1.95 (MATLAB) bridged via `oct2py`, unlocking `random_frog_select`.

### Phase A — Erode `paper_only` 7 → 5
- `vissa_select`: Python `auswahl 0.9.0` (LSX-UniWue) with a sklearn-1.6+
  `validate_data` compatibility shim.
- `on_pls`: Vendored `tomlof/OnPLS` GitHub Python implementation
  (`bindings/python/vendor/OnPLS/`) with two `numpy 2.x` scalar-conversion
  patches in `estimators.py`.

### Phases 50-52 — Three new methods from libPLS
Implemented in pls4all (kernel + ABI + Python binding) and parity-gated
against libPLS:

| Method | C++ | ABI symbol | Parity vs libPLS |
|--------|----:|------------|------------------|
| **ECR** (Liu 2013 Elastic Component Regression) | ~560 LOC | `n4m_ecr_fit` | rmse_rel ≈ 1e-7 (tight) |
| **IRIV** (Yun 2014 Iteratively Retains Informative Variables) | ~560 LOC | `n4m_iriv_select` | rmse_rel ≈ 0.7 (mask) |
| **IRF** (Yun 2013 Interval Random Frog) | ~650 LOC | `n4m_irf_select` | rmse_rel ≈ 0.8 (mask) |

ABI bumped 1.14 → 1.15.

### Phase 53 — VIP_SPA + DI-PLS auto-validation + IRIV unlock
- **VIP_SPA** (Favilla 2013 VIP + Araujo 2001 SPA composition) added as
  new pls4all method, mirroring `auswahl.VIP_SPA`. SPA greedy projections
  run on **raw masked X** to match `auswahl._spa._fit_spa` exactly; start
  is `argmax(VIP)` within the surviving subset. ABI bumped 1.15 → 1.16,
  symbol `n4m_vip_spa_select`. Parity vs auswahl: rmse_rel = 1.08 (mask).
- **DI-PLS** moved from `paper_only` to `ok`: `diPLSlib 2.5.0`
  (B-Analytics; Nikzad-Langerodi 2018 authors) wired as reference.
  rmse_rel = 5e-3 with tol = 2e-2.
- **IRIV unlock**: confirmed Octave Forge `statistics 1.7.7` (last
  release before the `datatypes >= 1.2.0` cascade) installs cleanly on
  Octave 10.3.0 and provides `ranksum`. MethodSpec auto-wires through
  `_octave_has_statistics()`; iriv_select rmse_rel = 0.69 vs libPLS.
- ChemometricsTools.jl audited on Julia 1.6.7 LTS (Julia 1.12 +
  ChemometricsTools 0.5.15 transitively unresolvable). The library
  exports no PLS variant that pls4all doesn't already have; ~12
  *chemometric* tools outside the PLS scope are out of scope for this
  parity gate (intended for a future dedicated chemometrics library
  layered on top of pls4all).

## Codex reviews

Three formal review passes (codex MCP, read-only mode). Major findings
applied:

1. **IRF** generator originally used `initial_intervals` for the per-step
   draw count and skipped a PLS refit per candidate. Fixed: `current_size`
   and Fisher-Yates partial shuffle + refit on V0 ∪ picks (matches libPLS
   `irf.m` exactly).
2. **IRIV** had a hand-coded NumPy fallback (policy violation: "external
   libraries only"). Removed; entry stays `paper_only` when Octave
   `statistics` is unavailable.
3. **VIP_SPA** originally autoscaled X for the SPA pass. Fixed: SPA on
   raw masked X to match `auswahl._spa._fit_spa` (auswahl only normalizes
   the current direction).
4. **VIP_SPA result** now exposes `n_selected` (actual count) alongside
   `top_k` (requested upper bound) since the spec says top_k is clamped.
5. `_VipSpaAuswahlReference` asserts `vip_threshold == 0.3` because
   auswahl hard-codes that constant; silent drift would falsely pass.
6. DI-PLS sklearn-1.8 shim is now signature-gated so it no-ops on older
   sklearn where `force_all_finite=` still works.

## Output artefacts

- `benchmarks/results/parity_gate/parity.csv` — 132 reference rows
- `benchmarks/results/parity_gate/summary.md` — narrative summary +
  weak-pass breakdown
- `benchmarks/results/parity_gate/timings.csv` — 3-axis timings (pls4all
  binding, Python ref, R ref). **Timings are descriptive only**: the host
  was under cross-tenant load and not a clean benchmark surface.

## Remaining `paper_only` and why

| Method | Reason kept paper-only |
|--------|------------------------|
| `fused_sparse_pls` (Yengo 2016) | `sgPLS` provides sparse-group PLS but not the fused-sparse variant; no other installable port. |
| `mir_pls` (Sjöblom 1998) | Multivariate Image Regression — niche image-PLS; no port in Python, R, or MATLAB. |
| `scars_select` (Zheng 2014) | Stability-CARS hybrid — MATLAB supplementary materials only; not in libPLS 1.95, not in auswahl, not in plsVarSel. |
| `sipls_select` (Nørgaard 2000) | Synergy iPLS — ChemometricsTools.jl has it only as a multi-step Julia recipe (`MakeIntervals` + `PartialLeastSquares` + `stackedweights`), not a callable function. mdatools provides `bipls` (backward) but not the synergy combinator. |
