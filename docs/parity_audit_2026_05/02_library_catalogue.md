# Parity audit — Step 2: external library catalogue

Date: 2026-05-16
Source: PyPI / CRAN / Bioconductor / Octave / Julia / MATLAB scans, plus
the in-tree `nirs4all/bench/AOM_v0/aompls` bench oracle.
Cross-references the inventory in `01_inventory.md`.

Policy reminder (updated 2026-05-16 after user clarification):
- External libraries only — **no hand-written NumPy mirrors inside the
  parity runner**.
- Sanctioned in-tree exceptions: (a) `nirs4all/bench/AOM_v0/aompls`
  (project bench oracle for AOM/POP); (b) **full Python reimplementations
  in `nirs4all/operators/models/`** count as external (e.g.
  `nirs4all.operators.models.sklearn.mbpls.MBPLS`,
  `…lwpls.LWPLS`, `…opls.OPLS`, `…ipls.iPLS`, `…aom_pls.AOMPLS`,
  `…pop_pls.POPPLS`). They are independent reimplementations of the
  same papers, not bindings to pls4all.
- **Not allowed**: nirs4all *bindings* that delegate to a lower-level
  library (e.g. anything that internally calls aompls package or
  pls4all). Those would be circular references.
- "Paper-only" entries should disappear if any external implementation
  in any language exists.

Parity-gate semantics (per user clarification): we compare **numerical
outputs** (predictions / coefficients / selected-index lists), not
output formats. A reference returning `(n,)` vs pls4all returning `(n,1)`
is fine — the runner already reshapes. The goal is "is the C++
implementation numerically correct?".

Timing-gate semantics (Step 4 deliverable): each method is timed in
**three modes**:
  1. **C++ direct** — via a small C++ benchmark binary that calls the
     internal C++ kernel without crossing the C ABI.
  2. **Python binding** — same call from `pls4all.<method>_fit(...)` —
     measures binding overhead.
  3. **External library** — the same call against the parity reference.
This lets us quantify both binding overhead and the C++ vs external
performance gap.

The **Confidence** column rates how well the candidate's algorithm
matches pls4all's: HIGH = same numerical formulation (sub-1e-8 expected),
MED = same algorithm family (1e-3 to 1e-1 expected), LOW = qualitative
(different convention; tolerance > 0.5).

The **Reach** column says where pls4all currently exposes the method:
public-fit (C ABI symbol), dispatch (model_fit + enums), pipeline (op),
aom-public (aom_global_select / per_component), internal-only (ABI gap).

---

## Already-available references (currently installed)

### Python (parity venv)
- `scikit-learn` 1.4.2 (PLSRegression, PLSCanonical, PLSSVD, PLSDA via OneHotEncoder, PCA, KernelPCA)
- `tensorly` 0.9.0 (PARAFAC, Tucker — for N-PLS approximation)
- `scipy` 1.11.4

### R (`/home/delete/miniconda3/envs/pls4all_r`)
- `pls` 2.8.5 — `plsr` (NIPALS, SIMPLS, kernelpls, oscorespls, svdpc, widekernel), `cppls`, `pcr`, `selectNcomp`
- `plsVarSel` 0.10.0 — `bve_pls`, `ga_pls`, `ipw_pls`, `mcuve_pls`, `rep_pls`, `shaving`, `spa_pls`, `T2_pls`, `WVC_pls`, `stpls`, `VIP`, `SR`, `sMC`, `RC`, `URC`, `LW`, `mRMR`, `lda_from_pls`, `truncation`, `PVR`, `PVS`, `covSel`, `filterPLSR`, `simulate_classes`, `simulate_data`
- `prospectr` 0.2.8 — `standardNormalVariate`, `msc`, `detrend`, `savitzkyGolay`, `gapDer`, `continuumRemoval`, `binning`, `movav`, `kenStone`, `naes`, `duplex`, `puchwein`, `shenkWest`, `blockNorm`, `blockScale`, `baseline`, `spliceCorrection`
- `mdatools` 0.15.0 — `pls`, `plsda`, `ipls` (interval PLS, both forward & backward), `prep.snv`, `prep.msc`, `prep.savgol`, `simca`, `ddsimca`, `vipscores`, `selratio`, plus T²/Q/DModX
- `spls` 2.3.2 — `spls`, `splsda` (sparse PLS / DA)
- `OmicsPLS` 2.1.0 — `o2m` (O2PLS)
- `multiway` 1.0.7 — `parafac`, `parafac2`, `tucker`, `mcr`, `MCR-ALS` and supporting tensor ops (no N-PLS regression directly; PARAFAC + OLS provides one)
- `kernlab` 0.9.33 — `kpca`, `gausspr`, `rvm` (kernel methods)
- `corpcor` — shrinkage covariance
- `softImpute` — matrix completion (missing-aware)
- `genalg` — GA (used by `plsVarSel::ga_pls`)
- `survival`, `risksetROC` — Cox PH backends
- `lars`, `glmnet`, `penalized` — sparse / penalized regression
- `caret`, `ipred`, `randomForest` — bagging / boosting wrappers
- `pcv` — Procrustes cross-validation
- `forecast`, `zoo` — rolling/recursive helpers
- `pls4all` (in-house R wrapper, already in repo)

### Octave 10.3.0 (in pls4all_r env, no `pkg list` yet)
- Available; can be wired with `oct2py` or via `subprocess`. Required for
  the libPLS MATLAB toolbox (CARS / MWPLS / Random Frog / SCARS).

### Julia 1.12.6
- Already used by the cross-binding parity gate (Phase 33). Available
  for any future Julia reference.

### Bench oracle
- `nirs4all/bench/AOM_v0/aompls` — AOM-PLS / POP-PLS reference (project policy).

---

## Per-method library catalogue

Methods from `01_inventory.md`, in the same numbering, plus library
candidates with install instructions.

### §1 Core PLS regression solvers

| # | pls4all method | Best Python ref | Best R ref | Other refs | Confidence | Notes |
|---|----------------|------------------|------------|------------|-------------|-------|
| 1 | NIPALS PLS1/PLS2 | `sklearn.cross_decomposition.PLSRegression` (NIPALS) | `pls::plsr(method='kernelpls'/'oscorespls'/'simpls')` — for NIPALS-equivalent see `pls::cppls(...,k=0)` or low-level `pls::oscorespls.fit` | Octave libPLS `pls_nipals`; Julia PartialLeastSquaresRegressor.jl | HIGH | sklearn has clean NIPALS. |
| 2 | Orthogonal-scores PLS | `sklearn.PLSRegression` (uses orthogonal scores internally) | `pls::oscorespls.fit` | — | HIGH | Direct API match. |
| 3 | SIMPLS | — (sklearn does NOT ship SIMPLS) | `pls::simpls.fit` | Julia: hand-written tiny | HIGH | Use R `pls::simpls.fit` directly. |
| 4 | Linear kernel-algorithm PLS | — | `pls::kernelpls.fit` | — | HIGH | R `pls::kernelpls.fit`. |
| 5 | Wide-kernel PLS | — | `pls::widekernelpls.fit` | — | HIGH | R `pls::widekernelpls.fit`. |
| 6 | SVD-based PLS | sklearn `PLSSVD` (cross-cov SVD), or `pls::simpls.fit` (algorithmic equivalent for predictive PLS) | `pls::simpls.fit` for predictive scores; **NOT `pls::svdpc.fit` — that is PCR, not SVD-PLS** (codex catch). | `bigPLSR` (CRAN, supports `simpls`/`nipals`/`kernelpls`/`widekernelpls`) | MED | The pls4all SVD solver builds X^T Y SVD and uses the dominant pair; closest external is `pls::simpls.fit` since SIMPLS does the same SVD step. Tolerance widened to 1e-2. |
| 7 | Power-iteration PLS | — (no maintained "power PLS" lib) | `pls::simpls.fit` as algorithmic ref (different solver, same fixed point) | `bigPLSR` | LOW | Power iteration converges to the dominant pair; compare predictions, not weights — tolerance ~1e-1. |
| 8 | Randomized SVD PLS | **NOT** `sklearn.utils.extmath.randomized_svd` (codex catch — pls4all's "randomized SVD" is a random-start power iter on `XᵀY`, not Halko/Martinsson randomized SVD with oversampling). Use `pls::simpls.fit` as the algorithmic ref. | `pls::simpls.fit` | `bigPLSR` | LOW | Tolerance widened to 1e-1; document that the solver name is misleading (`SOLVER_RANDOMIZED_SVD` is "random-init power iter", not Halko randomized SVD). |

### §2 Other core algos

| # | pls4all method | Best Python ref | Best R ref | Other refs | Confidence | Notes |
|---|----------------|------------------|------------|------------|-------------|-------|
| 9  | PLS-Canonical (NIPALS) | `sklearn.cross_decomposition.PLSCanonical(algorithm='nipals')` | `pls::plsr` w/ Mode-A | — | HIGH | sklearn direct match. |
| 10 | PLS-Canonical (SVD)    | `sklearn.PLSCanonical(algorithm='svd')` | `mixOmics::spls(mode='canonical')` (Bioc) | — | HIGH | sklearn covers it. |
| 11 | PLS-SVD                | `sklearn.PLSSVD` | — | — | HIGH | sklearn has `PLSSVD`. |
| 12 | PLS-DA                 | `sklearn.PLSRegression` on dummy-coded Y | `mdatools::plsda`; `pls::plsr` on dummy | `pyChemometrics`, `chemometrics` (R) | HIGH | Standard pattern. |
| 13 | OPLS                   | (codex catch: `pyopls` is mainly an OPLS *transformer/validation* package, not a full regressor — demote) | **PRIMARY: `ropls::opls` (Bioconductor)** | `mdatools` for OPLS-style diagnostics | HIGH (R) / LOW (pyopls) | Use Bioc `ropls::opls(predI=k, orthoI=m)` as the canonical reference; `pyopls` only as smoke-check. |
| 14 | OPLS-DA                | `pyopls 20.3` (transformer-only — not a full DA classifier) | **PRIMARY: `ropls::opls(y=factor, predI=k, orthoI=m)`** | — | HIGH (R) | Same: prefer Bioc. |
| 15 | OPLS-DA (multiclass)   | `pyopls` (loops one-vs-rest) | **PRIMARY: `ropls::opls`** with one-vs-rest loop | — | MED | One-vs-rest stacking convention may diverge between implementations. |
| 16 | PCR                    | `sklearn.decomposition.PCA + LinearRegression` (or `sklearn.pipeline.Pipeline`) | `pls::pcr` | — | HIGH | sklearn direct. |
| 17 | MB-PLS  (ABI gap → being added) | **PRIMARY: `nirs4all.operators.models.sklearn.mbpls.MBPLS`** (in-tree, full reimpl); also `mbpls 1.0.4` (Baum & Vermue) | `multiblock 0.8.10` (CRAN; `mbpls`); `mdatools` MB helpers | — | HIGH | Adding `n4m_mb_pls_fit` ABI shim in Step 3. (codex catch: install only `mbpls`, not `Multiblock-PLS` — namespace clash.) |
| 18 | LW-PLS  (ABI gap → being added) | **PRIMARY: `nirs4all.operators.models.sklearn.lwpls.LWPLS`** (in-tree, full reimpl) | `plsVarSel::LW` (loadings-weights, related but not identical) | Octave libPLS `lwpls.m` | HIGH | Adding `n4m_lw_pls_fit` ABI shim in Step 3. |
| 19 | PLS-LDA  (ABI gap → being added) | sklearn `PLSRegression + LDA` pipeline | `plsVarSel::lda_from_pls`, `lda_from_pls_cv` | — | HIGH | Adding `n4m_pls_lda_fit` ABI shim in Step 3. |
| 20 | PLS-Logistic (ABI gap → being added) | sklearn `LogisticRegression` on PLS scores | `pls::plsr` + `stats::glm` | — | HIGH | Adding `n4m_pls_logistic_fit` ABI shim in Step 3. |

### §3 Existing extra-PLS public-fit (already in registry)

These already have entries; library upgrades to consider:

| # | Registry name | Current status | Library upgrade candidate |
|---|----------------|----------------|----------------------------|
| 21 | `sparse_simpls` | R `spls` | also Python `sklearn` + soft-thresh (only for cross-check) |
| 22 | `di_pls` | **Python `diPLSlib 2.5.0`** (Phase 53) | The original Nikzad-Langerodi authors publish `diPLSlib` on PyPI. Tight parity `rmse_rel ≈ 5e-3` with `tol = 2e-2`. Requires a signature-gated `force_all_finite` → `ensure_all_finite` shim for sklearn 1.8+. |
| 23 | `recursive_pls` | sklearn + R pls | already covered. |
| 24 | `cppls` | paper-only | R `pls::cppls` exists but implements *Liland 2009 Canonical Powered PLS*, a different algorithm than Indahl 2005's Powered-PLS. **Add R `pls::cppls` as a qualitative reference and document the divergence**. |
| 25 | `weighted_pls` | sklearn | already covered. |
| 26 | `robust_pls` | paper-only | R `chemometrics::prm` (CRAN, Partial Robust M-regression — Serneels/Filzmoser/Hubert) — codex catch: function is `prm`, not `prm_pls`. Also Python `pyChemometrics 0.13.6`. |
| 27 | `ridge_pls` | sklearn | already covered. |
| 28 | `continuum_regression` | paper-only | **R `JICO::continuum`** (CRAN — Hsu/Park/Yang 2024 implementation of Stone-Brooks 1990 continuum regression). Codex catch: `chemometrics::cr_method` does NOT exist. JICO is the right reference. |
| 29 | `n_pls` | paper-only | R `multiway` ships `parafac` + `tucker` (PARAFAC isn't N-PLS but adjacent); R `npls` (NOT on CRAN — Bro's Toolbox is MATLAB-only). MATLAB nway toolbox via Octave + `oct2py` is the canonical reference. |
| 30 | `kernel_pls_*` (codex catch: `n4m_kernel_pls_fit` supports `kernel_type ∈ {0,1,2,3}` = LINEAR / RBF / POLYNOMIAL / SIGMOID — wire 4 sub-cells, not just RBF) | paper-only | LINEAR: `pls::kernelpls.fit`. RBF/POLY/SIGMOID: `kernlab::kernelMatrix(rbfdot/polydot/tanhdot)` Gram + `pls::plsr` on K. Python `ikpls 4.0.1` covers linear KPLS only. |
| 31 | `o2pls` | R `OmicsPLS` | already covered (qualitative). |
| 32 | `approximate_press` | paper-only | R `pls::plsr(validation='LOO')` returns true LOO-PRESS, useful as qualitative cross-ref. |
| 33 | `sparse_pls_da` | R `spls::splsda` | already covered. |
| 34 | `group_sparse_pls` | paper-only | R `sgPLS::sgPLS` / `sgPLSda` (CRAN, depends on mixOmics from Bioc). **Strong candidate** once mixOmics is installable. |
| 35 | `fused_sparse_pls` | paper-only | R `sgPLS::gPLS` — close but not fused. Real fused PLS: `glmnet` with fused penalty isn't standard. Likely stays paper-only. |
| 36 | `pls_monitoring` | paper-only | R `mdatools::pls` + `predict()$xdecomp$T2/$Q` already used for diagnostics; can be reused for monitoring smoke. |
| 37 | `one_se_rule` | paper-only | R `pls::selectNcomp(method='onesigma')` — **Strong direct candidate**. |
| 38 | `so_pls` | paper-only (had HDANOVA dep issue) | R `multiblock::sopls` 0.8.10 (CRAN) — **Re-attempt installation; HDANOVA is now CRAN**. |
| 39 | `on_pls` | paper-only | R `multiblock::onpls` — **Same; re-attempt with HDANOVA**. |
| 40 | `rosa` | paper-only | R `multiblock::rosa` — **Same**. |
| 41 | `bagging_pls` | paper-only | sklearn `BaggingRegressor(PLSRegression())` — RNG divergence noted but a soft tolerance check is meaningful. R `caret::bag(method='pls')` or `ipred::bagging`. |
| 42 | `boosting_pls` | paper-only | sklearn `AdaBoostRegressor(PLSRegression())`. R `mboost::glmboost(family=Gaussian())` for GLM-boosted PLS. |
| 43 | `random_subspace_pls` | paper-only | sklearn `BaggingRegressor(PLSRegression(), max_features=…)`. |
| 44 | `pls_glm` | paper-only (plsRglm install fail) | R `plsRglm 1.7.0` (CRAN). **Re-attempt — its hard dep `car` is in pls4all_r already**. |
| 45 | `pls_qda` | paper-only | R `plsVarSel::lda_from_pls` is LDA but `MASS::qda` on PLS scores gives PLS-QDA. **Strong candidate**. |
| 46 | `pls_cox` | paper-only (plsRcox install fail) | R `plsRcox 1.8.2` (CRAN). Python `lifelines.CoxPHFitter` on PLS scores as alt. **Re-attempt**. |
| 47 | `pds` | paper-only | R `prospectr::shenkWest` is a related transfer; `pcv::pcv` is procrustes cross-validation; no perfect PDS port. Could implement a PDS reference using R `pls::plsr` + windowing if we accept a paper-port. |
| 48 | `ds` | paper-only | Same: `prospectr::shenkWest` is closest; pure DS is `lm` per-band. **R `lm` per-band is the canonical "DS" — qualifies as external because base R is the reference**. |
| 49 | `mir_pls` | paper-only | No widely installable port; remains paper-only (Sjöblom 1998). |
| 50 | `missing_aware_nipals` | paper-only | R `softImpute::softImpute` for missing-value PLS (matrix completion + PLS). Or `mice` for imputation + `pls::plsr`. Qualitative match; tolerance widened. |
| 51 | `pls_diagnostic_t2` | R `mdatools` | already covered. |
| 52 | `pls_diagnostic_q` | R `mdatools` | already covered. |
| 53 | `pls_diagnostic_dmodx` | R `mdatools` | already covered. |

### §4 AOM / POP public ABI (Phase 6f)

| # | Method | Library | Notes |
|---|--------|---------|-------|
| 54 | Global AOM-SIMPLS selection | `nirs4all/bench/AOM_v0/aompls` (project bench oracle) | Sanctioned in-tree exception; matches §6b parity. |
| 55 | Per-component POP-PLS selection | `nirs4all/bench/AOM_v0/aompls` | Sanctioned in-tree exception; matches §6e parity. |
| 56 | AOM preprocessing fit/transform | `nirs4all/bench/AOM_v0/aompls.preprocessing` | Sanctioned in-tree exception. |

### §5 Preprocessing operators

Pipeline-level parity. R `prospectr` and `mdatools` cover most; for the
strict-linear AOM-only operators (FCK, Whittaker, finite-difference) the
`nirs4all/bench/AOM_v0/aompls` bench oracle is the canonical ref.

| # | Operator | Best Python ref | Best R ref | Other refs | Confidence |
|---|----------|------------------|------------|------------|-------------|
| 57 | Identity | (trivial) | (trivial) | — | HIGH |
| 58 | Center | `sklearn.preprocessing.StandardScaler(with_std=False)` | `base::scale(center=T,scale=F)` | — | HIGH |
| 59 | Autoscale | `sklearn.StandardScaler(with_mean=T,with_std=T)` | `base::scale(center=T,scale=T)` | — | HIGH |
| 60 | Pareto scaling | `chemotools.scaling.PointScaler` (or hand-written pareto via sklearn) | `mdatools::prep.autoscale(scale='pareto')` | — | HIGH |
| 61 | SNV | `chemotools.scatter.StandardNormalVariate` | `prospectr::standardNormalVariate` | — | HIGH |
| 62 | MSC | `chemotools.scatter.MultiplicativeScatterCorrection` | `prospectr::msc`; `mdatools::prep.msc` | — | HIGH |
| 63 | EMSC | `chemotools.scatter.ExtendedMultiplicativeScatterCorrection` | `prospectr` (no EMSC); R `EMSC` package (CRAN) | — | HIGH |
| 64 | Detrend (poly) | `chemotools.baseline.PolynomialDetrend` | `prospectr::detrend` | — | HIGH |
| 65 | Savitzky-Golay smoothing | `scipy.signal.savgol_filter` | `prospectr::savitzkyGolay`; `signal::sgolayfilt` | — | HIGH |
| 66 | Savitzky-Golay derivative | `scipy.signal.savgol_filter(..., deriv=k)` | `prospectr::savitzkyGolay(p=, w=, m=k)` | — | HIGH |
| 67 | Norris-Williams | `chemotools.derivative.NorrisWilliams` | `prospectr::gapDer` | — | HIGH |
| 68 | AsLS baseline | `pybaselines.whittaker.asls` | R `baseline::baseline.als` (CRAN) | — | HIGH |
| 69 | OSC | — (no maintained Python OSC) | R `mdatools` indirect; R `baseline` related; the canonical Wold OSC sits in `nirs4all/bench/AOM_v0/aompls.preprocessing.osc` if it ships. Else R `pcv` not OSC. | Octave libPLS `osc.m` | MED |
| 70 | EPO | — | R `EMSC` related; for true EPO use Roger 2003 — no maintained port; bench oracle. | Octave libPLS `epo.m` | MED |
| 71 | Wavelet denoise | `pywt.threshold` + `pywt.wavedec`/`waverec` | R `wavelets::dwt` + thresholding | — | HIGH |
| 72 | Finite difference | `numpy.diff` (trivial primitive) | `base::diff` | bench oracle | HIGH |
| 73 | Whittaker | `pybaselines.whittaker.whittaker_smooth` | R `ptw::whit2`; `MASS` via Whittaker; `baseline::baseline.als` overlap | bench oracle | HIGH |
| 74 | FCK | bench oracle | bench oracle | bench oracle | n/a (pls4all-specific operator) |

### §6 Variable selection methods

These were NOT exposed via the C ABI. **User decision (2026-05-16): add
the ABI shims now (Option A).** Plan: one `n4m_<selector>_select` symbol
per selector returning the existing generic `n4m_method_result_t**`,
which already has `_get_double_matrix` / `_get_int_vector` /
`_get_scalar` accessors keyed by string. So the cost is **23 new ABI
symbols** + ~5 new result keys per selector — not 180 — because the
result-accessor surface is reused.

Implementation order in Step 3:
  1. CARS, MCUVE, SPA, iPLS, GA-PLS, Random Frog (most-used, biggest
     parity wins because they each have a clear external counterpart).
  2. VIP / SR / sMC / RC / coefficient rankers (trivial — all just
     score vectors).
  3. EMCUVE, UVE, BVE, T2, WVC, WVC-threshold, REP, IPW, ST, SCARS,
     biPLS, siPLS, randomization, shaving (the long tail).

| # | Selector | Best R ref | Best Python ref | Other refs | Confidence (when wired) |
|---|----------|-------------|------------------|------------|-------------------------|
| 75 | VIP ranker | `plsVarSel::VIP`, `mdatools::vipscores` | `chemometrics 0.4.0` (`vip_pls`) | — | HIGH |
| 76 | Coefficient-magnitude ranker | `plsVarSel::RC` | `sklearn.PLSRegression.coef_` | — | HIGH |
| 77 | Selectivity-ratio ranker | `plsVarSel::SR`, `mdatools::selratio` | `chemometrics` | — | HIGH |
| 78 | Interval / iPLS / mwPLS | `mdatools::ipls` | — | Octave libPLS `mwpls.m` | HIGH |
| 79 | MCUVE | `plsVarSel::mcuve_pls`, `enpls::enpls.fs(method='mc')` | — | Octave libPLS `mcuve.m` | HIGH |
| 80 | UVE | `plsVarSel::mcuve_pls(MCUVE.threshold=NA)` (legacy mode) | — | — | HIGH |
| 81 | SPA | `plsVarSel::spa_pls` | — | Octave libPLS `spa.m` | HIGH |
| 82 | CARS | `enpls 6.1.1` (CARS bagging variant); for true CARS Li 2009: Octave libPLS `cars.m` | — | Octave libPLS canonical | HIGH (Octave) / MED (R enpls) |
| 83 | Random Frog | — | — | Octave libPLS `randomfrog.m` | HIGH (Octave) |
| 84 | SCARS | — | — | Octave libPLS `scars.m` (Zheng 2012) | HIGH (Octave) |
| 85 | GA-PLS | `plsVarSel::ga_pls` (uses `genalg`) | — | Octave libPLS `gapls.m` | HIGH |
| 86 | Shaving | `plsVarSel::shaving` | — | — | HIGH |
| 87 | BVE-PLS | `plsVarSel::bve_pls` | — | — | HIGH |
| 88 | T2-PLS | `plsVarSel::T2_pls` | — | — | HIGH |
| 89 | WVC-PLS | `plsVarSel::WVC_pls` | — | — | HIGH |
| 90 | EMCUVE | — (build by re-running `mcuve_pls` with vote rule) | — | Octave libPLS `mcuve.m` ensemble mode | MED |
| 91 | Randomization-test | `plsVarSel::truncation` (related); `permute::shuffle` for permuting Y + `pls::plsr` | — | — | HIGH |
| 92 | biPLS | `mdatools::ipls(method='backward')` | — | Octave libPLS `bipls.m` | HIGH |
| 93 | siPLS | — (no R/Python; pls4all extends mdatools forward iPLS) | — | Octave libPLS `sipls.m` | HIGH (Octave) |
| 94 | WVC-threshold | `plsVarSel::WVC_pls(threshold=…)` | — | — | HIGH |
| 95 | REP-PLS | `plsVarSel::rep_pls` | — | — | HIGH |
| 96 | IPW-PLS | `plsVarSel::ipw_pls` (and `ipw_pls_legacy`) | — | — | HIGH |
| 97 | ST-PLS | `plsVarSel::stpls` | — | — | HIGH |

---

## Install plan (Step 3 input)

### Install order (safest first; reordered per codex review)

#### Round A — pip into existing parity venv (low risk)
```bash
parity/python_generator/.venv/bin/pip install \
  chemotools==0.3.3 \
  pyopls==20.3.post1 \
  ikpls==4.0.1.post1 \
  hoggorm==0.13.3 \
  chemometrics==0.4.0 \
  mbpls==1.0.4 \
  lifelines==0.30.3 \
  pywavelets==1.9.0 \
  pybaselines==1.2.1 \
  oct2py==6.0.3
```
**Codex catch**: do NOT install both `mbpls` and `Multiblock-PLS` —
they expose the same `mbpls` module name. We install `mbpls 1.0.4`
(the maintained Baum & Vermue version).

- `chemotools`: SNV/MSC/EMSC/Pareto/Detrend/SG/Norris-Williams (operators §61-67)
- `pyopls`: OPLS smoke-check only (per codex demotion); primary OPLS ref is Bioc `ropls` in Round C
- `ikpls`: Improved Kernel PLS (linear) (§30 LINEAR sub-cell)
- `hoggorm`: PCR / PLSR1/2 / MB-PLS (§16, §17)
- `chemometrics 0.4.0` (Python): VIP scores
- `mbpls`: MB-PLS (§17)
- `lifelines`: Cox PH (§46)
- `pywavelets`: wavelets (§71)
- `pybaselines`: AsLS (§68), Whittaker (§73)
- `oct2py`: Python-Octave bridge for libPLS toolbox

#### Round B — Bioconductor install FIRST (codex catch: `sgPLS` Imports `mixOmics`, so install Bioc deps before CRAN packages that need them)
```r
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager", repos="https://cran.r-project.org")
BiocManager::install(c("mixOmics", "ropls"))
```
- `mixOmics`: required by `sgPLS`; provides sPLS / sPLSDA / DIABLO / MINT
- `ropls`: canonical R OPLS / OPLS-DA implementation (§13-14 primary ref)

#### Round C — CRAN install into pls4all_r (medium risk: build deps)
```r
install.packages(c(
  "enpls", "mvdalab", "plsdepot", "plsRglm", "plsRcox",
  "sgPLS", "wavelets", "chemometrics", "multiblock", "HDANOVA",
  "EMSC", "baseline", "ptw", "signal",
  "JICO", "bigPLSR", "mboost"
), repos="https://cran.r-project.org")
```
- `enpls 6.1.1`: Ensemble PLS, CARS-bagging, MCUVE-ensemble (§82, §90)
- `mvdalab 1.7`: MVDA toolbox (cross-ref VIP/SR)
- `plsdepot 0.3.1`: PLS toolkit (extra cross-checks)
- `plsRglm 1.7.0`: PLS-GLM (§44)
- `plsRcox 1.8.2`: PLS-Cox (§46)
- `sgPLS 1.8.1`: group / sparse-group PLS (§34) — needs mixOmics from Round B
- `wavelets`: wavelet (§71)
- `chemometrics 0.7.x` (R, NOT Python): `prm` (robust PLS §26)
- `multiblock 0.8.10` + `HDANOVA`: SO-PLS (§38), OnPLS (§39), ROSA (§40)
- `EMSC`: extended MSC reference (§63)
- `baseline`: AsLS reference (§68)
- `ptw`, `signal`: Whittaker (§73), SG (§65-66)
- **`JICO 0.1`**: continuum regression (§28) — codex addition
- **`bigPLSR 0.7.2`**: SIMPLS / NIPALS / kernelpls / widekernel cross-checks (§3-7) — codex addition
- **`mboost 2.9-11`**: PLS-boosting (§42)

#### Round D — Octave libPLS toolbox (CARS, MWPLS, GA-PLS, Random Frog, SCARS)
```bash
# libPLS has no formal package; download tar/zip from https://www.libpls.net
# and unpack into bindings/octave/libpls/
mkdir -p bindings/octave/libpls
curl -L https://www.libpls.net/download/libPLS_1.98.zip -o /tmp/libpls.zip
unzip /tmp/libpls.zip -d bindings/octave/libpls/
# Drive from Python via oct2py:
python -c "import oct2py; oc = oct2py.Oct2Py(); oc.addpath('bindings/octave/libpls'); print(oc.cars(...))"
```
This unlocks parity for CARS (§82), Random Frog (§83), SCARS (§84),
SPA (§81), MCUVE (§79), MWPLS / iPLS (§78, §92, §93), GA-PLS (§85). Not
all selectors require this — `plsVarSel` covers most. Octave + libPLS is
the **belt-and-suspenders** parity for Phase 5 selectors.

---

## Coverage projection (post-install)

Assuming Round A + Round B succeed cleanly:

| Family | Methods | Today (external ref) | After install |
|--------|---------|----------------------|----------------|
| §1 Core PLS solvers | 8 | 0 | 8 |
| §2 Other core | 12 | 0 | 10 (PLS-LDA / PLS-Logistic still need ABI) |
| §3 Extra-PLS | 33 | 11 | 26 (paper-only drops from 22 → 7) |
| §4 AOM/POP | 3 | 0 | 3 (bench oracle) |
| §5 Preprocessing | 18 | 0 | 18 |
| §6 Selectors | 23 | 0 | 0 (need ABI shim or cpp-test harness — Step 4 decision) |
| **TOTAL** | **97** | **11 of 97 (11%)** | **65 of 97 (67%) reachable; 0 selector parity until §6 ABI gap closes** |

If we add Octave libPLS (Round D) AND the §6 cpp-test harness (Step 4
option B), the 23 selectors come into play and coverage rises to
**88 / 97 = 91%**. The remaining 9 are MIR-PLS, `_or_`/randomization
quirks, AOM-FCK and a handful of paper-only cases where no implementation
exists in any language.
