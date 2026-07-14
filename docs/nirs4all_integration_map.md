# nirs4all model → n4m/pls4all estimator mapping (G3 prerequisite)

**Status:** authoritative mapping table for `map-model-classes`
(RELEASE_READINESS.md G3, line 401). This is the prerequisite the release
audit names for **G3 — drop-in integration of n4m into nirs4all**.

**Scope:** every model class exported by
`nirs4all/nirs4all/operators/models/sklearn/__init__.py` `__all__`, mapped to
its `pls4all.sklearn` (or full-`n4m`) counterpart, with a per-class verdict.
Every verdict below is backed by code that was read; file:line citations are
inline. Citations against `nirs4all-methods` are repo-relative to that repo;
citations against `nirs4all` are repo-relative to the `nirs4all` repo.

---

## 1. Dependency surface

There are **two distinct PyPI distributions** under `nirs4all-methods`, and
they do **not** overlap in what they expose:

| dist (PyPI) | import package | what it exposes | model fits? |
|---|---|---|---|
| `pls4all` (slim subset) | `pls4all`, `pls4all.sklearn` | the PLS models, classifiers, selectors, AOM/POP selection, calibration transfer, diagnostics | **YES** — all PLS `*_fit` + `MethodResult` + `Model`/N4MM payload plumbing |
| `nirs4all-methods` (full) | `n4m`, `n4m.sklearn` | preprocessing, filters, augmenters, splitters, interval generators, transfer (DS/PDS) | **NO model fits** |

- All the model estimators (`PLSRegression`, `OPLSRegression`, `PCR`,
  `SparsePLSRegression`, `PLSDAClassifier`, `MBPLSRegression`,
  `DIPLSRegression`, the `_in_sample.py` family, …) live in **`pls4all`**, not
  `n4m`. They are wired in `pls4all/sklearn/__init__.py:50-148` and the fit FFI
  lives in `pls4all/_methods.py` and `pls4all/_model.py`.
- The full **`n4m`** package's FFI (`bindings/python/src/n4m/_ffi_decls.py`)
  declares **only** `n4m_pp_*` (preprocessing), `n4m_filter_*`,
  `n4m_interval_generator_*`, and augmenter `*_fit`/`*_is_fitted` symbols. It
  has **no** `n4m_method_result_*`, no `n4m_pls_fit`/`n4m_model_*`, and no
  per-model `*_fit`. `n4m/python.py` is entirely preprocessing/filter/augmenter
  transforms (`n4m/python.py:134` `transform()`, `:441`, `:479`, `:631`, …).

> ### ⚠️ LOAD-BEARING FLAG: the FULL `n4m` package has no method-result FFI plumbing
>
> A "swap nirs4all models onto the full `n4m` package" is **not possible
> today**, even for the drop-in rows below. The PLS model surface
> (`MethodResult`, `Model`, `*_fit`) exists **only in the slim `pls4all`
> package**. The full `n4m` Python binding would need the method-result /
> model FFI subsystem **built first** before any model class could be served
> from it. Every "drop-in"/"adapter-needed" verdict below therefore assumes
> the target is **`pls4all.sklearn`**. If the integration is intended to ride
> on the full `n4m` package (so preprocessing + models come from one import),
> that FFI port is an additional, currently-missing prerequisite. This is
> consistent with RELEASE_READINESS.md G3 `consolidate-pypi-packaging`
> (line 415): "models live in dist `pls4all`, preprocessing in dist
> `nirs4all-methods`; a clean drop-in needs **both**".

**nirs4all currently has ZERO n4m/pls4all references** (greenfield). Verified
by grepping the whole `nirs4all` tree for `import n4m` / `from n4m` /
`import pls4all` / `from pls4all`: **0 matches** (the only `pls4all`-shaped
hits are the unrelated internal `pls4all_core` token). So the swap is
config/instance-level (nirs4all resolves models by class-path string or
instance), not a rewrite — matching RELEASE_READINESS.md G3 line 397-398.

### Native tuning / conformal / robustness capability boundary

This repository supplies the portable numerical engine and bindings. It does
not own the higher-level statistical orchestration that `nirs4all` Python and
Studio expose around tuning, conformal calibration, robustness reports and
workspace artifacts.

| Capability | `nirs4all-methods` / `n4m` / `pls4all` responsibility | `nirs4all` Python / Studio responsibility |
|---|---|---|
| PLS/PCR/SIMPLS model kernels | Native C++ kernels behind the stable C ABI; Python `pls4all.sklearn` wrappers for model fit/predict where the method-result FFI is present. | Pipeline orchestration, model class compatibility, workspace storage, `.n4a` packaging and user-facing run/predict APIs. |
| Pure-native estimator selection | `n4m.model_selection.finetune_estimator(...)` selects one native estimator configuration from an explicit validation plan and search space, returning an owning trial trace. | `nirs4all.run(tuning=...)` composes optimizer selection with dataset/cohort handling, terminal winner projection, optional conformal calibration, workspace persistence and Studio forms. |
| Full-DAG tuning | Out of scope for this repo; no DAG graph, branch/merge or host pipeline callback runner is exposed by `libn4m`. | Owned by `nirs4all`/DAG-ML. Current public subset remains fail-closed outside single-estimator or linear-chain shapes. |
| Conformal calibration | No native conformal guarantee or calibration artifact is emitted by `libn4m` or the bindings. | `nirs4all.calibrate()`, `predict_calibrated()`, `CalibratedRunResult`, conformal workspace/bundle storage and Studio interval views. |
| Robustness/generalization reports | No `RobustnessReport` type, workspace table, report export or Studio summary exists in `n4m`/`pls4all`. Native predictors can be used as frozen predictors by higher layers. | `nirs4all.robustness()`, `PredictResult.robustness()`, `RobustnessReport` JSON/Markdown/HTML/Parquet/artifact exports, workspace persistence and Studio robustness cards/preflight. |
| Bindings | Thin translation layers over the C ABI. Bindings must not reimplement tuning/conformal/robustness math or infer guarantees. | Bindings and Studio consume the public Python artifacts/registries or future portable contracts; any guarantee status and invalidation reasons must come from `nirs4all`, not binding-local heuristics. |

Load-bearing implication for downstream work: use `n4m` for native kernels and
single-estimator selection traces; use `nirs4all` for
`run(tuning=...) → calibrate/predict_calibrated() → robustness()` workflows,
workspace artifacts, keyword/effect registry, and Studio presentation. A
binding that wants to display conformal intervals or robustness summaries should
consume the `nirs4all` artifact/schema surfaces rather than synthesizing them
from raw native predictions.

## 2. Reproducibility caveat (stochastic methods are NOT bit-reproducible)

n4m's stochastic kernels (bagging, boosting, random-subspace, GPR, and the
randomized selectors CARS/RandomFrog/GA/PSO/VISSA/IRIV/…) seed a **PCG64**
stream with explicit integer seeds. nirs4all seeds the **numpy global RNG**.
These are different generators, so any stochastic swap **will not bit-reproduce
nirs4all's previous outputs** — only statistical equivalence can be claimed.
Verified: `PRODUCTION_AUDIT.md:372` ("nirs4all's numpy RNG ≠ n4m's PCG64") and
`benchmarks/cross_binding/donor_ops.py` declares `value_parity=False`
("stochastic — nirs4all RNG differs") for these. RELEASE_READINESS.md
`seeding-reproducibility-note` (line 428-430) tracks the same caveat. The
deterministic PLS/PCR/SIMPLS/OPLS/sparse rows are unaffected (they match the
oracle at ~1e-15).

## 3. Recommended integration mechanism

Two mechanisms are viable; **the delegating shim is recommended** for the
drop-in / adapter-needed rows.

**(A) Delegating shim (recommended).** Keep each nirs4all model class at its
existing import path (`nirs4all.operators.models.sklearn.<Name>`) and have its
`fit`/`predict` delegate to the `pls4all.sklearn` estimator. The shim must:
- **preserve the import path** (controllers and nirs4all `.n4a`/webapp configs resolve
  models by class-path string);
- **preserve `get_params` arg names** — nirs4all uses `n_components`,
  `scale`/`center`, `backend`, `solver`-free constructors, while `pls4all`
  uses `center_x`/`scale_x`/`center_y`/`scale_y`/`solver`. The shim's
  `__init__` keeps the nirs4all names and translates inside `fit` (see the
  per-row "constructor-arg mapping" column);
- **preserve `_estimator_type`** (declared on 17 of the 19 model files;
  `StackingRegressor`/`StackingClassifier` rely on it) and **`_webapp_meta`**
  (the studio UI catalog reads it — 19 model files carry one).

This is `adapter-shim-or-repoint` in RELEASE_READINESS.md G3 (line 406-409),
which explicitly notes "a naive import-path swap won't work" because the API
signatures differ.

**(B) Config references.** Alternatively reference `pls4all.sklearn.PLSRegression`
(etc.) directly in pipeline configs and drop the nirs4all wrapper. Simpler, but
loses the nirs4all `get_params` names, `_webapp_meta`, and any nirs4all-specific
`predict_proba`/decode behaviour; not recommended where the studio UI or
StackingRegressor introspection matters.

---

## 4. Mapping table (one row per nirs4all model class)

Constructor-arg mapping uses the convention **nirs4all → n4m/pls4all**.
"—" = no equivalent arg / not applicable.

| nirs4all class | n4m/pls4all target | constructor-arg mapping (nirs4all → n4m) | verdict | notes |
|---|---|---|---|---|
| **PLSDA** | `pls4all.sklearn.PLSDAClassifier` | `n_components → n_components`; (nirs4all has only `n_components`) | **adapter-needed** | nirs4all `PLSDA` exposes `predict_proba` (`plsda.py:55`); n4m `PLSDAClassifier` exposes only `predict`/`decision_function` (`_classification.py:119-132`), **no `predict_proba`/`classes_`-proba**. Shim must add proba (column-stack for binary, raw for multiclass) → see follow-ups. Both one-hot Y + argmax decode. |
| **IKPLS** | `pls4all.sklearn.PLSRegression` | `n_components→n_components`; `center→center_x`(+`center_y`); `scale→scale_x`(+`scale_y`); `algorithm`(1/2)→ — (n4m IKPLS variant via `solver`); `backend→` — | **adapter-needed** | IKPLS is the Dayal–MacGregor fast kernel; n4m has no `algorithm=1/2` IKPLS knob — closest is `PLSRegression(solver='kernel'/'wide-kernel')`. Numerically PLS1/PLS2-equivalent but not the same code path; `backend='jax'` has no n4m analogue. Constructor: `_init_` at `ikpls.py:97-104`. |
| **OPLS** | `pls4all.sklearn.OPLSRegression` | `n_components`(=#ortho)→`n_components`; `pls_components`→ — (n4m OPLS folds predictive+orthogonal into one `n_components`); `scale→scale_x/scale_y`; `backend→` — | **adapter-needed** | RELEASE_READINESS lists OPLS as "1:1 confirmed" (line 402). Verdict adapter-needed because nirs4all splits `n_components`(ortho) + `pls_components`(predictive) (`opls.py:93`) while n4m `OPLSRegression` takes a single `n_components` + `Deflation.ORTHOGONAL` (`_regression.py:155-187`); the two-knob→one-knob translation is the shim's job. |
| **OPLSDA** | `pls4all.sklearn.OPLSDAClassifier` | `n_components`(ortho)→`n_components`; `pls_components`→ —; `scale→scale_x` | **adapter-needed** | nirs4all `OPLSDA` exposes `predict_proba` (`oplsda.py:159`, delegating to inner PLSDA); n4m `OPLSDAClassifier` has **no `predict_proba`** (`_classification.py:135`). Same two-knob→one-knob issue as OPLS plus the missing-proba gap. |
| **PCR** | `pls4all.sklearn.PCR` | `n_components→n_components` (nirs4all has only `n_components`) | **drop-in** | nirs4all PCR = `PCA(k)+LinearRegression` (`pcr.py:38-41`); n4m `PCR` = `Algorithm.PCR + Solver.SVD` with lock-step center/scale (`_regression.py:190-224`). Same fit/predict contract, full predict-on-new-X. Note: nirs4all default `k=10`, n4m default `k=2` (shim passes `k` through). nirs4all PCR has **no `_estimator_type`** (defaults via RegressorMixin). |
| **MBPLS** | `pls4all.sklearn.MBPLSRegression` | `n_components→n_components`; `standardize→`(n4m centers internally); `method('NIPALS')→`(n4m mb forces NIPALS); `max_tol→`tol; **requires `block_sizes`**; `backend→` — | **adapter-needed** | RELEASE_READINESS "1:1 confirmed" (line 402). Adapter-needed because n4m `MBPLSRegression` **mandates `block_sizes=[k1,k2,…]`** (`_method_result_regressors.py:157-167`) — nirs4all `MBPLS` infers/accepts a list-of-blocks X instead (`mbpls.py:522`). Shim must derive `block_sizes`. n4m mb_pls fully predicts on new X (coef in original space, `pls.h:1316-1326`). |
| **DiPLS** | **no equivalent (semantic mismatch)** → `pls4all.sklearn.DIPLSRegression` is a *different* method | `lags`,`cv_splits`,`tol`,`max_iter` → ✗ no analogue | **keep-native** | **Name collision / wrong method.** nirs4all `DiPLS` = *Dynamic* PLS with time `lags` via `trendfitter` (`dipls.py:73-100`, tags `dynamic/time-series`). pls4all `DIPLSRegression` = *Domain-Invariant* PLS (Nikzad-Langerodi), needs `X_target` (`_method_result_regressors.py:101-128`). **These are not the same algorithm** — do NOT map. RELEASE_READINESS line 402 lists "DiPLS 1:1 confirmed" — that claim is **incorrect** (see §6). |
| **SparsePLS** | `pls4all.sklearn.SparsePLSRegression` or `SparseSimplsRegression` | `n_components→n_components`; `alpha→sparsity_lambda`; `max_iter→max_iter`; `tol→tol`; `scale→scale_x/scale_y`; `backend→` — | **adapter-needed** | RELEASE_READINESS "1:1 confirmed" (line 402). nirs4all `alpha` (regularization strength, `sparsepls.py:432-440`) maps to n4m `sparsity_lambda` (soft-threshold magnitude, `_regression.py:227-265`) — **scale/semantics differ**, so the shim must translate and parity-check, not pass straight through. Both predict on new X. |
| **LWPLS** | `pls4all.sklearn.LWPLSRegression` (in-sample-only) | `n_components→n_components`; `lambda_in_similarity→`(no n4m arg — n4m takes `n_neighbors`); `scale→`; `backend→`; `batch_size→` — | **blocked** | **Genuinely local.** n4m `lw_pls_fit` exports only `neighbor_indices` + per-sample local fits, **no global coefficients** (`pls.h:1342-1346`); the wrapper raises `NotImplementedError` on predict-on-new-X (`_in_sample.py:87-101`). nirs4all `LWPLS` *does* predict on new X (lazy per-query local fit, `lwpls.py:1021`). **C-ABI gap:** needs a new C entry-point that accepts a separate predict-X (refit-per-query). Cannot drop in today. |
| **SIMPLS** | `pls4all.sklearn.PLSRegression(solver='simpls')` | `n_components→n_components`; `center→center_x`(+`center_y`); `scale→scale_x`(+`scale_y`); `backend→` — | **adapter-needed** | Mappable-by-param (RELEASE_READINESS line 403). `PLSRegression` default solver is already `'simpls'` (`_regression.py:131`). Adapter-needed only for the arg-name translation (`center`/`scale`→`center_x`/`scale_x`/`center_y`/`scale_y`); full predict-on-new-X. |
| **IntervalPLS** | **no regressor equivalent** (n4m `IntervalSelector` is selection-only) | `n_intervals`,`interval_width`,`cv`,`scoring`,`mode`,`combination_method` → ✗ | **keep-native** | nirs4all `IntervalPLS` is a full **regressor** that selects intervals via CV then fits+predicts PLS on them (`ipls.py:1166-1205`). n4m's `IntervalSelector` (`_selection.py`) is a **`SelectorMixin`** — it transforms/masks columns but has **no `predict`**. No regressor counterpart → keep-native (or compose `IntervalSelector` + `PLSRegression` in a Pipeline as a future port). RELEASE_READINESS line 404 lists "IntervalPLS(regressor)" under "No n4m equivalent". |
| **RobustPLS** | `pls4all.sklearn.RobustPLSRegression` (currently in-sample-only wrapper) | `n_components→n_components`; `weighting('huber'/'tukey')→`(n4m default = PRM Fair-weight; legacy Huber via cfg); `c→huber_k`; `max_iter→max_irls_iter`; `tol,scale,center,backend→` — | **adapter-needed** | **NOT blocked.** The C ABI `n4m_robust_pls_fit` **exports `coefficients`, `x_mean`, `y_mean`** (`pls.h:773-777`) — the `_in_sample.py` wrapper (line 146) is merely *conservative* and refuses predict-on-new-X. A **pure-Python wrapper fix** (read coef like `_method_result_regressors.py`) restores predict-on-new-X with no C work. nirs4all `RobustPLS` predicts on new X (`robust_pls.py:780`). See §6 (stale docstring) and follow-ups. |
| **RecursivePLS** | `pls4all.sklearn.RecursivePLSRegression` (in-sample-only) — semantic mismatch | `forgetting_factor→`(no n4m arg — n4m takes `window_size`); `scale,center,backend→` — | **blocked** | **Different algorithm + no coef.** nirs4all `RecursivePLS` = exponentially-weighted online RPLS with a `forgetting_factor`, predicts on new X (`recursive_pls.py:555-568`, `:788`). n4m `recursive_pls_run` = fixed `window_size` moving-window **evaluator**, exports only `predictions` + `in_window` mask, **no coefficients** (`pls.h:728-731`); wrapper raises `NotImplementedError` (`_in_sample.py:200-214`). Neither the semantics nor predict-on-new-X line up. **C-ABI gap:** would need a forgetting-factor RPLS that exports a global coef. Keep-native is the pragmatic call until then. |
| **KOPLS** | **no equivalent** (n4m has KernelPLS, not Kernel-OPLS) | `n_components`,`n_ortho_components`,`kernel`,`gamma`,`degree`,`coef0`,`center`,`scale`,`backend` → ✗ | **keep-native** | n4m `KernelPLSRegression` is kernel-PLS (Rosipal–Trejo), **not** Kernel-OPLS (orthogonal kernel decomposition). No n4m K-OPLS kernel. `kopls.py:646-667`. |
| **KernelPLS** (from `nlpls.py`) | `pls4all.sklearn.KernelPLSRegression` (in-sample-only) | `n_components→n_components`; `kernel→kernel_type`(int enum); `gamma→gamma`; `degree→degree`; `coef0→coef0`; `center_kernel→`; `scale_y→`; `backend→` — | **blocked** | Mappable-by-param (RELEASE_READINESS line 403) **but in-sample-only**: n4m `kernel_pls_fit` exports `alpha`+`y_mean` but **not the kernel-centering statistics** needed for `K(X_new, X_train)`, so the wrapper raises `NotImplementedError` on new X (`_method_result_regressors.py:311-380`, esp. 367-380; `pls.h:869-873`). nirs4all `KernelPLS` predicts on new X (`nlpls.py:747`). **C-ABI gap:** export kernel-centering means. |
| **NLPLS** | `pls4all.sklearn.KernelPLSRegression` (in-sample-only) | same as KernelPLS above (`kernel→kernel_type`, etc.) | **blocked** | NLPLS and KernelPLS are the same `nlpls.py` family (NLPLS/KernelPLS/KPLS aliases, `nlpls.py:53` re-export). Same in-sample-only kernel-centering gap as the KernelPLS row. |
| **KPLS** (alias from `nlpls.py`) | `pls4all.sklearn.KernelPLSRegression` (in-sample-only) | same as KernelPLS | **blocked** | Alias of the `nlpls.py` KernelPLS (`nlpls.py:53`). Same kernel-centering C-ABI gap. |
| **OKLMPLS** | **no equivalent** | `featurizer`,`lambda_dyn`,`lambda_reg_y`,`max_iter`,`tol`,`warm_start_pls`,`standardize`,`backend`,`random_state` → ✗ | **keep-native** | Koopman-latent-map dynamic PLS with a pluggable `featurizer` (`oklmpls.py:766-789`); no n4m analogue. RELEASE_READINESS line 404. |
| **FCKPLS** | **no equivalent** | `alphas`,`sigmas`,`kernel_size`,`mode`,`kernel_type`,`standardize`,`backend` → ✗ | **keep-native** | Fractional convolutional-kernel PLS (`fckpls.py:595-614`); no n4m analogue. RELEASE_READINESS line 404. (Also exports `FractionalPLS`, `FractionalConvFeaturizer` — keep-native.) |
| **AOMPLSRegressor** | **no equivalent** (paper companion) | bank/operator args → ✗ | **keep-native** | Vendored `aom-nirs` (`aom_pls.py:20` re-exports `_aom_nirs.pls.estimators`). pls4all only ships AOM *selection* result handles (`aom_global_select`/`aom_per_component_select`), not the sklearn AOM-PLS regressor. |
| **POPPLSRegressor** | **no equivalent** (paper companion) | bank args → ✗ | **keep-native** | Vendored `aom-nirs` per-component AOM-PLS (`pop_pls.py:17`). |
| **AOMPLSAomlibRegressor** | **no equivalent** | `n_components`,`selection`,`cv`,`one_se`,`preprocessing`,`osc_n_components`,`asls_*`,`center`,`external_folds`, **`X_val`/`y_val`** → ✗ | **keep-native** | Wraps the external `aompls` C++ backend (`aom_pls_aomlib.py:149-209`). **Only model with `X_val`/`y_val` in its `fit` signature** (`aom_pls_aomlib.py:211-217`, currently unused). No n4m analogue. |
| **AOMPLSClassifier** | **no equivalent** (paper companion) | — | **keep-native** | Vendored `aom-nirs` AOM-PLS-DA (`aom_pls_classifier.py:9`). |
| **POPPLSClassifier** | **no equivalent** (paper companion) | — | **keep-native** | Vendored `aom-nirs` POP-PLS-DA (`pop_pls_classifier.py:5`). |
| **AOMRidge family** (`AOMRidgeRegressor`, `AOMRidgeClassifier`, `AOMRidgeBlender`, `AOMRidgeAutoSelector`, `AOMRidgePLS`, `AOMRidgePLSCV`, `AOMMultiKernelRidge`, `AOMKernelizer`, `AOMMultiBranchMKL`, `AOMLocalRidge`) | **partial equivalent**: native `n4m.aom_ridge_blender` for strict-linear compact/wide Ridge OOF blending; broader family still paper companion | — | **keep-native except covered blender subset** | Vendored `aom-nirs.ridge` (`aom_ridge.py:16-27`) still contains many non-covered AOMRidge variants. |
| **FastAOM family** (`FastAOMPLSRidge`, `FastAOMConfig`, `SingleChainPLSRidge`, `HardAOMChainPLSRidge`, `SoftAOMChainPLSRidge`, `SparseMultiKernelRidge`) | **no equivalent** (paper companion) | — | **keep-native** | All vendored `aom-nirs.fast` (`aom_fast.py:18-47`). No n4m counterpart. |
| **TabPFNNIRSRegressor** | **no equivalent** | `n_estimators`,`max_features`,`sg_*`,`osc_n_components`,`random_state`,`device`,`model_path` → ✗ | **keep-native** | Foundation-model regressor (`tabpfn_nirs.py:103-123`); no n4m analogue. Also note: **no `_estimator_type`** declared. |

> The model `__init__.py` `__all__` also exports a large set of AOM **operators**
> and **banks** (`LinearSpectralOperator`, `SavitzkyGolayOperator`,
> `WhittakerOperator`, `ComposedOperator`, `default_bank`, `compact_bank`, …)
> and FastAOM internals. These are **not estimators** (no `fit`/`predict` model
> contract) — they are preprocessing operators / config helpers and are out of
> scope for the model mapping. (Some have n4m preprocessing counterparts in the
> `n4m` package, but that is the preprocessing-parity surface, not model G3.)

---

## 5. Verdict counts

Tally over the **42 estimator names** in nirs4all's
`operators/models/sklearn/__init__.py` `__all__` (operators/banks excluded;
KernelPLS/NLPLS/KPLS are one underlying method exported under 3 names; the
AOMRidge family is 10 names, FastAOM 6 names):

| verdict | count | classes |
|---|---|---|
| **drop-in** | **1** | PCR |
| **adapter-needed** | **8** | PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, SparsePLS, SIMPLS, RobustPLS |
| **blocked** | **5** | LWPLS, RecursivePLS, KernelPLS, NLPLS, KPLS |
| **keep-native** | **28** | DiPLS, IntervalPLS, KOPLS, OKLMPLS, FCKPLS (+`FractionalPLS`,`FractionalConvFeaturizer`), AOMPLSRegressor, POPPLSRegressor, AOMPLSAomlibRegressor, AOMPLSClassifier, POPPLSClassifier, AOMRidge family (10), FastAOM family (6), TabPFNNIRSRegressor |

(1 + 8 + 5 + 28 = 42, matching the estimator names in `__all__`; the additional
`__all__` entries beyond these 42 are AOM **operators**/**banks**, not
estimators — see the note under the table.)

**RobustPLS is in `adapter-needed`, not `blocked`**, because its C ABI already
exports coefficients (`pls.h:773-777`) — the only work is a pure-Python wrapper
change (RELEASE_READINESS.md `in-sample-predict-coef-export`, line 410-414).

## 6. nirs4all classes with NO n4m equivalent (keep-native)

- **DiPLS** (dynamic/time-series — *not* the domain-invariant `DIPLSRegression`;
  name collision)
- **IntervalPLS** (as a regressor; n4m only has the selector)
- **KOPLS** (Kernel-OPLS)
- **OKLMPLS** (Koopman-latent dynamic PLS)
- **FCKPLS** (+ `FractionalPLS`, `FractionalConvFeaturizer`)
- **AOMPLSRegressor**, **POPPLSRegressor**, **AOMPLSAomlibRegressor**,
  **AOMPLSClassifier**, **POPPLSClassifier**
- **AOMRidge** family (10 classes)
- **FastAOM** family (6 classes)
- **TabPFNNIRSRegressor**

---

## 7. Concrete follow-up tasks (per class)

### 7a. Requires NEW n4m / C-ABI work (cannot be done in pure Python)

- **LWPLS — needs a C-ABI predict-on-new-X.** `lw_pls_fit` is local
  (`neighbor_indices` + per-sample fits, no global coef, `pls.h:1342-1346`).
  Add a C entry-point that takes (X_train, y_train, X_new) and runs the
  per-query local fit at predict time. (`in-sample-predict-coef-export` /
  the "genuinely local" half of RELEASE_READINESS line 413.)
- **KernelPLS / NLPLS / KPLS — needs kernel-centering export.** `kernel_pls_fit`
  exports `alpha`+`y_mean` but not the training-row/global-mean kernel-centering
  statistics (`pls.h:869-873`, wrapper `_method_result_regressors.py:367-380`).
  Add those to the C result so `K(X_new, X_train)` can be centered consistently.
- **RecursivePLS — needs a forgetting-factor RPLS that exports a global coef**
  (n4m's `recursive_pls_run` is window-based and prediction-only,
  `pls.h:728-731`). Different algorithm; either port nirs4all's EW-RPLS into the
  C core or keep native.
- **PLSDA / OPLSDA — `predict_proba` on the n4m classifier wrappers.** n4m
  `PLSDAClassifier`/`OPLSDAClassifier` expose only `decision_function`/`predict`
  (`_classification.py:119-132`, `:135`). nirs4all exposes `predict_proba`
  (`plsda.py:55`, `oplsda.py:159`). Either add `predict_proba`+`classes_`-proba
  to the n4m wrapper (pure-Python, no C work — it's a column-stack/raw-score
  transform) or have the shim synthesize it. (RELEASE_READINESS
  `plsda-predict-proba`, line 421.) *This one is pure-Python despite living in
  this subsection because it touches the n4m wrapper.*

### 7b. Pure-Python shim work (no n4m engine change)

- **RobustPLS — read coefficients in the wrapper.** `n4m_robust_pls_fit`
  already exports `coefficients`/`x_mean`/`y_mean` (`pls.h:773-777`); promote
  `RobustPLSRegression` from the in-sample contract (`_in_sample.py:146`) to a
  `_MethodResultRegressor` (the pattern in `_method_result_regressors.py`).
  **Also fix the stale docstring** at `_method_result_regressors.py:12-16` that
  wrongly says `robust_pls_fit`/`ridge_pls_fit`/`continuum_regression_fit`
  "expose only predictions" — the C ABI exports coefficients for all three
  (`pls.h:773-804`). **RidgePLS** and **ContinuumRegression** are the same
  pure-Python fix (not in nirs4all's `__all__`, but relevant if added).
- **PCR — straight delegating shim.** Translate `n_components` only; pass
  through. (drop-in.)
- **SIMPLS — arg-name shim.** Map `center→center_x/center_y`,
  `scale→scale_x/scale_y`; target `PLSRegression(solver='simpls')`.
- **OPLS / OPLSDA — two-knob→one-knob shim.** Collapse
  `n_components`(ortho)+`pls_components`(predictive) into the single n4m
  `n_components`; choose a convention and parity-check.
- **MBPLS — derive `block_sizes`.** nirs4all accepts list-of-blocks X; the n4m
  target mandates `block_sizes` (`_method_result_regressors.py:157-167`). Shim
  computes block sizes from the block list / config.
- **SparsePLS — translate `alpha`→`sparsity_lambda`** and parity-check (scales
  differ).
- **IKPLS — map `center`/`scale` and pick a solver** (`'kernel'`/`'wide-kernel'`);
  document that `algorithm=1/2` and `backend='jax'` have no n4m analogue.

### 7c. Packaging / cross-cutting

- **Build the method-result FFI in the full `n4m` package** *(prerequisite for
  any swap onto `n4m` rather than `pls4all`)* — see §1 flag. Until then, all
  drop-in/adapter targets must import from `pls4all.sklearn`.
- **Packaging decision** (`consolidate-pypi-packaging`, RELEASE_READINESS
  line 415): one dist re-exporting both `n4m`+`pls4all`, or two pinned dists —
  each bundles its own libn4m; verified not a double-free risk, only doubled
  payload.
- **Reproducibility note + statistical-equivalence test** for stochastic swaps
  (PCG64 vs numpy global RNG; §2).
- **`X_val`/`y_val` plumbing note**: only `AOMPLSAomlibRegressor` carries
  `X_val`/`y_val` in `fit` today (and ignores them); n4m model `fit(X, y)` has no
  such kwarg and would silently drop them. No action needed unless a mapped
  class starts consuming them (`fit-val-kwarg-support`, RELEASE_READINESS
  line 431).
- **`_estimator_type` gap**: PCR and TabPFNNIRSRegressor (and OPLSDA on the
  nirs4all side) don't declare `_estimator_type`; the delegating shim should set
  it explicitly to keep StackingRegressor/StackingClassifier introspection
  working.

---

## 8. Correctness concerns surfaced while building this map

1. **`DiPLS` is a name collision, not a mapping.** RELEASE_READINESS.md line 402
   lists "DiPLS" under "1:1 confirmed", but nirs4all `DiPLS` is **Dynamic** PLS
   (time `lags`, via `trendfitter`, `dipls.py:73`) while pls4all
   `DIPLSRegression` is **Domain-Invariant** PLS (Nikzad-Langerodi, needs
   `X_target`, `_method_result_regressors.py:101`). Mapping them would silently
   swap one algorithm for an unrelated one. → keep-native (or rename to avoid
   future confusion).
2. **Stale wrapper docstring overstates the in-sample limitation.**
   `pls4all/sklearn/_method_result_regressors.py:12-16` claims `robust_pls_fit`,
   `ridge_pls_fit`, and `continuum_regression_fit` "expose only predictions …
   not coefficients". The C ABI contradicts this: all three export
   `coefficients`/`x_mean`/`y_mean` (`pls.h:773-804`). The corresponding
   `_in_sample.py` wrappers are therefore *unnecessarily* blocked; this directly
   affects the RobustPLS verdict (adapter-needed, not blocked).
3. **The full `n4m` package cannot serve any model today** (no method-result
   FFI). Every drop-in/adapter target resolves to `pls4all.sklearn`. A
   single-import "preprocessing + models from `n4m`" story needs the FFI port
   first. (§1 flag.)
