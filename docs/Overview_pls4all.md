For a serious C++ PLS/NIRS library, I would not make a class by acronym. I would rather make a **composable PLS engine** with several axes: task type, solver, deflection, regularization, orthogonalization, blocks, variable selection, preprocessing, validation, domain adaptation, then possibly GPU/batch backend.

The existing structure confirms this structure: scikit-learn already exposes `PLSRegression`, `PLSCanonical`, `CCA` and `PLSSVD`, with logic around cross covariance and NIPALS/SVD; the R package `pls` exposes PLSR, PCR, CPPLS, NIPALS/orthogonal scores, SIMPLS, kernel PLS and wide-kernel PLS; `ropls` covers PLS(-DA)/OPLS(-DA) with diagnostics; `mixOmics` covers sPLS and MB-sPLS; and `plsVarSel` / `auswahl` show the enormous importance of variable selection wrappers around the PLS. ([Scikit-learn][1])

## Current Status (May 2026)

Snapshot of what is delivered in the `phase-49-vissa-pls` tag
(`0.97.0+abi.1.14.0`). The detailed roadmap is in
[`ROADMAP.md`](ROADMAP.md) ; phase notes are in `roadmap/phase-*.md`.
Details of parity gates by method are in `benchmarks/results/parity_gate/`.

| Overview Section | Status | Detail |
|----------------------|--------|---------|
| §1 Core family (PLS1/PLS2/PLSRegression/PLSSVD/PCR/CPPLS) | delivered except CPPLS | PLS1/PLS2, PLSRegression, PLSCanonical, PLSSVD, PCR — CPPLS postponed. |
| §2 Numerical solvers (NIPALS, OrthScores, SIMPLS, Kernel, WideKernel, SVD, Power, Randomized-SVD) | delivered | all 8 solvers are active in `n4m_model_fit`. |
| §3 Deflection modes (regression, canonical, orthogonal) | delivered for already active models | X-only / Y-only / symmetric deflection pushed back. |
| §4–§5 PLS-DA / PLS-LDA / PLS-logistic | delivered | dummy-coding PLS-DA, deterministic LDA/logistic heads in NumPy. |
| §6 OPLS / OPLS-DA / shared predictive multi-response | delivered | orthogonal deflection, shared predictive scores. |
| §7 Sparse / penalized | delivered (internal) | Sparse SIMPLS via soft-thresholding (`N4M_ALGO_SPARSE_PLS`), `fit_sparse_pls_da`, `fit_group_sparse_pls`, `fit_fused_sparse_pls`. |
| §8 Multi-blocks | delivered (internal) | MB-PLS delivered; Added `fit_o2pls`, `fit_so_pls`, `fit_on_pls`, `fit_rosa`. sPLS-DA in §7. |
| §9 Multiway / tensor PLS | delivered (internal) | `fit_n_pls` + `predict_n_pls` (Bro 3-way). PARAFAC-PLS / Tucker-PLS as future variants. |
| §10.1 Kernel linear algorithm / wide-kernel | delivered | solvers `KERNEL_ALGORITHM` and `WIDE_KERNEL`. |
| §10.2 RBF/polynomial non-linear kernel | delivered (internal) | `fit_kernel_pls` / `predict_kernel_pls` for RBF, polynomial, sigmoid via Gram dual. |
| §11 LW-PLS / local PLS | delivered (internal) | LW-PLS delivered (uniform-weight) — covers JIT-PLS. Adaptive PLS / weighted kernel postponed. |
| §12 Dynamic / recursive PLS | delivered (internal) | Recursive PLS moving-window via `fit_predict_recursive_pls`. Strict incremental update postponed. |
| §13 Domain adaptation (supervised DI-PLS, PDS, EPO/OSC) | partial | OSC, EPO and DI-PLS delivered (internal DI-PLS); PDS and DS postponed. |
| §14 Selection of variables | delivered for 5a–5u | rangers, intervals, biPLS, siPLS, stability, UVE, EMCUVE, randomization, SPA, CARS, Random Frog, SCARS, GA, Shaving, REP, IPW, ST, BVE, T2, WVC, WVC-threshold. |
| §15 AOM-PLS / POP-PLS | delivered 6a–6f | strict-linear AOM operators (identity, detrend, SG, fd, NW, Whittaker, FCK), AOM-SIMPLS global selection, POP-SIMPLS covariance per-component selection, public C ABI surface. |
| §16 NIRS Preprocessing | delivered | identity/center/autoscale/Pareto/SNV/MSC/EMSC/Detrend/SG/SG-derivative/Norris-Williams/ASLS/Wavelet(Haar)/OSC/EPO. |
| §17 PLS diagnostics / chemometrics | delivered (internal) | VIP, selectivity ratio, regression/classification metrics, T² Hotelling, Q-residuals, DModX (`cpp/src/core/pls_diagnostics.{hpp,cpp}`). Approximate-PRESS still to port. |
| §18 Validation and choice of components | delivered | splitters, CV engines, SIMPLS component selection, one-SE rule (`select_one_se_components`), `approximate_press` (PRESS + selection). Bayesian rules still to port. |
| §19 Monitoring / process control | delivered (internal) | Empirical thresholds T² / Q at configurable α + alarm flags (`pls_monitoring_fit` / `pls_monitoring_evaluate`). |
| §20 PLS Sets | delivered (internal) | `fit_bagging_pls`, `fit_boosting_pls`, `fit_random_subspace_pls`. |
| §20 PLS Sets | not delivered | postponed. |
| §21 GPU/batch variants | not delivered | Acceleration Roadmap (BLAS, OpenMP, CUDA) remains optional, never on the ABI critical path. |
| §22 Models to exclude | n/a | nothing to deliver. |
| §23 Realistic prioritization v0.1–v0.7 | v0.1–v0.6 mostly covered | v0.7 batch/GPU postponed. |

Performance benchmarks (multi-language: native C++, Python, R, etc.,
multi-size: 200/500/1000/2000/10000 samples × 100/1000/10000
variables) are being instrumented under `benchmarks/` — see
[`ROADMAP.md`](ROADMAP.md) “Benchmark Roadmap” section for details of
phases 7a (delivered), 7b, 7c, 7d, 7e (in preparation).

## 1. Core family: the basic PLS models

To be offered from the base.

```text
PLS1
PLS2
PLSRegression
PLSCanonical
PLSSVD
CCA / PLS mode B
PCR
CPPLS / powered PLS
```

Detail :

| Variant | Role |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `PLS1` | PLS regression with a single variable Y. |
| `PLS2` | Multi-output PLS regression.                                                                  |
| `PLSRegression` | Generic PLS1/PLS2 API according to the dimension of Y. |
| `PLSCanonical` | Symmetric relationship between two blocks X and Y. |
| `PLSSVD` | Direct SVD of the cross covariance `XᵀY`.                                                   |
| `CCA` / PLS mode B | Very close to certain canonical variants.                                                |
| `PCR` | Not strictly PLS, but essential baseline in chemometrics.                             |
| `CPPLS` | Canonical Powered PLS, useful as an intermediate variant between regression and discrimination. |

In the library, I would do at least:

```cpp
PLSRegressor
PLSCanonical
PLSSVD
PCRRegressor
CPPLSRegressor
```

## 2. Variants of numerical solvers

Here, we must distinguish **the statistical model** and **the calculation algorithm**.

```text
NIPALS
Orthogonal Scores PLS
SIMPLS
Kernel algorithm PLS
Wide-kernel PLS
SVD PLS
Power-method PLS
Randomized SVD PLS
QR/SVD-stabilized PLS
Missing-aware NIPALS
Batched PLS
```

Detail :

| Solveur               | Pourquoi l’avoir                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `NIPALS` | Historical reference, handles large `p` well, extensible, compatible missing values.       |
| `OrthogonalScoresPLS` | NIPALS/orthogonal scores version widely used in R packages. |
| `SIMPLS` | Fast, direct, elegant, often good for classic PLSR.                                    |
| `KernelPLSAlgorithm` | Warning: here “kernel” in the sense of an efficient linear algorithm, not RBF kernel.               |
| `WideKernelPLS` | Variant adapted to large matrices, a frequent case in spectroscopy.                         |
| `PLSSVD` | Very useful for transformation/covariance, less so for all iterative regression logic. |
| `PowerMethodPLS` | Alternative to complete SVD, useful for big problems.                                       |
| `RandomizedSVDPLS` | For very large matrices or GPU/batch backend.                                             |
| `StableSIMPLS` | SIMPLS with digital guardrails, because SIMPLS can be unstable in some cases.          |
| `MissingNIPALS` | Management of `NaN` without complete imputation.                                                  |

To be provided in the code:

```cpp
enum class PLSSolver {
    Nipals,
    OrthogonalScores,
    Simpls,
    KernelAlgorithm,
    WideKernel,
    SVD,
    PowerMethod,
    RandomizedSVD
};
```

## 3. Deflection modes

It’s fundamental. Many “PLS variants” are actually different choices of deflection.

```text
Regression deflation
Canonical deflation
Symmetric deflation
X-only deflation
Y-only deflation
X-and-Y deflation
Orthogonal score deflation
Residual deflation
Block-wise deflation
Sequential deflation
```

To expose explicitly:

```cpp
enum class DeflationMode {
    Regression,
    Canonical,
    Symmetric,
    XOnly,
    YOnly,
    XY,
    OrthogonalScores,
    BlockWise,
    Sequential
};
```

Also important:

```text
- score normalization;
- weight normalization;
- sign convention;
- rotation calculation mode;
- cumulative coefficients by number of components;
- saved intercepts and scalings;
- optional X / Y reconstruction.
```

## 4. Regression PLS and loss variants

To be proposed to cover real NIRS cases.

```text
Standard PLSR
Weighted PLS
Sample-weighted PLS
Robust PLS
Ridge-regularized PLS
Penalized PLS
Continuum regression
Monotonic inner relation PLS / MIR-PLS
Heteroscedastic PLS
Quantile PLS
Bayesian / probabilistic PLS
```

Priority :

| Variant |                                              Priority |
| -------------------------- | ----------------------------------------------------: |
| Standard PLSR              |                                         indispensable |
| Weighted PLS               |                                         indispensable |
| Robust PLS |                                            very useful |
| Ridge/regularized PLS |                                                 useful |
| Continuum regression |                                 useful but secondary |
| MIR-PLS | interesting for you, especially if you have already explored it |
| Bayesian/probabilistic PLS |                                 recherche / optionnel |
| Quantile PLS               |                                                 niche |

For NIRS, `WeightedPLS` is important: sample weights make it possible to manage unbalanced games, batches, local distances, or confidence measures.

## 5. PLS discriminante / classification

To propose because many NIRS papers do classification.

```text
PLS-DA
OPLS-DA
sPLS-DA
MB-PLS-DA
MB-sPLS-DA
PLS-LDA
PLS-QDA
PLS-logistic
PLS-GLM
PLS-Cox / survival PLS
One-vs-rest PLS-DA
Multiclass PLS-DA
Multilabel PLS-DA
```

Detail :

| Variant | Principle |
| ------------- | ------------------------------------------------------------ |
| `PLSDA` | Y one-hot or dummy, then PLS regression + decision rule. |
| `OPLSDA` | Orthogonalized version for separation/interpretation.       |
| `SparsePLSDA` | Integrated variable selection.                             |
| `PLSLDA` | PLS then LDA scores. Very useful and robust.                  |
| `PLSLogistic` | PLS scores then logistic regression.                       |
| `PLSGLM`      | Extension aux familles binomiale, Poisson, etc.              |
| `PLSCox` | For survival data, rather outside classic NIRS.          |

I would put `PLSDA`, `OPLSDA`, `PLSLDA` and `SparsePLSDA` in the advanced base.

## 6. Orthogonal PLS and interpretable variants

Essential for chemometrics, omics, spectroscopy.

```text
OPLS
OPLS-DA
O2PLS
OSC-PLS
DOSC-PLS
EPO-PLS
OnPLS
Orthogonalized PLS
Predictive-orthogonal decomposition
```

Detail :

| Variant | Role |
| --------- | ---------------------------------------------------------------------------------------- |
| `OPLS` | Separates predictive variation and orthogonal variation to Y. |
| `OPLSDA` | Classification with predictive/orthogonal components.                                |
| `O2PLS` | Models common and specific variation of two blocks.                                  |
| `OSCPLS`  | Orthogonal Signal Correction avant ou pendant PLS.                                       |
| `DOSCPLS` | Direct Orthogonal Signal Correction.                                                     |
| `EPOPLS` | External Parameter Orthogonalization, useful for temperature, humidity, instrument, etc. |
| `OnPLS` | Multi-blocks with global/local/single variation separation.                             |

For your NIRS context, I would prioritize:

```text
OPLS
OPLS-DA
OSC-PLS
EPO-PLS
O2PLS
```

## 7. Sparse PLS, penalized and integrated selection

To be offered early, but not necessarily from the MVP.

```text
sPLS
sPLS-DA
Group sparse PLS
Block sparse PLS
Fused sparse PLS
Elastic-net PLS
Ridge PLS
Lasso-loading PLS
Sparse kernel PLS
Sparse OPLS
Sparse MB-PLS
```

Detail :

| Variant | NIRS Utility |
| ---------------- | ---------------------------------------------------------------------- |
| `SparsePLS` | Selection of wavelengths directly in the model.              |
| `GroupSparsePLS` | Selection of spectral bands rather than isolated points.               |
| `FusedSparsePLS` | Favors contiguous wavelengths. Very spectrally relevant. |
| `ElasticNetPLS` | Combines sparsity and stability on correlated variables.                 |
| `SparseOPLS` | Interpretation + selection.                                            |
| `SparseMBPLS` | Multi-blocks + selection.                                               |

For NIRS, `GroupSparsePLS` and `FusedSparsePLS` are more coherent than pure point-to-point Lasso, because neighboring wavelengths are strongly correlated.

## 8. Multi-block/multi-table PLS

To be planned as a major axis of the bookstore.

```text
MB-PLS
MB-PLS-DA
MB-sPLS
MB-sPLS-DA
SO-PLS
SO-PLS-DA
ROSA
Hierarchical MB-PLS
Consensus PLS
Block-wise PLS
O2PLS multiblock
OnPLS
DIABLO-like MB-sPLS-DA
```

Detail :

| Variant | Role |
| ------------------- | -------------------------------------------------------------- |
| `MBPLS` | Several X blocks aligned on the same samples.          |
| `MBSparsePLS` | Multi-blocks + variable selection.                          |
| `MBPLSDA` | Multi-block classification.                                    |
| `SOPLS` | Sequential Orthogonalized PLS, blocks processed sequentially. |
| `ROSA`              | Response-Oriented Sequential Alternation.                      |
| `HierarchicalMBPLS` | Scores by block then higher model.                         |
| `ConsensusPLS` | Merging multiple models/blocks.                             |
| `OnPLS` | Common/local/unique variation decomposition.                 |
| `DIABLO`-like | Supervised multi-omics/multi-block classification version.     |

To be expected on the API side:

```cpp
struct Block {
    Matrix X;
    std::string name;
    std::vector<double> wavelengths;
    double block_weight;
};

MBPLSModel fit(const std::vector<Block>& blocks, const Matrix& Y);
```

## 9. Multiway / tensor PLS

Very useful if you want to cover spectral imaging, EEM, time series, hyperspectral cubes.

```text
N-PLS
Tri-PLS
Multiway PLS
Tensor PLS
PARAFAC-PLS
Tucker-PLS
N-way PLS-DA
Multiway MB-PLS
```

Example entries:

```text
samples × wavelengths × time
samples × x_pixels × y_pixels × wavelengths
samples × excitation × emission
samples × wavelengths × instruments
```

In my opinion, it's not MVP, but you have to predict the data structure.

## 10. Kernel and non-linear

Two families to clearly distinguish.

### 10.1 Kernel linear PLS algorithm

This is the historic “kernel PLS” algorithm to efficiently calculate linear PLS on large matrices. To be included.

```text
Kernel algorithm PLS
Wide-kernel PLS
```

### 10.2 Non-linear PLS kernel

Here we are talking about machine learning type kernels.

```text
RBF Kernel PLS
Polynomial Kernel PLS
Sigmoid Kernel PLS
Linear Kernel PLS
Precomputed Kernel PLS
Multiple Kernel PLS
Sparse Kernel PLS
Local Kernel PLS
```

To propose as an API:

```cpp
enum class KernelType {
    Linear,
    RBF,
    Polynomial,
    Sigmoid,
    Precomputed
};
```

But I would put that after the linear core, because validation and tuning quickly become cumbersome.

## 11. Local PLS, adaptive PLS, just-in-time PLS

Very important for large heterogeneous NIRS games.

```text
Local PLS
kNN-PLS
LW-PLS
Just-in-time PLS
Moving-window local PLS
Cluster-wise PLS
Mixture-of-PLS
Recursive local PLS
Adaptive local PLS
Local kernel PLS
Wavelet local PLS
```

Principle:

```text
For each prediction:
    1. find the relevant neighbors or samples ;
    2. weight the samples ;
    3. fit a local PLS ;
    4. predict the new sample.
```

It's expensive, but it's exactly the kind of thing a C++/batch/GPU backend can speed up.

To expect:

```cpp
LocalPLSRegressor
LWPLSRegressor
KNNPLSRegressor
ClusterPLSRegressor
MixturePLSRegressor
```

## 12. Dynamic PLS / process PLS / time-aware PLS

To be suggested if you want to cover processes, monitoring, time series, temporal phenotyping.

```text
Dynamic PLS
Lagged PLS
Time-lagged PLS
Recursive PLS
Moving-window PLS
Adaptive PLS
Online PLS
Exponentially weighted PLS
Dynamic OPLS
Dynamic MB-PLS
State-space PLS
```

Be careful with the name `DiPLS`: in recent NIRS literature, `di-PLS` can also refer to **domain-invariant PLS**, so I would separate:

```text
DPLS  = Dynamic PLS
diPLS = Domain-invariant PLS
```

## 13. Domain adaptation / calibration transfer

Very important in real NIRS: instrument, batch, temperature, humidity, site, species, matrix, physical form of the sample.

```text
Domain-invariant PLS / di-PLS
Transfer PLS
Calibration-transfer PLS
Instrument-transfer PLS
Standard-free transfer PLS
Spiked PLS
Slope/bias updated PLS
Direct Standardization + PLS
Piecewise Direct Standardization + PLS
EPO + PLS
OSC/EPO transfer PLS
Batch-corrected PLS
```

`di-PLS` is explicitly presented as a domain adaptation technique to reduce differences between related domains, for example to adapt an NIR calibration from one physical form of sample to another without new reference measurements. ([ScienceDirect][2])

I would put in the library:

```cpp
DomainInvariantPLS
TransferPLS
EPOPLS
PDSAdapter
SlopeBiasAdapter
SpikingAdapter
```

## 14. Selection of variables / wavelengths around the PLS

It is not always a “PLS variant” in the strict sense, but for a NIRS library, it is essential.

### 14.1 Scores/filter simples

```text
VIP
Regression coefficients
Absolute regression coefficients
Loading weights
Selectivity ratio
Significance Multivariate Correlation / sMC
Covariance selection / CovSel
T²-based selection
Q-residual-based selection
Jackknife coefficients
Bootstrap coefficients
Permutation importance
```

### 14.2 UVE / Monte Carlo / stability

```text
UVE-PLS
MCUVE-PLS
EMCUVE-PLS
Randomization test PLS
Stability selection PLS
Bootstrap UVE
Jackknife UVE
```

### 14.3 Interval / spectral band methods

```text
iPLS
biPLS
siPLS
mwPLS
moving-window PLS
interval random frog
recursive interval PLS
band selection PLS
windowed VIP
windowed CARS
```

### 14.4 Wrapper / metaheuristic methods

```text
GA-PLS
PSO-PLS
ACO-PLS
Simulated annealing PLS
Random Frog PLS
CARS-PLS
SCARS-PLS
CARS-SPA-PLS
SPA-PLS
VISSA-PLS
IRIV-PLS
VCPA-PLS
BOSS-PLS
```

### 14.5 Methods explicitly listed in `plsVarSel`

To integrate or reproduce properly:

```text
BVE-PLS
GA-PLS
IPW-PLS
MCUVE-PLS
REP-PLS
SPA-PLS
ST-PLS
Shaving
Truncation
FilterPLSR
PVR
PVS
T2-PLS
WVC-PLS
LDA-from-PLS
LDA-from-PLS-CV
```

`plsVarSel` lists precisely this type of functions around PLS variable selection, including `bve_pls`, `ga_pls`, `ipw_pls`, `mcuve_pls`, `rep_pls`, `spa_pls`, `stpls`, `VIP`, `T2_pls` and `WVC_pls`. ([CRAN][3])

C++ side:

```cpp
class VariableSelector {
public:
    virtual SelectionResult select(const Matrix& X, const Matrix& Y) = 0;
};

class VIPSelector;
class MCUVESelector;
class CARSSelector;
class SPASelector;
class IPLSSelector;
class GeneticPLSSelector;
class RandomFrogSelector;
```

## 15. AOM-PLS and variants linked to your axis

For you, this is probably the most differentiating thing.

```text
AOM-PLS
AOM-PLS1
AOM-PLS2
AOM-NIPALS
AOM-SIMPLS
AOM-KernelPLS
AOM-OPLS
AOM-PLS-DA
AOM-MB-PLS
AOM-sPLS
AOM-LWPLS
AOM-DomainInvariantPLS
POP-PLS
Hard-gated AOM-PLS
Soft-gated AOM-PLS
Sparse-gated AOM-PLS
Per-component AOM-PLS
Per-block AOM-PLS
Per-target AOM-PLS
```

I would clearly separate:

| Variant | Principle |
| -------------- | ---------------------------------------------------------- |
| `AOMPLS` | Weighted mixture of preprocessing operators in PLS. |
| `POPPLS` | Discrete choice of an operator per component.               |
| `SoftAOMPLS` | Differentiable or weighted mixture.                         |
| `HardAOMPLS` | Selection of a single operator.                           |
| `SparseAOMPLS` | Penalty for forcing few operators.                     |
| `BlockAOMPLS` | Different operators depending on the blocks.                     |
| `AOMOPLS` | AOM + predictive/orthogonal separation.                     |
| `AOMMBPLS` | AOM + multi-blocks.                                         |
| `AOMLWPLS` | AOM + local model.                                        |

For a C++ library, I would represent this with linear operators:

```cpp
class SpectralOperator {
public:
    virtual void apply(const Matrix& X, Matrix& out) const = 0;
};

class AOMPLSRegressor {
    std::vector<std::shared_ptr<SpectralOperator>> operators_;
    PLSSolver solver_;
    GatingMode gating_;
};
```

## 16. NIRS preprocessing to include in the library

Even if they are not PLS variants, a PLS/NIRS library must have them, otherwise it will not be standalone.

```text
Mean centering
Autoscaling
Pareto scaling
Robust scaling
SNV
MSC
EMSC
Detrend
Savitzky-Golay smoothing
Savitzky-Golay derivatives
Finite-difference derivatives
Norris-Williams derivatives
Wavelet transforms
Baseline correction
ASLS baseline
airPLS / arPLS / Whittaker baseline
Normalization by area
Min-max normalization
Vector normalization
Wavelength alignment
Spectral resampling
Splice correction
Dead-band removal
Saturation masking
Outlier masking
OSC
EPO
DOSC
```

For AOM, all these preprocessings should ideally be represented as reusable operators.

## 17. PLS diagnostics / chemometrics

To propose as a diagnostic API, not just plots.

```text
Scores
Loadings
Weights
Rotations
Y-loadings
Explained variance X
Explained variance Y
Regression coefficients per component
VIP
Selectivity ratio
Leverage
Hotelling T²
Q residuals / SPE
DModX
Score distance
Orthogonal distance
Residual distance
Contribution plots
Outlier flags
Applicability domain
```

`ropls` highlights R²/Q², permutation testing, outlier detection, VIP and regression coefficients as important elements of PLS/OPLS analysis. ([Bioconductor][4])

## 18. Validation, selection of the number of components, uncertainty

To be included very early. Otherwise the models will not be comparable.

```text
K-fold CV
LOO CV
Repeated K-fold
Monte Carlo CV
Bootstrap CV
Nested CV
Grouped CV
Blocked CV
Stratified CV
Time-series split
Venetian blinds CV
Kennard-Stone split
SPXY split
Duplex split
Permutation tests
Y-scrambling
Jackknife
Bootstrap confidence intervals
Coefficient confidence intervals
Prediction intervals
```

Metrics to expose:

```text
RMSEC
RMSECV
RMSEP
MSEC
MSECV
MSEP
R² calibration
R² validation
Q²
Bias
SEP
RPD
RPIQ
MAE
Median AE
Slope/intercept observed-vs-predicted
Sensitivity
Specificity
Balanced accuracy
AUC
MCC
```

The R package `pls` already exposes CV, LOO, MSEP/RMSEP/R² and jackknife; we must at least reproduce this level properly. ([CRAN][5])

## 19. Monitoring / process control avec PLS

For industrial applications or time series.

```text
PLS monitoring
PCA/PLS score charts
Hotelling T² chart
SPE/Q chart
Contribution plots
Fault detection PLS
Fault diagnosis PLS
Dynamic PLS monitoring
Recursive PLS monitoring
Batch process PLS
Multiblock process monitoring
```

This is not a priority for classic NIRS calibration, but very useful for “soft sensors”.

## 20. Ensembles de PLS

Interesting for robustness and large benchmarks.

```text
Bagging PLS
Boosting PLS
Random subspace PLS
Variable-space boosting PLS
Ensemble MCUVE
Ensemble CARS
Consensus PLS
Stacked PLS
Mixture-of-experts PLS
Cluster ensemble PLS
```

To use as wrappers:

```cpp
BaggingPLS
BoostingPLS
RandomSubspacePLS
ConsensusPLS
MixturePLS
```

## 21. GPU/batch variants to plan for in C++ design

Even if you start CPU C++ now, the API must allow it later:

```text
Batched PLS fit
Batched PLS predict
Batched CV
Batched preprocessing
Batched variable selection
Batched AOM-PLS
Batched local PLS
Batched bootstrap
Batched permutation tests
```

Ce n’est pas juste :

```text
fit_one_pls_on_gpu()
```

The real GPU gain will instead be:

```text
fit 10 000 PLS sur des combinaisons :
    preprocessing × variables × folds × n_components × targets
```

So plan now:

```cpp
struct PLSBatchJob {
    MatrixView X;
    MatrixView Y;
    PLSConfig config;
    Split split;
    PreprocessingPipeline pipeline;
};

std::vector<PLSResult> fit_batch(const std::vector<PLSBatchJob>& jobs);
```

## 22. Models to exclude or reject

I wouldn't put them in the initial core:

```text
PLS-SEM / Partial Least Squares Structural Equation Modeling
PLS path modeling
Neural PLS
Deep PLS
Highly specialized survival PLS
```

Note Phase 47 (2026-05-16): **Gaussian-process PLS** is now
implemented (`gpr_pls_fit`, parity sklearn rmse_rel ~ 2.3e-10). There
GP head is exposed as standalone primitive
(`fit_gp_on_scores`) to allow a future GPR-on-AOMPLS.

Note Phase 48-49 (2026-05-16): in the §14.4 list of
wrapper metaheuristics, **PSO-PLS** (`pso_select`, Kennedy &
Eberhart 1997, paper-only) and **VISSA-PLS** (`vissa_select`, Deng
et al. 2014, paper-only) are now delivered. The other
exotic metaheuristics (ACO, simulated annealing, IRIV, VCPA,
BOSS, CARS-SPA) remain postponed.

`PLS-SEM` is another world: sociometry, structural equation models, causal/latent graphs. It risks polluting a NIRS library.

## 23. Realistic prioritization

### MVP C++ v0.1

```text
PLS1
PLS2
PLSRegression
NIPALS
SIMPLS
PLSSVD
PCR
VIP
RMSE/R²/Q²
K-fold/LOO CV
predict/transform
coefficients per component
mean/scale/intercept serialization
```

### v0.2 chemometrics/NIRS

```text
SNV
MSC
EMSC
Detrend
Savitzky-Golay
derivatives
ASLS baseline
OSC
EPO
PLS-DA
PLS-LDA
OPLS
OPLS-DA
Hotelling T²
Q residuals
DModX
```

### v0.3 variable selection

```text
VIP selector
Coefficient selector
Selectivity ratio
iPLS
biPLS
moving-window PLS
UVE-PLS
MCUVE-PLS
CARS-PLS
SPA-PLS
GA-PLS
Random Frog
Shaving
BVE-PLS
```

### v0.4 advanced

```text
sPLS
sPLS-DA
Group sparse PLS
Fused sparse PLS
Kernel PLS
LW-PLS
Local PLS
Recursive/adaptive PLS
Domain-invariant PLS
Transfer PLS
```

### v0.5 multi-block

```text
MB-PLS
MB-PLS-DA
MB-sPLS
SO-PLS
ROSA
O2PLS
OnPLS
Hierarchical MB-PLS
```

### v0.6 AOM

```text
AOM-PLS
POP-PLS
AOM-SIMPLS
AOM-NIPALS
AOM-OPLS
AOM-MBPLS
Sparse AOM-PLS
AOM-LWPLS
```

### v0.7 GPU/batch

```text
batched PLS
batched CV
batched preprocessing
batched variable selection
batched AOM
batched local PLS
CUDA/cuBLAS or vendor backend
```

## Summary

The library should eventually offer these major families:

```text
1. PLS1 / PLS2 / PLSRegression
2. PLSCanonical / CCA-like / PLSSVD
3. PCR / CPPLS / continuum variants
4. NIPALS / SIMPLS / kernel algorithm / wide-kernel / SVD solvers
5. PLS-DA / OPLS-DA / sPLS-DA / PLS-LDA
6. OPLS / O2PLS / OSC-PLS / EPO-PLS
7. Sparse / group sparse / fused sparse / penalized PLS
8. MB-PLS / SO-PLS / MB-sPLS / OnPLS / ROSA
9. N-way / tensor PLS
10. Kernel nonlinear PLS
11. LW-PLS / local PLS / adaptive PLS
12. Dynamic / recursive / online PLS
13. Domain-invariant / transfer PLS
14. Variable selection : VIP, MCUVE, CARS, SPA, iPLS, GA, Random Frog, etc.
15. AOM-PLS and variants by component/block/operator
16. Diagnostics, validation, monitoring, and uncertainty
17. Batch/GPU-ready implementations
```

The C++ core must therefore be thought of as:

```text
PLS engine
+ solver strategy
+ deflation strategy
+ preprocessing operators
+ variable selectors
+ validation engine
+ diagnostics engine
+ serialization
+ batch execution
```

Not like a flat collection of 150 independent classes.

[1]: https://scikit-learn.org/stable/modules/cross_decomposition.html "1.8. Cross decomposition — scikit-learn 1.8.0 documentation"
[2]: https://www.sciencedirect.com/science/article/pii/S0039914021003829 "A brief note on application of domain-invariant PLS for adapting near-infrared spectroscopy calibrations between different physical forms of samples - ScienceDirect"
[3]: https://cran.rstudio.com/web/packages/plsVarSel/plsVarSel.pdf "plsVarSel: Variable Selection in Partial Least Squares"
[4]: https://www.bioconductor.org/packages//release/bioc/html/ropls.html "Bioconductor - ropls"
[5]: https://cran.r-project.org/web/packages/pls/pls.pdf "pls: Partial Least Squares and Principal Component Regression"
