Pour une librairie C++ PLS/NIRS sérieuse, je ne ferais pas une classe par acronyme. Je ferais plutôt un **moteur PLS composable** avec plusieurs axes : type de tâche, solveur, déflection, régularisation, orthogonalisation, blocs, sélection de variables, preprocessing, validation, adaptation de domaine, puis éventuellement backend GPU/batch.

L’existant confirme cette structuration : scikit-learn expose déjà `PLSRegression`, `PLSCanonical`, `CCA` et `PLSSVD`, avec une logique autour de la covariance croisée et de NIPALS/SVD ; le package R `pls` expose PLSR, PCR, CPPLS, NIPALS/orthogonal scores, SIMPLS, kernel PLS et wide-kernel PLS ; `ropls` couvre PLS(-DA)/OPLS(-DA) avec diagnostics ; `mixOmics` couvre sPLS et MB-sPLS ; et `plsVarSel` / `auswahl` montrent l’importance énorme des wrappers de sélection de variables autour de la PLS. ([Scikit-learn][1])

## 1. Famille cœur : les modèles PLS de base

À proposer dès le socle.

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

Détail :

| Variante           | Rôle                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `PLS1`             | Régression PLS avec une seule variable Y.                                                     |
| `PLS2`             | Régression PLS multi-sortie.                                                                  |
| `PLSRegression`    | API générique PLS1/PLS2 selon la dimension de Y.                                              |
| `PLSCanonical`     | Relation symétrique entre deux blocs X et Y.                                                  |
| `PLSSVD`           | SVD directe de la covariance croisée `XᵀY`.                                                   |
| `CCA` / PLS mode B | Très proche de certaines variantes canoniques.                                                |
| `PCR`              | Pas PLS strictement, mais baseline indispensable en chimiométrie.                             |
| `CPPLS`            | Canonical Powered PLS, utile comme variante intermédiaire entre régression et discrimination. |

Dans la librairie, je ferais au minimum :

```cpp
PLSRegressor
PLSCanonical
PLSSVD
PCRRegressor
CPPLSRegressor
```

## 2. Variantes de solveurs numériques

Ici, il faut distinguer **le modèle statistique** et **l’algorithme de calcul**.

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

Détail :

| Solveur               | Pourquoi l’avoir                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `NIPALS`              | Référence historique, gère bien les grands `p`, extensible, compatible missing values.       |
| `OrthogonalScoresPLS` | Version NIPALS/orthogonal scores très utilisée dans les packages R.                          |
| `SIMPLS`              | Rapide, direct, élégant, souvent bon pour PLSR classique.                                    |
| `KernelPLSAlgorithm`  | Attention : ici “kernel” au sens algorithme linéaire efficace, pas kernel RBF.               |
| `WideKernelPLS`       | Variante adaptée aux matrices larges, cas fréquent en spectroscopie.                         |
| `PLSSVD`              | Très utile pour transformation/covariance, moins pour toute la logique régression itérative. |
| `PowerMethodPLS`      | Alternative à SVD complète, utile pour gros problèmes.                                       |
| `RandomizedSVDPLS`    | Pour très grandes matrices ou backend GPU/batch.                                             |
| `StableSIMPLS`        | SIMPLS avec garde-fous numériques, car SIMPLS peut être instable dans certains cas.          |
| `MissingNIPALS`       | Gestion des `NaN` sans imputation complète.                                                  |

À prévoir dans le code :

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

## 3. Modes de déflection

C’est fondamental. Beaucoup de “variantes PLS” sont en fait des choix différents de déflection.

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

À exposer explicitement :

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

Important aussi :

```text
- normalisation des scores ;
- normalisation des poids ;
- convention de signe ;
- mode de calcul des rotations ;
- coefficients cumulés par nombre de composantes ;
- intercepts et scalings sauvegardés ;
- reconstruction X / Y optionnelle.
```

## 4. PLS de régression et variantes de perte

À proposer pour couvrir les cas NIRS réels.

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

Priorité :

| Variante                   |                                              Priorité |
| -------------------------- | ----------------------------------------------------: |
| Standard PLSR              |                                         indispensable |
| Weighted PLS               |                                         indispensable |
| Robust PLS                 |                                            très utile |
| Ridge/regularized PLS      |                                                 utile |
| Continuum regression       |                                 utile mais secondaire |
| MIR-PLS                    | intéressant pour toi, surtout si tu l’as déjà exploré |
| Bayesian/probabilistic PLS |                                 recherche / optionnel |
| Quantile PLS               |                                                 niche |

Pour la NIRS, `WeightedPLS` est important : les poids échantillons permettent de gérer des jeux déséquilibrés, des lots, des distances locales, ou des mesures de confiance.

## 5. PLS discriminante / classification

À proposer parce que beaucoup de papiers NIRS font de la classification.

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

Détail :

| Variante      | Principe                                                     |
| ------------- | ------------------------------------------------------------ |
| `PLSDA`       | Y one-hot ou dummy, puis régression PLS + règle de décision. |
| `OPLSDA`      | Version orthogonalisée pour séparation/interprétation.       |
| `SparsePLSDA` | Sélection de variables intégrée.                             |
| `PLSLDA`      | Scores PLS puis LDA. Très utile et robuste.                  |
| `PLSLogistic` | Scores PLS puis régression logistique.                       |
| `PLSGLM`      | Extension aux familles binomiale, Poisson, etc.              |
| `PLSCox`      | Pour données de survie, plutôt hors NIRS classique.          |

Je mettrais `PLSDA`, `OPLSDA`, `PLSLDA` et `SparsePLSDA` dans le socle avancé.

## 6. PLS orthogonale et variantes interprétables

Indispensable pour chimiométrie, omics, spectroscopie.

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

Détail :

| Variante  | Rôle                                                                                     |
| --------- | ---------------------------------------------------------------------------------------- |
| `OPLS`    | Sépare variation prédictive et variation orthogonale à Y.                                |
| `OPLSDA`  | Classification avec composantes prédictives/orthogonales.                                |
| `O2PLS`   | Modélise variation commune et spécifique de deux blocs.                                  |
| `OSCPLS`  | Orthogonal Signal Correction avant ou pendant PLS.                                       |
| `DOSCPLS` | Direct Orthogonal Signal Correction.                                                     |
| `EPOPLS`  | External Parameter Orthogonalization, utile pour température, humidité, instrument, etc. |
| `OnPLS`   | Multi-blocs avec séparation variation globale/locale/unique.                             |

Pour ton contexte NIRS, je mettrais en priorité :

```text
OPLS
OPLS-DA
OSC-PLS
EPO-PLS
O2PLS
```

## 7. PLS sparse, pénalisée et sélection intégrée

À proposer tôt, mais pas forcément dès le MVP.

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

Détail :

| Variante         | Utilité NIRS                                                           |
| ---------------- | ---------------------------------------------------------------------- |
| `SparsePLS`      | Sélection de longueurs d’onde directement dans le modèle.              |
| `GroupSparsePLS` | Sélection de bandes spectrales plutôt que points isolés.               |
| `FusedSparsePLS` | Favorise des longueurs d’onde contiguës. Très pertinent spectralement. |
| `ElasticNetPLS`  | Combine sparsité et stabilité sur variables corrélées.                 |
| `SparseOPLS`     | Interprétation + sélection.                                            |
| `SparseMBPLS`    | Multi-blocs + sélection.                                               |

Pour la NIRS, `GroupSparsePLS` et `FusedSparsePLS` sont plus cohérents que du Lasso pur point par point, car les longueurs d’onde voisines sont fortement corrélées.

## 8. PLS multi-blocs / multi-tables

À prévoir comme gros axe de la librairie.

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

Détail :

| Variante            | Rôle                                                           |
| ------------------- | -------------------------------------------------------------- |
| `MBPLS`             | Plusieurs blocs X alignés sur les mêmes échantillons.          |
| `MBSparsePLS`       | Multi-blocs + sélection de variables.                          |
| `MBPLSDA`           | Classification multi-blocs.                                    |
| `SOPLS`             | Sequential Orthogonalized PLS, blocs traités séquentiellement. |
| `ROSA`              | Response-Oriented Sequential Alternation.                      |
| `HierarchicalMBPLS` | Scores par bloc puis modèle supérieur.                         |
| `ConsensusPLS`      | Fusion de plusieurs modèles/blocs.                             |
| `OnPLS`             | Décomposition variation commune/locale/unique.                 |
| `DIABLO`-like       | Version classification supervisée multi-omics/multi-blocs.     |

À prévoir côté API :

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

Très utile si tu veux couvrir imagerie spectrale, EEM, séries temporelles, cubes hyperspectraux.

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

Exemples d’entrée :

```text
samples × wavelengths × time
samples × x_pixels × y_pixels × wavelengths
samples × excitation × emission
samples × wavelengths × instruments
```

À mon avis, ce n’est pas MVP, mais il faut prévoir la structure de données.

## 10. Kernel et non-linéaire

Deux familles à bien distinguer.

### 10.1 Kernel algorithm PLS linéaire

C’est l’algorithme “kernel PLS” historique pour calculer efficacement une PLS linéaire sur matrices larges. À inclure.

```text
Kernel algorithm PLS
Wide-kernel PLS
```

### 10.2 Kernel PLS non-linéaire

Là on parle de noyaux type machine learning.

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

À proposer en API :

```cpp
enum class KernelType {
    Linear,
    RBF,
    Polynomial,
    Sigmoid,
    Precomputed
};
```

Mais je mettrais ça après le cœur linéaire, parce que la validation et le tuning deviennent vite lourds.

## 11. Local PLS, adaptive PLS, just-in-time PLS

Très important pour gros jeux hétérogènes NIRS.

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

Principe :

```text
Pour chaque prédiction :
    1. trouver les voisins ou échantillons pertinents ;
    2. pondérer les échantillons ;
    3. fitter une PLS locale ;
    4. prédire le nouvel échantillon.
```

C’est coûteux, mais c’est exactement le genre de chose qu’un backend C++/batch/GPU peut accélérer.

À prévoir :

```cpp
LocalPLSRegressor
LWPLSRegressor
KNNPLSRegressor
ClusterPLSRegressor
MixturePLSRegressor
```

## 12. Dynamic PLS / process PLS / time-aware PLS

À proposer si tu veux couvrir procédés, monitoring, séries temporelles, phénotypage temporel.

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

Attention au nom `DiPLS` : dans la littérature récente NIRS, `di-PLS` peut aussi désigner **domain-invariant PLS**, donc je séparerais :

```text
DPLS  = Dynamic PLS
diPLS = Domain-invariant PLS
```

## 13. Domain adaptation / calibration transfer

Très important en NIRS réelle : instrument, lot, température, humidité, site, espèce, matrice, forme physique de l’échantillon.

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

`di-PLS` est explicitement présenté comme une technique d’adaptation de domaine permettant de réduire les différences entre domaines liés, par exemple pour adapter une calibration NIR d’une forme physique d’échantillon vers une autre sans nouvelles mesures de référence. ([ScienceDirect][2])

Je mettrais dans la librairie :

```cpp
DomainInvariantPLS
TransferPLS
EPOPLS
PDSAdapter
SlopeBiasAdapter
SpikingAdapter
```

## 14. Sélection de variables / longueurs d’onde autour de la PLS

Ce n’est pas toujours une “variante PLS” au sens strict, mais pour une librairie NIRS, c’est indispensable.

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

### 14.2 UVE / Monte Carlo / stabilité

```text
UVE-PLS
MCUVE-PLS
EMCUVE-PLS
Randomization test PLS
Stability selection PLS
Bootstrap UVE
Jackknife UVE
```

### 14.3 Méthodes intervalle / bandes spectrales

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

### 14.4 Méthodes wrapper / métaheuristiques

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

### 14.5 Méthodes explicitement listées dans `plsVarSel`

À intégrer ou reproduire proprement :

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

`plsVarSel` liste précisément ce type de fonctions autour de la sélection de variables PLS, notamment `bve_pls`, `ga_pls`, `ipw_pls`, `mcuve_pls`, `rep_pls`, `spa_pls`, `stpls`, `VIP`, `T2_pls` et `WVC_pls`. ([CRAN][3])

Côté C++ :

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

## 15. AOM-PLS et variantes liées à ton axe

Pour toi, c’est probablement le plus différenciant.

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

Je séparerais clairement :

| Variante       | Principe                                                   |
| -------------- | ---------------------------------------------------------- |
| `AOMPLS`       | Mélange pondéré d’opérateurs de preprocessing dans la PLS. |
| `POPPLS`       | Choix discret d’un opérateur par composante.               |
| `SoftAOMPLS`   | Mélange différentiable ou pondéré.                         |
| `HardAOMPLS`   | Sélection d’un opérateur unique.                           |
| `SparseAOMPLS` | Pénalité pour forcer peu d’opérateurs.                     |
| `BlockAOMPLS`  | Opérateurs différents selon les blocs.                     |
| `AOMOPLS`      | AOM + séparation prédictif/orthogonal.                     |
| `AOMMBPLS`     | AOM + multi-blocs.                                         |
| `AOMLWPLS`     | AOM + modèle local.                                        |

Pour une librairie C++, je représenterais ça avec des opérateurs linéaires :

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

## 16. Preprocessing NIRS à inclure dans la librairie

Même si ce ne sont pas des variantes de PLS, une librairie PLS/NIRS doit les avoir, sinon elle ne sera pas autonome.

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

Pour AOM, tous ces preprocessings doivent idéalement être représentés comme des opérateurs réutilisables.

## 17. Diagnostics PLS / chimiométrie

À proposer comme API de diagnostic, pas comme plots seulement.

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

`ropls` met en avant R²/Q², permutation testing, outlier detection, VIP et coefficients de régression comme éléments importants de l’analyse PLS/OPLS. ([Bioconductor][4])

## 18. Validation, sélection du nombre de composantes, incertitude

À inclure très tôt. Sinon les modèles ne seront pas comparables.

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

Métriques à exposer :

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

Le package R `pls` expose déjà CV, LOO, MSEP/RMSEP/R² et jackknife ; il faut au moins reproduire ce niveau-là proprement. ([CRAN][5])

## 19. Monitoring / process control avec PLS

Pour applications industrielles ou séries temporelles.

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

Ce n’est pas prioritaire pour NIRS calibration classique, mais très utile pour “soft sensors”.

## 20. Ensembles de PLS

Intéressant pour robustesse et gros benchmarks.

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

À prévoir comme wrappers :

```cpp
BaggingPLS
BoostingPLS
RandomSubspacePLS
ConsensusPLS
MixturePLS
```

## 21. Variantes GPU/batch à prévoir dès le design C++

Même si tu pars CPU C++ maintenant, il faut que l’API permette plus tard :

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

Le vrai gain GPU sera plutôt :

```text
fit 10 000 PLS sur des combinaisons :
    preprocessing × variables × folds × n_components × targets
```

Donc prévoir dès maintenant :

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

## 22. Modèles à exclure ou repousser

Je ne les mettrais pas dans le cœur initial :

```text
PLS-SEM / Partial Least Squares Structural Equation Modeling
PLS path modeling
Neural PLS
Deep PLS
Gaussian-process PLS
Highly specialized survival PLS
All metaheuristics exotic by défaut
```

`PLS-SEM` est un autre monde : sociométrie, modèles à équations structurelles, graphes causaux/latents. Ça risque de polluer une librairie NIRS.

## 23. Priorisation réaliste

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

### v0.2 chimiométrie/NIRS

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

### v0.3 sélection de variables

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

### v0.4 avancé

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

### v0.5 multi-blocs

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

## Synthèse

La librairie devrait proposer, à terme, ces grandes familles :

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
15. AOM-PLS et variantes par composante/bloc/opérateur
16. Diagnostics, validation, monitoring et incertitude
17. Batch/GPU-ready implementations
```

Le cœur C++ doit donc être pensé comme :

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

Pas comme une collection plate de 150 classes indépendantes.

[1]: https://scikit-learn.org/stable/modules/cross_decomposition.html "1.8. Cross decomposition — scikit-learn 1.8.0 documentation"
[2]: https://www.sciencedirect.com/science/article/pii/S0039914021003829 "A brief note on application of domain-invariant PLS for adapting near-infrared spectroscopy calibrations between different physical forms of samples - ScienceDirect"
[3]: https://cran.rstudio.com/web/packages/plsVarSel/plsVarSel.pdf "plsVarSel: Variable Selection in Partial Least Squares"
[4]: https://www.bioconductor.org/packages//release/bioc/html/ropls.html "Bioconductor - ropls"
[5]: https://cran.r-project.org/web/packages/pls/pls.pdf "pls: Partial Least Squares and Principal Component Regression"
