Oui. La bonne direction technique, pour ton objectif multi-bindings, c’est :

> **C++ interne, ABI C publique, zéro dépendance obligatoire, bindings très fins par langage.**

Il faut penser le projet comme un **moteur numérique portable**, pas comme une librairie Python/R avec du C++ derrière.

---

# 1. Décision d’architecture principale

## Architecture cible

```text
libnirspls-core
    C++17 interne
    aucune dépendance obligatoire
    pas d’I/O obligatoire
    pas de JSON/YAML obligatoire
    pas de STL exposée publiquement
    pas d’exceptions C++ hors du core

libnirspls-c
    ABI C stable
    headers C publics
    handles opaques
    buffers + strides
    status codes
    version d’ABI

bindings/
    python/
    r/
    matlab/
    js-wasm/
    android/
    autres/
```

Le C++ sert à coder proprement les modèles, les solveurs, les pipelines et les tests. Mais **tous les langages doivent appeler une interface C**, pas des classes C++.

Raison : une ABI C est beaucoup plus stable et portable qu’une ABI C++. Même côté libstdc++, il existe des histoires d’ABI, par exemple le “dual ABI” introduit dans GCC 5.1 pour des types comme `std::string` et `std::list`, ce qui illustre bien pourquoi il ne faut pas exposer des types C++ dans l’interface publique. ([gcc.gnu.org][1])

Donc :

```cpp
// Interdit dans l'API publique
std::vector<double>
std::string
Eigen::MatrixXd
std::unique_ptr<Model>
PLSModel::fit(...)
```

À la place :

```c
// API publique
typedef struct npls_context_t npls_context_t;
typedef struct npls_model_t npls_model_t;
typedef struct npls_config_t npls_config_t;

npls_status_t npls_model_fit(
    npls_context_t* ctx,
    const npls_config_t* cfg,
    const npls_matrix_view_t* X,
    const npls_matrix_view_t* Y,
    npls_model_t** out_model
);
```

---

# 2. Principe fort : zéro dépendance obligatoire

Je viserais cette règle :

```text
Required:
    - C++17 compiler
    - CMake
    - standard C/C++ library

Optional:
    - BLAS/LAPACK
    - OpenMP
    - CUDA/cuBLAS
    - Emscripten
    - Android NDK
    - MATLAB mex
```

Pas de dépendance obligatoire à :

```text
Eigen
Armadillo
Boost
OpenBLAS
MKL
Rcpp
pybind11
nanobind
embind
nlohmann/json
yaml-cpp
```

Certaines sont très bonnes, mais elles augmentent la surface de packaging. Pour ton objectif — R, Python, Android, JS, MATLAB, etc. — chaque dépendance devient un multiplicateur d’ennuis.

La bonne stratégie est :

```text
Backend de référence sans dépendance
    +
Backends optionnels accélérés
```

Exemple :

```text
nirspls-core-reference     // toujours disponible
nirspls-core-blas          // optionnel
nirspls-core-openmp        // optionnel
nirspls-core-cuda          // optionnel
nirspls-core-wasm          // optionnel
```

Mais l’API ne change jamais.

---

# 3. CMake comme unique système de build

CMake est le meilleur choix ici parce qu’il couvre naturellement le C++, les bibliothèques statiques/dynamiques, Android NDK, Emscripten, MATLAB via wrappers, et l’export de targets. La documentation CMake décrit notamment l’export/import de targets et de packages pour réutiliser une librairie depuis d’autres projets. ([cmake.org][2])

Structure recommandée :

```text
nirspls/
    CMakeLists.txt
    include/
        nirspls/
            npls.h              # API C publique
            npls_version.h
    src/
        core/
            linalg/
            pls/
            preprocessing/
            selection/
            validation/
            serialization/
        c_api/
    bindings/
        python/
        r/
        matlab/
        android/
        js/
    tests/
        c_api/
        numerical/
        regression/
        serialization/
    benchmarks/
    examples/
```

Targets CMake :

```text
npls_core          static lib C++ interne
npls_c             shared lib ABI C
npls_cli           outil test/benchmark
npls_tests         tests numériques
npls_wasm          build WebAssembly optionnel
npls_android       build Android optionnel
npls_cuda          backend GPU optionnel
```

---

# 4. Standard C++ recommandé

Je prendrais **C++17**, pas C++20, pour maximiser la portabilité.

```text
C++17 :
    - largement supporté
    - suffisant pour la structure interne
    - compatible Android/wasm/MATLAB plus facilement
    - moins de surprises de toolchain
```

À éviter dans le core portable :

```text
std::filesystem       // parfois pénible sur wasm/Android
coroutines
modules C++20
exceptions à travers ABI
RTTI obligatoire
threading complexe
```

Le core peut utiliser des exceptions en interne, mais toute exception doit être capturée avant de repasser l’interface C.

```cpp
extern "C" npls_status_t npls_model_fit(...) {
    try {
        // C++ interne
        return NPLS_OK;
    } catch (const std::bad_alloc&) {
        return NPLS_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return NPLS_ERR_INTERNAL;
    }
}
```

---

# 5. API C : design minimal et stable

## 5.1 Handles opaques

```c
typedef struct npls_context_t npls_context_t;
typedef struct npls_config_t npls_config_t;
typedef struct npls_model_t npls_model_t;
typedef struct npls_pipeline_t npls_pipeline_t;
typedef struct npls_cv_result_t npls_cv_result_t;
```

Les wrappers Python/R/MATLAB/JS ne voient jamais les classes C++.

---

## 5.2 Status codes

```c
typedef enum {
    NPLS_OK = 0,
    NPLS_ERR_INVALID_ARGUMENT = 1,
    NPLS_ERR_SHAPE_MISMATCH = 2,
    NPLS_ERR_NOT_FITTED = 3,
    NPLS_ERR_NUMERICAL_FAILURE = 4,
    NPLS_ERR_OUT_OF_MEMORY = 5,
    NPLS_ERR_UNSUPPORTED = 6,
    NPLS_ERR_ABI_MISMATCH = 7,
    NPLS_ERR_INTERNAL = 255
} npls_status_t;
```

Pas d’exception côté API C.

---

## 5.3 Contexte d’exécution

```c
npls_status_t npls_context_create(npls_context_t** out_ctx);
void npls_context_destroy(npls_context_t* ctx);

npls_status_t npls_context_set_seed(npls_context_t* ctx, uint64_t seed);
npls_status_t npls_context_set_backend(npls_context_t* ctx, npls_backend_t backend);
const char* npls_context_last_error(const npls_context_t* ctx);
```

Le contexte permet :

```text
- erreurs locales ;
- seed reproductible ;
- backend CPU/GPU ;
- allocateur optionnel ;
- logs optionnels ;
- pas d’état global.
```

---

## 5.4 Matrix view avec strides

Indispensable pour accepter Python, R, MATLAB, JS et Android sans imposer un layout unique.

```c
typedef enum {
    NPLS_DTYPE_F64 = 1,
    NPLS_DTYPE_F32 = 2
} npls_dtype_t;

typedef struct {
    void* data;
    int64_t rows;
    int64_t cols;
    int64_t row_stride;
    int64_t col_stride;
    npls_dtype_t dtype;
} npls_matrix_view_t;
```

Ça permet de passer :

```text
- NumPy row-major ou Fortran-major ;
- R column-major ;
- MATLAB column-major ;
- TypedArray wasm ;
- buffers Android ;
- sous-vues éventuelles.
```

En interne, tu peux copier si nécessaire vers un layout canonique.

Je recommande :

```text
API publique :
    accepte strides quelconques

Core interne :
    canonicalise si besoin

Backend BLAS :
    préfère column-major

Backend wasm/mobile :
    préfère buffers compacts
```

---

## 5.5 Configuration stable

Deux options :

### Option A — struct versionnée

```c
typedef struct {
    uint32_t struct_size;
    int32_t algorithm;
    int32_t solver;
    int32_t n_components;
    int32_t scale_x;
    int32_t center_x;
    int32_t scale_y;
    int32_t center_y;
    double tol;
    int32_t max_iter;
} npls_pls_config_t;
```

Avantage : rapide, simple.

Inconvénient : chaque ajout doit être géré avec soin.

### Option B — config opaque avec setters

```c
npls_status_t npls_config_create(npls_config_t** out_cfg);
void npls_config_destroy(npls_config_t* cfg);

npls_status_t npls_config_set_algorithm(npls_config_t* cfg, npls_algorithm_t algo);
npls_status_t npls_config_set_solver(npls_config_t* cfg, npls_solver_t solver);
npls_status_t npls_config_set_n_components(npls_config_t* cfg, int32_t n_components);
npls_status_t npls_config_set_center_x(npls_config_t* cfg, int enabled);
npls_status_t npls_config_set_scale_x(npls_config_t* cfg, int enabled);
```

Je recommande **Option B** pour la stabilité ABI. Tu peux ajouter des setters sans casser l’interface existante.

---

# 6. Pas de JSON/YAML dans le core

Même si les configs NIRS sont naturellement déclaratives, je ne mettrais pas de parser JSON/YAML dans le core C++.

Mauvais choix pour le core :

```text
npls_fit_from_json(...)
npls_model_to_yaml(...)
```

Bon choix :

```text
Core :
    config typée
    enums
    arrays
    buffers

Bindings :
    Python peut lire YAML/JSON
    R peut lire listes/data.frames
    JS peut lire JSON
    MATLAB peut lire struct
```

Donc :

```text
Python dict  → wrapper Python → npls_config_t
R list       → wrapper R      → npls_config_t
JS object    → wrapper JS     → npls_config_t
MATLAB struct→ wrapper MATLAB → npls_config_t
```

Le core reste indépendant.

---

# 7. Sérialisation : buffer binaire versionné

Pour Android/JS/MATLAB, il faut pouvoir exporter/importer un modèle sans dépendre de Python/R.

Je ferais :

```c
npls_status_t npls_model_export_size(
    const npls_model_t* model,
    size_t* out_size
);

npls_status_t npls_model_export_to_buffer(
    const npls_model_t* model,
    void* buffer,
    size_t buffer_size
);

npls_status_t npls_model_import_from_buffer(
    npls_context_t* ctx,
    const void* buffer,
    size_t buffer_size,
    npls_model_t** out_model
);
```

Format interne :

```text
magic: "NPLS"
format_version
abi_version
endianness
dtype
algorithm
solver
n_samples_fit
n_features
n_targets
n_components
flags
arrays:
    x_mean
    x_scale
    y_mean
    y_scale
    weights W
    loadings P
    y_loadings Q
    rotations R
    coefficients B
    intercept
    preprocessing states
checksum optionnel
```

Important :

```text
- pas de pointeurs sérialisés ;
- pas de std::string sérialisée brute ;
- pas d’objet C++ ;
- format little-endian explicite ;
- versionné ;
- import compatible entre langages.
```

Pour l’humain, les bindings peuvent générer un sidecar JSON :

```text
model.nplsbin
model.metadata.json
```

Mais le core n’a pas besoin de parser JSON.

---

# 8. Cœur numérique : pas de grosse lib obligatoire

## 8.1 Linalg minimale à implémenter

Pour démarrer, il faut peu de choses :

```text
dot
axpy
scal
norm2
center/scale
matrix-vector product
matrix-matrix small/medium product
transpose view
copy/convert layout
power iteration
Gram matrix
orthogonalization
deflation
```

Pour la PLS NIPALS et une bonne partie des variantes, tu peux aller très loin avec ça.

Tu peux éviter de dépendre d’une SVD complète au début.

## 8.2 Solveurs PLS initiaux

Je ferais d’abord :

```text
NIPALS PLS1
NIPALS PLS2
Orthogonal scores PLS
SIMPLS minimal
PLSSVD minimal
PCR minimal
```

Mais en termes de code, le MVP devrait surtout sécuriser :

```text
PLSRegression
    solver = NIPALS
    y_dim = 1 ou >1

predict
transform
coefficients
VIP
CV
serialization
```

SIMPLS et PLSSVD peuvent venir après si la base est propre.

---

# 9. Types de modèles internes

Ne fais pas 150 classes plates. Fais un moteur configurable.

```cpp
enum class Algorithm {
    PLSRegression,
    PLSCanonical,
    PLSDA,
    OPLS,
    MBPLS,
    AOMPLS
};

enum class Solver {
    Nipals,
    Simpls,
    OrthogonalScores,
    SVD,
    PowerMethod
};

enum class DeflationMode {
    Regression,
    Canonical,
    XOnly,
    XY,
    Orthogonal
};
```

En interne :

```cpp
struct PLSFitOptions {
    Algorithm algorithm;
    Solver solver;
    DeflationMode deflation;
    int n_components;
    bool center_x;
    bool scale_x;
    bool center_y;
    bool scale_y;
    double tol;
    int max_iter;
};
```

Mais cette struct interne ne doit pas être exposée directement.

---

# 10. Modèle interne PLS

Structure interne typique :

```cpp
struct PLSModelData {
    int n_features;
    int n_targets;
    int n_components;

    Vector x_mean;
    Vector x_scale;
    Vector y_mean;
    Vector y_scale;

    Matrix W;      // x weights
    Matrix P;      // x loadings
    Matrix Q;      // y loadings
    Matrix T;      // scores, optionnel
    Matrix U;      // y scores, optionnel
    Matrix R;      // rotations
    Matrix B;      // coefficients
    Vector intercept;

    PreprocessingPipelineState preprocessing;
    Diagnostics diagnostics;
};
```

À noter : `T` et `U` peuvent être énormes. Je ne les stockerais pas toujours.

Config :

```text
store_scores = false par défaut
store_diagnostics = false par défaut
store_training_residuals = false par défaut
```

Sinon les modèles exportés deviennent trop gros.

---

# 11. Deux builds : calibration et prédiction

Très important pour Android/JS.

Je séparerais conceptuellement :

## Build complet

Pour Python/R/MATLAB :

```text
fit
predict
transform
cross-validation
variable selection
diagnostics
serialization
```

## Build léger

Pour Android/JS/browser :

```text
import model
predict
transform
preprocessing
applicability domain simple
```

Donc :

```text
libnirspls-full
libnirspls-predict
```

Le build `predict` doit être minuscule, stable, facile à embarquer.

---

# 12. Bindings par langage

## 12.1 Python

Je ne commencerais pas avec pybind11.

Même si pybind11 est très pratique, il ajoute une dépendance et impose un vrai module d’extension CPython. Pour minimiser les problèmes, je commencerais par :

```text
Python wrapper pur
    ctypes
    NumPy
    shared library npls_c
```

Python peut aussi utiliser l’ABI stable CPython si tu choisis plus tard un module C natif avec Limited API ; la doc Python précise que les extensions ciblant la Stable ABI doivent utiliser un sous-ensemble limité de l’API C. ([Python documentation][3])

Mais pour la v1 :

```text
ctypes + NumPy + libnpls_c
```

API Python :

```python
model = npls.PLSRegression(n_components=10, solver="nipals")
model.fit(X, y)
yp = model.predict(X_test)
model.save("model.nplsbin")
```

Le wrapper convertit :

```text
NumPy array → npls_matrix_view_t
dict/config → npls_config_t
```

---

## 12.2 R

Je ne commencerais pas avec Rcpp.

Rcpp est très agréable, mais si tu veux réduire les dépendances, fais un adaptateur R natif via `.Call`. Le manuel “Writing R Extensions” documente les interfaces natives de R comme `.C`, `.Call`, `.Fortran` et `.External`. ([CRAN][4])

Structure :

```text
bindings/r/
    R/
        PLSRegression.R
    src/
        r_gateway.c
        npls_core_sources...
```

Le gateway R :

```c
SEXP npls_r_fit_pls(SEXP X, SEXP Y, SEXP config);
SEXP npls_r_predict(SEXP model_handle, SEXP X);
SEXP npls_r_export(SEXP model_handle);
SEXP npls_r_import(SEXP raw_buffer);
```

R API :

```r
model <- npls_pls(n_components = 10, solver = "nipals")
model <- fit(model, X, y)
pred <- predict(model, Xtest)
save_npls(model, "model.nplsbin")
```

Pour CRAN/Bioconductor, le plus simple est probablement de compiler le core depuis les sources du package R, plutôt que de dépendre d’une lib système installée.

---

## 12.3 MATLAB

MATLAB doit passer par un wrapper MEX.

MathWorks documente les MEX comme des fonctions MATLAB appelant du C/C++ ; la doc indique qu’un fichier MEX se comporte comme une fonction MATLAB. ([MathWorks][5])

Architecture :

```text
MATLAB class
    +
MEX gateway
    +
npls_c ABI
```

Exemple :

```matlab
model = npls.PLSRegression("n_components", 10);
model = model.fit(X, y);
ypred = model.predict(Xtest);
model.save("model.nplsbin");
```

Implémentation :

```text
npls_fit_mex.cpp
npls_predict_mex.cpp
npls_export_mex.cpp
npls_import_mex.cpp
```

Tu peux utiliser l’API C++ MEX moderne, mais ne la mets que dans le binding MATLAB. Le core ne doit jamais dépendre de MATLAB. MathWorks documente l’API C++ MEX moderne séparément. ([MathWorks][6])

---

## 12.4 JavaScript / WebAssembly

Pour JS, il faut compiler le core en WASM avec Emscripten. MDN documente le fait qu’un module C/C++ peut être compilé en WebAssembly avec des outils comme Emscripten. ([MDN Web Docs][7])

Je ne commencerais pas avec `embind`.

Je ferais :

```text
C ABI exports
    +
Emscripten
    +
TypeScript wrapper
```

Emscripten fournit `ccall()` et `cwrap()` pour appeler des fonctions C compilées depuis JavaScript. ([Emscripten][8])

API JS :

```ts
const model = await NPLSModel.load(buffer);
const ypred = model.predict(X);
```

Usage prioritaire JS :

```text
prediction only
visualisation
démo web
nirs4all Studio
```

Je repousserais le fit complet côté navigateur. Le fit complet en wasm est possible, mais il va complexifier la mémoire, les performances et la gestion des gros datasets.

---

## 12.5 Android

Android doit passer par :

```text
C++ core
    +
C ABI
    +
JNI gateway
    +
Kotlin/Java wrapper
```

Le NDK Android supporte CMake pour compiler du C/C++ via l’Android Gradle Plugin ou directement avec CMake. ([Android Developers][9])

Package cible :

```text
AAR
    arm64-v8a/libnpls.so
    x86_64/libnpls.so
    Kotlin wrapper
```

Priorité :

```text
predict-only d’abord
fit ensuite seulement si besoin
```

API Kotlin :

```kotlin
val model = NplsModel.load(assetBytes)
val yPred = model.predict(x)
```

Je commencerais par :

```text
arm64-v8a
x86_64 pour émulateur
```

J’éviterais les ABIs 32-bit sauf besoin explicite.

---

# 13. GPU : backend optionnel, jamais dépendance core

La GPU doit être pensée comme un backend séparé :

```text
libnirspls-core
libnirspls-c
libnirspls-cuda       optionnel
```

API :

```c
typedef enum {
    NPLS_BACKEND_REFERENCE_CPU = 0,
    NPLS_BACKEND_NATIVE_CPU = 1,
    NPLS_BACKEND_BLAS = 2,
    NPLS_BACKEND_CUDA = 3
} npls_backend_t;
```

Si CUDA n’est pas disponible :

```text
npls_context_set_backend(ctx, NPLS_BACKEND_CUDA)
    → NPLS_ERR_UNSUPPORTED
```

La règle importante :

> **Même API, backends différents.**

Le GPU ne doit pas changer les bindings Python/R/MATLAB/JS.

Et surtout, le GPU ne doit pas être pensé comme :

```text
accélérer un fit PLS isolé
```

mais comme :

```text
accélérer des lots de fits
```

Donc il faut prévoir une API batch :

```c
npls_status_t npls_fit_batch(
    npls_context_t* ctx,
    const npls_batch_fit_request_t* request,
    npls_batch_fit_result_t** out_result
);
```

Workloads GPU utiles :

```text
CV massive
bootstrap
permutation tests
MCUVE
CARS
iPLS / moving-window PLS
AOM-PLS
LW-PLS
grid de preprocessings
multi-target massif
imagerie hyperspectrale
```

Le GPU peut arriver plus tard, mais l’API doit être prête dès maintenant.

---

# 14. Préprocessing : opérateurs stateful

Pour la NIRS, il faut traiter les preprocessings comme des objets `fit/transform`.

Exemples :

```text
Center
Scale
SNV
MSC
EMSC
Detrend
Savitzky-Golay
Derivative
ASLS baseline
OSC
EPO
Wavelength resampling
Dead-band masking
```

Architecture :

```cpp
class Preprocessor {
public:
    virtual void fit(const MatrixView& X, const VectorView* y) = 0;
    virtual void transform(const MatrixView& X, MatrixView& out) const = 0;
    virtual void serialize(BinaryWriter& w) const = 0;
};
```

Mais API C :

```c
npls_pipeline_t* pipeline;
npls_pipeline_add_operator(pipeline, NPLS_OP_SNV, params);
npls_pipeline_add_operator(pipeline, NPLS_OP_SAVGOL, params);
```

Important : `fit` et `transform` séparés pour éviter les fuites de données.

```text
CV fold:
    fit preprocessing sur train
    transform train
    transform validation
    fit PLS sur train transformé
    predict validation transformé
```

C’est probablement plus important que de gagner 20 % de temps de calcul.

---

# 15. AOM-PLS : design dès le début, implémentation plus tard

Même si tu ne codes pas AOM-PLS au départ, le design doit le permettre.

Il faut donc représenter les preprocessings comme des opérateurs composables :

```cpp
OperatorBank:
    Identity
    SNV
    MSC
    SG_derivative_1
    SG_derivative_2
    ASLS
    Detrend
    OSC
    ...
```

Puis AOM peut devenir :

```text
PLS solver
    +
operator bank
    +
gating strategy
```

Config future :

```c
npls_config_set_algorithm(cfg, NPLS_ALGO_AOM_PLS);
npls_config_set_gating(cfg, NPLS_GATING_SOFT);
npls_config_set_operator_bank(cfg, bank);
```

Gating modes :

```text
soft
hard
sparse
per-component
per-block
per-target
```

Mais ne commence pas par AOM. Commence par rendre le moteur PLS et preprocessing béton.

---

# 16. Variable selection : wrappers autour du moteur PLS

Pour MCUVE, CARS, iPLS, SPA, etc., ne les mets pas dans le solveur PLS.

Fais :

```text
VariableSelector
    utilise PLSRegressor
    retourne selected_features
```

Côté API :

```c
npls_status_t npls_select_variables(
    npls_context_t* ctx,
    const npls_variable_selection_config_t* cfg,
    const npls_matrix_view_t* X,
    const npls_matrix_view_t* Y,
    npls_selection_result_t** out_result
);
```

Puis :

```text
VIPSelector
MCUVESelector
CARSSelector
IPLSSelector
SPASelector
GeneticPLSSelector
RandomFrogSelector
```

Le cœur reste modulaire.

---

# 17. Tests : priorité absolue

Le projet ne vaut que si les résultats sont fiables.

Je ferais quatre niveaux de tests.

## 17.1 Tests numériques unitaires

```text
dot
norm
center/scale
deflation
NIPALS convergence
PLS coefficients
predict
serialization roundtrip
```

## 17.2 Tests de propriétés

```text
fit → predict identique après export/import
fit avec X copié row-major ou col-major donne même résultat
scaling sauvegardé correctement
n_components=1 puis n_components=k cohérent
pas de NaN silencieux
```

## 17.3 Tests de non-régression

Datasets synthétiques fixés :

```text
small_pls1
small_pls2
collinear
p >> n
n >> p
multi-target
classification dummy Y
spectral-like smooth data
```

## 17.4 Tests contre références externes

À faire dans les bindings, pas forcément dans le core :

```text
scikit-learn
R pls
mdatools
ropls
mixOmics
nirs4all actuel
```

Objectif :

```text
tolérance float64 stricte pour cas simples
tolérance plus large pour variantes algorithmiques
```

---

# 18. Mémoire : règle stricte

Ne jamais faire :

```text
le wrapper alloue
le core libère
```

ou l’inverse.

Deux options sûres :

## Option A — l’utilisateur fournit les buffers

```c
npls_predict(model, X, Y_pred_out);
```

## Option B — le core alloue, le core libère

```c
npls_array_t* out;
npls_predict_alloc(model, X, &out);
npls_array_free(out);
```

Je recommande les deux :

```text
API haute performance :
    user-provided buffers

API pratique :
    allocated outputs with npls_free
```

Mais jamais :

```text
free()
delete
mxFree
R_Free
PyMem_Free
```

à travers les frontières.

---

# 19. Versioning ABI

Dans `npls.h` :

```c
#define NPLS_ABI_VERSION_MAJOR 1
#define NPLS_ABI_VERSION_MINOR 0
#define NPLS_ABI_VERSION_PATCH 0
```

Fonctions :

```c
uint32_t npls_get_abi_version_major(void);
uint32_t npls_get_abi_version_minor(void);
uint32_t npls_get_abi_version_patch(void);

const char* npls_get_build_info(void);
const char* npls_get_version_string(void);
```

Règle :

```text
ABI major change:
    changement cassant

ABI minor change:
    ajout de fonctions ou enums compatibles

ABI patch:
    bugfix
```

Noms de symboles préfixés :

```text
npls_*
```

Export uniquement de ces symboles. Tout le reste en `hidden`.

---

# 20. Packaging recommandé

## Core

```text
CMake install
static lib
shared lib
headers
```

## Python

```text
wheel
    npls_c.dll/.so/.dylib
    Python ctypes wrapper
    NumPy dependency
```

## R

```text
R package
    src/ core + r_gateway
    .Call interface
```

## MATLAB

```text
toolbox
    mex files
    class wrappers
    model import/export
```

## JS

```text
npm package
    npls.wasm
    npls.js
    npls.d.ts
```

## Android

```text
AAR
    JNI wrapper
    .so par ABI
    Kotlin API
```

---

# 21. Ce que je ne ferais pas

Je ne ferais pas :

```text
- API publique en C++ ;
- pybind11 dès le départ ;
- Rcpp dès le départ ;
- embind dès le départ ;
- Eigen obligatoire ;
- OpenBLAS obligatoire ;
- CUDA obligatoire ;
- JSON/YAML dans le core ;
- modèle sérialisé sous forme d’objet C++ ;
- allocation mémoire floue ;
- preprocessing sans séparation fit/transform ;
- fit complet JS/Android comme priorité initiale.
```

Le piège serait de rendre le projet agréable à coder dans un langage, mais pénible à distribuer partout.

---

# 22. Roadmap technique proposée

## Phase 0 — Socle ABI et build

```text
- repo CMake
- npls.h
- context/config/model handles
- matrix_view avec strides
- status codes
- version ABI
- tests C simples
- build Linux/macOS/Windows
```

Objectif : avoir une librairie C appelable.

---

## Phase 1 — PLS CPU référence

```text
- linalg minimale sans dépendance
- PLS1 NIPALS
- PLS2 NIPALS
- center/scale
- predict
- transform
- coefficients
- export/import binaire
- tests numériques
```

Objectif : un modèle PLS fiable, sérialisable, portable.

---

## Phase 2 — Bindings minimaux

```text
Python:
    ctypes + NumPy

R:
    .Call

MATLAB:
    MEX

JS:
    WASM predict-only

Android:
    JNI predict-only
```

Objectif : prouver que l’architecture multi-langage fonctionne.

---

## Phase 3 — Chimiométrie/NIRS

```text
- SNV
- MSC
- EMSC
- Savitzky-Golay
- derivatives
- detrend
- ASLS baseline
- CV leakage-safe
- VIP
- RMSEC/RMSECV/RMSEP/R²/Q²/RPD/RPIQ
```

Objectif : devenir utile pour la NIRS réelle.

---

## Phase 4 — Variantes PLS avancées

```text
- SIMPLS
- kernel algorithm PLS
- OPLS
- PLS-DA
- PLS-LDA
- MB-PLS
- LW-PLS
```

Objectif : couvrir les variantes majeures sans exploser l’architecture.

---

## Phase 5 — Variable selection

```text
- VIP selector
- coefficient selector
- iPLS
- moving-window PLS
- MCUVE
- CARS
- SPA
```

Objectif : rendre la librairie vraiment chimiométrique.

---

## Phase 6 — AOM-PLS

```text
- operator bank
- soft gating
- hard gating
- sparse gating
- per-component AOM
- AOM-NIPALS
- AOM-SIMPLS
- AOM-OPLS
```

Objectif : différenciation scientifique forte.

---

## Phase 7 — Backends accélérés

```text
- BLAS optionnel
- OpenMP optionnel
- batch CV
- batch variable selection
- CUDA optionnel
```

Objectif : accélérer les workloads massifs, pas seulement un fit isolé.

---

# 23. Direction finale

Je résumerais la direction technique ainsi :

```text
1. C++17 interne.
2. ABI C stable obligatoire.
3. Zéro dépendance obligatoire.
4. CMake uniquement.
5. Matrix views avec strides.
6. Handles opaques.
7. Pas de STL/C++ dans l’API publique.
8. Pas de JSON/YAML dans le core.
9. Sérialisation binaire versionnée.
10. Wrappers fins par langage.
11. Build predict-only pour mobile/web.
12. Backends accélérés optionnels.
13. Tests numériques très stricts.
14. AOM/MB/OPLS comme extensions du moteur, pas comme architecture séparée.
```

La phrase de conception :

> **Un moteur PLS/NIRS portable en C++, exposé comme une petite ABI C stable, avec des bindings qui ne font que traduire les objets natifs du langage vers cette ABI.**

C’est la voie la plus saine si tu veux fournir Python, R, MATLAB, Android, JS, puis éventuellement Julia, C#, Rust, Go ou autre sans réécrire le cœur.

[1]: https://gcc.gnu.org/onlinedocs/libstdc%2B%2B/manual/using_dual_abi.html?utm_source=chatgpt.com "Dual ABI - GCC, the GNU Compiler Collection"
[2]: https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html?utm_source=chatgpt.com "Importing and Exporting Guide"
[3]: https://docs.python.org/3/c-api/stable.html?utm_source=chatgpt.com "C API Stability — Python 3.14.5 documentation"
[4]: https://cran.r-project.org/doc/manuals/r-devel/R-exts.html?utm_source=chatgpt.com "Writing R Extensions - CRAN - R Project"
[5]: https://www.mathworks.com/help/matlab/call-mex-file-functions.html?utm_source=chatgpt.com "Call C/C++ MEX Functions from MATLAB"
[6]: https://www.mathworks.com/help/matlab/matlab_external/cpp-mex-api.html?utm_source=chatgpt.com "C++ MEX API - MATLAB & Simulink"
[7]: https://developer.mozilla.org/en-US/docs/WebAssembly/Guides/C_to_Wasm?utm_source=chatgpt.com "Compiling a new C/C++ module to WebAssembly"
[8]: https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html?highlight=cwrap&utm_source=chatgpt.com "Interacting with code"
[9]: https://developer.android.com/ndk/guides/cmake?utm_source=chatgpt.com "CMake - NDK"
