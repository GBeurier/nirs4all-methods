# Audit parity/benchmark et proposition de framework de validation

Date: 2026-05-20  
Portee: `pls4all` et `chemometrics4all`

Ce document analyse l'etat courant des mecanismes de parity checks, de
benchmarks, de references externes et de documentation automatique. Il propose
un design de mini framework de validation reproductible, deterministe,
perenne et documentable pour les publications scientifiques.

## Synthese courte

Les deux projets ont deja une base solide: generation deterministe des donnees,
comparaison cross-binding, snapshots d'oracles externes, CSV riches en
metadonnees, tableaux de documentation rafraichis automatiquement et dashboard
interactif. Le probleme principal n'est pas l'absence de validation, mais le
fait que la logique est repartie entre plusieurs sources de verite: registres
Python, lockfiles JSON, scripts de benchmark par backend, CSV historiques,
snapshots binaires, fragments de refresh dashboard et generateurs de docs.

La direction recommandee est de transformer l'existant en un framework
declaratif sous `benchmarks/validation/`, avec:

- un registre de methodes, datasets, references, comparateurs et tolerances;
- un manifeste d'environnement reproductible;
- des drivers backend/reference pluggables;
- un schema de resultats et de snapshots stable;
- une generation de documentation "Validation contract" par methode.

Le framework n'a pas besoin d'etre embarque dans les bibliotheques publiques.
Il doit vivre dans les repositories comme infrastructure de validation et de
publication.

## Fonctionnement actuel

### pls4all

`pls4all` est le systeme le plus mature. La validation est organisee autour de
deux couches.

La couche historique `benchmarks/run.py` et `benchmarks/runners/*` produit des
benchmarks plus anciens (`aom_global`, `pls_regression`, `matrix`) avec CSV
d'accuracy et de timing sous `benchmarks/results/`. Elle reste utile, mais le
systeme actif de documentation/dashboard est plutot le cross-binding.

La couche cross-binding est pilotee par:

- `benchmarks/parity_timing/registry.py`: catalogue canonique des methodes PLS,
  references externes, parametres de cellule, tolerances, `paper_only`, notes;
- `benchmarks/parity_timing/_comparator.py`: comparateurs purs NumPy pour la
  binding parity et la reference parity;
- `benchmarks/parity_timing/lockfile.py` et
  `benchmarks/parity_timing/truth_sources.lock.json`: snapshot structurel du
  registre, host-insensitive;
- `benchmarks/cross_binding/orchestrator.py`: generation dataset, execution des
  backends, collecte des timings, generation et chargement d'oracles;
- `benchmarks/cross_binding/scripts/*`: drivers Python, C++, R, MATLAB/Octave,
  JS/WASM, Julia et references externes;
- `benchmarks/cross_binding/data/.reference_oracles/`: oracles `.npy` +
  metadonnees `.json`;
- `benchmarks/cross_binding/results/*.csv`: resultats, refresh partiels et
  matrices;
- `docs/_extras/build_landing.py`: transformation des CSV en `bench-data.json`;
- `docs/_extras/build_methods.py` et `docs/_extras/method_benchmark_tables.py`:
  generation des pages methodes et rafraichissement automatique des sections
  `### Benchmarks` pendant `sphinx-build`.

Le modele de parity est maintenant explicite:

- Gate 1, binding consistency: les bindings `pls4all` sont compares a la route
  native C++ de reference, tolerance max-abs stricte, typiquement `1e-6`.
- Gate 2, external-reference validity: chaque ligne reussie est comparee a
  l'oracle externe canonique de la methode. La metrique principale est
  `rmse_rel`, avec echappatoire absolue pres de zero.
- Les methodes `paper_only` sont documentees mais ne peuvent pas pretendre a un
  gate executable.

Points factuels observes localement:

- `truth_sources.lock.json` declare 73 methodes, dont 4 `paper_only`.
- Le dossier `.reference_oracles` contient 407 oracles `.npy`.
- Le dossier `benchmarks/cross_binding/data` contient 223 CSV de donnees
  deterministes.
- Le snapshot docs `docs/_extras/method-benchmarks.json` contient 72 blocs de
  benchmarks de methode.
- Le `full_matrix.csv` courant est partiel dans le workspace audite, donc les
  tableaux s'appuient aussi sur les fichiers `dashboard_refresh_*.csv` et le
  snapshot de documentation.

### chemometrics4all

`chemometrics4all` reprend la logique de `pls4all`, mais avec un scope plus
large: preprocessing, baseline correction, splitters, filtres, diagnostics,
resampling, augmentation et calibration transfer.

Les fichiers structurants sont:

- `benchmarks/benchmark_registry.json`: registre humain minimal des methodes
  benchmarkables;
- `benchmarks/truth_sources.lock.json`: sources de verite verrouillees
  (packages, projets, litterature);
- `benchmarks/check_truth_sources.py`: verification de coherence registry /
  lockfile;
- `benchmarks/cross_binding/orchestrator.py`: grand orchestrateur contenant
  aussi `Dataset`, `ReferenceSpec`, `MethodSpec`, references, comparateurs,
  snapshots et drivers;
- `benchmarks/reference_snapshots/cross_binding/**/*.npz`: snapshots des sorties
  de references externes;
- `benchmarks/cross_binding/results/*.csv`: resultats de benchmark;
- `benchmarks/cross_binding/validate_dashboard_payload.py`: garde-fou du JSON
  dashboard;
- `docs/_extras/build_landing.py`, `build_methods.py`,
  `method_benchmark_tables.py`: generation dashboard et documentation.

Le modele est legerement different de `pls4all`:

- La reference canonique externe produit un snapshot `.npz` avec metadata JSON
  embarquee.
- Les lignes C++/Python/R sont comparees au snapshot si `gate_c4a=True`.
- Les references dont le contrat est utile mais non identique sont marquees
  `context`, pas faussement "exactes".
- Les bindings internes sont compares au C++ lorsque la route native est
  disponible.

Points factuels observes localement:

- `benchmark_registry.json` declare 26 methodes, mais les resultats/dashboard
  couvrent 66 algorithmes. Le registre JSON n'est donc pas la source de verite
  effective du runner.
- `truth_sources.lock.json` contient 18 sources verrouillees.
- `cross_binding/results/full_matrix.csv` contient 999 lignes, 66 algorithmes,
  14 backends et 28 colonnes.
- `reference_snapshots/cross_binding` contient 211 snapshots `.npz`.
- Le snapshot docs `docs/_extras/method-benchmarks.json` contient 66 blocs.

### Documentation et dashboard

Les deux projets ont maintenant un schema proche:

1. Les runners produisent des CSV.
2. Les CSV sont transformes en payload JSON dashboard (`docs/_static/bench-data.json`).
3. Les pages methodes contiennent un bloc `### Benchmarks`.
4. Le hook Sphinx `source-read` remplace ce bloc a chaque build par le dernier
   bloc calcule depuis CSV/payload/snapshot.
5. Les dashboards HTML lisent `bench-data.json`.

C'est efficace pour garder les tableaux a jour, mais la documentation par
methode reste encore incomplete pour la validation scientifique: elle affiche
les sources/comparateurs et les timings, mais pas toujours un contrat exhaustif
lisible par humain sur les parametres, datasets, role de la reference,
tolerances, backend exact, hash de snapshot et commande de reproduction.

## Points forts

- Les deux projets ont deja une vraie separation entre benchmark timing et
  parity numerique.
- `pls4all` distingue clairement binding parity et external-reference parity.
- `chemometrics4all` introduit correctement la notion de references `context`
  quand le contrat externe ne correspond pas exactement.
- Les datasets sont generes de maniere deterministe par taille et seed.
- `pls4all` force les backends cross-language a lire les memes CSV de donnees,
  ce qui evite les divergences de RNG entre Python, R et MATLAB/Octave.
- Les resultats CSV transportent beaucoup de provenance: versions, seeds,
  chemins de predictions/oracles, schema de timing, raisons d'echec.
- Les snapshots d'oracles permettent de garder un gate reference actif meme
  quand la reference externe n'est pas relancee.
- Les pages de documentation se rafraichissent automatiquement avec les nouveaux
  runs, ce qui limite la derive entre docs et resultats.
- Les generateurs de docs sont deja capables de marquer les references
  canoniques et les divergences.

## Critiques et limites

### Sources de verite multiples

`pls4all` a `registry.py`, `truth_sources.lock.json`, les CSV, les oracles, les
snapshots docs et les scripts backend. `chemometrics4all` a en plus un
`benchmark_registry.json` qui ne couvre pas encore tout ce que l'orchestrateur
execute. Cela rend difficile de repondre simplement a: "quel est le contrat de
validation de cette methode ?".

### Configuration code-heavy

Les references sont souvent declarees directement en Python, avec factories,
imports, wrappers R et notes dans le meme fichier. C'est puissant, mais:

- la revue est difficile;
- l'ajout d'une methode demande de modifier du code executable;
- les declarations ne sont pas facilement extractibles vers une publication;
- les dependances optionnelles peuvent influencer le registre au moment de
  l'import.

### Environnement encore trop implicite

On observe des chemins absolus et locaux, par exemple vers des environnements
conda R/Octave ou vers le checkout local `nirs4all`. Les CSV capturent des
versions, mais il manque un manifeste environnemental stable qui dise: Python,
R, packages, BLAS, compilateur, CMake preset, commit git, et variables de
threading attendues.

### Resultats fragmentes

Les matrices completes, refresh partiels, probes, snapshots docs et anciens CSV
coexistent dans `results/`. Les builders savent merger certains fichiers, mais
le contrat de priorite est ad hoc. Cela rend les dashboards pratiques, mais
moins faciles a auditer scientifiquement.

### Snapshots pas assez gouvernes

Les snapshots sont utiles, mais le process de refresh doit devenir plus formel:
quand a-t-on le droit de les regenerer ? quel changement de reference les
justifie ? quel hash de dataset et quel hash d'environnement ont produit la
sortie ? comment relier un snapshot a une ligne de publication ?

### Tolerances melangees

Les tolerances expriment plusieurs choses:

- stricte equivalence numerique;
- drift acceptable entre implementations;
- diagnostic qualitatif;
- simple presence d'une reference.

`pls4all` commence a distinguer release gate et diagnostic wide tolerance, mais
cette distinction doit devenir un champ de schema, pas seulement une convention.

### Documentation encore trop resultat-centree

Les pages methodes montrent les sources et les timings, mais la "fiche de
validation" devrait etre un objet de premier ordre: reference canonique,
parametres exacts, dataset, seed, output compare, comparateur, tolerance, role
de chaque backend, statut de snapshot, commande reproductible.

## Proposition de framework

### Objectif

Construire un mini framework interne de validation, commun dans l'esprit aux
deux projets, mais instanciable par repo. Il doit permettre:

- d'ajouter une methode sans modifier un orchestrateur monolithique;
- d'ajouter un dataset de test en declaratif;
- d'ajouter une reference externe avec role et contrat explicites;
- de relancer smoke/full/publication runs de facon reproductible;
- de produire automatiquement la documentation method-level;
- de fournir une trace exploitable dans une publication scientifique.

### Layout propose

```text
benchmarks/
  validation/
    README.md
    schema/
      method.schema.json
      dataset.schema.json
      reference.schema.json
      run_manifest.schema.json
    registry/
      methods.yaml
      datasets.yaml
      references.yaml
      suites.yaml
      tolerances.yaml
    drivers/
      abi_cpp.py
      abi_python.py
      abi_r.py
      abi_matlab.py
      ref_python.py
      ref_r.py
    comparators/
      numeric.py
      sign_invariant.py
      masks.py
      classification.py
    snapshots/
      cross_binding/
    results/
      runs/<run_id>/
        manifest.json
        results.parquet
        results.csv
        summary.json
    tools/
      validate_registry.py
      run_suite.py
      refresh_snapshots.py
      render_docs.py
```

Le code public de la lib ne depend pas de ce framework. Les bindings publics
sont seulement appeles par les drivers.

### Registre de methode

Chaque methode devrait avoir un contrat declaratif proche de:

```yaml
method_id: airpls
label: AirPLS
family: baseline
operation: fit_transform
status: release_gate

inputs:
  primary: X
  auxiliaries: []
  dtype: float64

parameters:
  lam: 1.0e5
  max_iter: 20
  conv_tol: 1.0e-3

implementations:
  c_abi:
    symbol: c4a_pp_airpls
    args: [lam, max_iter, conv_tol]
  python_abi:
    callable: chemometrics4all.python.airpls
  python_sklearn:
    class: chemometrics4all.sklearn.AirPLS
  r:
    expr: airpls(X, lam = 1e5, max_iter = 20L)

references:
  canonical: pybaselines_airpls
  comparators: []
  context: []

datasets:
  smoke: nirs_smoke_100x50
  publication: nirs_baseline_grid_v1

output:
  kind: matrix
  comparator: max_abs_and_rms

tolerances:
  binding:
    metric: max_abs_diff
    threshold: 1.0e-6
    gate: release
  reference:
    metric: max_abs_diff
    threshold: 1.0e-6
    gate: release

documentation:
  contract_note: Same-contract pybaselines reference, row-wise baseline.
  citations: [pybaselines_2021]
```

Pour `pls4all`, `operation` peut etre `fit_predict`, `select_mask`,
`diagnostic`, `press`, etc. Pour les methodes qualitatives, le schema doit
dire explicitement `gate: diagnostic` ou `gate: context`, jamais masquer cela
dans une tolerance large.

### Registre de reference

Les references doivent etre verrouillees separement:

```yaml
source_id: pybaselines_airpls
type: package_reference
language: Python
package: pybaselines
version: 1.2.1
callable: pybaselines.whittaker.airpls
role_allowed: [canonical, comparator]
contract:
  same_output_shape: true
  boundary_mode: rowwise_1d
  deterministic: true
environment:
  lock: validation/env/python-refs.lock
publication:
  citation_key: pybaselines_2021
```

Les references `context` doivent exiger un `contract_mismatch` explicite. Une
reference sans contrat compatible ne doit jamais participer a un gate release.

### Registre de datasets

Les datasets doivent etre des objets identifiables, hashables et publiables:

```yaml
dataset_id: nirs_smoke_100x50
generator: synthetic_spectra_v1
n_samples: 100
n_features: 50
seed: 20260556
dtype: float64
outputs:
  X_hash: sha256:...
  y_hash: sha256:...
notes: Smooth synthetic NIR-like spectra with multiplicative scatter and noise.
```

Pour les publications, il faut distinguer:

- datasets synthetiques deterministes;
- datasets fixtures issus de donnees publiques;
- datasets proprietaires ou non redistribuables, avec hash et description.

### Comparateurs

Les comparateurs doivent etre nommes et versionnes:

- `max_abs`: bindings ABI, transformations simples;
- `rmse_rel`: predictions regression;
- `sign_invariant_columns`: PCA/SVD, scores/loadings;
- `mask_jaccard` ou `mask_distance`: selection de variables;
- `indices_equal`: splitters;
- `set_equal` / `ordered_equal`: sorties d'indices;
- `finite_shape_only`: diagnostic contextuel, non release;
- `custom:<id>`: comparateur specifique mais documente.

Chaque comparateur doit produire une structure stable:

```json
{
  "metric": "rmse_rel",
  "value": 2.4e-8,
  "threshold": 1e-6,
  "ok": true,
  "gate": "release",
  "note": "reference rms normalized"
}
```

### Manifestes de run

Chaque execution devrait produire un `manifest.json` immutable:

```json
{
  "run_id": "2026-05-20T14-32-10Z_pls4all_smoke",
  "repo": "pls4all",
  "git_sha": "...",
  "dirty": true,
  "suite": "smoke",
  "command": "python -m benchmarks.validation.run_suite --suite smoke",
  "environment_id": "ubuntu-py313-r443-openblas",
  "environment_hash": "sha256:...",
  "cpu": "...",
  "blas": "...",
  "thread_policy": {
    "OMP_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "MKL_NUM_THREADS": 1
  },
  "registry_hash": "sha256:...",
  "dataset_hashes": ["sha256:..."],
  "snapshot_policy": "read_only"
}
```

Les resultats de cellules doivent referencer `run_id`, `method_id`,
`dataset_id`, `backend_id`, `reference_id`, `comparator_id` et
`snapshot_id`. Le CSV peut rester pour compatibilite dashboard, mais un format
canonique type Parquet ou JSONL faciliterait l'audit.

### Snapshots

Les snapshots devraient contenir ou accompagner:

- output dtype, shape, hash;
- method_id, reference_id, dataset_id;
- params exacts;
- package version ou git SHA;
- environment_hash;
- comparator attendu;
- justification du refresh.

Deux commandes separees sont importantes:

```bash
python -m benchmarks.validation.run_suite --suite smoke --snapshot-policy read-only
python -m benchmarks.validation.refresh_snapshots --method airpls --reason "reference contract changed"
```

Le mode normal de CI doit etre read-only. Le refresh doit etre un acte explicite
et revu.

### Environnement reproductible

Un framework perenne doit figer plusieurs niveaux:

- Python: `uv.lock` ou `requirements-lock.txt` pour les refs Python;
- R: `renv.lock` ou script d'installation versionne;
- MATLAB/Octave: version attendue, packages, fallback;
- JS/WASM/Julia si presents: lockfiles natifs;
- C/C++: CMake preset, compilateur, flags, BLAS/OpenMP;
- systeme: image Docker/Apptainer ou au minimum fichier `environment.yml`;
- hardware: CPU, RAM, BLAS, OS, thread policy.

Les timings ne seront jamais parfaitement portables, mais le protocole peut
l'etre. Pour les publications, le run manifest devient la source de citation
experimentale.

### Documentation par methode

Chaque page methode devrait inclure une section generee:

```markdown
### Validation contract

- MethodSpec: `airpls`
- Dataset(s): `nirs_smoke_100x50`, `nirs_smoke_100x500`, ...
- Parameters: `lam=1e5`, `max_iter=20`
- Canonical reference: `pybaselines.whittaker.airpls` (`pybaselines 1.2.1`)
- Reference role: release gate
- Comparator: `max_abs_and_rms`
- Tolerance: `max_abs_diff <= 1e-6`
- ABI routes tested: C ABI, Python ABI, sklearn wrapper, R binding
- Snapshot: `sha256:...`, generated from run `...`
- Last validated: run `...`, git SHA `...`
- Reproduce:
  `python -m benchmarks.validation.run_suite --method airpls --suite smoke`
```

La section `### Benchmarks` peut rester centree sur timings/divergences. La
section `Validation contract` doit documenter le protocole, pas seulement le
dernier resultat.

## Design incremental recommande

### Phase 1: unifier les schemas sans casser l'existant

- Ajouter `benchmarks/validation/registry/*.yaml`.
- Generer ces YAML depuis les registres actuels pour bootstrap.
- Ajouter un validateur de schema.
- Ne pas changer les runners existants immediatement.
- Ajouter aux docs une section `Validation contract` construite depuis le YAML.

### Phase 2: extraire les MethodSpec hors des orchestrateurs

- `pls4all`: garder `registry.py` comme adaptateur temporaire, mais faire du
  YAML la source de documentation et de lock.
- `chemometrics4all`: sortir `MethodSpec` et `ReferenceSpec` de
  `cross_binding/orchestrator.py` vers un module/registre dedie.
- Faire echouer CI si `benchmark_registry.json`, specs runner et docs ne
  couvrent pas les memes methodes.

### Phase 3: formaliser snapshots et run manifests

- Ajouter `run_id`, `registry_hash`, `environment_hash`, `dataset_id` dans les
  CSV/resultats.
- Passer les snapshots en read-only par defaut.
- Ajouter une commande explicite de refresh avec raison obligatoire.
- Produire un `summary.json` par run.

### Phase 4: stabiliser l'environnement

- Ajouter un environnement de validation reproductible:
  - Docker/Apptainer pour publication;
  - lock Python/R/JS/Julia;
  - scripts d'installation R/Octave;
  - CMake preset benchmark.
- Documenter la difference entre "CI smoke", "nightly full" et "publication
  run".

### Phase 5: publicability

- Generer automatiquement une annexe methodologique:
  - versions;
  - datasets;
  - references;
  - comparateurs;
  - tolerances;
  - hardware;
  - commit SHA;
  - hash des resultats.
- Exporter une table compacte par methode pour publications.

## Critique globale par projet

### pls4all

Forces:

- Registre riche, avec 73 methodes et une politique de references documentee.
- Separation conceptuelle claire entre binding parity et reference parity.
- Comparateurs purs et reutilisables.
- Support multi-langage tres large.
- Oracles externes nombreux et deja utilises par la doc.
- Timing adaptatif robuste pour cellules lentes.

Limites:

- `registry.py` est devenu un gros fichier executable qui fait a la fois
  catalogue, wrappers, probes R, politique de references et documentation.
- Plusieurs chemins absolus rendent la reproductibilite fragile hors machine
  locale.
- Les references optionnelles rendent une partie du registre host-sensitive.
- Les tolerances larges melangent parfois "validation scientifique" et "presence
  qualitative d'une reference".
- Les fichiers `results/` contiennent beaucoup de probes/refresh; le resultat
  "canonique actuel" n'est pas assez explicite.
- Le `full_matrix.csv` audite est partiel; la doc fonctionne grace au merge
  refresh/snapshot, ce qui est pratique mais peu lisible comme artefact de
  publication.

Verdict: base excellente pour un framework, mais a modulariser. Le risque n'est
pas technique, il est documentaire et operationnel: il faut rendre le contrat
de validation explicite, hashable et exportable.

### chemometrics4all

Forces:

- Bon modele de snapshots `.npz` avec metadata embarquee.
- Bonne distinction entre references canoniques, comparateurs et context rows.
- Orchestrateur capable de tester C ABI, Python ABI, sklearn wrapper, R et refs.
- Dashboard deja alimente par de vrais resultats locaux.
- Documentation methodes maintenant proche de `pls4all`.

Limites:

- `benchmark_registry.json` est en retard par rapport a l'orchestrateur: 26
  methodes declarees contre 66 algorithmes observes dans les resultats.
- `orchestrator.py` porte trop de responsabilites: schema, registry, refs,
  comparateurs, snapshots, execution, CSV.
- La politique de tolerance externe est moins formalisee que dans `pls4all`.
- La couverture C++/R/externe reste heterogene selon les methodes.
- Les references `nirs4all` sont utiles, mais doivent etre traitees comme
  projet externe git-pinned avec commit/hash explicite pour publications.

Verdict: `chemometrics4all` a un design de snapshots plus propre, mais il lui
manque une source de verite unique. Le premier chantier est de reconcilier
registry JSON, MethodSpec runtime, resultats et docs.

## Recommandation finale

Ne pas jeter l'existant. Le bon mouvement est de mettre une couche declarative
au-dessus, puis de faire converger les runners vers elle.

Priorite immediate:

1. Ajouter une section `Validation contract` generee par methode.
2. Creer un schema commun `method/reference/dataset/comparator/run_manifest`.
3. Faire de `benchmark_registry` / `MethodSpec` une source unique, ou au moins
   verifier automatiquement leur coherence.
4. Verrouiller et documenter l'environnement de validation.
5. Formaliser le cycle de vie des snapshots.

Quand ces cinq points seront en place, les dashboards resteront utiles pour
explorer les timings, et les pages methodes deviendront suffisamment precises
pour soutenir une section "Validation and benchmarking protocol" dans des
publications scientifiques.
