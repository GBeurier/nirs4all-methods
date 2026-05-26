# Roadmap d'implementation du framework de validation

Date: 2026-05-20

Objectif: transformer les mecanismes parity/benchmark de `pls4all` puis
`chemometrics4all` en framework interne reproductible, declaratif et connecte
aux dashboards et a la documentation methode par methode.

## Principe de migration

Le framework doit s'ajouter au-dessus de l'existant avant de remplacer les
orchestrateurs. L'objectif initial est de rendre les contrats de validation
explicites, exportables et consommables par la doc/dashboard, sans casser les
runs actuels.

Ordre impose:

1. `pls4all` d'abord.
2. Tests et validation du framework sur `pls4all`.
3. Portage du meme design dans `chemometrics4all`.

## Phase PLS-1: socle declaratif

Livrables:

- `benchmarks/validation/`
- modeles dataclass ou typed dict pour methodes, references, datasets,
  comparateurs, suites et run manifests;
- export deterministe depuis `benchmarks.parity_timing.registry`;
- fichiers registry generes sous `benchmarks/validation/registry/`;
- validateur qui compare le registry genere au registry Python courant.

Contraintes:

- zero dependance runtime nouvelle si possible;
- format JSON stable au depart, meme si le design cible accepte YAML;
- pas de modification des runners existants;
- generation deterministe, stable en diff.

Commandes attendues:

```bash
python -m benchmarks.validation.export_registry --write
python -m benchmarks.validation.validate_registry
```

## Phase PLS-2: lien docs et dashboard

Livrables:

- payload `docs/_static/validation-contracts.json`;
- injection d'une section `### Validation contract` dans les pages methodes;
- ajout minimal d'un bloc de metadonnees validation dans `bench-data.json`
  pour relier dashboard et framework;
- conservation du refresh automatique `### Benchmarks`.

La section methode doit contenir au minimum:

- MethodSpec;
- datasets smoke/benchmark;
- reference canonique et role;
- comparateur;
- tolerances binding/reference;
- backends ABI testes;
- commande de reproduction.

Commandes attendues:

```bash
python -m benchmarks.validation.export_registry --write --docs
python docs/_extras/method_benchmark_tables.py
sphinx-build -E -b html docs docs/_build/html --keep-going
```

## Phase PLS-3: wrapper de run framework

Livrables:

- `python -m benchmarks.validation.run_suite --suite smoke`;
- mode `--refresh-docs`;
- generation d'un `run_manifest.json` par execution;
- mode snapshot read-only par defaut;
- documentation du cycle de refresh snapshot.

Le wrapper peut appeler l'orchestrateur existant. Le but n'est pas de reecrire
le runner, mais de normaliser son interface et ses artefacts.

## Phase PLS-4: tests et verrouillage

Livrables:

- tests unitaires du validateur et des exports;
- garde-fou CI possible: registry genere == registry courant;
- test Sphinx;
- test dashboard payload;
- documentation `benchmarks/validation/README.md`.

## Phase C4A-1: portage du socle

Apres validation sur `pls4all`, appliquer la meme structure a
`chemometrics4all`.

Priorites specifiques:

- reconcilier `benchmark_registry.json` et les `MethodSpec` de
  `cross_binding/orchestrator.py`;
- sortir progressivement les declarations du monolithe orchestrateur;
- conserver les snapshots `.npz` avec metadata embarquee;
- documenter explicitement `canonical`, `comparator`, `context`.

## Phase C4A-2: docs/dashboard

Memes livrables que PLS-2:

- `validation-contracts.json`;
- section `Validation contract`;
- lien data indirect dans dashboard;
- build Sphinx force.

## Phase C4A-3: consolidation commune

Quand les deux repos ont le meme framework:

- aligner les schemas JSON;
- aligner la terminologie `release`, `diagnostic`, `context`,
  `paper_only`;
- factoriser seulement si utile, sans creer une dependance runtime entre libs;
- produire une annexe methodologique exportable pour publications.

## Criteres d'acceptation

Pour chaque repo:

- le registry validation se regenere sans diff inattendu;
- le validateur detecte une divergence MethodSpec/registry;
- chaque page methode expose son contrat de validation;
- les tableaux benchmark continuent a se rafraichir;
- le dashboard se construit avec le payload enrichi;
- `sphinx-build -E -b html docs docs/_build/html --keep-going` passe;
- les tests existants de dashboard/parity passent.

## Risques

- Le workspace courant contient beaucoup de changements non relies. Ne pas
  nettoyer ni reverter hors perimetre.
- `pls4all/results/full_matrix.csv` peut etre partiel; le framework ne doit pas
  supposer qu'une matrice complete est disponible.
- Les references optionnelles R/MATLAB peuvent rendre le registry host-sensitive;
  l'export doit distinguer structure declaree et references resolues.
- Les tolerances qualitatives doivent rester visibles comme diagnostic, pas
  comme release gate.
