# nirs4all-methods — Stopping Point Handoff

**Date arrêt**: 2026-05-19
**Dernier commit pushé**: à venir (post-audit bindings/gates/release)
**Branch**: `main`
**Repo**: https://github.com/GBeurier/nirs4all-methods

## État global

| | Valeur |
|---|---|
| Project version | 0.1.0 |
| ABI version | **1.9.0** |
| ABI symbols exported | **498** |
| C++ tests | local tests build `build/local-tests`: **266 PASS / 0 FAIL** |
| Reference parity gate (Gate 2) | **76 PASS / 0 FAIL / 27 SKIP** |
| Binding parity Python (Gate 1) | **57 binding-parity suites PASS** @ tol 1e-6; full Python tests **64 PASS** |
| Binding parity R (Gate 1) | **175 PASS / 0 FAIL / 0 SKIP**; R source tarball self-contained; `R CMD check --no-manual --no-build-vignettes` from `/tmp` = **Status OK** |
| Python package | sdist + wheel build OK; wheel smoke import OK without `N4M_LIB_PATH` after staging bundled `libn4m` |
| Dashboard perf | Small pls4all NIRS timing CSV + payload generated from C++/Python/R bindings plus external reference methods; reference gates use stored output snapshots under `benchmarks/reference_snapshots/cross_binding/` |
| Commits sur main | ~17 |
| n4m.h | ~2580 lignes |

## Phases complétées (0-23)

- **Phase 0** Bootstrap (clone pls4all → strip → rename p4a→n4m)
- **Phase 1** PCG64 RNG bit-exact NumPy parity
- **Phases 2-10** 52 preprocessings (scatter, derivatives, baselines, wavelets, signal-conv, orthog, feat-sel, resampling)
- **Phase 11** 9 splitters (NEW n4m_split_*)
- **Phases 12-14** 11 filter methods (Y-outlier 4 + X-outlier 6 + leverage/quality/composite)
- **Phases 15-18** 39 augmenters (2 v2-deferred as stubs)
- **Phase 19** Signal type detection + 8 NIRS metrics + Hotelling T²/Q-residuals
- **Phase 20** Transfer metrics (9-metric vector)
- **Phase 21** FCK static transformer
- **Phase 21.5** Pre-binding ABI cleanup (M2 splitter `_handle_t` rename + M3 filter `_fit`/`_apply` split + splitter enums to n4m.h) + parity comparator modules
- **Phase 22** Python binding (ctypes + expanded sklearn-compatible surface; binding parity PASS)
- **Phase 23** R binding (.Call + idiomatic R functions across the 48 benchmarked methods; inverse transforms exposed where ABI supports them; vendored core for CRAN-style source builds)
- **Phase 26a** Benchmarks/dashboard aligned with pls4all contract: C ABI + Python + R + external references, thread/size sweeps, real `bench-data.json`

## Phases restantes

| # | Phase | Status |
|---|---|---|
| **24** | MATLAB binding | pending |
| **25** | JS/WASM binding | pending |
| **26** | Full benchmarks + GitHub Pages + pls4all re-pull | benchmarks/dashboard implemented locally; nightly workflow/GitHub Pages polish pending |
| **27** | CARS/MCUVE feature selection | pending; requires internal PLS callback/model substrate |
| **28** | PLS/AOM model families from nirs4all | pending; likely belongs after/with pls4all model substrate |

## Reprendre le travail

### Continuation immédiate

Pour reprendre, commence par:
```bash
cd /home/delete/nirs4all/nirs4all-methods
git pull origin main
cmake --preset dev-debug && cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/n4m_tests   # doit donner 266/266
./build/dev-debug/cpp/cli/n4m_cli --version  # doit afficher 0.1.0+abi.1.9.0
```

Pour valider les bindings Python:
```bash
cd bindings/python && pip install -e . && cd ../..
N4M_LIB_PATH=$(pwd)/build/dev-debug/cpp/src/libn4m.so.1.9.0 \
  python -m pytest bindings/python/tests/test_binding_parity.py -v
# expected: binding parity PASS
```

Pour valider le R CRAN-style (nécessite R installé):
```bash
R CMD build bindings/r/n4m
tmp=$(mktemp -d)
cp n4m_0.1.0.tar.gz "$tmp/"
(cd "$tmp" && R CMD check --no-manual --no-build-vignettes n4m_0.1.0.tar.gz)
# expected: Status OK
```

Pour valider le wheel Python avec libn4m embarquée:
```bash
bindings/python/scripts/build_libn4m_in_wheel.sh
rm -rf bindings/python/dist
python -m build --sdist --wheel bindings/python
python -m venv /tmp/n4m-wheel-smoke
/tmp/n4m-wheel-smoke/bin/python -m pip install bindings/python/dist/*.whl
env -u N4M_LIB_PATH /tmp/n4m-wheel-smoke/bin/python -c "import n4m as n4m; print(n4m.version())"
```

Pour les parity gates complètes:
```bash
# Gate 0 + 2 (fixture determinism + reference_parity + C++ tests)
python parity/python_generator/scripts/run_parity_gate.py
# expected: Stage 1 PASS, Stage 2 76 PASS / 0 FAIL / 27 SKIP, Stage 3 PASS

# Stage 4 binding_parity = Python + R activated via parity-gate.yml in CI
```

### Décisions architecturales déjà prises

1. **2 parity gates** (verbatim pls4all):
   - `binding_parity()` (Gate 1): max|pred-ref| ≤ 1e-6, binding ↔ libn4m
   - `reference_parity()` (Gate 2): RMSE relative, libn4m ↔ canonical Python
   - Implémentés en `parity/binding_parity.py` + `parity/reference_parity.py`
2. **27 SKIPs en Gate 2** mis de côté (à rediscuter): 5 wavelet layout + 5 splitter PCG64 stochastic + 17 augmenter stochastic
3. **Pas de matrix de workflow multi-binding** — un seul `parity-gate.yml` avec Stage 4 conditionnels per binding (présence du dir)
4. **Internal NumPy fixtures** (`parity/python_generator/src/n4m_parity_*_ref/`) restent des fixtures internes de parité, pas des références externes de benchmark
5. **Stratégie batch parallèle**: phases indépendantes lancées en parallèle via git worktrees, intégrées en commit unique. A bien marché pour Phases 7-21 + Phases 6,13,15-18.
6. **R/CRAN**: le package R vendore `src/libn4m` et compile statiquement dans `n4m.so`; ne dépend plus de `N4M_LIB_PATH`.
7. **EPO d-aware ABI**: `n4m_pp_epo_transform_with_d(handle, X, d, d_len, out)` est exporté et utilisé par Python `EPO.fit_transform(X, d)` et par la ligne benchmark C++ `epo`. `n4m_pp_epo_transform(handle, X, out)` reste le contrat sans `d` et donc pass-through à `d = d_mean`.
8. **Inverse transforms bindings**: les bindings doivent exposer `inverse_transform` quand l'ABI le supporte (`MSC`, `BaselineCenter`). Pour les projections lossy (`OSC`, `EPO`), l'ABI retourne `N4M_ERR_UNSUPPORTED`; on expose l'appel côté Python mais on ne fabrique pas de faux inverse.

### Décisions à prendre pour rediscussion

- **27 SKIPs Gate 2**: doit-on essayer d'aligner les implémentations n4m avec nirs4all (perte de C-side determinism garantie) ou shipper comme divergences documentées?
- **`n4m_array_t` reintroduction**: deferred to Phase 9 (FlexiblePCA needed variable-shape output) but never landed. À voir si Phase 24 MATLAB ou Phase 25 JS/WASM le requiert.
- **Tier-2 binding coverage**: Python couvre maintenant la majorité des opérateurs non-modèle exposés en ABI; R couvre le premier bloc CRAN-safe dont preprocessings/baselines/signal conversions avec inverses MSC/BaselineCenter.
- **CARS/MCUVE**: absents du core/ABI; à implémenter après une décision sur le callback PLS interne.
- **Modèles PLS/AOM/Ridge nirs4all**: hors scope actuel de nirs4all-methods; nécessitent un design ABI modèle distinct et probablement réutilisation/coordination avec pls4all.

### Phase 24 — MATLAB binding (à faire)

Pattern similaire à Phase 22 mais via MEX dispatcher:
- `bindings/matlab/+n4m/` — classdefs
- `bindings/matlab/mex/` — MEX C++ glue
- `binding_parity` test runner via MATLAB unittest (ou Octave fallback)
- Tolerance 1e-6 attendue (IEEE 754 double identique)

### Phase 25 — JS/WASM binding (à faire)

Via Emscripten:
- `bindings/js/CMakeLists.txt` — emcc compile libn4m → libn4m.wasm + libn4m.js
- `bindings/js/npm/` — npm package
- Tier-2: subset (preprocessings stateless principalement, lightweight filters, signal detect — défer ops banded/iteratifs)
- Test via Node.js + jest

### Phase 26 — Benchmarks + GitHub Pages

- **Re-pull pls4all** depuis sa branche actuelle pour refresh CRAN/bench infra (user instruction depuis Phase 0)
- `benchmarks/cross_binding/` — timing matrix now covers direct C ABI for every registered method, Python, installed R, and local `nirs4all`/NumPy/SciPy/sklearn/pybaselines/PyWavelets/R base/R stats references where credible comparators exist
- Current snapshot refresh command: `python benchmarks/cross_binding/orchestrator.py --repeat 5 --size-preset small --threads 1 --write-reference-snapshots`
- Full NIRS size matrix matches pls4all via `--size-preset pls4all`; current generated dashboard intentionally uses only the small pls4all block (`100x50`, `100x500`, `100x2500`)
- Dashboard payload: `python docs/_extras/build_landing.py --results benchmarks/cross_binding/results --out docs/_static/bench-data.json`
- Current dashboard: 48 methods, 3 sizes, 1 thread setting; generated locally from the current cross-binding CSV
- Canonical external reference rows use `reference_role=canonical` and render with a reference-method icon, not a gate icon; `cpp` carries the reference gate vs the stored canonical external snapshot, and non-canonical externals carry comparator parity vs that same snapshot
- Exact comparator rows are now restricted to credible same-contract comparisons, but every timed reference row still receives a parity verdict. Loose-contract externals such as nirs4all Savitzky-Golay edge-mode mismatch, nirs4all wavelet output-shape mismatch, nirs4all FCK convention drift, sklearn/nirs4all K-bins RNG/index-order mismatch, and SciPy APIs with different semantics are gated against the stored canonical snapshot and surface as divergent/error when they do not match.
- Timing now follows the pls4all warmup convention: unmeasured warmup calls, then `repeat` timed calls with median per-call wall time; default `repeat=5`.
- `bench-nightly.yml` workflow weekly cron avec >20% regression gate
- Docs Sphinx auto-deployed to `https://gbeurier.github.io/nirs4all-methods/`

## Fichiers clés à connaître

| Fichier | Rôle |
|---|---|
| `cpp/include/n4m/n4m.h` | Public C ABI — 2580 lignes, §1-§29 |
| `cpp/include/n4m/n4m_version.h` | ABI 1.9.0 / project 0.1.0 |
| `cpp/src/core/` | Engines C internes (preprocessing/, augmentations/, splitters/, filters/, utilities/, common/) |
| `cpp/src/c_api/` | extern "C" wrappers |
| `cpp/abi/expected_symbols_*.txt` | 498 symboles ABI |
| `parity/fixtures/*_v1.json` | 104 fixtures parity |
| `parity/binding_parity.py` | Gate 1 comparator |
| `parity/reference_parity.py` | Gate 2 comparator |
| `parity/run_reference_parity.py` | Gate 2 runner (103 suites) |
| `parity/python_generator/scripts/run_parity_gate.py` | Master orchestrator (Stage 1+2+3+4) |
| `parity/tolerances.md` | Per-op tolerances + binding_tol column |
| `parity/divergences.md` | Pas encore créé (cf. SKIPs decisions) |
| `bindings/python/` | Phase 22 binding |
| `bindings/r/n4m/` | Phase 23 binding |
| `benchmarks/cross_binding/orchestrator.py` | Cross-binding timing matrix (C ABI/Python/R/external) |
| `benchmarks/cross_binding/results/full_matrix.csv` | CSV dashboard final local run |
| `docs/_static/bench-data.json` | Payload dashboard généré depuis CSV final |
| `.github/workflows/parity-gate.yml` | CI gate (Stages 1+2+3+4) |
| `docs/reviews/PHASES.md` | Verdict aggregate |
| `docs/reviews/DEFERRALS.md` | Cross-phase tracker |
| `roadmap/phase-*.md` | Per-phase briefs |
| `CHANGELOG.md` | Version history Keep-a-Changelog |

## Worktrees (locked) — à nettoyer quand bindings/* committed

```
.claude/worktrees/agent-a2f069496633f1cb9   (P7 signal conversion)
.claude/worktrees/agent-a31c030eb1c9da7da   (P9 FlexiblePCA/SVD)
.claude/worktrees/agent-a40029a940a5f7368   (P12 Y-outlier)
.claude/worktrees/agent-a4d683af3b4d9c636   (P11 splitters)
.claude/worktrees/agent-a62b667d41c6c951a   (P10 resampling)
.claude/worktrees/agent-a62f804de4a887c03   (P14 filter meta)
.claude/worktrees/agent-a737bd14b02708e7b   (P19 signal+metrics)
.claude/worktrees/agent-acf7d86e13d2d30ec   (P8 OSC/EPO)
.claude/worktrees/agent-ae0957170d00f1e1b   (P21 FCK)
.claude/worktrees/agent-aeb108cd1920422ff   (P20 transfer metrics)
.claude/worktrees/agent-a5893245d6c36424d   (P6 wavelets)
.claude/worktrees/agent-a402c58f3cc0f19b4   (P13 X-outlier)
.claude/worktrees/agent-a23aefc9f7105e3c3   (P15 noise+drift)
.claude/worktrees/agent-a9860ac0187876fda   (P16 wavelength+spectral)
.claude/worktrees/agent-a930356face899912   (P17 mixup+physical)
.claude/worktrees/agent-ac746cf475ecf61cf   (P18 edge+splines+random)
.claude/worktrees/agent-ac966dcc666177491   (P22 Python binding)
.claude/worktrees/agent-aeca18f184655fdec   (P23 R binding)
```

Une fois bindings/python et bindings/r committed sur main:
```bash
git worktree list   # liste les worktrees actifs
git worktree remove .claude/worktrees/agent-XXX --force
```

ou laisser tels quels pour audit trail.

## Pour aller plus loin

Le repo est dans un état stable pour la couche C++ non-modèle + 2 bindings
(Python + R) à parity gates verts. Le scope opérateurs non-modèles déjà
implémenté en ABI est largement couvert; les vrais manques fonctionnels restent
CARS/MCUVE et les familles de modèles PLS/AOM/Ridge issues de `nirs4all`.

Les prochaines phases 24/25/26-polish sont **moins critiques** et peuvent attendre des décisions stratégiques:
- MATLAB / WASM bindings: nice-to-have, dépend de la cible utilisateur (MATLAB = labs académiques, WASM = web dashboards)
- Benchmarks: la matrice locale est en place; restent surtout le cron nightly, le seuil de régression et la publication GitHub Pages
- Re-pull pls4all: pour synchroniser les améliorations d'infra (CRAN, nightly bench) ajoutées par pls4all depuis Phase 0

Tout le repo est sur GitHub publique: https://github.com/GBeurier/nirs4all-methods
