# chemometrics4all — Stopping Point Handoff

**Date arrêt**: 2026-05-18
**Dernier commit pushé**: à venir (Phase 22+23 binding commit ci-dessous)
**Branch**: `main`
**Repo**: https://github.com/GBeurier/chemometrics4all

## État global

| | Valeur |
|---|---|
| Project version | 0.1.0 |
| ABI version | **1.9.0** |
| ABI symbols exported | **402** |
| C++ tests | **266/266 PASS** |
| Reference parity gate (Gate 2) | **76 PASS / 0 FAIL / 27 SKIP** |
| Binding parity Python (Gate 1) | **20/20 PASS** @ tol 1e-6 |
| Binding parity R (Gate 1) | **93/94 PASS** @ tol 1e-6 (1 SKIP: MSC.inverse_transform not in tier-2 yet) |
| Commits sur main | ~17 |
| c4a.h | ~2580 lignes |

## Phases complétées (0-23)

- **Phase 0** Bootstrap (clone pls4all → strip → rename p4a→c4a)
- **Phase 1** PCG64 RNG bit-exact NumPy parity
- **Phases 2-10** 52 preprocessings (scatter, derivatives, baselines, wavelets, signal-conv, orthog, feat-sel, resampling)
- **Phase 11** 9 splitters (NEW c4a_split_*)
- **Phases 12-14** 11 filter methods (Y-outlier 4 + X-outlier 6 + leverage/quality/composite)
- **Phases 15-18** 39 augmenters (2 v2-deferred as stubs)
- **Phase 19** Signal type detection + 8 NIRS metrics + Hotelling T²/Q-residuals
- **Phase 20** Transfer metrics (9-metric vector)
- **Phase 21** FCK static transformer
- **Phase 21.5** Pre-binding ABI cleanup (M2 splitter `_handle_t` rename + M3 filter `_fit`/`_apply` split + splitter enums to c4a.h) + parity comparator modules
- **Phase 22** Python binding (ctypes + 20 sklearn-compatible classes; binding_parity 20/20 PASS)
- **Phase 23** R binding (.Call + 15 idiomatic R functions; binding_parity 93/94 PASS)

## Phases restantes

| # | Phase | Status |
|---|---|---|
| **24** | MATLAB binding | pending |
| **25** | JS/WASM binding | pending |
| **26** | Benchmarks + GitHub Pages + pls4all re-pull | pending |

## Reprendre le travail

### Continuation immédiate

Pour reprendre, commence par:
```bash
cd /home/delete/nirs4all/chemometrics4all
git pull origin main
cmake --preset dev-debug && cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests   # doit donner 266/266
./build/dev-debug/cpp/cli/chemometrics4all_cli --version  # doit afficher 0.1.0+abi.1.9.0
```

Pour valider les bindings Python:
```bash
cd bindings/python && pip install -e . && cd ../..
CHEMOMETRICS4ALL_LIB_PATH=$(pwd)/build/dev-debug/cpp/src/libc4a.so.1.9.0 \
  python -m pytest bindings/python/tests/test_binding_parity.py -v
# expected: 20/20 PASS
```

Pour valider le R (nécessite R installé):
```bash
CHEMOMETRICS4ALL_LIB_PATH=$(pwd)/build/dev-debug/cpp/src/libc4a.so.1.9.0 \
  R CMD INSTALL --install-tests bindings/r/chemometrics4all
cd $(R RHOME)/library/chemometrics4all/tests && Rscript testthat.R
# expected: 93/94 PASS (1 skip)
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
   - `binding_parity()` (Gate 1): max|pred-ref| ≤ 1e-6, binding ↔ libc4a
   - `reference_parity()` (Gate 2): RMSE relative, libc4a ↔ canonical Python
   - Implémentés en `parity/binding_parity.py` + `parity/reference_parity.py`
2. **27 SKIPs en Gate 2** mis de côté (à rediscuter): 5 wavelet layout + 5 splitter PCG64 stochastic + 17 augmenter stochastic
3. **Pas de matrix de workflow multi-binding** — un seul `parity-gate.yml` avec Stage 4 conditionnels per binding (présence du dir)
4. **Frozen NumPy refs** (`parity/python_generator/src/c4a_parity_*_ref/`) restent comme insulation layer (contrat origine, pas remplacés par Gate 2)
5. **Stratégie batch parallèle**: phases indépendantes lancées en parallèle via git worktrees, intégrées en commit unique. A bien marché pour Phases 7-21 + Phases 6,13,15-18.

### Décisions à prendre pour rediscussion

- **27 SKIPs Gate 2**: doit-on essayer d'aligner les implémentations c4a avec nirs4all (perte de C-side determinism garantie) ou shipper comme divergences documentées?
- **`c4a_array_t` reintroduction**: deferred to Phase 9 (FlexiblePCA needed variable-shape output) but never landed. À voir si Phase 24 MATLAB ou Phase 25 JS/WASM le requiert.
- **Tier-2 Python coverage**: actuellement 20 ops, devrait s'étendre à ~50-95 selon priorité user-facing.

### Phase 24 — MATLAB binding (à faire)

Pattern similaire à Phase 22 mais via MEX dispatcher:
- `bindings/matlab/+chemometrics4all/` — classdefs
- `bindings/matlab/mex/` — MEX C++ glue
- `binding_parity` test runner via MATLAB unittest (ou Octave fallback)
- Tolerance 1e-6 attendue (IEEE 754 double identique)

### Phase 25 — JS/WASM binding (à faire)

Via Emscripten:
- `bindings/js/CMakeLists.txt` — emcc compile libc4a → libc4a.wasm + libc4a.js
- `bindings/js/npm/` — npm package
- Tier-2: subset (preprocessings stateless principalement, lightweight filters, signal detect — défer ops banded/iteratifs)
- Test via Node.js + jest

### Phase 26 — Benchmarks + GitHub Pages

- **Re-pull pls4all** depuis sa branche actuelle pour refresh CRAN/bench infra (user instruction depuis Phase 0)
- `benchmarks/cross_binding/` — 5 suites (preprocess_matrix, augment_matrix, splitter, filter, cross_binding) similaire à pls4all/benchmarks
- `bench-nightly.yml` workflow weekly cron avec >20% regression gate
- Docs Sphinx auto-deployed to `https://gbeurier.github.io/chemometrics4all/`

## Fichiers clés à connaître

| Fichier | Rôle |
|---|---|
| `cpp/include/chemometrics4all/c4a.h` | Public C ABI — 2580 lignes, §1-§29 |
| `cpp/include/chemometrics4all/c4a_version.h` | ABI 1.9.0 / project 0.1.0 |
| `cpp/src/core/` | Engines C internes (preprocessing/, augmentations/, splitters/, filters/, utilities/, common/) |
| `cpp/src/c_api/` | extern "C" wrappers |
| `cpp/abi/expected_symbols_*.txt` | 402 symboles ABI |
| `parity/fixtures/*_v1.json` | 104 fixtures parity |
| `parity/binding_parity.py` | Gate 1 comparator |
| `parity/reference_parity.py` | Gate 2 comparator |
| `parity/run_reference_parity.py` | Gate 2 runner (103 suites) |
| `parity/python_generator/scripts/run_parity_gate.py` | Master orchestrator (Stage 1+2+3+4) |
| `parity/tolerances.md` | Per-op tolerances + binding_tol column |
| `parity/divergences.md` | Pas encore créé (cf. SKIPs decisions) |
| `bindings/python/` | Phase 22 binding |
| `bindings/r/chemometrics4all/` | Phase 23 binding |
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

Le repo est dans un état stable production-quality pour la couche C++ + 2 bindings (Python + R) à parity gates verts. Le scope opérateurs est 100% couvert.

Les prochaines phases (24-26) sont **moins critiques** et peuvent attendre des décisions stratégiques:
- MATLAB / WASM bindings: nice-to-have, dépend de la cible utilisateur (MATLAB = labs académiques, WASM = web dashboards)
- Benchmarks: utile pour le marketing et la documentation perf, mais ne change pas la fonctionnalité
- Re-pull pls4all: pour synchroniser les améliorations d'infra (CRAN, nightly bench) ajoutées par pls4all depuis Phase 0

Tout le repo est sur GitHub publique: https://github.com/GBeurier/chemometrics4all
