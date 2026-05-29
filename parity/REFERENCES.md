# Parity references — donors, dependencies, and how to avoid matrix holes

This document records **which external library is the reference (donor) for each
method family**, the **dependencies** needed to run those references, and the
**setup + doctor** scripts that guarantee the divergence/benchmark matrix is
never left with a hole *because a reference library was not installed*.

See [`../PRODUCTION_AUDIT.md`](../PRODUCTION_AUDIT.md) §1 for the full parity model
(donor → snapshot → oracle; C++ native + external libs + backends vs oracle at
**1e-8**; bindings vs C++ native at **1e-12**).

## The rule

> Every method has **one documented donor** chosen as the most credible
> implementation for the community. Methods that originate in the Python
> **nirs4all** library (augmentations, scatter/baseline preprocessing, splitters)
> use **nirs4all** as the donor. The only exception is a genuine *paper-only*
> method, which must be explicitly marked as such.

A "missing value" in the matrix must always mean *paper-only / not-applicable*,
**never** "the reference library was not installed". The doctor below enforces that.

## Setup (one command)

```bash
N4M_R_ENV=/home/delete/miniconda3/envs/pls4all_r scripts/setup_parity_references.sh
```

Installs all reference dependencies into the parity venv + R env and finishes by
running the doctor with `--strict`. Idempotent.

## Doctor (verify before any sweep / in CI)

```bash
N4M_R_ENV=/home/delete/miniconda3/envs/pls4all_r \
  parity/python_generator/.venv/bin/python parity/scripts/check_references.py --strict
```

Prints, per dependency layer, what is installed; exits non-zero (with `--strict`)
if any **required** reference dependency is missing. `--json` for machine output.

## Dependency layers

| Layer | Where | What |
|---|---|---|
| **Python references** | parity venv (`parity/python_generator/.venv`) | `scikit-learn`, `numpy`, `scipy`, `auswahl` (RandomFrog/IRF/VIP_SPA/VISSA), `pyswarms` (BinaryPSO), `tensorly` (N-PLS), `ikpls`, `diPLSlib==2.5.0` (di_pls — pinned; the registry shims target this API) |
| **R references** | `$N4M_R_ENV` (R 4.3.x) | `plsVarSel`, `enpls`, `multiblock`, `plsRglm`, `plsRcox`, `chemometrics`, `JICO`, `ropls`, `mixOmics`, `sgPLS`, `mdatools`, `mboost`, `softImpute`, `survival`, `survAUC`, `rms`, `permute`, `prospectr`, `pls`, `spls`, `OmicsPLS`, `kernlab`, `corpcor`, `genalg`, `baseline`, `EMSC`, `wavelets`, `plsdepot`, `mvdalab`, `HDANOVA`, `multiway` |
| **Octave / libPLS** | Octave (from `$N4M_R_ENV` or system) + vendored `bindings/octave/libPLS_1.95` (called via a **one-shot `octave-cli --eval`**, not a persistent oct2py session) | `ecr` (Elastic Component Regression), `cars` (CARS selector) |

### Notes / footguns

- **`OCTAVE_HOME` is required for a conda/relocatable Octave.** Without it, `octave-cli`
  cannot find its standard m-file library and `version`/`pkg` are undefined, which makes
  the bridge fail. `_resolve_octave()` (registry) and the doctor derive `OCTAVE_HOME` as the
  prefix above `bin/octave-cli` automatically; override with `OCTAVE_EXECUTABLE` / `OCTAVE_HOME`.
  The libPLS bridge uses a **one-shot `octave-cli --eval`** (`_octave_run_libpls`); a persistent
  `oct2py` session stalls on Octave 10, so `oct2py` is now **optional/unused**.
- **R `mbpls` is intentionally NOT installed.** The PyPI/CRAN `mbpls` is broken against
  sklearn ≥1.8 and unused — `mb_pls` uses the in-tree `nirs4all...sklearn.mbpls` reference.
- **`rpy2` is not used** — the registry shells out to `Rscript`, not in-process R.
- The donor↔method assignment is currently authoritative in
  `benchmarks/parity_timing/registry.py` (`MethodSpec.python_reference` / `r_reference` /
  `extra_references`). `PRODUCTION_AUDIT.md` §7 tracks migrating this into the catalog
  (`catalog/methods/<id>.yaml`) as the single source of truth, with a `validate.py` gate
  that fails on any method lacking a documented reference that is not marked `paper_only`.
- The ≈104 non-PLS donor methods use **nirs4all** as the donor, frozen as IEEE-754 hex
  fixtures under `parity/fixtures/` and asserted bit-exact by the C++ doctests.

## Per-method donor summary (PLS family)

| Donor (`reference_kind`) | Count | Examples |
|---|---:|---|
| external lib | 65 | sklearn (`pls`, `pcr`, `pls_lda`), R `pls` (`cppls`, solver variants), R `plsVarSel` (selectors), R `multiblock` (`so_pls`), matlab/libPLS (`ecr`, `cars`), `diPLSlib` (`di_pls`), `ikpls` (cross-check) |
| `nirs4all_sanctioned` | 5 | `aom_pls`, `aom_preprocess`, `lw_pls`, `mb_pls`, `pop_pls` |
| `paper_only` (no reference) | 3 | `fused_sparse_pls`, `mir_pls`, `sipls_select` |
| nirs4all (donor methods) | ≈104 | augmentation, preprocessing, filters, splitters |
