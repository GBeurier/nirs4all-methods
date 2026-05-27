# Parity reference environments

Pinned, reproducible environments for the **external** reference libraries the
parity comparator checks n4m against (sklearn, ikpls, R `pls`/`ropls`/`mixOmics`,
Octave, …). Each external snapshot records the `env_id` + lock hash it was
generated in, so a verdict is never silently compared against a moving
dependency.

> **Phase-C-min status.** This directory currently ships the recipe **schema**
> + one **real pinned recipe** (`env-py-sklearn-1.4/`) + a local dev env
> (`../../dev/environment.yml`). The full per-(framework, version) matrix, the
> GHCR build/publish CI (`parity-env-build.yml`), and the in-container reference
> generators are **deferred** (full Phase C, catalog-coupled). The golden
> `n4m_self` snapshots produced today (`parity/generators/run.py`) need **no**
> external env — they run against the in-repo libn4m.

## Recipe layout

```
parity/environments/<env_id>/
  meta.yaml          # machine-readable descriptor (schema below)
  requirements.lock  # fully-pinned dependency set (python: pip-compile output;
                     # R: renv.lock; octave: pkg list) — the source of truth
  Dockerfile         # builds the env from requirements.lock (reproducible)
```

`<env_id>` convention: `env-<lang>-<framework>-<major.minor>`, e.g.
`env-py-sklearn-1.4`, `env-r-pls-2.8`, `env-octave-6.4`.

## `meta.yaml` schema

| field | meaning |
|-------|---------|
| `env_id` | unique id, matches the directory name |
| `language` | `python` \| `r` \| `octave` |
| `framework` | reference library this env pins (`scikit-learn`, `pls`, …) |
| `version` | the framework's pinned version |
| `lock_file` | path to the pinned lock (relative to this dir) |
| `description` | one line on what this env is for |
| `consumed_by` | the generator(s) that run inside it (deferred Docker generators) |

The **lock hash** referenced by snapshots is `sha256(requirements.lock)`; it is
computed on demand (no need to store it here). A snapshot whose recorded lock
hash differs from the current recipe is refused by the comparator (full Phase C,
deferred).

## Adding a recipe

1. Create `parity/environments/<env_id>/` with the three files above.
2. Pin `requirements.lock` exactly (no ranges). For Python, `pip-compile`.
3. Fill `meta.yaml`.
4. (Deferred) wire it into `parity-env-build.yml` so CI publishes the image to
   GHCR as `<env_id>-<short_lock_hash>`.
