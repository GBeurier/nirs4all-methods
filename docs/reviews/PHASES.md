# Phase Verdict Index

Aggregate index of per-phase review outcomes. Each phase is reviewed by
Opus (after-the-fact, before commit) and Codex (every two phases, on the
state at the time of the latest commit). Phases that have not yet shipped
appear with empty cells.

| Phase | Topic | Opus verdict | Codex verdict | Commit SHA | Tests |
| --- | --- | --- | --- | --- | --- |
| 0 | Bootstrap (chassis, ABI, CI) | ACCEPT (9 high-conf fixes applied) | — | _initial_ | 7 / 7 |
| 1 | Common infrastructure (PCG64 RNG) | ACCEPT (2 nits + 1 medium fixed) | — | (pushed) | 20 / 20 |
| 2 | Stateless scatter preprocessings | REVISE → 3 high-conf fixes applied | ACCEPT with Phase 3 prerequisites | (pushed) | 34 / 34 |
| 3 | Stateful scatter preprocessings + c4a.h cleanup | — | — | — | 48 / 48 |
| 4 | Derivatives & smoothing | — | — | — | — |
| 5 | Baseline correction (AsLS family) | — | — | — | — |
| 6 | Wavelets & denoising | — | — | — | — |
| 7 | Signal conversion | — | — | — | — |
| 8 | Orthogonalization | — | — | — | — |
| 9 | Feature selection | — | — | — | — |
| 10 | Resampling, cropping, targets | — | — | — | — |
| 11 | Splitters (9 algorithms) | — | — | — | — |
| 12 | Filters: Y-based outliers | — | — | — | — |
| 13 | Filters: X-based outliers (6 methods) | — | — | — | — |
| 14 | Filters: leverage, quality, composite | — | — | — | — |
| 15 | Augmentations: noise + drift | — | — | — | — |
| 16 | Augmentations: wavelength + spectral | — | — | — | — |
| 17 | Augmentations: mixup + physical | — | — | — | — |
| 18 | Augmentations: edge + splines + random | — | — | — | — |
| 19 | Signal type detection + NIRS metrics | — | — | — | — |
| 20 | Transfer metrics | — | — | — | — |
| 21 | FCK static transformer | — | — | — | — |
| 22 | Python binding (sklearn) | — | — | — | — |
| 23 | R binding (.Call + tidymodels) | — | — | — | — |
| 24 | MATLAB binding | — | — | — | — |
| 25 | JS / WASM binding | — | — | — | — |
| 26 | Benchmarks + GH Pages | — | — | — | — |

## Review cadence

* **Opus post-review** — runs after every phase, before commit. Transcript
  lives at `docs/reviews/phase-<N>/opus-post.md`.
* **Codex review** — runs every two phases on the latest commit. Transcript
  lives at `docs/reviews/phase-<N>/codex-post.md`.

## ABI version timeline

| ABI version | Phase | Symbols | Notes |
| --- | --- | --- | --- |
| 1.0.0 | 0 | 29 | Bootstrap (context, matrix view, status). |
| 1.1.0 | 1 | 36 | + 7 PCG64 RNG symbols. |
| 1.2.0 | 2 | 57 | + 21 stateless preprocessing symbols. |
| 1.3.0 | 3 | 79 | + 22 stateful preprocessing symbols; c4a.h trimmed from 2208 to 680 lines. |
