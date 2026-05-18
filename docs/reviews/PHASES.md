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
| 3 | Stateful scatter preprocessings + c4a.h cleanup | ACCEPT (3 follow-ups applied) | — | `98c9e25` | 51 / 51 |
| 4 | Derivatives & smoothing | ACCEPT (5 medium/low deferred) | ACCEPT (Phases 3+4 combined) | `92be7b7` | 61 / 61 |
| 5a | Baseline correction (Detrend + AsLS family core) | REVISE → 3 fixes applied (algo docs, AirPLS tolerance dispatch, frozen-ref validation script) | — (Codex due after Phase 5b) | (pushed) | 69 / 69 |
| 5b | Baseline correction (ModPoly/IModPoly/SNIP/RollingBall/IAsLS/BEADS) + LogTransform split | — | — | `ad6e3be` | 82 / 82 |
| 6 | Wavelets & denoising (6 ops, `c4a_pp_wavelet_*`) | — (parallel batch) | — | (batch commit) | 12 / 12 (subset) |
| 7 | Signal conversion | — (parallel batch) | — | (batch commit) | 11 / 11 (subset) |
| 8 | Orthogonalization | — (parallel batch) | — | (batch commit) | 6 / 6 (subset) |
| 9 | Feature selection | — (parallel batch) | — | (batch commit) | 6 / 6 (subset) |
| 10 | Resampling, cropping, targets | — (parallel batch) | — | (batch commit) | 10 / 10 (subset) |
| 11 | Splitters (9 algorithms) | — (parallel batch) | — | (batch commit) | 18 / 18 (subset) |
| 12 | Filters: Y-based outliers | — (parallel batch) | — | (batch commit) | 12 / 12 (subset) |
| 13 | Filters: X-based outliers (6 methods, `c4a_filter_x_outlier_*`) | — (parallel batch) | — | (batch commit) | 12 / 12 (subset) |
| 14 | Filters: leverage, quality, composite | — (parallel batch) | — | (batch commit) | 6 / 6 (subset) |
| 15 | Augmentations: noise + drift (7 ops, `c4a_aug_*`) | — (parallel batch) | — | (batch commit) | 14 / 14 (subset) |
| 16 | Augmentations: wavelength + spectral (10 ops, `c4a_aug_*`) | — (parallel batch) | — | (batch commit) | 20 / 20 (subset) |
| 17 | Augmentations: mixup + physical (10 ops, `c4a_aug_*`) | — (parallel batch) | — | (batch commit) | 25 / 25 (subset) |
| 18 | Augmentations: edge + splines + random (12 ops, `c4a_aug_*`) | — (parallel batch) | — | (batch commit) | 12 / 12 (subset) |
| 19 | Signal type detection + NIRS metrics | — (parallel batch) | — | (batch commit) | 12 / 12 (subset) |
| 20 | Transfer metrics | — (parallel batch) | — | (batch commit) | 3 / 3 (subset) |
| 21 | FCK static transformer | — (parallel batch) | — | (batch commit) | 4 / 4 (subset) |
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
| 1.4.0 | 4 | 96 | + 17 derivative & smoothing symbols. |
| 1.5.0 | 5a | 108 | + 12 baseline correction core symbols (Detrend + AsLS family). |
| 1.6.0 | 5b | 126 | + 18 baseline correction rest (ModPoly, IModPoly, SNIP, RollingBall, IAsLS, BEADS) + LogTransform fit/transform split + LDLT refactor. |
| 1.7.0 | 7-21 (parallel batch) | 251 | + 125 symbols across signal conversion, orthogonalisation, feature selection, resampling, splitters, filters, signal type / NIRS metrics, transfer metrics, FCK static transformer. |
| 1.8.0 | 6, 13, 15-18 (parallel batch) | 400 | + 149 symbols across wavelets (27 + 4 enums), X-outlier filter (5 + 1 enum), and the new `c4a_aug_*` ABI category (117 augmenter symbols + helper opcodes). |
