# Phase 7c - R binding scaffolding

Status: shipped locally as `phase-7-comprehensive-benchmark` (rolled up
with 7b/7d/7e).

Goal: ship the minimum R package that lets the comprehensive benchmark
matrix run on R when `Rscript` + the package are available. No
mandatory R dependency — Rcpp is intentionally NOT used (we stick to
plain `.Call` to preserve the zero-mandatory-deps promise of libp4a).

Delivered:

- `bindings/r/pls4all/` standard R package layout:
  - `DESCRIPTION`, `NAMESPACE`, `R/pls4all-package.R`,
    `R/version.R`, `R/model.R`.
  - `src/r_gateway.c` — `.Call` gateway over `libp4a`: version queries
    + fit/predict for all 9 shipped PLS regression solvers.
  - `src/r_init.c` — `R_registerRoutines` so `R CMD check` is happy.
  - `src/Makevars` + `src/Makevars.win` — link against `libp4a` via
    `PLS4ALL_LIB_DIR` configure var.
  - `tests/testthat/test-version.R` — version smoke.
- `bindings/r/README.md` documents how to build / install / smoke and
  the available solver names.
- Matrix benchmark runner detects `Rscript` and the installed
  package; when missing it records `r_not_available` /
  `r_package_missing` and skips R accuracy gating.

Build / install (on a host with R):

```bash
cmake --build --preset dev-release --parallel
cd bindings/r
R CMD INSTALL --configure-vars="\
  PLS4ALL_INCLUDE_DIR=$PWD/../../cpp/include \
  PLS4ALL_LIB_DIR=$PWD/../../build/dev-release/cpp/src" pls4all
```

This phase explicitly does NOT ship:

- AOM/POP R wrappers (Phase 6f exposed the C ABI; R bindings TBD).
- sklearn-style estimator classes; the matrix benchmark only needs
  fit/predict.
- Parity tests against R `pls` / `ropls` / `mixOmics`. Those land with
  the wider binding roadmap.

This host does not have R installed, so the package source compiles
only against the C ABI on a future host. The source files are checked
in, and the matrix runner skips R rows safely on hosts without
`Rscript`.
