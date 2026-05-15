# Phase 31l — Batch 12: PLS monitoring + one-SE rule

**Status:** shipped — local tag `phase-31l-batch-12-monitoring-onese`
(`0.82.0+abi.1.12.0`).

This is the last batch of the §1-§30 Overview taxonomy promotion to the
public C ABI. Every internal kernel that was shipped in phases 0 — 30
now has a public entry point and a parity-gate row (external reference
when available, paper-only with citation otherwise).

## Methods (both paper-only)

| Method | Internal kernel | Public C ABI entry | Paper |
|--------|-----------------|--------------------|-------|
| PLS monitoring | `pls_monitoring_fit` + `pls_monitoring_evaluate` | `p4a_pls_monitoring_run` | Kourti & MacGregor 1995 |
| One-SE rule | `select_one_se_components` | `p4a_one_se_rule_compute` | Breiman et al. 1984 |

## Why paper-only

- **PLS monitoring:** Industrial monitoring tools (Aspen, GE Proficy,
  Umetrics SIMCA-online) are proprietary; R `mvprocess` is not on CRAN.
  No widely installable port of percentile-based T²/Q thresholding.
- **One-SE rule:** Generic CART-era helper. R `pls::selectNcomp` has a
  similar one-SE-like option, but operates on a different RMSE shape
  and includes its own CV step — we expose the rule as a standalone
  helper rather than a full CV+selection method, so a 1:1 external
  reference isn't meaningful.

## ABI delta

- 2 new public symbols, total 159.
- C ABI minor 11 → 12.

## Local gate

- 256 internal C++ tests.
- Parity gate: 13 external PASS, 24 paper-only smoke PASS, 0 numpy.
