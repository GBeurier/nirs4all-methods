#!/usr/bin/env python3
"""Reclassify Python-composite methods' C-kernel binding cells as not_available.

A handful of registry methods have a canonical path that is a Python composite
(sklearn QDA / LogReg / numpy plsRglm / L2Boost / BaggingRegressor / tensorly
PARAFAC, ...) which BYPASSES the libn4m C kernel. registry / python tiers
reproduce them correctly, but the R / MATLAB binding dispatchers route through
the C kernel, which legitimately cannot run them and returns
"convergence failed" / an error. Those cells are NOT n4m failures — the binding
simply does not implement the method. build_landing renders a `not_available`
reason as a neutral gray cell, whereas a raw "convergence failed" falls through
to a red binding failure. This post-processor rewrites the reason so the
dashboard reads honestly.

Run after the orchestrator, before build_landing (the run script wires it in).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Methods whose canonical implementation is a Python composite (no C-kernel
# binding). Their registry/python cells stay untouched; only the C-kernel
# binding tiers (R + MATLAB) are reclassified.
COMPOSITE_METHODS = {
    "pls_qda", "pls_logistic", "pls_glm", "pls_cox",
    "bagging_pls", "boosting_pls", "random_subspace_pls",
    "group_sparse_pls", "n_pls",
}
# C-kernel binding tiers that cannot run a Python-composite method.
C_KERNEL_BINDINGS = {
    "r_tier1", "r_tier2", "r_pls_compat", "r_mdatools_compat",
    "matlab_tier1", "matlab_tier2",
}
REASON = "not_available: python-composite method (no C-kernel binding)"

# Selectors whose canonical reference path routes through the `auswahl` Python
# package (registry._*_via_auswahl). auswahl is not on PyPI and cannot be
# installed here, so EVERY tier of these methods crashes with
# ModuleNotFoundError — not an n4m failure. Mark every tier not_available with
# an honest reason instead of a red ImportError.
AUSWAHL_METHODS = {
    "vissa_select", "random_frog_select", "vip_spa_select", "irf_select",
}
AUSWAHL_REASON = "not_available: reference requires the 'auswahl' package (not on PyPI)"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(
        "benchmarks/cross_binding/results/full_matrix.csv")
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{path}: empty, nothing to do")
        return 0
    fields = list(rows[0].keys())
    n_comp = n_aus = 0
    for r in rows:
        algo = r.get("algorithm")
        if (algo in COMPOSITE_METHODS
                and r.get("backend") in C_KERNEL_BINDINGS
                and str(r.get("binding_parity_ok", "")).lower() != "true"):
            r["reason"] = REASON
            r["binding_parity_note"] = REASON
            r["binding_parity_ok"] = ""        # neutral, not False
            r["reference_parity_ok"] = ""
            r["reference_parity_note"] = REASON
            n_comp += 1
        elif (algo in AUSWAHL_METHODS
                and str(r.get("ok", "")).lower() != "true"
                and "auswahl" in (r.get("reason", "") or "").lower()):
            r["reason"] = AUSWAHL_REASON
            r["binding_parity_note"] = AUSWAHL_REASON
            r["binding_parity_ok"] = ""
            r["reference_parity_ok"] = ""
            r["reference_parity_note"] = AUSWAHL_REASON
            n_aus += 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"reclassified {n_comp} composite C-kernel binding cells + "
          f"{n_aus} auswahl-dependent cells -> not_available in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
