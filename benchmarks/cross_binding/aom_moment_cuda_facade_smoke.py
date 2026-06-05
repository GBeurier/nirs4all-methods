#!/usr/bin/env python3
"""CUDA smoke for the logical AOM and moment Python facades.

The binding loads one libn4m per Python process, so this script spawns a child
interpreter with ``N4M_LIB_PATH`` pointing at the CUDA build. It then verifies
that ``n4m.moment`` and ``n4m.aom`` are thin aliases over the public runtime and
that a wide PLS1 moment screen uses the CUDA device CV path on one GPU.

This is a correctness/route smoke, not a timing benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CUDA_LIB = REPO / "build" / "cuda-on" / "cpp" / "src" / "libn4m.so"
DEFAULT_PYTHONPATH = REPO / "bindings" / "python" / "src"
DEFAULT_OUT = REPO / "benchmarks" / "cross_binding" / "aom_moment_cuda_facade_smoke.json"


CHILD_CODE = r"""
import json
import sys

import numpy as np

import n4m
import n4m.aom as aom
import n4m.moment as moment


def main():
    payload = json.loads(sys.stdin.read())
    n_samples = int(payload["n_samples"])
    n_features = int(payload["n_features"])
    cv = int(payload["cv"])
    seed = int(payload["seed"])

    if n_samples <= cv:
        raise ValueError("n_samples must be greater than cv")
    if n_features < 1024:
        raise ValueError("n_features must be >= 1024 to smoke the CUDA PLS1 device loop")

    assert n4m.aom is aom
    assert n4m.moment is moment
    assert callable(n4m.moments)
    assert moment.moments is n4m.moments
    assert moment.sweep_run is n4m.sweep_run
    assert moment.aom_moment_screen_refit_campaign is n4m.aom_moment_screen_refit_campaign
    assert moment.aom_screen_refit_candidate_pool is n4m.aom_screen_refit_candidate_pool
    assert moment.aom_refit_candidates is n4m.aom_refit_candidates
    assert moment.aom_chain_fixed_fit_run is n4m.aom_chain_fixed_fit_run
    assert moment.build_aom_strict_chain_grid is n4m.build_aom_strict_chain_grid
    assert moment.decode_aom_chains is n4m.decode_aom_chains
    assert moment.aom_candidate_table is n4m.aom_candidate_table
    assert moment.aom_evaluate_candidates is n4m.aom_evaluate_candidates
    assert moment.aom_candidate_operator_summary is n4m.aom_candidate_operator_summary
    assert moment.aom_candidate_route_summary is n4m.aom_candidate_route_summary
    assert moment.aom_candidate_rank_diagnostics is n4m.aom_candidate_rank_diagnostics
    assert moment.aom_candidate_report_records is n4m.aom_candidate_report_records
    assert moment.aom_save_candidate_report is n4m.aom_save_candidate_report
    assert moment.aom_load_candidate_report is n4m.aom_load_candidate_report
    assert moment.NativeAOMFixedCandidateRegressor is n4m.NativeAOMFixedCandidateRegressor
    assert (
        moment.NativeAOMMomentScreenRefitRegressor
        is n4m.NativeAOMMomentScreenRefitRegressor
    )
    assert (
        moment.NativeAOMMomentPLSScreenRefitRegressor
        is n4m.NativeAOMMomentPLSScreenRefitRegressor
    )
    assert (
        moment.NativeAOMMomentRidgeScreenRefitRegressor
        is n4m.NativeAOMMomentRidgeScreenRefitRegressor
    )
    assert aom.aom_preprocess is n4m.aom_preprocess
    assert aom.aom_chain_sweep_run is n4m.aom_chain_sweep_run
    assert aom.aom_moment_screen_refit_campaign is n4m.aom_moment_screen_refit_campaign

    cuda_available = int(n4m.lib.n4m_backend_is_available(5))
    if cuda_available != 1:
        raise RuntimeError("CUDA backend is not available from the loaded libn4m")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.7 * X[:, 0]
        - 0.25 * X[:, min(7, n_features - 1)]
        + 0.05 * rng.standard_normal(n_samples)
    )
    folds = np.arange(n_samples, dtype=np.int32) % cv
    pre = aom.aom_preprocess(X, y, operators=["identity"], gating_mode="soft")
    if pre["transformed"].shape != X.shape:
        raise RuntimeError("aom.aom_preprocess returned an unexpected transformed shape")
    if pre["operator_kinds"].tolist() != [0]:
        raise RuntimeError("aom.aom_preprocess identity operator kind was not reported")

    moment_res = moment.sweep_run(
        X,
        y,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1],
        heads=("pls",),
        scale_x=False,
        score_only=True,
    )
    chains = [[("identity", ())], [("savgol_smooth", (5, 2))]]
    aom_res = aom.aom_chain_sweep_run(
        X,
        y,
        chains,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[],
        pls_components=[1],
        heads=("pls",),
        scale_x=False,
        score_only=True,
        moment_policy="force_moments",
    )

    moment_device_cv = int(moment_res.get("n_pls_moment_cuda_device_cv_fits", 0))
    aom_device_cv = int(aom_res.get("n_pls_moment_cuda_device_cv_fits", 0))
    if moment_device_cv <= 0:
        raise RuntimeError("moment.sweep_run did not report CUDA device CV fits")
    if aom_device_cv <= 0:
        raise RuntimeError("aom.aom_chain_sweep_run did not report CUDA device CV fits")

    print(json.dumps({
        "report_schema": "n4m.aom_moment_cuda_facade_smoke.v1",
        "library_path": n4m.library_path(),
        "abi": ".".join(str(value) for value in n4m.abi_version()),
        "cuda_available": cuda_available,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "facades": {
            "aom_aliases_top_level": True,
            "moment_aliases_top_level": True,
            "n4m_moments_stays_callable": True,
            "aom_moment_screen_refit_aliases_top_level": True,
            "aom_moment_screen_refit_estimators_alias_top_level": True,
            "aom_moment_winner_reuse_aliases_top_level": True,
            "aom_moment_audit_report_aliases_top_level": True,
            "aom_moment_route_summary_alias_top_level": True,
            "aom_preprocess_aliases_top_level": True,
        },
        "aom_preprocess": {
            "n_operators": int(pre["n_operators"]),
            "mode": int(pre["mode"]),
            "operator_kinds": [int(value) for value in pre["operator_kinds"]],
        },
        "moment": {
            "selected_cv_rmse": float(moment_res["selected_cv_rmse"]),
            "n_candidates": int(moment_res["n_candidates"]),
            "n_pls_moment_cuda_device_cv_fits": moment_device_cv,
            "n_pls_moment_host_cv_fits": int(moment_res.get("n_pls_moment_host_cv_fits", 0)),
            "n_pls_materialized_cv_fits": int(moment_res.get("n_pls_materialized_cv_fits", 0)),
        },
        "aom": {
            "selected_cv_rmse": float(aom_res["selected_cv_rmse"]),
            "n_candidates": int(aom_res["n_candidates"]),
            "n_pls_moment_cuda_device_cv_fits": aom_device_cv,
            "n_pls_moment_host_cv_fits": int(aom_res.get("n_pls_moment_host_cv_fits", 0)),
            "n_pls_materialized_cv_fits": int(aom_res.get("n_pls_materialized_cv_fits", 0)),
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-lib", default=str(DEFAULT_CUDA_LIB))
    parser.add_argument("--pythonpath", default=str(DEFAULT_PYTHONPATH))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5150)
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env["N4M_LIB_PATH"] = str(Path(args.cuda_lib))
    env["PYTHONPATH"] = str(Path(args.pythonpath))
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    payload = {
        "n_samples": int(args.n_samples),
        "n_features": int(args.n_features),
        "cv": int(args.cv),
        "seed": int(args.seed),
    }
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(CHILD_CODE)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CUDA facade smoke failed with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    report = json.loads(proc.stdout)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
