"""Command-line entry point for the parity generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from . import suites


REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "parity" / "fixtures"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _serialize(fixture: dict) -> bytes:
    """Stable JSON encoding — sorted keys, no extra whitespace, LF endings."""
    return (
        json.dumps(fixture, sort_keys=True, indent=2, separators=(",", ": "), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _generate_all() -> dict[str, bytes]:
    """Run every checked-in fixture producer; return id -> bytes mapping."""
    return {
        "synthetic_small_pls1_v1":    _serialize(suites.synthetic_small_pls1_v1()),
        "synthetic_small_pls2_v1":    _serialize(suites.synthetic_small_pls2_v1()),
        "synthetic_tiny_centered_v1": _serialize(suites.synthetic_tiny_centered_v1()),
        "synthetic_simpls_tiny_pls1_v1": _serialize(suites.synthetic_simpls_tiny_pls1_v1()),
        "synthetic_simpls_small_pls2_v1": _serialize(suites.synthetic_simpls_small_pls2_v1()),
        "synthetic_svd_tiny_pls1_v1": _serialize(suites.synthetic_svd_tiny_pls1_v1()),
        "synthetic_svd_small_pls2_v1": _serialize(suites.synthetic_svd_small_pls2_v1()),
        "synthetic_pls_svd_tiny_v1": _serialize(suites.synthetic_pls_svd_tiny_v1()),
        "synthetic_pls_svd_small_v1": _serialize(suites.synthetic_pls_svd_small_v1()),
        "synthetic_pipeline_identity_v1": _serialize(suites.synthetic_pipeline_identity_v1()),
        "synthetic_pipeline_center_v1": _serialize(suites.synthetic_pipeline_center_v1()),
        "synthetic_pipeline_autoscale_v1": _serialize(suites.synthetic_pipeline_autoscale_v1()),
        "synthetic_pipeline_pareto_v1": _serialize(suites.synthetic_pipeline_pareto_v1()),
        "synthetic_pipeline_snv_v1": _serialize(suites.synthetic_pipeline_snv_v1()),
        "synthetic_pipeline_msc_v1": _serialize(suites.synthetic_pipeline_msc_v1()),
        "synthetic_pipeline_emsc_v1": _serialize(suites.synthetic_pipeline_emsc_v1()),
        "synthetic_pipeline_detrend_poly_v1": _serialize(suites.synthetic_pipeline_detrend_poly_v1()),
        "synthetic_pipeline_savgol_smooth_v1": _serialize(suites.synthetic_pipeline_savgol_smooth_v1()),
        "synthetic_pipeline_savgol_derivative_v1": _serialize(suites.synthetic_pipeline_savgol_derivative_v1()),
        "synthetic_pipeline_asls_v1": _serialize(suites.synthetic_pipeline_asls_v1()),
        "synthetic_pipeline_norris_williams_v1": _serialize(suites.synthetic_pipeline_norris_williams_v1()),
        "synthetic_pipeline_wavelet_haar_v1": _serialize(suites.synthetic_pipeline_wavelet_haar_v1()),
        "synthetic_pipeline_osc_v1": _serialize(suites.synthetic_pipeline_osc_v1()),
        "synthetic_pipeline_epo_v1": _serialize(suites.synthetic_pipeline_epo_v1()),
        "synthetic_metrics_regression_v1": _serialize(suites.synthetic_metrics_regression_v1()),
        "synthetic_classification_binary_metrics_v1": _serialize(suites.synthetic_classification_binary_metrics_v1()),
        "synthetic_classification_multiclass_metrics_v1": _serialize(suites.synthetic_classification_multiclass_metrics_v1()),
        "synthetic_classification_calibration_curve_v1": _serialize(suites.synthetic_classification_calibration_curve_v1()),
        "synthetic_pls_lda_multiclass_v1": _serialize(suites.synthetic_pls_lda_multiclass_v1()),
        "synthetic_pls_logistic_multiclass_v1": _serialize(suites.synthetic_pls_logistic_multiclass_v1()),
        "synthetic_mb_pls_block_weighted_v1": _serialize(suites.synthetic_mb_pls_block_weighted_v1()),
        "synthetic_lw_pls_local_window_v1": _serialize(suites.synthetic_lw_pls_local_window_v1()),
        "synthetic_variable_importance_pls2_v1": _serialize(suites.synthetic_variable_importance_pls2_v1()),
        "synthetic_variable_selection_rankers_v1": _serialize(suites.synthetic_variable_selection_rankers_v1()),
        "synthetic_interval_selection_moving_window_v1": _serialize(suites.synthetic_interval_selection_moving_window_v1()),
        "synthetic_coefficient_stability_mcuve_v1": _serialize(suites.synthetic_coefficient_stability_mcuve_v1()),
        "synthetic_uve_artificial_variables_v1": _serialize(suites.synthetic_uve_artificial_variables_v1()),
        "synthetic_emcuve_pls_ensemble_v1": _serialize(suites.synthetic_emcuve_pls_ensemble_v1()),
        "synthetic_spa_pls_projection_v1": _serialize(suites.synthetic_spa_pls_projection_v1()),
        "synthetic_cars_pls_competitive_v1": _serialize(suites.synthetic_cars_pls_competitive_v1()),
        "synthetic_random_frog_pls_v1": _serialize(suites.synthetic_random_frog_pls_v1()),
        "synthetic_scars_pls_stability_v1": _serialize(suites.synthetic_scars_pls_stability_v1()),
        "synthetic_ga_pls_wrapper_v1": _serialize(suites.synthetic_ga_pls_wrapper_v1()),
        "synthetic_shaving_pls_recursive_v1": _serialize(suites.synthetic_shaving_pls_recursive_v1()),
        "synthetic_bve_pls_backward_v1": _serialize(suites.synthetic_bve_pls_backward_v1()),
        "synthetic_t2_pls_hotelling_v1": _serialize(suites.synthetic_t2_pls_hotelling_v1()),
        "synthetic_wvc_pls_numeric_v1": _serialize(suites.synthetic_wvc_pls_numeric_v1()),
        "synthetic_component_coefficients_pls2_v1": _serialize(suites.synthetic_component_coefficients_pls2_v1()),
        "synthetic_validation_kfold_balanced_v1": _serialize(suites.synthetic_validation_kfold_balanced_v1()),
        "synthetic_validation_leave_one_out_v1": _serialize(suites.synthetic_validation_leave_one_out_v1()),
        "synthetic_validation_holdout_v1": _serialize(suites.synthetic_validation_holdout_v1()),
        "synthetic_validation_external_folds_v1": _serialize(suites.synthetic_validation_external_folds_v1()),
        "synthetic_validation_repeated_kfold_v1": _serialize(suites.synthetic_validation_repeated_kfold_v1()),
        "synthetic_validation_monte_carlo_v1": _serialize(suites.synthetic_validation_monte_carlo_v1()),
        "synthetic_validation_kennard_stone_v1": _serialize(suites.synthetic_validation_kennard_stone_v1()),
        "synthetic_validation_spxy_v1": _serialize(suites.synthetic_validation_spxy_v1()),
        "synthetic_cv_kfold_nipals_pls1_v1": _serialize(suites.synthetic_cv_kfold_nipals_pls1_v1()),
        "synthetic_cv_kfold_nipals_pls2_v1": _serialize(suites.synthetic_cv_kfold_nipals_pls2_v1()),
        "synthetic_component_cv_simpls_pls2_v1": _serialize(suites.synthetic_component_cv_simpls_pls2_v1()),
        "synthetic_power_tiny_pls1_v1": _serialize(suites.synthetic_power_tiny_pls1_v1()),
        "synthetic_power_small_pls2_v1": _serialize(suites.synthetic_power_small_pls2_v1()),
        "synthetic_randomized_svd_tiny_pls1_v1": _serialize(suites.synthetic_randomized_svd_tiny_pls1_v1()),
        "synthetic_randomized_svd_small_pls2_v1": _serialize(suites.synthetic_randomized_svd_small_pls2_v1()),
        "synthetic_canonical_tiny_pls1_v1": _serialize(suites.synthetic_canonical_tiny_pls1_v1()),
        "synthetic_canonical_small_pls2_v1": _serialize(suites.synthetic_canonical_small_pls2_v1()),
        "synthetic_canonical_svd_tiny_pls1_v1": _serialize(suites.synthetic_canonical_svd_tiny_pls1_v1()),
        "synthetic_canonical_svd_small_pls2_v1": _serialize(suites.synthetic_canonical_svd_small_pls2_v1()),
        "synthetic_pls_da_binary_v1": _serialize(suites.synthetic_pls_da_binary_v1()),
        "synthetic_pls_da_multiclass_v1": _serialize(suites.synthetic_pls_da_multiclass_v1()),
        "synthetic_opls_tiny_pls1_v1": _serialize(suites.synthetic_opls_tiny_pls1_v1()),
        "synthetic_opls_small_pls1_v1": _serialize(suites.synthetic_opls_small_pls1_v1()),
        "synthetic_opls_small_pls2_v1": _serialize(suites.synthetic_opls_small_pls2_v1()),
        "synthetic_opls_da_binary_v1": _serialize(suites.synthetic_opls_da_binary_v1()),
        "synthetic_opls_da_multiclass_v1": _serialize(suites.synthetic_opls_da_multiclass_v1()),
        "synthetic_kernel_tiny_pls1_v1": _serialize(suites.synthetic_kernel_tiny_pls1_v1()),
        "synthetic_kernel_wide_pls2_v1": _serialize(suites.synthetic_kernel_wide_pls2_v1()),
        "synthetic_wide_kernel_pls1_v1": _serialize(suites.synthetic_wide_kernel_pls1_v1()),
        "synthetic_wide_kernel_pls2_v1": _serialize(suites.synthetic_wide_kernel_pls2_v1()),
        "synthetic_oscores_tiny_pls1_v1": _serialize(suites.synthetic_oscores_tiny_pls1_v1()),
        "synthetic_oscores_small_pls2_v1": _serialize(suites.synthetic_oscores_small_pls2_v1()),
        "synthetic_pcr_tiny_pls1_v1": _serialize(suites.synthetic_pcr_tiny_pls1_v1()),
        "synthetic_pcr_small_pls2_v1": _serialize(suites.synthetic_pcr_small_pls2_v1()),
    }


def _regenerate(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = _generate_all()
    manifest = {"schema_version": 1, "fixtures": {}}
    for fid, payload in payloads.items():
        path = out_dir / f"{fid}.json"
        path.write_bytes(payload)
        manifest["fixtures"][fid] = {
            "path": f"{fid}.json",
            "sha256": _sha256(payload),
            "bytes": len(payload),
        }
    (out_dir / "manifest.json").write_bytes(_serialize(manifest))
    print(f"Wrote {len(payloads)} fixtures + manifest.json to {out_dir}")
    return 0


def _check(out_dir: Path) -> int:
    payloads = _generate_all()
    failures = 0
    for fid, payload in payloads.items():
        existing = out_dir / f"{fid}.json"
        if not existing.exists():
            print(f"FAIL: {fid}.json is missing from {out_dir}", file=sys.stderr)
            failures += 1
            continue
        on_disk = existing.read_bytes()
        if on_disk != payload:
            print(
                f"FAIL: {fid}.json differs (on-disk {_sha256(on_disk)[:12]}..."
                f", regenerated {_sha256(payload)[:12]}...)",
                file=sys.stderr,
            )
            failures += 1
    if failures == 0:
        print(f"OK: all {len(payloads)} fixtures match the on-disk copies bit-for-bit")
    return 0 if failures == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="generate-fixtures",
                                description="Regenerate pls4all parity fixtures.")
    p.add_argument("--out", type=Path, default=FIXTURE_DIR,
                   help="Output directory (default: parity/fixtures).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--regenerate", action="store_true",
                   help="Overwrite the on-disk fixtures with freshly generated ones.")
    g.add_argument("--check", action="store_true",
                   help="Regenerate in memory and compare against on-disk fixtures bit-for-bit.")
    args = p.parse_args(argv)
    return _regenerate(args.out) if args.regenerate else _check(args.out)


if __name__ == "__main__":
    sys.exit(main())
