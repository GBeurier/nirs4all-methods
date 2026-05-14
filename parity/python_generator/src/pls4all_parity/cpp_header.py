"""Generate compact C++ fixture headers from parity JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "parity" / "fixtures"
HEADER_SPECS = {
    "simpls": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "simpls_fixtures.hpp",
        "fixtures": (
            "synthetic_simpls_tiny_pls1_v1",
            "synthetic_simpls_small_pls2_v1",
        ),
        "struct": "SimplsFixture",
        "array": "kSimplsFixtures",
    },
    "svd": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "svd_fixtures.hpp",
        "fixtures": (
            "synthetic_svd_tiny_pls1_v1",
            "synthetic_svd_small_pls2_v1",
        ),
        "struct": "SvdFixture",
        "array": "kSvdFixtures",
    },
    "pls-svd": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "pls_svd_fixtures.hpp",
        "fixtures": (
            "synthetic_pls_svd_tiny_v1",
            "synthetic_pls_svd_small_v1",
        ),
        "struct": "PlsSvdFixture",
        "array": "kPlsSvdFixtures",
    },
    "pipeline": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "pipeline_fixtures.hpp",
        "fixtures": (
            "synthetic_pipeline_identity_v1",
            "synthetic_pipeline_center_v1",
            "synthetic_pipeline_autoscale_v1",
            "synthetic_pipeline_pareto_v1",
            "synthetic_pipeline_snv_v1",
            "synthetic_pipeline_msc_v1",
            "synthetic_pipeline_emsc_v1",
            "synthetic_pipeline_detrend_poly_v1",
            "synthetic_pipeline_savgol_smooth_v1",
            "synthetic_pipeline_savgol_derivative_v1",
            "synthetic_pipeline_asls_v1",
            "synthetic_pipeline_norris_williams_v1",
            "synthetic_pipeline_wavelet_haar_v1",
            "synthetic_pipeline_osc_v1",
            "synthetic_pipeline_epo_v1",
        ),
        "struct": "PipelineFixture",
        "array": "kPipelineFixtures",
    },
    "metrics": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "metrics_fixtures.hpp",
        "fixtures": (
            "synthetic_metrics_regression_v1",
        ),
        "struct": "MetricsFixture",
        "array": "kMetricsFixtures",
    },
    "classification-metrics": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "classification_metrics_fixtures.hpp",
        "fixtures": (
            "synthetic_classification_binary_metrics_v1",
        ),
        "struct": "ClassificationMetricsFixture",
        "array": "kClassificationMetricsFixtures",
    },
    "classification-extensions": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "classification_extension_fixtures.hpp",
        "fixtures": (
            "synthetic_classification_multiclass_metrics_v1",
            "synthetic_classification_calibration_curve_v1",
        ),
        "struct": "ClassificationExtensionFixture",
        "array": "kClassificationExtensionFixtures",
    },
    "pls-lda": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "pls_lda_fixtures.hpp",
        "fixtures": (
            "synthetic_pls_lda_multiclass_v1",
        ),
        "struct": "PlsLdaFixture",
        "array": "kPlsLdaFixtures",
    },
    "pls-logistic": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "pls_logistic_fixtures.hpp",
        "fixtures": (
            "synthetic_pls_logistic_multiclass_v1",
        ),
        "struct": "PlsLogisticFixture",
        "array": "kPlsLogisticFixtures",
    },
    "mb-pls": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "mb_pls_fixtures.hpp",
        "fixtures": (
            "synthetic_mb_pls_block_weighted_v1",
        ),
        "struct": "MbPlsFixture",
        "array": "kMbPlsFixtures",
    },
    "lw-pls": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "lw_pls_fixtures.hpp",
        "fixtures": (
            "synthetic_lw_pls_local_window_v1",
        ),
        "struct": "LwPlsFixture",
        "array": "kLwPlsFixtures",
    },
    "variable-importance": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "variable_importance_fixtures.hpp",
        "fixtures": (
            "synthetic_variable_importance_pls2_v1",
        ),
        "struct": "VariableImportanceFixture",
        "array": "kVariableImportanceFixtures",
    },
    "component-coefficients": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "component_coefficients_fixtures.hpp",
        "fixtures": (
            "synthetic_component_coefficients_pls2_v1",
        ),
        "struct": "ComponentCoefficientsFixture",
        "array": "kComponentCoefficientsFixtures",
    },
    "validation": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "validation_fixtures.hpp",
        "fixtures": (
            "synthetic_validation_kfold_balanced_v1",
            "synthetic_validation_leave_one_out_v1",
            "synthetic_validation_holdout_v1",
        ),
        "struct": "ValidationFixture",
        "array": "kValidationFixtures",
    },
    "advanced-validation": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "advanced_validation_fixtures.hpp",
        "fixtures": (
            "synthetic_validation_external_folds_v1",
            "synthetic_validation_repeated_kfold_v1",
            "synthetic_validation_monte_carlo_v1",
            "synthetic_validation_kennard_stone_v1",
            "synthetic_validation_spxy_v1",
        ),
        "struct": "AdvancedValidationFixture",
        "array": "kAdvancedValidationFixtures",
    },
    "cross-validation": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "cross_validation_fixtures.hpp",
        "fixtures": (
            "synthetic_cv_kfold_nipals_pls1_v1",
            "synthetic_cv_kfold_nipals_pls2_v1",
        ),
        "struct": "CrossValidationFixture",
        "array": "kCrossValidationFixtures",
    },
    "component-cv": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "component_cv_fixtures.hpp",
        "fixtures": (
            "synthetic_component_cv_simpls_pls2_v1",
        ),
        "struct": "ComponentCvFixture",
        "array": "kComponentCvFixtures",
    },
    "power": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "power_fixtures.hpp",
        "fixtures": (
            "synthetic_power_tiny_pls1_v1",
            "synthetic_power_small_pls2_v1",
        ),
        "struct": "PowerFixture",
        "array": "kPowerFixtures",
    },
    "randomized-svd": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "randomized_svd_fixtures.hpp",
        "fixtures": (
            "synthetic_randomized_svd_tiny_pls1_v1",
            "synthetic_randomized_svd_small_pls2_v1",
        ),
        "struct": "RandomizedSvdFixture",
        "array": "kRandomizedSvdFixtures",
    },
    "canonical": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "canonical_fixtures.hpp",
        "fixtures": (
            "synthetic_canonical_tiny_pls1_v1",
            "synthetic_canonical_small_pls2_v1",
        ),
        "struct": "CanonicalFixture",
        "array": "kCanonicalFixtures",
    },
    "canonical-svd": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "canonical_svd_fixtures.hpp",
        "fixtures": (
            "synthetic_canonical_svd_tiny_pls1_v1",
            "synthetic_canonical_svd_small_pls2_v1",
        ),
        "struct": "CanonicalSvdFixture",
        "array": "kCanonicalSvdFixtures",
    },
    "pls-da": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "pls_da_fixtures.hpp",
        "fixtures": (
            "synthetic_pls_da_binary_v1",
            "synthetic_pls_da_multiclass_v1",
        ),
        "struct": "PlsDaFixture",
        "array": "kPlsDaFixtures",
    },
    "opls": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "opls_fixtures.hpp",
        "fixtures": (
            "synthetic_opls_tiny_pls1_v1",
            "synthetic_opls_small_pls1_v1",
            "synthetic_opls_small_pls2_v1",
        ),
        "struct": "OplsFixture",
        "array": "kOplsFixtures",
    },
    "opls-da": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "opls_da_fixtures.hpp",
        "fixtures": (
            "synthetic_opls_da_binary_v1",
            "synthetic_opls_da_multiclass_v1",
        ),
        "struct": "OplsDaFixture",
        "array": "kOplsDaFixtures",
    },
    "pcr": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "pcr_fixtures.hpp",
        "fixtures": (
            "synthetic_pcr_tiny_pls1_v1",
            "synthetic_pcr_small_pls2_v1",
        ),
        "struct": "PcrFixture",
        "array": "kPcrFixtures",
    },
    "kernel": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "kernel_fixtures.hpp",
        "fixtures": (
            "synthetic_kernel_tiny_pls1_v1",
            "synthetic_kernel_wide_pls2_v1",
        ),
        "struct": "KernelFixture",
        "array": "kKernelFixtures",
    },
    "wide-kernel": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "wide_kernel_fixtures.hpp",
        "fixtures": (
            "synthetic_wide_kernel_pls1_v1",
            "synthetic_wide_kernel_pls2_v1",
        ),
        "struct": "WideKernelFixture",
        "array": "kWideKernelFixtures",
    },
    "orthogonal-scores": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "orthogonal_scores_fixtures.hpp",
        "fixtures": (
            "synthetic_oscores_tiny_pls1_v1",
            "synthetic_oscores_small_pls2_v1",
        ),
        "struct": "OrthogonalScoresFixture",
        "array": "kOrthogonalScoresFixtures",
    },
}


def _symbol(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def _shape(shape: list[int]) -> tuple[int, int]:
    if len(shape) == 1:
        return 1, int(shape[0])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"unsupported 3-D fixture field shape: {shape}")


def _format_double(value: float) -> str:
    return f"{float(value):.17g}"


def _emit_array(lines: list[str], name: str, values: list[float]) -> None:
    lines.append(f"inline const double {name}[] = {{")
    row: list[str] = []
    for value in values:
        row.append(_format_double(value))
        if len(row) == 4:
            lines.append(f"    {', '.join(row)},")
            row.clear()
    if row:
        lines.append(f"    {', '.join(row)},")
    lines.append("};")
    lines.append("")


def _matrix_ref(array_name: str, payload: dict[str, Any]) -> str:
    rows, cols = _shape(payload["shape"])
    sign = "true" if payload.get("sign_invariant", False) else "false"
    return (
        f"MatrixRef{{{rows}, {cols}, {array_name}, "
        f"sizeof({array_name}) / sizeof(double), {sign}}}"
    )


def _load_fixture(fixture_id: str, fixture_dir: Path) -> dict[str, Any]:
    with (fixture_dir / f"{fixture_id}.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def generate(fixture_ids: Sequence[str],
             fixture_dir: Path,
             out: Path,
             family: str,
             struct_name: str,
             array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_components;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef coefficients;",
        "    MatrixRef intercept;",
        "    MatrixRef x_mean;",
        "    MatrixRef x_scale;",
        "    MatrixRef y_mean;",
        "    MatrixRef y_scale;",
        "    MatrixRef weights_w;",
        "    MatrixRef loadings_p;",
        "    MatrixRef y_loadings_q;",
        "    MatrixRef rotations_r;",
        "    MatrixRef scores_t;",
        "    MatrixRef predict_train;",
        "};",
        "",
    ]

    entries: list[str] = []
    fields = (
        ("X", ("data", "X"), "x"),
        ("Y", ("data", "Y"), "y"),
        ("coefficients", ("expected", "coefficients"), "coefficients"),
        ("intercept", ("expected", "intercept"), "intercept"),
        ("x_mean", ("expected", "x_mean"), "x_mean"),
        ("x_scale", ("expected", "x_scale"), "x_scale"),
        ("y_mean", ("expected", "y_mean"), "y_mean"),
        ("y_scale", ("expected", "y_scale"), "y_scale"),
        ("weights_W", ("expected", "weights_W"), "weights_w"),
        ("loadings_P", ("expected", "loadings_P"), "loadings_p"),
        ("y_loadings_Q", ("expected", "y_loadings_Q"), "y_loadings_q"),
        ("rotations_R", ("expected", "rotations_R"), "rotations_r"),
        ("scores_T", ("expected", "scores_T"), "scores_t"),
        ("predict_train", ("expected", "predict_train"), "predict_train"),
    )

    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        refs: dict[str, str] = {}
        for _public_name, path, suffix in fields:
            payload: dict[str, Any] = fixture
            for key in path:
                payload = payload[key]
            field_array_name = f"{prefix}_{suffix}"
            _emit_array(lines, field_array_name, payload["values"])
            refs[suffix] = _matrix_ref(field_array_name, payload)
        n_components = int(fixture["generator"]["params"]["n_components"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {n_components},\n"
            f"        {refs['x']},\n"
            f"        {refs['y']},\n"
            f"        {refs['coefficients']},\n"
            f"        {refs['intercept']},\n"
            f"        {refs['x_mean']},\n"
            f"        {refs['x_scale']},\n"
            f"        {refs['y_mean']},\n"
            f"        {refs['y_scale']},\n"
            f"        {refs['weights_w']},\n"
            f"        {refs['loadings_p']},\n"
            f"        {refs['y_loadings_q']},\n"
            f"        {refs['rotations_r']},\n"
            f"        {refs['scores_t']},\n"
            f"        {refs['predict_train']}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _pipeline_operator(name: str) -> tuple[str, list[float]]:
    mapping = {
        "identity": "P4A_OP_IDENTITY",
        "center": "P4A_OP_CENTER",
        "autoscale": "P4A_OP_AUTOSCALE",
        "pareto": "P4A_OP_PARETO_SCALE",
        "snv": "P4A_OP_SNV",
        "msc": "P4A_OP_MSC",
    }
    if name.startswith("detrend_poly_"):
        degree = float(int(name.rsplit("_", 1)[1]))
        return "P4A_OP_DETREND_POLY", [degree]
    if name.startswith("emsc_"):
        degree = float(int(name.rsplit("_", 1)[1]))
        return "P4A_OP_EMSC", [degree]
    if name.startswith("savgol_smooth_"):
        _prefix, _kind, window, poly_degree = name.split("_")
        return "P4A_OP_SAVGOL_SMOOTH", [float(window), float(poly_degree)]
    if name.startswith("savgol_derivative_"):
        _prefix, _kind, window, poly_degree, derivative_order, delta = name.split("_")
        return "P4A_OP_SAVGOL_DERIVATIVE", [
            float(window),
            float(poly_degree),
            float(derivative_order),
            float(delta),
        ]
    if name.startswith("asls_"):
        _prefix, lam, asymmetry, iterations = name.split("_")
        return "P4A_OP_ASLS_BASELINE", [
            float(lam),
            float(asymmetry),
            float(iterations),
        ]
    if name.startswith("norris_williams_"):
        _prefix, _kind, segment, gap, derivative_order = name.split("_")
        return "P4A_OP_NORRIS_WILLIAMS", [
            float(segment),
            float(gap),
            float(derivative_order),
        ]
    if name.startswith("wavelet_haar_"):
        _prefix, _kind, levels, threshold = name.split("_")
        return "P4A_OP_WAVELET_DENOISE", [
            float(levels),
            float(threshold),
        ]
    if name == "osc":
        return "P4A_OP_OSC", []
    if name.startswith("osc_"):
        _prefix, max_iter, tol = name.split("_")
        return "P4A_OP_OSC", [
            float(max_iter),
            float(tol),
        ]
    if name == "epo":
        return "P4A_OP_EPO", []
    if name.startswith("epo_"):
        _prefix, max_iter, tol = name.split("_")
        return "P4A_OP_EPO", [
            float(max_iter),
            float(tol),
        ]
    return mapping[name], []


def generate_pipeline(fixture_ids: Sequence[str],
                      fixture_dir: Path,
                      out: Path,
                      family: str,
                      struct_name: str,
                      array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "pls4all/p4a.h"',
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    const p4a_operator_kind_t* operators;",
        "    std::size_t n_operators;",
        "    const double* params;",
        "    const std::int32_t* n_params;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef transform_train;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        x_name = f"{prefix}_x"
        y_name = f"{prefix}_y"
        transformed_name = f"{prefix}_transform_train"
        ops_name = f"{prefix}_operators"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_array(lines, y_name, fixture["data"]["Y"]["values"])
        _emit_array(lines, transformed_name, fixture["expected"]["transform_train"]["values"])
        parsed_ops = [
            _pipeline_operator(op)
            for op in fixture["generator"]["params"]["operators"]
        ]
        operators = [symbol for symbol, _params in parsed_ops]
        flat_params = [value for _symbol_name, params in parsed_ops for value in params]
        n_params = [len(params) for _symbol_name, params in parsed_ops]
        params_name = f"{prefix}_params"
        n_params_name = f"{prefix}_n_params"
        lines.append(f"inline const p4a_operator_kind_t {ops_name}[] = {{")
        lines.append(f"    {', '.join(operators)},")
        lines.append("};")
        lines.append("")
        lines.append(f"inline const double {params_name}[] = {{")
        if flat_params:
            lines.append(f"    {', '.join(_format_double(v) for v in flat_params)},")
        else:
            lines.append("    0.0,")
        lines.append("};")
        lines.append("")
        lines.append(f"inline const std::int32_t {n_params_name}[] = {{")
        lines.append(f"    {', '.join(str(v) for v in n_params)},")
        lines.append("};")
        lines.append("")
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {ops_name},\n"
            f"        sizeof({ops_name}) / sizeof(p4a_operator_kind_t),\n"
            f"        {params_name},\n"
            f"        {n_params_name},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {_matrix_ref(y_name, fixture['data']['Y'])},\n"
            f"        {_matrix_ref(transformed_name, fixture['expected']['transform_train'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_metrics(fixture_ids: Sequence[str],
                     fixture_dir: Path,
                     out: Path,
                     family: str,
                     struct_name: str,
                     array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    MatrixRef observed;",
        "    MatrixRef predicted;",
        "    MatrixRef expected;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        observed_name = f"{prefix}_observed"
        predicted_name = f"{prefix}_predicted"
        expected_name = f"{prefix}_expected"
        _emit_array(lines, observed_name, fixture["data"]["Y_true"]["values"])
        _emit_array(lines, predicted_name, fixture["data"]["Y_pred"]["values"])
        _emit_array(lines, expected_name, fixture["expected"]["metrics"]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {_matrix_ref(observed_name, fixture['data']['Y_true'])},\n"
            f"        {_matrix_ref(predicted_name, fixture['data']['Y_pred'])},\n"
            f"        {_matrix_ref(expected_name, fixture['expected']['metrics'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _emit_int_array(lines: list[str], name: str, values: list[int]) -> None:
    lines.append(f"inline const std::int64_t {name}[] = {{")
    row: list[str] = []
    for value in values:
        row.append(str(int(value)))
        if len(row) == 8:
            lines.append(f"    {', '.join(row)},")
            row.clear()
    if row:
        lines.append(f"    {', '.join(row)},")
    lines.append("};")
    lines.append("")


def _emit_i32_array(lines: list[str], name: str, values: list[int]) -> None:
    lines.append(f"inline const std::int32_t {name}[] = {{")
    row: list[str] = []
    for value in values:
        row.append(str(int(value)))
        if len(row) == 8:
            lines.append(f"    {', '.join(row)},")
            row.clear()
    if row:
        lines.append(f"    {', '.join(row)},")
    lines.append("};")
    lines.append("")


def _index_ref(array_name: str) -> str:
    return f"IndexRef{{{array_name}, sizeof({array_name}) / sizeof(std::int64_t)}}"


def _cv_index_ref(array_name: str) -> str:
    return f"CvIndexRef{{{array_name}, sizeof({array_name}) / sizeof(std::int64_t)}}"


def _class_label_ref(array_name: str, payload: dict[str, Any]) -> str:
    rows, cols = _shape(payload["shape"])
    return (
        f"ClassLabelRef{{{rows}, {cols}, {array_name}, "
        f"sizeof({array_name}) / sizeof(std::int32_t)}}"
    )


def generate_validation(fixture_ids: Sequence[str],
                        fixture_dir: Path,
                        out: Path,
                        family: str,
                        struct_name: str,
                        array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct IndexRef {",
        "    const std::int64_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    const char* kind;",
        "    std::int64_t n_samples;",
        "    std::int32_t n_splits;",
        "    std::int64_t holdout_start;",
        "    std::int64_t holdout_count;",
        "    IndexRef train_offsets;",
        "    IndexRef train_indices;",
        "    IndexRef test_offsets;",
        "    IndexRef test_indices;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        expected = fixture["expected"]
        names = {
            "train_offsets": f"{prefix}_train_offsets",
            "train_indices": f"{prefix}_train_indices",
            "test_offsets": f"{prefix}_test_offsets",
            "test_indices": f"{prefix}_test_indices",
        }
        for field, name in names.items():
            _emit_int_array(lines, name, expected[field]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f'        "{params["kind"]}",\n'
            f"        {int(params['n_samples'])},\n"
            f"        {int(params['n_splits'])},\n"
            f"        {int(params['holdout_start'])},\n"
            f"        {int(params['holdout_count'])},\n"
            f"        {_index_ref(names['train_offsets'])},\n"
            f"        {_index_ref(names['train_indices'])},\n"
            f"        {_index_ref(names['test_offsets'])},\n"
            f"        {_index_ref(names['test_indices'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_advanced_validation(fixture_ids: Sequence[str],
                                 fixture_dir: Path,
                                 out: Path,
                                 family: str,
                                 struct_name: str,
                                 array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct IndexRef {",
        "    const std::int64_t* values;",
        "    std::size_t size;",
        "};",
        "",
        "struct MatrixRef {",
        "    std::int64_t rows;",
        "    std::int64_t cols;",
        "    const double* values;",
        "    std::size_t size;",
        "    bool sign_invariant;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    const char* kind;",
        "    std::int64_t n_samples;",
        "    std::int32_t n_splits;",
        "    std::int32_t n_repeats;",
        "    std::int64_t test_count;",
        "    std::int64_t train_count;",
        "    std::uint64_t seed;",
        "    IndexRef fold_ids;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    IndexRef train_offsets;",
        "    IndexRef train_indices;",
        "    IndexRef test_offsets;",
        "    IndexRef test_indices;",
        "};",
        "",
    ]

    entries: list[str] = []
    empty_index_ref = "IndexRef{nullptr, 0}"
    empty_matrix_ref = "MatrixRef{0, 0, nullptr, 0, false}"

    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        data = fixture["data"]
        expected = fixture["expected"]
        names = {
            "fold_ids": f"{prefix}_fold_ids",
            "X": f"{prefix}_X",
            "Y": f"{prefix}_Y",
            "train_offsets": f"{prefix}_train_offsets",
            "train_indices": f"{prefix}_train_indices",
            "test_offsets": f"{prefix}_test_offsets",
            "test_indices": f"{prefix}_test_indices",
        }
        fold_ref = empty_index_ref
        x_ref = empty_matrix_ref
        y_ref = empty_matrix_ref
        if "fold_ids" in data:
            _emit_int_array(lines, names["fold_ids"], data["fold_ids"]["values"])
            fold_ref = _index_ref(names["fold_ids"])
        if "X" in data:
            _emit_array(lines, names["X"], data["X"]["values"])
            x_ref = _matrix_ref(names["X"], data["X"])
        if "Y" in data:
            _emit_array(lines, names["Y"], data["Y"]["values"])
            y_ref = _matrix_ref(names["Y"], data["Y"])
        for field in ("train_offsets", "train_indices", "test_offsets", "test_indices"):
            _emit_int_array(lines, names[field], expected[field]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f'        "{params["kind"]}",\n'
            f"        {int(params['n_samples'])},\n"
            f"        {int(params['n_splits'])},\n"
            f"        {int(params['n_repeats'])},\n"
            f"        {int(params['test_count'])},\n"
            f"        {int(params['train_count'])},\n"
            f"        {int(params['seed'])}ULL,\n"
            f"        {fold_ref},\n"
            f"        {x_ref},\n"
            f"        {y_ref},\n"
            f"        {_index_ref(names['train_offsets'])},\n"
            f"        {_index_ref(names['train_indices'])},\n"
            f"        {_index_ref(names['test_offsets'])},\n"
            f"        {_index_ref(names['test_indices'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_classification_metrics(fixture_ids: Sequence[str],
                                    fixture_dir: Path,
                                    out: Path,
                                    family: str,
                                    struct_name: str,
                                    array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct ClassLabelRef {",
        "    std::int64_t rows;",
        "    std::int64_t cols;",
        "    const std::int32_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    double threshold;",
        "    ClassLabelRef labels;",
        "    MatrixRef scores;",
        "    MatrixRef expected;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        labels_name = f"{prefix}_labels"
        scores_name = f"{prefix}_scores"
        expected_name = f"{prefix}_expected"
        params = fixture["generator"]["params"]
        _emit_i32_array(lines, labels_name, fixture["data"]["labels"]["values"])
        _emit_array(lines, scores_name, fixture["data"]["scores"]["values"])
        _emit_array(lines, expected_name, fixture["expected"]["metrics"]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {_format_double(params['threshold'])},\n"
            f"        {_class_label_ref(labels_name, fixture['data']['labels'])},\n"
            f"        {_matrix_ref(scores_name, fixture['data']['scores'])},\n"
            f"        {_matrix_ref(expected_name, fixture['expected']['metrics'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_classification_extensions(fixture_ids: Sequence[str],
                                       fixture_dir: Path,
                                       out: Path,
                                       family: str,
                                       struct_name: str,
                                       array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct ClassificationExtensionLabelRef {",
        "    std::int64_t rows;",
        "    std::int64_t cols;",
        "    const std::int32_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    const char* kind;",
        "    std::int32_t n_classes;",
        "    std::int32_t n_bins;",
        "    ClassificationExtensionLabelRef labels;",
        "    MatrixRef scores;",
        "    MatrixRef summary_metrics;",
        "    MatrixRef per_class_metrics;",
        "    MatrixRef confusion_matrix;",
        "    MatrixRef calibration_curve;",
        "};",
        "",
    ]

    entries: list[str] = []
    empty_matrix_ref = "MatrixRef{0, 0, nullptr, 0, false}"
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        labels_name = f"{prefix}_labels"
        scores_name = f"{prefix}_scores"
        names = {
            "summary_metrics": f"{prefix}_summary_metrics",
            "per_class_metrics": f"{prefix}_per_class_metrics",
            "confusion_matrix": f"{prefix}_confusion_matrix",
            "calibration_curve": f"{prefix}_calibration_curve",
        }
        _emit_i32_array(lines, labels_name, fixture["data"]["labels"]["values"])
        _emit_array(lines, scores_name, fixture["data"]["scores"]["values"])
        refs = {field: empty_matrix_ref for field in names}
        for field, array_name_for_field in names.items():
            if field in fixture["expected"]:
                _emit_array(lines, array_name_for_field, fixture["expected"][field]["values"])
                refs[field] = _matrix_ref(array_name_for_field, fixture["expected"][field])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f'        "{params["kind"]}",\n'
            f"        {int(params['n_classes'])},\n"
            f"        {int(params['n_bins'])},\n"
            f"        {_class_label_ref(labels_name, fixture['data']['labels']).replace('ClassLabelRef', 'ClassificationExtensionLabelRef')},\n"
            f"        {_matrix_ref(scores_name, fixture['data']['scores'])},\n"
            f"        {refs['summary_metrics']},\n"
            f"        {refs['per_class_metrics']},\n"
            f"        {refs['confusion_matrix']},\n"
            f"        {refs['calibration_curve']}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_pls_lda(fixture_ids: Sequence[str],
                     fixture_dir: Path,
                     out: Path,
                     family: str,
                     struct_name: str,
                     array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct PlsLdaLabelRef {",
        "    std::int64_t rows;",
        "    std::int64_t cols;",
        "    const std::int32_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_classes;",
        "    std::int32_t n_components;",
        "    MatrixRef X;",
        "    PlsLdaLabelRef labels;",
        "    PlsLdaLabelRef predictions;",
        "    MatrixRef decision_scores;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        x_name = f"{prefix}_X"
        labels_name = f"{prefix}_labels"
        pred_name = f"{prefix}_predictions"
        scores_name = f"{prefix}_decision_scores"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_i32_array(lines, labels_name, fixture["data"]["labels"]["values"])
        _emit_i32_array(lines, pred_name, fixture["expected"]["predictions"]["values"])
        _emit_array(lines, scores_name, fixture["expected"]["decision_scores"]["values"])
        label_ref = _class_label_ref(labels_name, fixture["data"]["labels"]).replace("ClassLabelRef", "PlsLdaLabelRef")
        pred_ref = _class_label_ref(pred_name, fixture["expected"]["predictions"]).replace("ClassLabelRef", "PlsLdaLabelRef")
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {int(params['n_classes'])},\n"
            f"        {int(params['n_components'])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {label_ref},\n"
            f"        {pred_ref},\n"
            f"        {_matrix_ref(scores_name, fixture['expected']['decision_scores'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_pls_logistic(fixture_ids: Sequence[str],
                          fixture_dir: Path,
                          out: Path,
                          family: str,
                          struct_name: str,
                          array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct PlsLogisticLabelRef {",
        "    std::int64_t rows;",
        "    std::int64_t cols;",
        "    const std::int32_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_classes;",
        "    std::int32_t n_components;",
        "    MatrixRef X;",
        "    PlsLogisticLabelRef labels;",
        "    PlsLogisticLabelRef predictions;",
        "    MatrixRef decision_scores;",
        "    MatrixRef probabilities;",
        "    MatrixRef intercepts;",
        "    MatrixRef coefficients;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        x_name = f"{prefix}_X"
        labels_name = f"{prefix}_labels"
        pred_name = f"{prefix}_predictions"
        scores_name = f"{prefix}_decision_scores"
        probabilities_name = f"{prefix}_probabilities"
        intercepts_name = f"{prefix}_intercepts"
        coefficients_name = f"{prefix}_coefficients"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_i32_array(lines, labels_name, fixture["data"]["labels"]["values"])
        _emit_i32_array(lines, pred_name, fixture["expected"]["predictions"]["values"])
        _emit_array(lines, scores_name, fixture["expected"]["decision_scores"]["values"])
        _emit_array(lines, probabilities_name, fixture["expected"]["probabilities"]["values"])
        _emit_array(lines, intercepts_name, fixture["expected"]["intercepts"]["values"])
        _emit_array(lines, coefficients_name, fixture["expected"]["coefficients"]["values"])
        label_ref = _class_label_ref(labels_name, fixture["data"]["labels"]).replace("ClassLabelRef", "PlsLogisticLabelRef")
        pred_ref = _class_label_ref(pred_name, fixture["expected"]["predictions"]).replace("ClassLabelRef", "PlsLogisticLabelRef")
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {int(params['n_classes'])},\n"
            f"        {int(params['n_components'])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {label_ref},\n"
            f"        {pred_ref},\n"
            f"        {_matrix_ref(scores_name, fixture['expected']['decision_scores'])},\n"
            f"        {_matrix_ref(probabilities_name, fixture['expected']['probabilities'])},\n"
            f"        {_matrix_ref(intercepts_name, fixture['expected']['intercepts'])},\n"
            f"        {_matrix_ref(coefficients_name, fixture['expected']['coefficients'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_mb_pls(fixture_ids: Sequence[str],
                    fixture_dir: Path,
                    out: Path,
                    family: str,
                    struct_name: str,
                    array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct MbPlsBlockRef {",
        "    const std::int64_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_components;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MbPlsBlockRef block_sizes;",
        "    MatrixRef predictions;",
        "    MatrixRef coefficients;",
        "    MatrixRef intercept;",
        "    MatrixRef x_mean;",
        "    MatrixRef x_scale;",
        "    MatrixRef block_weights;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        expected = fixture["expected"]
        x_name = f"{prefix}_X"
        y_name = f"{prefix}_Y"
        block_sizes_name = f"{prefix}_block_sizes"
        pred_name = f"{prefix}_predictions"
        coefficients_name = f"{prefix}_coefficients"
        intercept_name = f"{prefix}_intercept"
        mean_name = f"{prefix}_x_mean"
        scale_name = f"{prefix}_x_scale"
        weights_name = f"{prefix}_block_weights"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_array(lines, y_name, fixture["data"]["Y"]["values"])
        _emit_int_array(lines, block_sizes_name, params["block_sizes"])
        _emit_array(lines, pred_name, expected["predictions"]["values"])
        _emit_array(lines, coefficients_name, expected["coefficients"]["values"])
        _emit_array(lines, intercept_name, expected["intercept"]["values"])
        _emit_array(lines, mean_name, expected["x_mean"]["values"])
        _emit_array(lines, scale_name, expected["x_scale"]["values"])
        _emit_array(lines, weights_name, expected["block_weights"]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {int(params['n_components'])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {_matrix_ref(y_name, fixture['data']['Y'])},\n"
            f"        MbPlsBlockRef{{{block_sizes_name}, sizeof({block_sizes_name}) / sizeof(std::int64_t)}},\n"
            f"        {_matrix_ref(pred_name, expected['predictions'])},\n"
            f"        {_matrix_ref(coefficients_name, expected['coefficients'])},\n"
            f"        {_matrix_ref(intercept_name, expected['intercept'])},\n"
            f"        {_matrix_ref(mean_name, expected['x_mean'])},\n"
            f"        {_matrix_ref(scale_name, expected['x_scale'])},\n"
            f"        {_matrix_ref(weights_name, expected['block_weights'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_lw_pls(fixture_ids: Sequence[str],
                    fixture_dir: Path,
                    out: Path,
                    family: str,
                    struct_name: str,
                    array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct LwPlsIndexRef {",
        "    std::int64_t rows;",
        "    std::int64_t cols;",
        "    const std::int64_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_components;",
        "    std::int32_t n_neighbors;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef predictions;",
        "    LwPlsIndexRef neighbor_indices;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        expected = fixture["expected"]
        x_name = f"{prefix}_X"
        y_name = f"{prefix}_Y"
        pred_name = f"{prefix}_predictions"
        neighbors_name = f"{prefix}_neighbor_indices"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_array(lines, y_name, fixture["data"]["Y"]["values"])
        _emit_array(lines, pred_name, expected["predictions"]["values"])
        _emit_int_array(lines, neighbors_name, expected["neighbor_indices"]["values"])
        neighbor_rows, neighbor_cols = _shape(expected["neighbor_indices"]["shape"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {int(params['n_components'])},\n"
            f"        {int(params['n_neighbors'])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {_matrix_ref(y_name, fixture['data']['Y'])},\n"
            f"        {_matrix_ref(pred_name, expected['predictions'])},\n"
            f"        LwPlsIndexRef{{{neighbor_rows}, {neighbor_cols}, {neighbors_name}, sizeof({neighbors_name}) / sizeof(std::int64_t)}}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_variable_importance(fixture_ids: Sequence[str],
                                 fixture_dir: Path,
                                 out: Path,
                                 family: str,
                                 struct_name: str,
                                 array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_components;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef vip;",
        "    MatrixRef selectivity_ratio;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        x_name = f"{prefix}_X"
        y_name = f"{prefix}_Y"
        vip_name = f"{prefix}_vip"
        sr_name = f"{prefix}_selectivity_ratio"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_array(lines, y_name, fixture["data"]["Y"]["values"])
        _emit_array(lines, vip_name, fixture["expected"]["vip"]["values"])
        _emit_array(lines, sr_name, fixture["expected"]["selectivity_ratio"]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {int(params['n_components'])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {_matrix_ref(y_name, fixture['data']['Y'])},\n"
            f"        {_matrix_ref(vip_name, fixture['expected']['vip'])},\n"
            f"        {_matrix_ref(sr_name, fixture['expected']['selectivity_ratio'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_component_coefficients(fixture_ids: Sequence[str],
                                    fixture_dir: Path,
                                    out: Path,
                                    family: str,
                                    struct_name: str,
                                    array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_components;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef coefficients_by_component;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        x_name = f"{prefix}_X"
        y_name = f"{prefix}_Y"
        coef_name = f"{prefix}_coefficients_by_component"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_array(lines, y_name, fixture["data"]["Y"]["values"])
        _emit_array(lines, coef_name, fixture["expected"]["coefficients_by_component"]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {int(params['n_components'])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {_matrix_ref(y_name, fixture['data']['Y'])},\n"
            f"        {_matrix_ref(coef_name, fixture['expected']['coefficients_by_component'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_cross_validation(fixture_ids: Sequence[str],
                              fixture_dir: Path,
                              out: Path,
                              family: str,
                              struct_name: str,
                              array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        "struct CvIndexRef {",
        "    const std::int64_t* values;",
        "    std::size_t size;",
        "};",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_components;",
        "    std::int32_t n_splits;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef predictions;",
        "    MatrixRef metrics;",
        "    CvIndexRef test_offsets;",
        "    CvIndexRef test_indices;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        expected = fixture["expected"]
        x_name = f"{prefix}_X"
        y_name = f"{prefix}_Y"
        pred_name = f"{prefix}_predictions"
        metrics_name = f"{prefix}_metrics"
        offsets_name = f"{prefix}_test_offsets"
        indices_name = f"{prefix}_test_indices"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_array(lines, y_name, fixture["data"]["Y"]["values"])
        _emit_array(lines, pred_name, expected["predictions"]["values"])
        _emit_array(lines, metrics_name, expected["metrics"]["values"])
        _emit_int_array(lines, offsets_name, expected["test_offsets"]["values"])
        _emit_int_array(lines, indices_name, expected["test_indices"]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {int(params['n_components'])},\n"
            f"        {int(params['n_splits'])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {_matrix_ref(y_name, fixture['data']['Y'])},\n"
            f"        {_matrix_ref(pred_name, expected['predictions'])},\n"
            f"        {_matrix_ref(metrics_name, expected['metrics'])},\n"
            f"        {_cv_index_ref(offsets_name)},\n"
            f"        {_cv_index_ref(indices_name)}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_component_cv(fixture_ids: Sequence[str],
                          fixture_dir: Path,
                          out: Path,
                          family: str,
                          struct_name: str,
                          array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    const char* solver;",
        "    std::int32_t n_components_max;",
        "    std::int32_t n_splits;",
        "    std::int32_t best_n_components;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef metrics_by_component;",
        "};",
        "",
    ]

    entries: list[str] = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        params = fixture["generator"]["params"]
        expected = fixture["expected"]
        x_name = f"{prefix}_X"
        y_name = f"{prefix}_Y"
        metrics_name = f"{prefix}_metrics_by_component"
        _emit_array(lines, x_name, fixture["data"]["X"]["values"])
        _emit_array(lines, y_name, fixture["data"]["Y"]["values"])
        _emit_array(lines, metrics_name, expected["metrics_by_component"]["values"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f'        "{params["solver"]}",\n'
            f"        {int(params['n_components_max'])},\n"
            f"        {int(params['n_splits'])},\n"
            f"        {int(expected['best_n_components']['values'][0])},\n"
            f"        {_matrix_ref(x_name, fixture['data']['X'])},\n"
            f"        {_matrix_ref(y_name, fixture['data']['Y'])},\n"
            f"        {_matrix_ref(metrics_name, expected['metrics_by_component'])}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(HEADER_SPECS), default="simpls",
                        help="Fixture family to emit when fixture ids are omitted.")
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("fixture_ids", nargs="*")
    args = parser.parse_args(argv)
    spec = HEADER_SPECS[args.family]
    out = args.out or spec["out"]
    fixture_ids = args.fixture_ids or list(spec["fixtures"])
    if args.family == "pipeline":
        generate_pipeline(fixture_ids, args.fixture_dir, out, args.family,
                          str(spec["struct"]), str(spec["array"]))
    elif args.family == "metrics":
        generate_metrics(fixture_ids, args.fixture_dir, out, args.family,
                         str(spec["struct"]), str(spec["array"]))
    elif args.family == "classification-metrics":
        generate_classification_metrics(fixture_ids, args.fixture_dir, out, args.family,
                                        str(spec["struct"]), str(spec["array"]))
    elif args.family == "classification-extensions":
        generate_classification_extensions(fixture_ids, args.fixture_dir, out, args.family,
                                           str(spec["struct"]), str(spec["array"]))
    elif args.family == "pls-lda":
        generate_pls_lda(fixture_ids, args.fixture_dir, out, args.family,
                         str(spec["struct"]), str(spec["array"]))
    elif args.family == "pls-logistic":
        generate_pls_logistic(fixture_ids, args.fixture_dir, out, args.family,
                              str(spec["struct"]), str(spec["array"]))
    elif args.family == "mb-pls":
        generate_mb_pls(fixture_ids, args.fixture_dir, out, args.family,
                        str(spec["struct"]), str(spec["array"]))
    elif args.family == "lw-pls":
        generate_lw_pls(fixture_ids, args.fixture_dir, out, args.family,
                        str(spec["struct"]), str(spec["array"]))
    elif args.family == "variable-importance":
        generate_variable_importance(fixture_ids, args.fixture_dir, out, args.family,
                                     str(spec["struct"]), str(spec["array"]))
    elif args.family == "component-coefficients":
        generate_component_coefficients(fixture_ids, args.fixture_dir, out, args.family,
                                        str(spec["struct"]), str(spec["array"]))
    elif args.family == "validation":
        generate_validation(fixture_ids, args.fixture_dir, out, args.family,
                            str(spec["struct"]), str(spec["array"]))
    elif args.family == "advanced-validation":
        generate_advanced_validation(fixture_ids, args.fixture_dir, out, args.family,
                                     str(spec["struct"]), str(spec["array"]))
    elif args.family == "cross-validation":
        generate_cross_validation(fixture_ids, args.fixture_dir, out, args.family,
                                  str(spec["struct"]), str(spec["array"]))
    elif args.family == "component-cv":
        generate_component_cv(fixture_ids, args.fixture_dir, out, args.family,
                              str(spec["struct"]), str(spec["array"]))
    else:
        generate(fixture_ids, args.fixture_dir, out, args.family,
                 str(spec["struct"]), str(spec["array"]))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
