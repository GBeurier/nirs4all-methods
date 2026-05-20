"""Export the chemometrics4all validation registry to deterministic JSON.

The exporter intentionally does not import
``benchmarks.cross_binding.orchestrator``. That module loads numpy,
ctypes, bindings, and local shared libraries at import time. Here we
only need its declarative surface: method ids, reference declarations,
dataset sizes, comparators, and the seed policy. Those are extracted
with ``ast`` and enriched with ``benchmarks/benchmark_registry.json`` /
``benchmarks/truth_sources.lock.json``.

Usage
-----

    python -m benchmarks.validation.export_registry
    python -m benchmarks.validation.export_registry --write
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from .models import (
    SCHEMA_VERSION,
    ComparatorRecord,
    DatasetRecord,
    Manifest,
    MethodRecord,
    ReferenceLink,
    ReferenceRecord,
    SuiteRecord,
    ToleranceRecord,
)

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = ROOT / "benchmarks" / "cross_binding" / "orchestrator.py"
BENCHMARK_REGISTRY_PATH = ROOT / "benchmarks" / "benchmark_registry.json"
TRUTH_SOURCES_LOCK_PATH = ROOT / "benchmarks" / "truth_sources.lock.json"

REGISTRY_DIR = Path(__file__).resolve().parent / "registry"
METHODS_PATH = REGISTRY_DIR / "methods.json"
REFERENCES_PATH = REGISTRY_DIR / "references.json"
DATASETS_PATH = REGISTRY_DIR / "datasets.json"
COMPARATORS_PATH = REGISTRY_DIR / "comparators.json"
TOLERANCES_PATH = REGISTRY_DIR / "tolerances.json"
SUITES_PATH = REGISTRY_DIR / "suites.json"
MANIFEST_PATH = REGISTRY_DIR / "manifest.json"

REGISTRY_FILES: tuple[Path, ...] = (
    METHODS_PATH,
    REFERENCES_PATH,
    DATASETS_PATH,
    COMPARATORS_PATH,
    TOLERANCES_PATH,
    SUITES_PATH,
    MANIFEST_PATH,
)

REF_BUILDERS: dict[str, dict[str, Any]] = {
    "nirs_ref": {
        "backend": "ref.nirs4all",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "numpy_ref": {
        "backend": "ref.numpy",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "sklearn_ref": {
        "backend": "ref.sklearn",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "scipy_ref": {
        "backend": "ref.scipy",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "statsmodels_ref": {
        "backend": "ref.statsmodels",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "cowarp_ref": {
        "backend": "ref.cowarp",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "pywavelets_ref": {
        "backend": "ref.pywavelets",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "pycaltransfer_ref": {
        "backend": "ref.pycaltransfer",
        "language": "Python",
        "compare": True,
        "gate_c4a": True,
    },
    "r_prospectr_ref": {
        "backend": "ref.r.prospectr",
        "language": "R",
        "compare": False,
        "gate_c4a": False,
    },
}

COMPARATOR_IDS: dict[str, str] = {
    "outputs_close": "default_allclose",
    "outputs_close_iasls": "iasls_allclose",
    "outputs_close_sign_invariant_columns": "sign_invariant_columns",
    "outputs_close_savgol_valid_window": "savgol_valid_window",
}


def _parse_orchestrator() -> ast.Module:
    return ast.parse(ORCHESTRATOR_PATH.read_text(encoding="utf-8"))


def _literal(node: ast.AST | None, default: Any = None) -> Any:
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return default


def _expr(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _call_name(node: ast.AST | None) -> str:
    if not isinstance(node, ast.Call):
        return ""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _bool_keyword(call: ast.Call, name: str, default: bool) -> bool:
    return bool(_literal(_keyword(call, name), default))


def _str_keyword(call: ast.Call, name: str, default: str = "") -> str:
    value = _literal(_keyword(call, name), default)
    return "" if value is None else str(value)


def _int_keyword(call: ast.Call, name: str) -> int | None:
    value = _literal(_keyword(call, name), None)
    return None if value is None else int(value)


def _comparator_id(node: ast.AST | None) -> str:
    if node is None:
        return "default_allclose"
    if isinstance(node, ast.Constant) and node.value is None:
        return "default_allclose"
    if isinstance(node, ast.Name):
        return COMPARATOR_IDS.get(node.id, node.id)
    return COMPARATOR_IDS.get(_expr(node), _expr(node) or "default_allclose")


def _entry_label(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and node.value is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name == "py_transform" and node.args:
            return _expr(node.args[0])
        if name in {"c_same", "c_stateful"} and node.args:
            first = _literal(node.args[0], None)
            if first is not None:
                return str(first)
    return _expr(node)


def _find_assignment(tree: ast.Module, name: str) -> ast.AST | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node.value
    return None


def _extract_sizes(tree: ast.Module) -> tuple[tuple[int, int], ...]:
    value = _literal(_find_assignment(tree, "PLS4ALL_BENCHMARK_SIZES"), [])
    return tuple((int(n), int(p)) for n, p in value)


def _extract_smoke_sizes(
    tree: ast.Module,
    sizes: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    value = _literal(_find_assignment(tree, "PLS4ALL_SMALL_SIZES"), None)
    if value is not None:
        return tuple((int(n), int(p)) for n, p in value)
    return sizes[:3]


def _extract_seed_base(tree: ast.Module) -> int:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not node.args or _literal(node.args[0]) != "--seed":
            continue
        value = _literal(_keyword(node, "default"), None)
        if value is not None:
            return int(value)
    return 20260519


def _method_calls(tree: ast.Module) -> list[ast.Call]:
    value = _find_assignment(tree, "METHODS")
    if not isinstance(value, ast.Tuple):
        return []
    return [
        elt for elt in value.elts
        if isinstance(elt, ast.Call) and _call_name(elt) == "MethodSpec"
    ]


def _reference_calls(node: ast.AST | None) -> list[ast.Call]:
    if node is None:
        return []
    if isinstance(node, (ast.Tuple, ast.List)):
        return [
            elt for elt in node.elts
            if isinstance(elt, ast.Call)
        ]
    if isinstance(node, ast.Call):
        return [node]
    return []


def _parse_reference(call: ast.Call, method_comparator: str) -> ReferenceLink:
    name = _call_name(call)
    if name == "ReferenceSpec":
        backend = str(_literal(call.args[0], "ref.unknown")) if call.args else "ref.unknown"
        library = str(_literal(call.args[1], backend)) if len(call.args) > 1 else backend
        language = _str_keyword(call, "language", "Python")
        compare = _bool_keyword(call, "compare", True)
        gate = _bool_keyword(call, "gate_c4a", True)
        contract_note = _str_keyword(call, "contract_note")
        max_cols = _int_keyword(call, "max_cols")
        comparator = _comparator_id(_keyword(call, "comparator"))
        if comparator == "default_allclose":
            comparator = method_comparator
        return ReferenceLink(
            backend=backend,
            library=library,
            language=language,
            compare=compare,
            gate_c4a=gate,
            contract_note=contract_note,
            max_cols=max_cols,
            comparator=comparator,
            r_expr=_str_keyword(call, "r_expr"),
        )

    spec = REF_BUILDERS.get(name, {})
    backend = spec.get("backend", f"ref.{name or 'unknown'}")
    language = spec.get("language", "Python")
    library = str(_literal(call.args[0], backend)) if call.args else backend
    r_expr = ""
    if name == "r_prospectr_ref" and len(call.args) > 1:
        r_expr = str(_literal(call.args[1], ""))
    compare = _bool_keyword(call, "compare", bool(spec.get("compare", True)))
    gate = _bool_keyword(call, "gate_c4a", bool(spec.get("gate_c4a", True)))
    comparator = _comparator_id(_keyword(call, "comparator"))
    if comparator == "default_allclose":
        comparator = method_comparator
    return ReferenceLink(
        backend=str(backend),
        library=library,
        language=str(language),
        compare=compare,
        gate_c4a=gate,
        contract_note=_str_keyword(call, "contract_note"),
        max_cols=_int_keyword(call, "max_cols"),
        comparator=comparator,
        r_expr=r_expr,
    )


def _load_benchmark_registry() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        data = json.loads(BENCHMARK_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    methods = {
        str(record.get("method_id")): record
        for record in data.get("methods", [])
        if isinstance(record, dict) and record.get("method_id")
    }
    return methods, data


def _fallback_dataset(seed_base: int) -> dict[str, Any]:
    return {
        "n_samples": 100,
        "n_features": 50,
        "seed": seed_base + 37,
    }


def _quality(comparator_id: str) -> str:
    if comparator_id == "default_allclose":
        return "strict"
    if comparator_id == "iasls_allclose":
        return "relaxed-atol"
    if comparator_id in {"sign_invariant_columns", "savgol_valid_window"}:
        return "contract-specific"
    return "custom"


def build_methods(
    tree: ast.Module | None = None,
    benchmark_methods: dict[str, dict[str, Any]] | None = None,
) -> list[MethodRecord]:
    tree = tree or _parse_orchestrator()
    benchmark_methods = benchmark_methods if benchmark_methods is not None else _load_benchmark_registry()[0]
    seed_base = _extract_seed_base(tree)
    records: list[MethodRecord] = []

    for call in _method_calls(tree):
        method_id = str(_literal(call.args[0], "")) if call.args else ""
        if not method_id:
            continue
        label = str(_literal(call.args[1], method_id)) if len(call.args) > 1 else method_id
        family = str(_literal(call.args[2], "other")) if len(call.args) > 2 else "other"
        py_entry = _entry_label(call.args[3]) if len(call.args) > 3 else ""
        cpp_entry = _entry_label(call.args[4]) if len(call.args) > 4 else ""
        r_expr = str(_literal(call.args[5], "") or "") if len(call.args) > 5 else ""
        refs_node = call.args[6] if len(call.args) > 6 else _keyword(call, "references")
        compare_internal = _bool_keyword(call, "compare_internal", True)
        max_cols = _int_keyword(call, "max_cols")
        method_comparator = _comparator_id(_keyword(call, "comparator"))
        references = [
            _parse_reference(ref_call, method_comparator)
            for ref_call in _reference_calls(refs_node)
        ]
        registry_row = benchmark_methods.get(method_id, {})
        records.append(MethodRecord(
            method_id=method_id,
            label=str(registry_row.get("label") or label),
            family=str(registry_row.get("family") or family),
            py_class=py_entry,
            c_prefix=cpp_entry,
            r_expr=r_expr,
            max_cols=max_cols,
            compare_internal=compare_internal,
            comparator=method_comparator,
            tolerance_quality=_quality(method_comparator),
            operation=str(registry_row.get("operation") or "cross_binding_callable"),
            truth_sources=[
                str(source) for source in registry_row.get("truth_sources", [])
            ],
            metrics=[
                str(metric) for metric in registry_row.get(
                    "metrics",
                    ["max_abs_diff", "rms_diff", "rel_l2_diff", "shape_equal"],
                )
            ],
            default_dataset=dict(registry_row.get("default_dataset") or _fallback_dataset(seed_base)),
            references=references,
        ))

    records.sort(key=lambda item: item.method_id)
    return records


def build_references(methods: list[MethodRecord]) -> list[ReferenceRecord]:
    by_backend: dict[str, dict[str, Any]] = {}
    for method in methods:
        for ref in method.references:
            bucket = by_backend.setdefault(
                ref.backend,
                {"backend": ref.backend, "language": ref.language, "methods": set()},
            )
            bucket["methods"].add(method.method_id)
    return [
        ReferenceRecord(
            backend=record["backend"],
            language=record["language"],
            methods=sorted(record["methods"]),
        )
        for record in sorted(by_backend.values(), key=lambda row: row["backend"])
    ]


def build_datasets(tree: ast.Module | None = None) -> list[DatasetRecord]:
    tree = tree or _parse_orchestrator()
    sizes = _extract_sizes(tree)
    smoke = set(_extract_smoke_sizes(tree, sizes))
    seed_base = _extract_seed_base(tree)
    records = [
        DatasetRecord(
            id=f"{n}x{p}",
            n=n,
            p=p,
            seed_base=seed_base,
            in_smoke_preset=(n, p) in smoke,
        )
        for n, p in sizes
    ]
    records.sort(key=lambda item: (item.n, item.p))
    return records


def build_comparators() -> list[ComparatorRecord]:
    return [
        ComparatorRecord(
            id="default_allclose",
            name="default allclose gate",
            description=(
                "Flattened-output shape check followed by numpy.allclose "
                "with equal_nan=True."
            ),
            metric="allclose(max_abs_diff, rtol, atol)",
            default_rtol=1e-5,
            default_atol=1e-8,
            contract_note="Default C/Python/reference parity comparator.",
        ),
        ComparatorRecord(
            id="iasls_allclose",
            name="IAsLS relaxed absolute-tolerance gate",
            description=(
                "Same flattened allclose gate as default_allclose, with "
                "a higher absolute tolerance for iterative IAsLS baselines."
            ),
            metric="allclose(max_abs_diff, rtol, atol)",
            default_rtol=1e-5,
            default_atol=5e-6,
            contract_note="Used by iasls only.",
        ),
        ComparatorRecord(
            id="sign_invariant_columns",
            name="sign-invariant component gate",
            description=(
                "Column signs are aligned before allclose, matching PCA/SVD "
                "component sign indeterminacy."
            ),
            metric="sign-aligned allclose",
            default_rtol=1e-5,
            default_atol=1e-8,
            contract_note="Used by component projection methods.",
        ),
        ComparatorRecord(
            id="savgol_valid_window",
            name="Savitzky-Golay valid-window gate",
            description=(
                "Compares the shared interior window when a reference drops "
                "boundary columns."
            ),
            metric="valid-window allclose",
            default_rtol=1e-5,
            default_atol=1e-8,
            contract_note="Used for prospectr Savitzky-Golay parity.",
        ),
    ]


def build_tolerances(methods: list[MethodRecord]) -> list[ToleranceRecord]:
    by_comparator = {record.id: record for record in build_comparators()}
    records: list[ToleranceRecord] = []
    for method in methods:
        comparator = by_comparator.get(method.comparator, by_comparator["default_allclose"])
        records.append(ToleranceRecord(
            method=method.method_id,
            comparator=comparator.id,
            binding_tolerance=comparator.default_rtol,
            reference_rtol=comparator.default_rtol,
            reference_atol=comparator.default_atol,
            quality=method.tolerance_quality,
        ))
    records.sort(key=lambda item: item.method)
    return records


def build_suites(
    methods: list[MethodRecord],
    datasets: list[DatasetRecord],
) -> list[SuiteRecord]:
    method_ids = [method.method_id for method in methods]
    smoke_datasets = [dataset.id for dataset in datasets if dataset.in_smoke_preset]
    all_datasets = [dataset.id for dataset in datasets]
    comparators = [record.id for record in build_comparators()]
    return [
        SuiteRecord(
            id="smoke",
            description=(
                "Fast deterministic cross-binding validation subset: all "
                "methods against PLS4ALL_SMALL_SIZES."
            ),
            methods=method_ids,
            datasets=smoke_datasets,
            comparators=comparators,
        ),
        SuiteRecord(
            id="benchmark",
            description=(
                "Full deterministic cross-binding benchmark sweep: all "
                "methods against PLS4ALL_BENCHMARK_SIZES."
            ),
            methods=method_ids,
            datasets=all_datasets,
            comparators=comparators,
        ),
    ]


def build_manifest(counts: dict[str, int]) -> Manifest:
    return Manifest(
        files=[path.name for path in REGISTRY_FILES if path.name != "manifest.json"],
        counts=dict(counts),
    )


def build_payload() -> dict[str, Any]:
    tree = _parse_orchestrator()
    benchmark_methods, _registry = _load_benchmark_registry()
    methods = build_methods(tree, benchmark_methods)
    references = build_references(methods)
    datasets = build_datasets(tree)
    comparators = build_comparators()
    tolerances = build_tolerances(methods)
    suites = build_suites(methods, datasets)
    counts = {
        "methods": len(methods),
        "references": len(references),
        "datasets": len(datasets),
        "comparators": len(comparators),
        "tolerances": len(tolerances),
        "suites": len(suites),
    }
    manifest = build_manifest(counts)
    return {
        METHODS_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "methods": [record.to_dict() for record in methods],
        },
        REFERENCES_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "references": [record.to_dict() for record in references],
        },
        DATASETS_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "datasets": [record.to_dict() for record in datasets],
        },
        COMPARATORS_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "comparators": [record.to_dict() for record in comparators],
        },
        TOLERANCES_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "tolerances": [record.to_dict() for record in tolerances],
        },
        SUITES_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "suites": [record.to_dict() for record in suites],
        },
        MANIFEST_PATH.name: manifest.to_dict(),
    }


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_payload(
    payload: dict[str, Any],
    registry_dir: Path = REGISTRY_DIR,
) -> list[Path]:
    registry_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, data in sorted(payload.items()):
        path = registry_dir / name
        path.write_text(dumps(data), encoding="utf-8")
        written.append(path)
    return written


def _cmd_write() -> int:
    written = write_payload(build_payload())
    for path in written:
        print(f"wrote {path}")
    return 0


def _cmd_print() -> int:
    payload = build_payload()
    manifest = payload[MANIFEST_PATH.name]
    counts = manifest.get("counts", {})
    print("validation registry payload (in-memory)")
    print(f"  schema_version: {manifest.get('schema_version')}")
    print(f"  source: {manifest.get('source')}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh benchmarks/validation/registry/*.json.",
    )
    args = parser.parse_args(argv)
    if args.write:
        return _cmd_write()
    return _cmd_print()


if __name__ == "__main__":
    sys.exit(main())
