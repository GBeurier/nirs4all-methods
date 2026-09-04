#!/usr/bin/env python3
"""Generate the evidence-bound V1 capability matrix.

The matrix is a statement about code evidence only.  A native row requires a
public-header declaration and membership in *every* shipped ABI snapshot; an
adapter row additionally names the generated ctypes declarations it relies on.
The generator fails closed instead of emitting an "unverified" capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = ROOT / "docs" / "architecture"
OUTPUT_JSON = ARCHITECTURE / "v1_estimator_stateful_preprocessor_matrix.json"
OUTPUT_DOC = ARCHITECTURE / "v1_estimator_stateful_preprocessor_matrix.md"
SCHEMA_PATH = ARCHITECTURE / "v1_estimator_stateful_preprocessor_matrix.schema.json"
HEADERS = ROOT / "cpp" / "include" / "n4m"
CATALOG = ROOT / "catalog" / "methods"
PYTHON_FFI = ROOT / "bindings" / "python" / "src" / "n4m" / "_ffi_decls.py"
PYTHON_NATIVE = ROOT / "bindings" / "python" / "src" / "n4m" / "_impl" / "native.py"
PYTHON_MOMENT_STACK = (
    ROOT / "bindings" / "python" / "src" / "n4m" / "_impl" / "native_sweeps.py"
)
POLICY = ROOT / "CONTRIBUTING.md"
ABI_SNAPSHOTS = {
    platform: ROOT / "cpp" / "abi" / f"expected_symbols_{platform}.txt"
    for platform in ("linux", "macos", "windows")
}

PUBLIC_SYMBOL_RE = re.compile(
    r"\bN4M_API\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)+(?P<symbol>n4m_[a-z0-9_]+)\s*\(",
    re.MULTILINE,
)
FFI_SYMBOL_RE = re.compile(r'\(\s*"(?P<symbol>n4m_[a-z0-9_]+)"\s*,')
ALGORITHM_RE = re.compile(r"\b(N4M_ALGO_[A-Z0-9_]+)\s*=")
STATEFUL_ACTIONS = (
    "inverse_transform",
    "transform_with_d",
    "output_cols",
    "is_fitted",
    "transform",
    "threshold",
    "create",
    "destroy",
    "apply",
    "fit",
)
STATEFUL_HEADER_GLOBS = (
    "cpp/include/n4m/transform/*.h",
    "cpp/include/n4m/domain_adaptation.h",
    "cpp/include/n4m/decomposition.h",
    "cpp/include/n4m/feature_selection.h",
    "cpp/include/n4m/outlier_detection.h",
)
CLASSIFICATIONS = ("native", "adapter", "refused", "not-promised")


class MatrixError(RuntimeError):
    """Raised when the matrix has incomplete or contradictory evidence."""


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _header_symbols(paths: Iterable[Path]) -> dict[str, set[str]]:
    return {
        _relative(path): set(PUBLIC_SYMBOL_RE.findall(path.read_text(encoding="utf-8")))
        for path in sorted(set(paths))
    }


def _header_index(header_symbols: dict[str, set[str]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for header, symbols in header_symbols.items():
        for symbol in symbols:
            index[symbol].append(header)
    return {symbol: sorted(paths) for symbol, paths in index.items()}


def _snapshot_symbols() -> dict[str, set[str]]:
    return {
        platform: {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("n4m_")
        }
        for platform, path in ABI_SNAPSHOTS.items()
    }


def _input_files() -> list[Path]:
    """Return every true generation input in deterministic order.

    Deliberately no VCS metadata is included: changing a commit alone cannot
    change this list or the rendered artifacts.
    """
    paths = [*HEADERS.rglob("*.h"), *CATALOG.glob("*.yaml")]
    paths.extend(ABI_SNAPSHOTS.values())
    paths.extend((PYTHON_FFI, PYTHON_NATIVE, PYTHON_MOMENT_STACK, POLICY))
    return sorted(set(paths), key=_relative)


def _input_digest() -> dict[str, Any]:
    records = []
    digest = hashlib.sha256()
    for path in _input_files():
        content_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = _relative(path)
        records.append({"path": relative, "sha256": content_digest})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest.encode("ascii"))
        digest.update(b"\n")
    return {"algorithm": "sha256", "files": records, "sha256": digest.hexdigest()}


def _catalog_models() -> list[dict[str, Any]]:
    entries = []
    for path in sorted(CATALOG.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("category") != "models":
            continue
        if not isinstance(data.get("method_id"), str):
            raise MatrixError(f"catalog model lacks method_id: {_relative(path)}")
        symbols = data.get("abi_symbols", [])
        if not isinstance(symbols, list) or not all(
            isinstance(s, str) for s in symbols
        ):
            raise MatrixError(
                f"catalog model has invalid abi_symbols: {_relative(path)}"
            )
        entries.append(
            {
                "catalog_file": _relative(path),
                "method_id": data["method_id"],
                "abi_symbols": sorted(symbols),
                "bindings": data.get("bindings", {}),
            }
        )
    if len({entry["method_id"] for entry in entries}) != len(entries):
        raise MatrixError("catalog model IDs are not unique")
    return sorted(entries, key=lambda entry: entry["method_id"])


def _catalog_method_ids() -> list[str]:
    """Return the full catalog universe for boundary-candidate discovery."""
    identifiers = []
    for path in sorted(CATALOG.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        identifier = data.get("method_id")
        if isinstance(identifier, str):
            identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise MatrixError("catalog method IDs are not unique")
    return sorted(identifiers)


def _model_header_symbols(header_index: dict[str, list[str]]) -> set[str]:
    return {
        symbol
        for symbol in header_index
        if re.fullmatch(r"n4m_estimators_[a-z0-9_]+_(?:fit|run)", symbol)
        or re.fullmatch(
            r"n4m_ensemble_(?:bagging_pls|boosting_pls|random_subspace_pls)_fit", symbol
        )
        or re.fullmatch(r"n4m_domain_adaptation_(?:di_pls|ds|pds)_fit", symbol)
    }


def _native_row(
    identifier: str,
    kind: str,
    symbols: Iterable[str],
    header_index: dict[str, list[str]],
    snapshots: dict[str, set[str]],
    **extra: Any,
) -> dict[str, Any]:
    api_symbols = sorted(set(symbols))
    missing_headers = [symbol for symbol in api_symbols if symbol not in header_index]
    missing_abi = {
        platform: sorted(set(api_symbols) - available)
        for platform, available in snapshots.items()
    }
    if missing_headers or any(missing_abi.values()):
        details = []
        if missing_headers:
            details.append(f"not declared: {missing_headers}")
        for platform, missing in missing_abi.items():
            if missing:
                details.append(f"missing {platform} ABI: {missing}")
        raise MatrixError(
            f"native row {identifier} has no complete ABI proof ({'; '.join(details)})"
        )
    return {
        "id": identifier,
        "kind": kind,
        "classification": "native",
        "api_symbols": api_symbols,
        "source_headers": sorted(
            {h for symbol in api_symbols for h in header_index[symbol]}
        ),
        "abi_proof": {
            platform: _relative(path) for platform, path in ABI_SNAPSHOTS.items()
        },
        **extra,
    }


def _stateful_prefix(symbol: str) -> tuple[str, str] | None:
    """Split a lifecycle suffix without the old greedy-regex ambiguity."""
    for action in STATEFUL_ACTIONS:  # longest actions intentionally precede transform
        suffix = f"_{action}"
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)], action
    return None


def _stateful_rows(
    header_symbols: dict[str, set[str]],
    header_index: dict[str, list[str]],
    snapshots: dict[str, set[str]],
) -> list[dict[str, Any]]:
    operations: dict[str, dict[str, str]] = defaultdict(dict)
    for symbol in sorted({s for symbols in header_symbols.values() for s in symbols}):
        parsed = _stateful_prefix(symbol)
        if parsed is not None:
            prefix, action = parsed
            operations[prefix][action] = symbol

    rows = []
    all_stateful_symbols = {s for symbols in header_symbols.values() for s in symbols}
    for prefix, actions in sorted(operations.items()):
        if not {"create", "destroy", "fit"}.issubset(actions) or not (
            "transform" in actions or "apply" in actions
        ):
            continue
        symbols = sorted(s for s in all_stateful_symbols if s.startswith(f"{prefix}_"))
        lifecycle = {
            "create": actions["create"],
            "destroy": actions["destroy"],
            "fit": actions["fit"],
            "operation": actions.get("transform", actions.get("apply")),
        }
        for optional in (
            "is_fitted",
            "inverse_transform",
            "transform_with_d",
            "threshold",
            "output_cols",
        ):
            if optional in actions:
                lifecycle[optional] = actions[optional]
        rows.append(
            _native_row(
                prefix.removeprefix("n4m_"),
                "stateful_preprocessor",
                symbols,
                header_index,
                snapshots,
                lifecycle=lifecycle,
            )
        )
    return rows


def _moment_stack_adapter(snapshots: dict[str, set[str]]) -> dict[str, Any]:
    symbols = ["n4m_model_selection_sweep_run"]
    ffi = set(FFI_SYMBOL_RE.findall(PYTHON_FFI.read_text(encoding="utf-8")))
    adapter_source = PYTHON_NATIVE.read_text(encoding="utf-8")
    stack_source = PYTHON_MOMENT_STACK.read_text(encoding="utf-8")
    if not set(symbols) <= ffi or not all(
        symbol in adapter_source for symbol in symbols
    ):
        raise MatrixError("moment_stack lacks declared Python FFI proof")
    if "class NativeMomentStackRegressor" not in stack_source:
        raise MatrixError("moment_stack catalog adapter class was not found")
    if any(
        symbol not in symbols_for_platform
        for symbols_for_platform in snapshots.values()
        for symbol in symbols
    ):
        raise MatrixError(
            "moment_stack FFI proof is absent from a platform ABI snapshot"
        )
    return {
        "id": "models.ensembles.moment_stack",
        "kind": "model",
        "classification": "adapter",
        "catalog_file": "catalog/methods/models.ensembles.moment_stack.yaml",
        "adapter_file": _relative(PYTHON_MOMENT_STACK),
        "api_symbols": symbols,
        "ffi_proof": {
            "declarations": _relative(PYTHON_FFI),
            "call_site": _relative(PYTHON_NATIVE),
            "adapter_class": "NativeMomentStackRegressor",
        },
        "boundary": "Python composes native sweep results and a Python Ridge meta-model; no all-native claim.",
    }


def _model_rows(
    header_index: dict[str, list[str]], snapshots: dict[str, set[str]]
) -> list[dict[str, Any]]:
    catalog = _catalog_models()
    catalog_symbols = {s for entry in catalog for s in entry["abi_symbols"]}
    header_symbols = _model_header_symbols(header_index)
    if catalog_symbols != header_symbols:
        raise MatrixError(
            "model universe mismatch: "
            f"catalog-only={sorted(catalog_symbols - header_symbols)}, "
            f"header-only={sorted(header_symbols - catalog_symbols)}"
        )
    rows = []
    for entry in catalog:
        if entry["abi_symbols"]:
            rows.append(
                _native_row(
                    entry["method_id"],
                    "model",
                    entry["abi_symbols"],
                    header_index,
                    snapshots,
                    catalog_file=entry["catalog_file"],
                )
            )
        elif entry["method_id"] == "models.ensembles.moment_stack":
            rows.append(_moment_stack_adapter(snapshots))
        else:
            raise MatrixError(
                f"catalog model {entry['method_id']} has no native or declared adapter proof"
            )
    if {row["id"] for row in rows} != {entry["method_id"] for entry in catalog}:
        raise MatrixError("a catalog model was not classified exactly once")
    return rows


def _generic_model_row(
    header_index: dict[str, list[str]], snapshots: dict[str, set[str]]
) -> dict[str, Any]:
    symbols = (
        "n4m_config_set_algorithm",
        "n4m_model_fit",
        "n4m_model_destroy",
        "n4m_model_predict",
        "n4m_model_predict_alloc",
        "n4m_model_transform",
        "n4m_model_transform_alloc",
    )
    algorithms = sorted(
        ALGORITHM_RE.findall((HEADERS / "n4m.h").read_text(encoding="utf-8"))
    )
    if not algorithms:
        raise MatrixError("generic model API exposes no n4m_algorithm_t routes")
    return _native_row(
        "generic_n4m_model_api",
        "generic_model_api",
        symbols,
        header_index,
        snapshots,
        algorithms=algorithms,
    )


def _derived_boundary_rows(
    header_index: dict[str, list[str]], catalog_method_ids: list[str]
) -> list[dict[str, Any]]:
    """Report potential extension/HPO surface from names, never invented examples."""
    public_symbols = sorted(header_index)
    extension_terms = ("plugin", "register", "callback", "executor", "external")
    hpo_terms = ("hpo", "optimizer", "objective", "sweep")

    def candidates(terms: tuple[str, ...]) -> dict[str, list[str]]:
        return {
            "api_symbols": [
                symbol
                for symbol in public_symbols
                if any(term in symbol for term in terms)
            ],
            "catalog_ids": [
                identifier
                for identifier in catalog_method_ids
                if any(term in identifier for term in terms)
            ],
        }

    policy = POLICY.read_text(encoding="utf-8")
    if "There is no plugin API" not in policy:
        raise MatrixError("closed-library plugin policy evidence is missing")
    return [
        {
            "id": "derived_runtime_extension_boundary",
            "kind": "extension_boundary",
            "classification": "refused",
            "policy_file": _relative(POLICY),
            "potential_external_or_unsupported": candidates(extension_terms),
            "reason": "Potential extension names are mechanically derived; policy supplies no runtime plugin API.",
        },
        {
            "id": "derived_hpo_boundary",
            "kind": "optimization_boundary",
            "classification": "not-promised",
            "potential_external_or_unsupported": candidates(hpo_terms),
            "reason": "Potential HPO/executor names are mechanically derived; this does not assert a host-objective executor.",
        },
    ]


def validate_matrix(matrix: dict[str, Any]) -> None:
    """Validate the checked-in schema and cross-field proof invariants."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _validate_schema_contract(matrix, schema)
    ids = [row["id"] for row in matrix["rows"]]
    if len(ids) != len(set(ids)):
        raise MatrixError("matrix row IDs are not unique")
    snapshots = _snapshot_symbols()
    for row in matrix["rows"]:
        if row["classification"] == "native":
            if set(row["abi_proof"]) != set(ABI_SNAPSHOTS):
                raise MatrixError(f"native row {row['id']} lacks a platform ABI proof")
            for platform, available in snapshots.items():
                missing = set(row["api_symbols"]) - available
                if missing:
                    raise MatrixError(
                        f"native row {row['id']} missing {platform}: {sorted(missing)}"
                    )
        if row["classification"] == "adapter" and not row.get("ffi_proof"):
            raise MatrixError(f"adapter row {row['id']} lacks FFI proof")


def _schema_error(message: str) -> None:
    raise MatrixError(f"matrix schema validation failed: {message}")


def _validate_schema_contract(matrix: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate the closed 1.1 schema without a runtime-only dependency.

    The adjacent Draft 2020-12 JSON Schema is the portable contract. This
    focused validator enforces every generated shape and is intentionally kept
    in the standard-library generator so the documented docs CI environment
    need not install a second validation package.
    """
    if schema.get("$id") != "https://nirs4all.org/schema/v1-capability-matrix-1.1.json":
        _schema_error("unexpected schema identifier")
    if set(matrix) != {
        "schema_version",
        "scope",
        "generation",
        "classifications",
        "rows",
    }:
        _schema_error("top-level keys are not closed")
    if matrix["schema_version"] != schema["properties"]["schema_version"]["const"]:
        _schema_error("unsupported schema_version")
    if matrix["classifications"] != list(CLASSIFICATIONS):
        _schema_error("classifications must match the closed enum")
    generation = matrix["generation"]
    if set(generation) != {
        "command",
        "deterministic",
        "input_digest",
        "platform_abi_snapshots",
    }:
        _schema_error("generation keys are not closed")
    if generation["deterministic"] is not True:
        _schema_error("generation.deterministic must be true")
    if set(generation["platform_abi_snapshots"]) != set(ABI_SNAPSHOTS):
        _schema_error("platform ABI snapshot keys are not closed")
    digest = generation["input_digest"]
    if (
        set(digest) != {"algorithm", "files", "sha256"}
        or digest["algorithm"] != "sha256"
    ):
        _schema_error("input digest must be a closed sha256 object")
    if not isinstance(digest["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", digest["sha256"]
    ):
        _schema_error("input digest is not sha256")
    if not digest["files"]:
        _schema_error("input digest has no files")
    for record in digest["files"]:
        if set(record) != {"path", "sha256"} or not isinstance(record["path"], str):
            _schema_error("input digest file record is invalid")
        if not isinstance(record["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["sha256"]
        ):
            _schema_error("input digest file hash is invalid")

    allowed = {
        "id",
        "kind",
        "classification",
        "api_symbols",
        "source_headers",
        "abi_proof",
        "ffi_proof",
        "catalog_file",
        "adapter_file",
        "boundary",
        "algorithms",
        "lifecycle",
        "policy_file",
        "potential_external_or_unsupported",
        "reason",
    }
    lifecycle_keys = {
        "create",
        "destroy",
        "fit",
        "operation",
        "is_fitted",
        "inverse_transform",
        "transform_with_d",
        "threshold",
        "output_cols",
    }
    for row in matrix["rows"]:
        if not {"id", "kind", "classification"} <= set(row) or set(row) - allowed:
            _schema_error("row keys are not closed")
        if (
            not isinstance(row["id"], str)
            or not row["id"]
            or not isinstance(row["kind"], str)
            or not row["kind"]
        ):
            _schema_error("row id and kind must be non-empty strings")
        if row["classification"] not in CLASSIFICATIONS:
            _schema_error("row classification is outside the closed enum")
        if row["classification"] == "native":
            if not {"api_symbols", "source_headers", "abi_proof"} <= set(row):
                _schema_error("native row lacks required proof keys")
        if row["classification"] == "adapter" and not {
            "api_symbols",
            "ffi_proof",
        } <= set(row):
            _schema_error("adapter row lacks required proof keys")
        if "api_symbols" in row and (
            not row["api_symbols"]
            or len(row["api_symbols"]) != len(set(row["api_symbols"]))
            or not all(
                isinstance(symbol, str) and symbol.startswith("n4m_")
                for symbol in row["api_symbols"]
            )
        ):
            _schema_error("API symbol proof is invalid")
        if "lifecycle" in row:
            lifecycle = row["lifecycle"]
            if (
                not {"create", "destroy", "fit", "operation"} <= set(lifecycle)
                or set(lifecycle) - lifecycle_keys
            ):
                _schema_error("lifecycle keys are invalid")


def build_matrix() -> dict[str, Any]:
    """Build a deterministic, fail-closed payload without VCS-derived input."""
    header_symbols = _header_symbols(HEADERS.rglob("*.h"))
    stateful_headers = [
        path for glob in STATEFUL_HEADER_GLOBS for path in ROOT.glob(glob)
    ]
    stateful_header_symbols = _header_symbols(stateful_headers)
    header_index = _header_index(header_symbols)
    snapshots = _snapshot_symbols()
    rows = _model_rows(header_index, snapshots)
    rows.append(_generic_model_row(header_index, snapshots))
    rows.extend(_stateful_rows(stateful_header_symbols, header_index, snapshots))
    rows.extend(_derived_boundary_rows(header_index, _catalog_method_ids()))
    matrix = {
        "schema_version": "1.1",
        "scope": "V1 catalogued models, generic model routes, stateful preprocessors, adapters, and derived boundaries",
        "generation": {
            "command": "python3 docs/_extras/generate_v1_capability_matrix.py --check",
            "deterministic": True,
            "input_digest": _input_digest(),
            "platform_abi_snapshots": {
                platform: _relative(path) for platform, path in ABI_SNAPSHOTS.items()
            },
        },
        "classifications": list(CLASSIFICATIONS),
        "rows": sorted(rows, key=lambda row: row["id"]),
    }
    validate_matrix(matrix)
    return matrix


def render_markdown(matrix: dict[str, Any]) -> str:
    counts = Counter((row["kind"], row["classification"]) for row in matrix["rows"])
    models = [row["id"] for row in matrix["rows"] if row["kind"] == "model"]
    stateful = [
        row["id"] for row in matrix["rows"] if row["kind"] == "stateful_preprocessor"
    ]
    return "\n".join(
        (
            "# V1 capability matrix",
            "",
            "This generated matrix is limited to code evidence. Native rows require a public declaration and all Linux, macOS, and Windows ABI snapshots; adapters name their FFI declarations and call path.",
            "",
            "```bash",
            "python3 docs/_extras/generate_v1_capability_matrix.py --check",
            "```",
            "",
            f"Input digest (not a Git revision): `{matrix['generation']['input_digest']['sha256']}`.",
            "",
            "| Scope | Classification | Count |",
            "| --- | --- | ---: |",
            *[
                f"| {kind} | {classification} | {count} |"
                for (kind, classification), count in sorted(counts.items())
            ],
            "",
            "The model universe is exactly the `category: models` catalog entries plus the generic `n4m_model_fit` algorithm routes. Runtime-extension and HPO candidates are derived from public/catalog names; they do not create an unsupported capability claim.",
            "",
            "## Classified catalog models",
            "",
            ", ".join(f"`{identifier}`" for identifier in models),
            "",
            "## Stateful preprocessors",
            "",
            ", ".join(f"`{identifier}`" for identifier in stateful),
            "",
        )
    )


def _serialized(matrix: dict[str, Any]) -> str:
    return json.dumps(matrix, indent=2, sort_keys=True) + "\n"


def write_or_check(check: bool) -> int:
    matrix = build_matrix()
    expected = {OUTPUT_JSON: _serialized(matrix), OUTPUT_DOC: render_markdown(matrix)}
    stale = [
        path
        for path, content in expected.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if check:
        if stale:
            print(
                "capability matrix is stale:",
                *(_relative(path) for path in stale),
                sep="\n  ",
                file=sys.stderr,
            )
            return 1
        return 0
    for path, content in expected.items():
        path.write_text(content, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if committed outputs differ"
    )
    return write_or_check(parser.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
