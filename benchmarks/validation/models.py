"""Typed records for the validation framework's JSON snapshot.

The framework wraps ``benchmarks/cross_binding/orchestrator.py`` and
exports a deterministic, host-insensitive structural view. The
dataclasses below describe the shape of each JSON file under
``benchmarks/validation/registry/``. They are intentionally stdlib-only:
the model layer itself does not import numpy, ctypes, or the chemometrics4all
package.

All dataclasses are frozen and convert to plain dicts via ``to_dict()``
so the export step can serialise them with ``json.dumps(..., indent=2,
sort_keys=True, ensure_ascii=True)`` and produce byte-stable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
SCHEMA_DESCRIPTION = (
    "Declarative validation contracts derived from "
    "benchmarks/cross_binding/orchestrator.py and enriched from "
    "benchmarks/benchmark_registry.json. Phase C4A-1 of the validation "
    "framework port - structural data only, host-insensitive, regenerated "
    "by `python -m benchmarks.validation.export_registry --write`."
)


def _scalarise(value: Any) -> Any:
    """Coerce a parameter / dataset cell value to a JSON-friendly scalar.

    Mirrors the helper used in the pls4all snapshot. The validation
    package does not depend on numpy directly, but we still handle the
    ``tolist`` / ``item`` duck-type protocol in case a caller stores a
    numpy scalar somewhere on the snapshot path.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, (list, tuple)):
        return [_scalarise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _scalarise(v) for k, v in sorted(value.items())}
    return str(value)


def normalise_mapping(params: dict[str, Any]) -> dict[str, Any]:
    """Sorted, JSON-friendly copy of a mapping (cell params, datasets...)."""
    return {str(k): _scalarise(v) for k, v in sorted(params.items())}


@dataclass(frozen=True)
class ReferenceLink:
    """One reference declared by a MethodSpec.

    Mirrors a ``ReferenceSpec(...)`` row from the cross-binding
    orchestrator minus the Python callables (factories, comparators).
    The structural fields here are everything the docs + dashboard need
    to render parity / context badges without resolving the factory.
    """
    backend: str
    library: str
    language: str
    compare: bool
    gate_c4a: bool
    contract_note: str
    max_cols: int | None
    comparator: str
    r_expr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "library": self.library,
            "language": self.language,
            "compare": bool(self.compare),
            "gate_c4a": bool(self.gate_c4a),
            "contract_note": self.contract_note,
            "max_cols": (None if self.max_cols is None
                          else int(self.max_cols)),
            "comparator": self.comparator,
            "r_expr": self.r_expr,
        }


@dataclass(frozen=True)
class MethodRecord:
    """One row in ``methods.json``.

    Mirrors a ``MethodSpec(...)`` entry in
    ``benchmarks/cross_binding/orchestrator.py`` and adds the enrichments
    available from ``benchmarks/benchmark_registry.json``: operation
    kind, truth source ids, default dataset cell.
    """
    method_id: str
    label: str
    family: str
    py_class: str
    c_prefix: str
    r_expr: str
    max_cols: int | None
    compare_internal: bool
    comparator: str
    tolerance_quality: str
    operation: str
    truth_sources: list[str]
    metrics: list[str]
    default_dataset: dict[str, Any]
    references: list[ReferenceLink]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "label": self.label,
            "family": self.family,
            "py_class": self.py_class,
            "c_prefix": self.c_prefix,
            "r_expr": self.r_expr,
            "max_cols": (None if self.max_cols is None
                          else int(self.max_cols)),
            "compare_internal": bool(self.compare_internal),
            "comparator": self.comparator,
            "tolerance_quality": self.tolerance_quality,
            "operation": self.operation,
            "truth_sources": sorted(self.truth_sources),
            "metrics": sorted(self.metrics),
            "default_dataset": (normalise_mapping(self.default_dataset)
                                 if self.default_dataset else {}),
            "references": [r.to_dict() for r in self.references],
        }


@dataclass(frozen=True)
class ReferenceRecord:
    """Aggregate row in ``references.json`` - one per backend slot.

    The runtime orchestrator declares each reference under a backend
    identifier (``ref.nirs4all``, ``ref.sklearn``, ``ref.r.prospectr``,
    ...). The aggregate record lists every method that wires the backend
    and notes the canonical language for the slot (for UI grouping).
    """
    backend: str
    language: str
    methods: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "language": self.language,
            "methods": sorted(self.methods),
        }


@dataclass(frozen=True)
class DatasetRecord:
    """One synthetic dataset cell from the cross-binding orchestrator.

    The orchestrator generates ``(n, p)`` cells via
    ``synthetic_spectra(n, p, seed)``. The ``seed`` is derived from a
    base seed plus a per-size offset (see ``orchestrator.main``); the
    snapshot captures the base seed and the formula in the manifest so
    consumers can rebuild the per-cell seed without re-reading
    orchestrator.py.
    """
    id: str
    n: int
    p: int
    seed_base: int
    in_smoke_preset: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "n": int(self.n),
            "p": int(self.p),
            "seed_base": int(self.seed_base),
            "in_smoke_preset": bool(self.in_smoke_preset),
        }


@dataclass(frozen=True)
class ComparatorRecord:
    """One numerical parity comparator exposed by ``orchestrator.py``.

    Mirrors the module-level callables (``outputs_close``,
    ``outputs_close_iasls``, ``outputs_close_sign_invariant_columns``,
    ``outputs_close_savgol_valid_window``). Every callable returns
    ``(ok, reason)`` for a left/right pair; the structural fields here
    are enough for the docs to describe what gate each method runs.
    """
    id: str
    name: str
    description: str
    metric: str
    default_rtol: float
    default_atol: float
    contract_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metric": self.metric,
            "default_rtol": float(self.default_rtol),
            "default_atol": float(self.default_atol),
            "contract_note": self.contract_note,
        }


@dataclass(frozen=True)
class ToleranceRecord:
    """Per-method tolerance band (the contract surfaced in the docs)."""
    method: str
    comparator: str
    binding_tolerance: float
    reference_rtol: float
    reference_atol: float
    quality: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "comparator": self.comparator,
            "binding_tolerance": float(self.binding_tolerance),
            "reference_rtol": float(self.reference_rtol),
            "reference_atol": float(self.reference_atol),
            "quality": self.quality,
        }


@dataclass(frozen=True)
class SuiteRecord:
    """Named bundle of methods + datasets + comparators."""
    id: str
    description: str
    methods: list[str]
    datasets: list[str]
    comparators: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "methods": sorted(self.methods),
            "datasets": sorted(self.datasets),
            "comparators": sorted(self.comparators),
        }


@dataclass(frozen=True)
class Manifest:
    """``manifest.json`` - schema metadata + counts + seed formula."""
    schema_version: int = SCHEMA_VERSION
    description: str = SCHEMA_DESCRIPTION
    source: str = "benchmarks/cross_binding/orchestrator.py"
    benchmark_registry: str = "benchmarks/benchmark_registry.json"
    truth_sources_lock: str = "benchmarks/truth_sources.lock.json"
    seed_formula: str = (
        "synthetic_spectra(n, p, seed=seed_base + 1009 * size_index + 37)"
    )
    files: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "description": self.description,
            "source": self.source,
            "benchmark_registry": self.benchmark_registry,
            "truth_sources_lock": self.truth_sources_lock,
            "seed_formula": self.seed_formula,
            "files": sorted(self.files),
            "counts": {k: int(v) for k, v in sorted(self.counts.items())},
        }
