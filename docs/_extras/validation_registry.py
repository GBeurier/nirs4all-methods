"""Read the committed validation-framework snapshot.

The declarative validation framework under
``benchmarks/validation/registry/`` exports a deterministic JSON view of
the parity / timing registry (Phase PLS-1). Phase PLS-2 wires that
snapshot into:

  * per-method documentation (``docs/_extras/build_methods.py`` renders
    a ``### Validation contract`` section per method);
  * the landing dashboard payload (``docs/_extras/build_landing.py``
    embeds a compact ``validation`` block in ``bench-data.json``).

This module is the single read-only entry point that both consumers go
through. It is stdlib-only (matches the validation framework's own
contract) and degrades gracefully when the snapshot is incomplete -
docs / dashboard builds on a fresh clone with missing files do not
crash, they just publish an empty validation surface.

Usage
-----

.. code-block:: python

    from validation_registry import load_snapshot

    snapshot = load_snapshot()
    if snapshot.is_loaded:
        contract = snapshot.method_contract("pls")
        payload  = snapshot.dashboard_payload()

The module deliberately does not import anything from
``benchmarks.parity_timing``; the snapshot is host-insensitive and
already captures everything Phase PLS-2 needs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

#: Default location of the committed snapshot. Tests override this via
#: the ``registry_dir`` argument to :func:`load_snapshot`.
DEFAULT_REGISTRY_DIR: Path = (
    ROOT / "benchmarks" / "validation" / "registry"
)

#: Files the snapshot must expose for the loader to consider it
#: complete. Matches ``benchmarks.validation.export_registry.REGISTRY_FILES``
#: but kept independent so the docs build does not import the
#: validation framework's Python package (which itself imports
#: ``benchmarks.parity_timing.registry``).
SNAPSHOT_FILES: tuple[str, ...] = (
    "manifest.json",
    "methods.json",
    "references.json",
    "datasets.json",
    "comparators.json",
    "tolerances.json",
    "suites.json",
)


# ---------------------------------------------------------------------------
# Loaded view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationSnapshot:
    """In-memory view of ``benchmarks/validation/registry/*.json``.

    ``is_loaded`` is true when every expected file was read successfully.
    Partial / malformed snapshots still load (so callers can inspect
    what is available) but ``is_loaded`` flips to ``False`` and the
    ``errors`` list explains what is missing - useful both for the
    Sphinx warning logs and the strict-mode failure path.
    """

    registry_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    methods: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    comparators: list[dict[str, Any]] = field(default_factory=list)
    tolerances: list[dict[str, Any]] = field(default_factory=list)
    suites: list[dict[str, Any]] = field(default_factory=list)
    errors: tuple[str, ...] = ()

    @property
    def is_loaded(self) -> bool:
        """True when every expected file parsed without errors."""
        return not self.errors and bool(self.manifest) and bool(self.methods)

    # ------------------------------------------------------------------
    # Per-method lookups
    # ------------------------------------------------------------------

    def method_contract(self, name: str) -> dict[str, Any] | None:
        """Return the full contract for ``name`` or ``None`` if absent.

        The returned dict combines fields from ``methods.json`` and
        ``tolerances.json`` and adds:

        - ``comparator_detail`` - the matching entry from
          ``comparators.json`` (metric, default tolerance, ...);
        - ``suites`` - the suites that include this method, with
          ``datasets`` reduced to the cells the suite touches;
        - ``datasets`` - the union of dataset records (``{id, n, p}``)
          referenced by those suites.

        Callers should treat the result as read-only.
        """
        method = self._methods_by_name().get(name)
        if method is None:
            return None
        tolerance = self._tolerances_by_method().get(name, {})
        comparator_id = method.get("comparator") or tolerance.get("comparator")
        comparator_detail = self._comparators_by_id().get(comparator_id, {})

        suites = []
        dataset_ids: set[str] = set()
        for suite in self.suites:
            if name in suite.get("methods", []):
                suite_datasets = list(suite.get("datasets", []))
                dataset_ids.update(suite_datasets)
                suites.append({
                    "id": suite.get("id"),
                    "description": suite.get("description", ""),
                    "datasets": suite_datasets,
                    "comparators": list(suite.get("comparators", [])),
                })

        datasets_by_id = self._datasets_by_id()
        datasets = sorted(
            (datasets_by_id[d] for d in dataset_ids if d in datasets_by_id),
            key=lambda r: (int(r.get("n", 0)), int(r.get("p", 0))),
        )

        return {
            "name": name,
            "description": method.get("description", ""),
            "notes": method.get("notes", ""),
            "paper_only": method.get("paper_only", ""),
            "reference_kind": method.get("reference_kind", ""),
            "canonical_reference_role": method.get(
                "canonical_reference_role"),
            "has_python_reference": bool(
                method.get("has_python_reference", False)),
            "has_r_reference": bool(method.get("has_r_reference", False)),
            "extra_reference_roles": list(
                method.get("extra_reference_roles", [])),
            "prediction_key": method.get("prediction_key", ""),
            "cell_params": dict(method.get("cell_params", {})),
            "rmse_rel_tol": method.get("rmse_rel_tol"),
            "tolerance_quality": method.get("tolerance_quality", ""),
            "needs_x_target": bool(method.get("needs_x_target", False)),
            "needs_sample_weights": bool(
                method.get("needs_sample_weights", False)),
            "needs_labels": bool(method.get("needs_labels", False)),
            "needs_group_assignment": bool(
                method.get("needs_group_assignment", False)),
            "comparator": comparator_id,
            "comparator_detail": dict(comparator_detail),
            "binding_tolerance": tolerance.get("binding_tolerance"),
            "reference_tolerance": tolerance.get(
                "reference_tolerance"),
            "suites": sorted(suites, key=lambda s: s.get("id") or ""),
            "datasets": datasets,
        }

    def method_names(self) -> list[str]:
        """Sorted list of MethodSpec names exposed by the snapshot."""
        return sorted(self._methods_by_name())

    # ------------------------------------------------------------------
    # Dashboard payload
    # ------------------------------------------------------------------

    def dashboard_payload(self) -> dict[str, Any]:
        """Compact payload embedded as ``bench-data.json["validation"]``.

        Designed for the landing dashboard's tooltip / info panels:
        we expose per-method tolerance + reference structure + comparator
        keyed by method name (so the JS layer can ``payload.validation
        .methods[algo]`` in O(1)). The schema mirrors the JSON files
        themselves to keep diffing straightforward.

        Returns an empty stub (``schema_version=0``) when the snapshot
        is missing - the dashboard's existing UI still renders, the
        new ``validation`` block is just empty.
        """
        if not self.is_loaded:
            return {
                "schema_version": 0,
                "available": False,
                "errors": list(self.errors),
                "methods": {},
                "comparators": {},
                "datasets": {},
                "suites": [],
            }

        tolerance_by_method = self._tolerances_by_method()
        compact_methods: dict[str, dict[str, Any]] = {}
        for record in self.methods:
            name = record.get("name")
            if not name:
                continue
            tolerance = tolerance_by_method.get(name, {})
            compact_methods[name] = {
                "comparator": record.get("comparator"),
                "reference_kind": record.get("reference_kind"),
                "canonical_reference_role": record.get(
                    "canonical_reference_role"),
                "tolerance_quality": record.get("tolerance_quality"),
                "binding_tolerance": tolerance.get("binding_tolerance"),
                "reference_tolerance": tolerance.get(
                    "reference_tolerance"),
                "rmse_rel_tol": record.get("rmse_rel_tol"),
                "has_python_reference": bool(
                    record.get("has_python_reference", False)),
                "has_r_reference": bool(record.get("has_r_reference", False)),
                "extra_reference_roles": list(
                    record.get("extra_reference_roles", [])),
                "needs_x_target": bool(record.get("needs_x_target", False)),
                "needs_sample_weights": bool(
                    record.get("needs_sample_weights", False)),
                "needs_labels": bool(record.get("needs_labels", False)),
                "needs_group_assignment": bool(
                    record.get("needs_group_assignment", False)),
                "prediction_key": record.get("prediction_key"),
                "paper_only": record.get("paper_only") or "",
            }

        comparators = {
            c.get("id"): {
                "name": c.get("name"),
                "metric": c.get("metric"),
                "default_tolerance": c.get("default_tolerance"),
                "tolerance_field": c.get("tolerance_field"),
                "description": c.get("description", ""),
            }
            for c in self.comparators
            if c.get("id")
        }

        datasets = {
            d.get("id"): {
                "n": d.get("n"),
                "p": d.get("p"),
                "seed_base": d.get("seed_base"),
            }
            for d in self.datasets
            if d.get("id")
        }

        suites = [
            {
                "id": s.get("id"),
                "description": s.get("description", ""),
                "methods": list(s.get("methods", [])),
                "datasets": list(s.get("datasets", [])),
                "comparators": list(s.get("comparators", [])),
            }
            for s in self.suites
        ]

        return {
            "schema_version": int(self.manifest.get("schema_version") or 0),
            "available": True,
            "source": self.manifest.get("source"),
            "counts": dict(self.manifest.get("counts", {})),
            "methods": compact_methods,
            "comparators": comparators,
            "datasets": datasets,
            "suites": suites,
        }

    # ------------------------------------------------------------------
    # Internal helpers (memoised via local builds; the dataclass itself
    # is frozen so we cache on access through plain method calls).
    # ------------------------------------------------------------------

    def _methods_by_name(self) -> dict[str, dict[str, Any]]:
        return {m.get("name"): m for m in self.methods if m.get("name")}

    def _tolerances_by_method(self) -> dict[str, dict[str, Any]]:
        return {
            t.get("method"): t
            for t in self.tolerances
            if t.get("method")
        }

    def _comparators_by_id(self) -> dict[str, dict[str, Any]]:
        return {c.get("id"): c for c in self.comparators if c.get("id")}

    def _datasets_by_id(self) -> dict[str, dict[str, Any]]:
        return {d.get("id"): d for d in self.datasets if d.get("id")}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> tuple[Any, str | None]:
    """Read and parse a JSON file, returning ``(value, error_message)``.

    ``error_message`` is ``None`` on success; otherwise it carries a
    human-readable description suitable for either Sphinx logs or the
    test assertions further down.
    """
    if not path.exists():
        return None, f"missing file: {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name}: {exc}"


def load_snapshot(
    registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
) -> ValidationSnapshot:
    """Load the validation snapshot from ``registry_dir``.

    Missing or unparseable files do not raise; they accumulate into
    :attr:`ValidationSnapshot.errors` so callers can decide whether to
    warn, fall back, or hard-fail. The default location is the
    committed snapshot under ``benchmarks/validation/registry/``.
    """
    registry_dir = Path(registry_dir)
    errors: list[str] = []

    def _list_or_empty(value: Any, key: str) -> list[dict[str, Any]]:
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return [r for r in value[key] if isinstance(r, dict)]
        return []

    manifest_data, manifest_err = _read_json_file(
        registry_dir / "manifest.json")
    if manifest_err:
        errors.append(manifest_err)
    manifest = manifest_data if isinstance(manifest_data, dict) else {}

    methods_data, methods_err = _read_json_file(
        registry_dir / "methods.json")
    if methods_err:
        errors.append(methods_err)
    methods = _list_or_empty(methods_data, "methods")

    references_data, references_err = _read_json_file(
        registry_dir / "references.json")
    if references_err:
        errors.append(references_err)
    references = _list_or_empty(references_data, "references")

    datasets_data, datasets_err = _read_json_file(
        registry_dir / "datasets.json")
    if datasets_err:
        errors.append(datasets_err)
    datasets = _list_or_empty(datasets_data, "datasets")

    comparators_data, comparators_err = _read_json_file(
        registry_dir / "comparators.json")
    if comparators_err:
        errors.append(comparators_err)
    comparators = _list_or_empty(comparators_data, "comparators")

    tolerances_data, tolerances_err = _read_json_file(
        registry_dir / "tolerances.json")
    if tolerances_err:
        errors.append(tolerances_err)
    tolerances = _list_or_empty(tolerances_data, "tolerances")

    suites_data, suites_err = _read_json_file(
        registry_dir / "suites.json")
    if suites_err:
        errors.append(suites_err)
    suites = _list_or_empty(suites_data, "suites")

    return ValidationSnapshot(
        registry_dir=registry_dir,
        manifest=manifest,
        methods=methods,
        references=references,
        datasets=datasets,
        comparators=comparators,
        tolerances=tolerances,
        suites=suites,
        errors=tuple(errors),
    )


__all__ = [
    "DEFAULT_REGISTRY_DIR",
    "SNAPSHOT_FILES",
    "ValidationSnapshot",
    "load_snapshot",
]
