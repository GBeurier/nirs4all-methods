"""Internal validation framework - declarative registry layer.

Phase C4A-1 port of the pls4all validation framework: project the
chemometrics4all cross-binding orchestrator (``benchmarks/cross_binding/
orchestrator.py``) onto a deterministic JSON snapshot under
``benchmarks/validation/registry/`` so docs and dashboard consumers can
inspect comparator / reference / tolerance contracts without importing
the heavy orchestrator module.

Public surface used by the CLIs and tests:

- ``models``              - dataclasses describing the JSON payload shape
- ``export_registry``     - build/write the registry JSON snapshot
- ``validate_registry``   - fail non-zero on drift vs the committed snapshot
"""

from __future__ import annotations

__all__ = ["models", "export_registry", "validate_registry"]
