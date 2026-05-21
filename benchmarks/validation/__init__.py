"""Internal validation framework - declarative registry layer.

Phase PLS-1 of `VALIDATION_FRAMEWORK_ROADMAP.md`:
expose the existing `benchmarks.parity_timing.registry` as a deterministic
JSON surface (methods, references, datasets, comparators, tolerances,
suites). The framework wraps the existing registry - it does not rewrite
the orchestrator and does not require R / MATLAB packages to be installed.

Public surface used by the CLIs and tests:

- `models`              - dataclasses describing the JSON payload shape
- `export_registry`     - build/write the registry JSON snapshot
- `validate_registry`   - fail non-zero on drift vs the committed snapshot
"""

from __future__ import annotations

__all__ = ["models", "export_registry", "validate_registry"]
