# SPDX-License-Identifier: CECILL-2.1
"""Test-only adapter exposing the legacy AOM/moment facade surface.

The public ``n4m`` package no longer re-exports the flat AOM/moment functions
and ``Native*`` estimator names at the top level (Phase 3a clean break). These
large campaign/wrapper tests exercise that exact behaviour, which now lives in
the private implementation facades. This module re-publishes those facades under
the historical local names (``aom``, ``moment``, ``native``, ``native_sklearn``)
plus a combined ``legacy`` namespace, so the test bodies keep running against the
real (ABI-2) implementation without losing any assertion.

It is not part of the shipped public surface.
"""
from types import SimpleNamespace

from n4m._impl import aom_facade as aom
from n4m._impl import moment_facade as moment
from n4m._impl import native

# native_sklearn historically pointed at n4m.sklearn; every attribute these tests
# read off it lives on the AOM facade.
native_sklearn = aom

# Combined namespace mirroring the removed top-level ``n4m.<symbol>`` surface:
# moment facade is the superset, with the AOM-only extras layered on top.
_combined: dict[str, object] = {}
for _fac in (moment, aom):
    for _name in getattr(_fac, "__all__", []):
        _combined[_name] = getattr(_fac, _name)
for _name in ("aom_chain_score_campaign", "aom_pls", "pop_pls"):
    _combined[_name] = getattr(aom, _name)
legacy = SimpleNamespace(**_combined)

__all__ = ["aom", "moment", "native", "native_sklearn", "legacy"]
