# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 5a baseline correction operators.

Validated once against ``pybaselines==1.1.4`` (as pinned in
``parity/python_generator/pyproject.toml``).  After validation these
modules become the canonical parity floor for chemometrics4all — pybaselines
itself is no longer in the build path of the parity fixture generator.

This insulation gives us two things:

  1. Upstream pybaselines releases can change implementation details
     (sparse solver, convergence-test ordering, tolerance defaults) without
     breaking the c4a parity fixtures.
  2. The reference is self-contained: ~500 LOC of pure NumPy / scipy.sparse
     that anyone can read and audit against the original paper.

Each operator implementation cites the paper it derives from.  Default
parameters match pybaselines 1.1.4.
"""

from .detrend import detrend
from .asls   import asls
from .airpls import airpls
from .arpls  import arpls

__all__ = ["detrend", "asls", "airpls", "arpls"]
