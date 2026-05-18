# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 5a + 5b baseline correction operators.

Validated once against ``pybaselines==1.1.4`` (as pinned in
``parity/python_generator/pyproject.toml``).  After validation these
modules become the canonical parity floor for chemometrics4all — pybaselines
itself is no longer in the build path of the parity fixture generator.

This insulation gives us two things:

  1. Upstream pybaselines releases can change implementation details
     (sparse solver, convergence-test ordering, tolerance defaults) without
     breaking the c4a parity fixtures.
  2. The reference is self-contained: pure NumPy / scipy.sparse that anyone
     can read and audit against the original paper.

Each operator implementation cites the paper it derives from.  Default
parameters match pybaselines 1.1.4.

Phase 5a (4 ops): detrend, asls, airpls, arpls.
Phase 5b (6 ops): modpoly, imodpoly, snip, rolling_ball, iasls, beads.
"""

from .detrend       import detrend
from .asls          import asls
from .airpls        import airpls
from .arpls         import arpls
from .modpoly       import modpoly
from .imodpoly      import imodpoly
from .snip          import snip
from .rolling_ball  import rolling_ball
from .iasls         import iasls
from .beads         import beads

__all__ = [
    "detrend", "asls", "airpls", "arpls",
    "modpoly", "imodpoly", "snip", "rolling_ball", "iasls", "beads",
]
