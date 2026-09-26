# SPDX-License-Identifier: CECILL-2.1

"""Domain-invariant native regression over source and target cohorts."""

from n4m._impl import native as _native
from n4m._impl.domain_invariant import DIPLS

di_pls = _native.di_pls

__all__ = ["DIPLS", "di_pls"]
