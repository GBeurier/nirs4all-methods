# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 8 orthogonalization preprocessing
operators (OSC, EPO).

Validated once against ``nirs4all==0.8.x``
(``operators/transforms/orthogonalization.py``). After validation these
modules become the canonical parity floor for chemometrics4all — nirs4all
itself is no longer in the build path of the parity fixture generator.

Phase 1 design ban: chemometrics4all does NOT ship PLS models, so the
OSC reference here uses the SVD-fallback path documented in the Phase 8
brief. For a univariate target ``y`` the SVD of the rank-1 matrix
``y @ x^T`` reduces to the normalised cross-correlation ``X^T y / ||X^T y||``,
which is bit-equal to nirs4all's PLS-weight extraction.

Phase 8 (2 ops): osc, epo.
"""

from .epo import epo, epo_fit_transform
from .osc import osc, osc_fit_transform

__all__ = ["osc", "osc_fit_transform", "epo", "epo_fit_transform"]
