# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later

from n4m._impl import (
    DetectorRollOffAugmenter,
    EdgeArtifactsAugmenter,
    EdgeCurvatureAugmenter,
    MoistureAugmenter,
    StrayLightAugmenter,
    TemperatureAugmenter,
    TruncatedPeakAugmenter,
)

__all__ = [
    "DetectorRollOffAugmenter",
    "EdgeArtifactsAugmenter",
    "EdgeCurvatureAugmenter",
    "MoistureAugmenter",
    "StrayLightAugmenter",
    "TemperatureAugmenter",
    "TruncatedPeakAugmenter",
]
