# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import NativeAOMMomentPLSExactScreenRefitRegressor as AOMMomentPLSExactScreenRefitRegressor
from n4m._impl import NativeAOMMomentPLSScreenRefitRegressor as AOMMomentPLSScreenRefitRegressor
from n4m._impl import NativeAOMMomentRidgeScreenRefitRegressor as AOMMomentRidgeScreenRefitRegressor
from n4m._impl import NativeAOMMomentScreenRefitRegressor as AOMMomentScreenRefitRegressor
from n4m._impl import NativeAOMSavgolFocusRegressor as AOMSavgolFocusRegressor
from n4m._impl import NativeAOMStagedChainCampaignRegressor as AOMStagedChainCampaignRegressor
from n4m._impl import NativeAOMStrictFamilyLiteRegressor as AOMStrictFamilyLiteRegressor
from n4m._impl import native as _native

aom_chain_score_campaign = _native.aom_chain_score_campaign
aom_chain_screen_refit_campaign = _native.aom_chain_screen_refit_campaign
aom_moment_screen_refit_campaign = _native.aom_moment_screen_refit_campaign
aom_staged_chain_campaign = _native.aom_staged_chain_campaign

__all__ = [
    "AOMMomentPLSExactScreenRefitRegressor",
    "AOMMomentPLSScreenRefitRegressor",
    "AOMMomentRidgeScreenRefitRegressor",
    "AOMMomentScreenRefitRegressor",
    "AOMSavgolFocusRegressor",
    "AOMStagedChainCampaignRegressor",
    "AOMStrictFamilyLiteRegressor",
    "aom_chain_score_campaign",
    "aom_chain_screen_refit_campaign",
    "aom_moment_screen_refit_campaign",
    "aom_staged_chain_campaign",
]
