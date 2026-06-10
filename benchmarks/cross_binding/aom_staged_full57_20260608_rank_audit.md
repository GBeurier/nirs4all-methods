# AOM rank audit summary

> **Offline audit only.** Production selection uses train-CV;
> `selection_uses_test_set` is `False` for all rows. Do not use
> audit rankings to route or select by dataset name, source or id.

Rows with audit data: **51** of 51 total.

| Dataset | Selected | Oracle | CV RMSE | Test RMSE | Test rank | Oracle RMSE | Gap ratio | CV/test Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PHOSPHORUS/MP_spxyG | ridge:10 identity | pls:1 savgol_derivative(11,2,2) | 0.0147 | 0.0215 | 7 | 0.0195 | 0.1035 | -0.6905 |
| PHOSPHORUS/V25_spxyG | ridge:1 savgol_smooth(7,2) | pls:1 savgol_derivative(11,2,2) | 0.2281 | 0.2947 | 6 | 0.2047 | 0.4400 | -0.8810 |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD | ridge:10 identity | pls:2 savgol_derivative(11,2,2) | 3.0799 | 4.1612 | 6 | 3.1096 | 0.3382 | -0.8286 |
| PLUMS/Firmness_spxy70 | pls:2 detrend_poly(2) | ridge:0.1 detrend_poly(2) | 0.4319 | 0.3517 | 6 | 0.2825 | 0.2449 | -0.0857 |
| ECOSIS_LeafTraits/Ccar_spxyG_block2deg | ridge:10 savgol_smooth(5,2) -> finite_difference(1) | ridge:10 norris_williams(5,5,1) | 100.6190 | 12.4524 | 6 | 10.1307 | 0.2292 | 0.2381 |
| ALPINE/ALPINE_P_291_KS | ridge:1 identity | ridge:10 detrend_poly(2) | 0.0797 | 0.0586 | 6 | 0.0549 | 0.0674 | 0.3333 |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra | ridge:10 norris_williams(5,5,1) | pls:2 detrend_poly(1) | 3.0842 | 4.8506 | 6 | 4.6425 | 0.0448 | -0.5476 |
| COLZA/N_woOutlier | ridge:0.1 savgol_smooth(7,2) | ridge:10 detrend_poly(1) -> norris_williams(5,5,1) | 0.3527 | 0.2861 | 6 | 0.2775 | 0.0308 | 0.2619 |
| BEEFMARBLING/Beef_Marbling_RandomSplit | ridge:10 detrend_poly(1) -> norris_williams(5,5,1) | ridge:10 savgol_smooth(5,2) -> finite_difference(1) | 72.0961 | 73.4547 | 6 | 72.4922 | 0.0133 | 0.3333 |
| BISCUIT/Biscuit_Sucrose_40_RandomSplit | pls:2 finite_difference(1) | ridge:10 savgol_smooth(7,2) | 1.8361 | 1.2820 | 5 | 0.8018 | 0.5989 | -0.4524 |
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | ridge:0.1 savgol_smooth(7,2) | pls:1 detrend_poly(2) | 63.0871 | 41.3984 | 5 | 26.2460 | 0.5773 | -0.7381 |
| COLZA/C_woOutlier | ridge:0.1 savgol_smooth(7,2) | ridge:10 norris_williams(5,5,1) | 1.5525 | 1.9148 | 5 | 1.6533 | 0.1581 | -0.1190 |
| BERRY/brix_groupSampleID_stratDateVar_balRows | pls:2 savgol_derivative(11,2,2) | ridge:0.1 identity | 4.2234 | 4.0224 | 5 | 3.5811 | 0.1232 | 0.0857 |
| BEER/Beer_OriginalExtract_60_YbaseSplit | ridge:0.1 detrend_poly(1) | pls:2 detrend_poly(1) | 0.3478 | 0.2527 | 5 | 0.2293 | 0.1020 | -0.3929 |
| BEER/Beer_OriginalExtract_60_KS | ridge:0.1 detrend_poly(1) | pls:2 detrend_poly(2) | 0.2531 | 0.2334 | 5 | 0.2193 | 0.0641 | -0.5357 |
| PHOSPHORUS/NP_spxyG | ridge:0.1 savgol_smooth(7,2) | pls:2 detrend_poly(1) | 0.0653 | 0.1074 | 5 | 0.1033 | 0.0393 | 0.2857 |
| WOOD_density/WOOD_N_402_Olale | ridge:10 detrend_poly(1) | ridge:10 detrend_poly(2) | 0.0462 | 0.0478 | 5 | 0.0467 | 0.0256 | 0.6667 |
| DarkResp/Rd25_spxy70 | ridge:10 detrend_poly(1) | ridge:10 savgol_smooth(7,2) | 0.2143 | 0.1767 | 5 | 0.1730 | 0.0214 | 0.2143 |
| PHOSPHORUS/Pi_spxyG | ridge:0.1 identity | pls:2 detrend_poly(1) | 0.0549 | 0.1680 | 4 | 0.1499 | 0.1205 | 0.1905 |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR | ridge:10 detrend_poly(1) | ridge:10 detrend_poly(2) | 3.6327 | 3.5307 | 4 | 3.3682 | 0.0482 | 0.6190 |


## Mismatch patterns (post-hoc audit only)

> Rows where the train-CV winner was not the test oracle
> (`test_rank_delta > 0` or `oracle_gap_ratio > 0`).
> **Production selection is unchanged** - post-hoc audit surface only.
> Do not use these patterns to route or select by dataset name, source or id.

### By head pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| pls -> ridge | 4 | 0.2501 | 0.1841 | 5 | 4 |
| ridge -> pls | 11 | 0.2123 | 0.1205 | 5 | 4 |
| ridge -> ridge | 25 | 0.0815 | 0.0256 | 3 | 2 |

### By chain pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| detrend_poly(1) -> detrend_poly(1) | 2 | 0.0886 | 0.0886 | 3.5000 | 2.5000 |
| detrend_poly(1) -> detrend_poly(2) | 5 | 0.0332 | 0.0256 | 4 | 3 |
| detrend_poly(1) -> norris_williams(5,5,1) -> detrend_poly(1) | 1 | 0.0653 | 0.0653 | 2 | 1 |
| detrend_poly(1) -> norris_williams(5,5,1) -> savgol_smooth(5,2) -> finite_difference(1) | 1 | 0.0133 | 0.0133 | 6 | 5 |
| detrend_poly(1) -> savgol_derivative(7,2,1) -> detrend_poly(1) | 1 | 1.0778 | 1.0778 | 2 | 1 |
| detrend_poly(1) -> savgol_derivative(7,2,1) -> detrend_poly(1) -> norris_williams(5,5,1) | 1 | 0.0080 | 0.0080 | 3 | 2 |
| detrend_poly(1) -> savgol_smooth(7,2) | 1 | 0.0214 | 0.0214 | 5 | 4 |
| detrend_poly(2) -> detrend_poly(2) | 1 | 0.2449 | 0.2449 | 6 | 5 |
| detrend_poly(2) -> identity | 1 | 0.0392 | 0.0392 | 3 | 2 |
| finite_difference(1) -> detrend_poly(1) -> norris_williams(5,5,1) | 1 | 0.0257 | 0.0257 | 2 | 1 |
| finite_difference(1) -> norris_williams(5,5,1) | 1 | 0.2761 | 0.2761 | 3 | 2 |
| finite_difference(1) -> savgol_smooth(7,2) | 1 | 0.5989 | 0.5989 | 5 | 4 |
| identity -> detrend_poly(1) | 2 | 0.1750 | 0.1750 | 3.5000 | 2.5000 |
| identity -> detrend_poly(2) | 1 | 0.0674 | 0.0674 | 6 | 5 |
| identity -> identity | 1 | 0.0049 | 0.0049 | 4 | 3 |
| identity -> savgol_derivative(11,2,2) | 2 | 0.2209 | 0.2209 | 6.5000 | 5.5000 |
| identity -> savgol_smooth(5,2) | 1 | 0.0021 | 0.0021 | 2 | 1 |
| norris_williams(5,5,1) -> detrend_poly(1) | 1 | 0.0448 | 0.0448 | 6 | 5 |
| norris_williams(5,5,1) -> detrend_poly(1) -> norris_williams(5,5,1) | 1 | 0.0073 | 0.0073 | 2 | 1 |
| savgol_derivative(11,2,2) -> detrend_poly(1) | 2 | 0.0420 | 0.0420 | 2.5000 | 1.5000 |
| savgol_derivative(11,2,2) -> identity | 1 | 0.1232 | 0.1232 | 5 | 4 |
| savgol_smooth(5,2) -> finite_difference(1) -> norris_williams(5,5,1) | 1 | 0.2292 | 0.2292 | 6 | 5 |
| savgol_smooth(5,2) -> savgol_smooth(7,2) | 1 | 0.0108 | 0.0108 | 2 | 1 |
| savgol_smooth(7,2) -> detrend_poly(1) | 2 | 0.0389 | 0.0389 | 4.5000 | 3.5000 |
| savgol_smooth(7,2) -> detrend_poly(1) -> norris_williams(5,5,1) | 1 | 0.0308 | 0.0308 | 6 | 5 |
| savgol_smooth(7,2) -> detrend_poly(2) | 2 | 0.2910 | 0.2910 | 3.5000 | 2.5000 |
| savgol_smooth(7,2) -> identity | 2 | 0.0022 | 0.0022 | 3 | 2 |
| savgol_smooth(7,2) -> norris_williams(5,5,1) | 1 | 0.1581 | 0.1581 | 5 | 4 |
| savgol_smooth(7,2) -> savgol_derivative(11,2,2) | 1 | 0.4400 | 0.4400 | 6 | 5 |

