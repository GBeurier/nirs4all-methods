# AOM rank audit summary

> **Offline audit only.** Production selection uses train-CV;
> `selection_uses_test_set` is `False` for all rows. Do not use
> audit rankings to route or select by dataset name, source or id.

Rows with audit data: **8** of 8 total.

| Dataset | Selected | Oracle | CV RMSE | Test RMSE | Test rank | Oracle RMSE | Gap ratio | CV/test Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BEEFMARBLING/Beef_Marbling_RandomSplit | ridge:10 detrend_poly(1) -> norris_williams(5,5,1) | ridge:10 savgol_smooth(5,2) -> finite_difference(1) | 72.0961 | 73.4547 | 6 | 72.4922 | 0.0133 | 0.3333 |
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | ridge:0.1 savgol_smooth(7,2) | pls:1 detrend_poly(2) | 63.0871 | 41.3984 | 5 | 26.2460 | 0.5773 | -0.7381 |
| WOOD_density/WOOD_N_402_Olale | ridge:10 detrend_poly(1) | ridge:10 detrend_poly(2) | 0.0462 | 0.0478 | 5 | 0.0467 | 0.0256 | 0.6667 |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR | ridge:10 detrend_poly(1) | ridge:10 detrend_poly(2) | 3.6327 | 3.5307 | 4 | 3.3682 | 0.0482 | 0.6190 |
| DIESEL/DIESEL_bp50_246_hla-b | ridge:1 detrend_poly(2) | ridge:1 identity | 3.9599 | 2.8132 | 3 | 2.7072 | 0.0392 | 0.5476 |
| DIESEL/DIESEL_bp50_246_b-a | ridge:1 detrend_poly(1) | ridge:1 detrend_poly(2) | 3.0879 | 3.3790 | 2 | 3.3080 | 0.0215 | 0.7619 |
| MANURE21/All_manure_CaO_SPXY_strat_Manure_type | ridge:0.1 identity | ridge:0.1 savgol_smooth(5,2) | 15.3097 | 7.7581 | 2 | 7.7416 | 0.0021 | 0.8333 |
| MANURE21/All_manure_P2O5_SPXY_strat_Manure_type | ridge:0.1 identity | ridge:0.1 identity | 3.6567 | 2.5417 | 1 | 2.5417 | 0.0000 | 0.9762 |


## Mismatch patterns (post-hoc audit only)

> Rows where the train-CV winner was not the test oracle
> (`test_rank_delta > 0` or `oracle_gap_ratio > 0`).
> **Production selection is unchanged** - post-hoc audit surface only.
> Do not use these patterns to route or select by dataset name, source or id.

### By head pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| ridge -> pls | 1 | 0.5773 | 0.5773 | 5 | 4 |
| ridge -> ridge | 6 | 0.0250 | 0.0235 | 3.5000 | 2.5000 |

### By chain pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| detrend_poly(1) -> detrend_poly(2) | 3 | 0.0318 | 0.0256 | 4 | 3 |
| detrend_poly(1) -> norris_williams(5,5,1) -> savgol_smooth(5,2) -> finite_difference(1) | 1 | 0.0133 | 0.0133 | 6 | 5 |
| detrend_poly(2) -> identity | 1 | 0.0392 | 0.0392 | 3 | 2 |
| identity -> savgol_smooth(5,2) | 1 | 0.0021 | 0.0021 | 2 | 1 |
| savgol_smooth(7,2) -> detrend_poly(2) | 1 | 0.5773 | 0.5773 | 5 | 4 |

