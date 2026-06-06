# AOM rank audit summary

> **Offline audit only.** Production selection uses train-CV;
> `selection_uses_test_set` is `False` for all rows. Do not use
> audit rankings to route or select by dataset name, source or id.

Rows with audit data: **4** of 4 total.

| Dataset | Selected | Oracle | CV RMSE | Test RMSE | Test rank | Oracle RMSE | Gap ratio | CV/test Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BEEFMARBLING/Beef_Marbling_RandomSplit | ridge:10 detrend_poly(1) -> norris_williams(5,5,1) | ridge:10 savgol_smooth(5,2) -> finite_difference(1) | 72.0961 | 73.4547 | 6 | 72.4922 | 0.0133 | 0.3333 |
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | ridge:0.1 savgol_smooth(7,2) | pls:1 detrend_poly(2) | 63.0871 | 41.3984 | 5 | 26.2460 | 0.5773 | -0.7381 |
| DIESEL/DIESEL_bp50_246_hla-b | ridge:1 detrend_poly(2) | ridge:1 identity | 3.9599 | 2.8132 | 3 | 2.7072 | 0.0392 | 0.5476 |
| DIESEL/DIESEL_bp50_246_b-a | ridge:1 detrend_poly(1) | ridge:1 detrend_poly(2) | 3.0879 | 3.3790 | 2 | 3.3080 | 0.0215 | 0.7619 |


## Mismatch patterns (post-hoc audit only)

> Rows where the train-CV winner was not the test oracle
> (`test_rank_delta > 0` or `oracle_gap_ratio > 0`).
> **Production selection is unchanged** - post-hoc audit surface only.
> Do not use these patterns to route or select by dataset name, source or id.

### By head pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| ridge -> pls | 1 | 0.5773 | 0.5773 | 5 | 4 |
| ridge -> ridge | 3 | 0.0246 | 0.0215 | 3 | 2 |

### By chain pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| detrend_poly(1) -> detrend_poly(2) | 1 | 0.0215 | 0.0215 | 2 | 1 |
| detrend_poly(1) -> norris_williams(5,5,1) -> savgol_smooth(5,2) -> finite_difference(1) | 1 | 0.0133 | 0.0133 | 6 | 5 |
| detrend_poly(2) -> identity | 1 | 0.0392 | 0.0392 | 3 | 2 |
| savgol_smooth(7,2) -> detrend_poly(2) | 1 | 0.5773 | 0.5773 | 5 | 4 |

