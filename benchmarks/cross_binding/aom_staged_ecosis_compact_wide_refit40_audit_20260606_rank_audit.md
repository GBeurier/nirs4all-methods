# AOM rank audit summary

> **Offline audit only.** Production selection uses train-CV;
> `selection_uses_test_set` is `False` for all rows. Do not use
> audit rankings to route or select by dataset name, source or id.

Rows with audit data: **1** of 1 total.

| Dataset | Selected | Oracle | CV RMSE | Test RMSE | Test rank | Oracle RMSE | Gap ratio | CV/test Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | ridge:0.1 savgol_smooth(7,2) | ridge:10 finite_difference(1) | 63.0871 | 41.3984 | 41 | 10.5817 | 2.9123 | 0.0667 |


## Mismatch patterns (post-hoc audit only)

> Rows where the train-CV winner was not the test oracle
> (`test_rank_delta > 0` or `oracle_gap_ratio > 0`).
> **Production selection is unchanged** - post-hoc audit surface only.
> Do not use these patterns to route or select by dataset name, source or id.

### By head pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| ridge -> ridge | 1 | 2.9123 | 2.9123 | 41 | 40 |

### By chain pair

| Pair | Count | Mean gap ratio | Median gap ratio | Median test rank | Median rank delta |
|---|---:|---:|---:|---:|---:|
| savgol_smooth(7,2) -> finite_difference(1) | 1 | 2.9123 | 2.9123 | 41 | 40 |

