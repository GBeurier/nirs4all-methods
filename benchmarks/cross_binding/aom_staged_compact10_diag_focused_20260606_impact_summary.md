# AOM impact group summary

## by_head_stage_option

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| ridge\|baseline\|detrend_poly(2) | 5 | 3 | 3 | 1.6 | 0.27435019124695614 |
| ridge\|baseline\|detrend_poly(1) | 5 | 2 | 2 | 1.8 | 0.24037848977839804 |
| ridge\|smooth\|savgol_smooth(21,3) | 4 | 2 | 2 | 2.75 | 0.08291959037550312 |
| ridge\|smooth\|savgol_smooth(11,3) | 4 | 1 | 1 | 3.25 | 0.0800634825416983 |
| ridge\|derivative\|norris_williams(5,5,1) | 2 | 0 | 2 | 1.0 |  |
| ridge\|derivative\|norris_williams(7,7,1) | 3 | 0 | 1 | 2.0 | 0.26355352957342415 |
| ridge\|smooth\|savgol_smooth(15,3) | 4 | 0 | 0 | 2.5 | 0.09771698780462001 |
| ridge\|smooth\|savgol_smooth(7,2) | 1 | 0 | 0 | 4.0 | 0.13016499212596244 |

## by_operator

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| detrend_poly | 6 | 5 | 5 | 2.0 | -1.0467840423359167 |
| savgol_smooth | 5 | 3 | 3 | 2.2 | 0.13688254733733363 |
| norris_williams | 4 | 0 | 3 | 1.25 | 0.26355352957342415 |
| identity | 4 | 0 | 0 | 4.5 | 0.0 |
| savgol_derivative | 7 | 0 | 0 | 6.285714285714286 | -2.6446013543562064 |

## by_stage_family

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| baseline | 6 | 5 | 5 | 2.0 | -1.0467840423359167 |
| smooth | 5 | 3 | 3 | 2.2 | 0.13688254733733363 |
| derivative | 7 | 0 | 3 | 3.7142857142857144 | -1.1523728010622396 |
| identity | 4 | 0 | 0 | 4.5 | 0.0 |

## by_stage_option

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| baseline\|detrend_poly(2) | 6 | 3 | 3 | 2.5 | -1.0467840423359167 |
| smooth\|savgol_smooth(21,3) | 4 | 2 | 2 | 2.75 | 0.08291959037550312 |
| baseline\|detrend_poly(1) | 6 | 2 | 2 | 2.8333333333333335 | -1.1182077990803392 |
| smooth\|savgol_smooth(11,3) | 4 | 1 | 1 | 3.25 | 0.0800634825416983 |
| derivative\|norris_williams(7,7,1) | 3 | 0 | 1 | 2.0 | 0.26355352957342415 |
| smooth\|savgol_smooth(15,3) | 4 | 0 | 0 | 2.5 | 0.09771698780462001 |
| derivative\|norris_williams(5,5,1) | 3 | 0 | 2 | 3.3333333333333335 |  |
| smooth\|savgol_smooth(7,2) | 1 | 0 | 0 | 4.0 | 0.13016499212596244 |
