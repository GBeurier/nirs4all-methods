# AOM impact group summary

## by_head_stage_option

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| ridge\|baseline\|detrend_poly(1) | 9 | 4 | 4 | 2.7777777777777777 | -0.1340252521704303 |
| ridge\|identity\|identity | 6 | 2 | 2 | 3.6666666666666665 | 0.0 |
| ridge\|baseline\|detrend_poly(2) | 6 | 1 | 1 | 2.3333333333333335 | 0.11100055988263614 |
| ridge\|derivative\|norris_williams(5,5,1) | 5 | 1 | 2 | 2.8 | -0.004059802642524243 |
| ridge\|smooth\|savgol_smooth(7,2) | 6 | 1 | 1 | 3.3333333333333335 | 0.014103789512566566 |
| pls\|derivative\|savgol_derivative(11,2,2) | 6 | 1 | 1 | 6.0 | -1.0832328057925493 |
| ridge\|smooth\|savgol_smooth(5,2) | 7 | 0 | 0 | 3.7142857142857144 | 0.0032701272764754903 |
| ridge\|derivative\|savgol_derivative(7,2,1) | 3 | 0 | 0 | 4.0 |  |

## by_operator

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| detrend_poly | 9 | 5 | 5 | 2.3333333333333335 | -0.11583264967144609 |
| identity | 6 | 2 | 2 | 3.6666666666666665 | 0.0 |
| norris_williams | 5 | 1 | 2 | 2.8 | -0.004059802642524243 |
| savgol_smooth | 10 | 1 | 1 | 3.6 | 0.018304126369476714 |
| savgol_derivative | 7 | 1 | 1 | 4.571428571428571 | -1.0832328057925493 |
| finite_difference | 8 | 0 | 0 | 5.75 | -0.6197827228479578 |

## by_stage_family

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| baseline | 9 | 5 | 5 | 2.3333333333333335 | -0.11583264967144609 |
| derivative | 9 | 2 | 3 | 3.6666666666666665 | -0.41025307321271776 |
| identity | 6 | 2 | 2 | 3.6666666666666665 | 0.0 |
| smooth | 10 | 1 | 1 | 3.6 | 0.018304126369476714 |

## by_stage_option

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| baseline\|detrend_poly(1) | 9 | 4 | 4 | 2.7777777777777777 | -0.1340252521704303 |
| identity\|identity | 6 | 2 | 2 | 3.6666666666666665 | 0.0 |
| derivative\|norris_williams(5,5,1) | 5 | 1 | 2 | 2.8 | -0.004059802642524243 |
| baseline\|detrend_poly(2) | 7 | 1 | 1 | 3.0 | -0.5087291490678997 |
| smooth\|savgol_smooth(7,2) | 6 | 1 | 1 | 3.3333333333333335 | 0.014103789512566566 |
| derivative\|savgol_derivative(11,2,2) | 6 | 1 | 1 | 6.0 | -1.0832328057925493 |
| smooth\|savgol_smooth(5,2) | 10 | 0 | 0 | 4.1 | 0.0054238845455668065 |
| derivative\|savgol_derivative(7,2,1) | 4 | 0 | 0 | 4.5 | -0.009263737987928546 |
