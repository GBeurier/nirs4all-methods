# AOM impact group summary

## by_head_stage_option

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| ridge\|baseline\|detrend_poly(1) | 7 | 4 | 4 | 2.0 | -0.1679252996901987 |
| ridge\|identity\|identity | 5 | 2 | 2 | 3.2 | 0.0 |
| ridge\|baseline\|detrend_poly(2) | 5 | 1 | 1 | 2.4 | 0.13721771535634805 |
| ridge\|smooth\|savgol_smooth(7,2) | 6 | 1 | 1 | 3.3333333333333335 | 0.01410378951331086 |
| ridge\|derivative\|norris_williams(5,5,1) | 4 | 0 | 1 | 3.25 | -0.004059802642524243 |
| ridge\|smooth\|savgol_smooth(5,2) | 7 | 0 | 0 | 3.7142857142857144 | 0.0032701272766147566 |
| ridge\|derivative\|savgol_derivative(7,2,1) | 2 | 0 | 0 | 4.0 |  |
| ridge\|derivative\|finite_difference(1) | 2 | 0 | 0 | 5.0 | 0.057924015941585694 |

## by_operator

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| detrend_poly | 7 | 5 | 5 | 1.8571428571428572 | -0.15943237432305918 |
| identity | 5 | 2 | 2 | 3.2 | 0.0 |
| savgol_smooth | 8 | 1 | 1 | 3.625 | 0.018726417465762812 |
| norris_williams | 4 | 0 | 1 | 3.25 | -0.004059802642524243 |
| savgol_derivative | 5 | 0 | 0 | 5.8 | -1.4942198148196921 |
| finite_difference | 6 | 0 | 0 | 6.666666666666667 | -0.8331362986622014 |

## by_stage_family

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| baseline | 7 | 5 | 5 | 1.8571428571428572 | -0.15943237432305918 |
| identity | 5 | 2 | 2 | 3.2 | 0.0 |
| smooth | 8 | 1 | 1 | 3.625 | 0.018726417465762812 |
| derivative | 7 | 0 | 1 | 4.428571428571429 | -0.5502483968384635 |

## by_stage_option

| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |
|---|---:|---:|---:|---:|---:|
| baseline\|detrend_poly(1) | 7 | 4 | 4 | 2.0 | -0.1679252996901987 |
| identity\|identity | 5 | 2 | 2 | 3.2 | 0.0 |
| baseline\|detrend_poly(2) | 6 | 1 | 1 | 3.1666666666666665 | -0.6978276150686978 |
| smooth\|savgol_smooth(7,2) | 6 | 1 | 1 | 3.3333333333333335 | 0.01410378951331086 |
| derivative\|norris_williams(5,5,1) | 4 | 0 | 1 | 3.25 | -0.004059802642524243 |
| smooth\|savgol_smooth(5,2) | 8 | 0 | 0 | 4.25 | 0.0032701272766147566 |
| derivative\|savgol_derivative(7,2,1) | 3 | 0 | 0 | 5.333333333333333 | -0.009263737987928143 |
| derivative\|finite_difference(1) | 6 | 0 | 0 | 6.666666666666667 | -0.8331362986622014 |
