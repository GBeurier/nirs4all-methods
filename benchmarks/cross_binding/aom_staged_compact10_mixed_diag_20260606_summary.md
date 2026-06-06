# AOM staged compact10 mixed diagnostics summary

Date: 2026-06-06

Scope:

- Runner: `benchmarks/cross_binding/run_aom_staged_real_cohort.py`
- Output CSV: `aom_staged_real_cohort_compact10_mixed_diag_20260606.csv`
- Diagnostics dir: `aom_staged_compact10_mixed_diag_20260606/`
- Profile: `--plan compact --heads ridge,pls --max-chains 12 --top-k 12
  --refit-top-k 6 --refit-per-head-top-k 2 --scale-x-grid false,true`
- Property filter: `--max-features 1200`
- CUDA: one visible GPU, `--cuda-pls-parallel-folds`,
  `--cuda-pls-min-device-features 1`, `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`

Run facts:

| Metric | Value |
|---|---:|
| Dataset rows | 10 |
| OK rows | 8 |
| Property-skipped rows | 2 |
| Selection uses test set | false for every OK row |
| Screen PLS CUDA / host CV fits | 960 / 0 |
| Refit PLS CUDA / host CV fits | 220 / 0 |
| Diagnostics JSON files | 8 |
| `impact_groups.csv` rows | 169 |

Best impact-group counts by dataset, using train-CV `refit_cv_rmse`:

| Group kind | Winner counts |
|---|---|
| `by_operator` | `detrend_poly`: 5, `identity`: 2, `savgol_smooth`: 1 |
| `by_stage_family` | `baseline`: 5, `identity`: 2, `smooth`: 1 |
| `by_stage_option` | `baseline|detrend_poly(1)`: 4, `identity|identity`: 2, `baseline|detrend_poly(2)`: 1, `smooth|savgol_smooth(7,2)`: 1 |
| `by_head_stage_option` | `ridge|baseline|detrend_poly(1)`: 4, `ridge|identity|identity`: 2, `ridge|baseline|detrend_poly(2)`: 1, `ridge|smooth|savgol_smooth(7,2)`: 1 |

Selected train-CV candidates:

| Dataset | Selected head | Scale X | Selected chain | Refit CV RMSE | Held-out RMSEP |
|---|---|---:|---|---:|---:|
| BEEFMARBLING/Beef_Marbling_RandomSplit | ridge | true | `detrend_poly(1) -> norris_williams(5,5,1)` | 72.09612491682333 | 73.45472692579786 |
| DIESEL/DIESEL_bp50_246_b-a | ridge | true | `detrend_poly(1)` | 3.08788515762756 | 3.379018650442519 |
| DIESEL/DIESEL_bp50_246_hla-b | ridge | true | `detrend_poly(2)` | 3.9599077693924336 | 2.813216233179541 |
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | ridge | false | `savgol_smooth(7,2)` | 63.0871183090006 | 41.398414151482235 |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR | ridge | true | `detrend_poly(1)` | 3.6327043889036603 | 3.530696563164442 |
| MANURE21/All_manure_CaO_SPXY_strat_Manure_type | ridge | true | `identity` | 15.309663393104419 | 7.7580618866024045 |
| MANURE21/All_manure_P2O5_SPXY_strat_Manure_type | ridge | true | `identity` | 3.6567277444658592 | 2.5416505080848246 |
| WOOD_density/WOOD_N_402_Olale | ridge | true | `detrend_poly(1)` | 0.04623582177979116 | 0.04784775858729744 |

Interpretation:

- The compact mixed profile still selects `ridge` everywhere.
- Within this low budget, the strongest reusable preprocessing signal is
  baseline detrending, especially `detrend_poly(1)`.
- Identity remains best on the two MANURE rows in this cohort.
- SavGol appears as the best train-CV family only on the ECOSIS Chla+b split;
  this matches the previous PLS-only observation that ECOSIS is the main case
  where a non-Ridge/SavGol-heavy direction deserves more budget.
- The diagnostics are now rich enough to drive the next incremental campaign:
  compare `baseline|detrend_poly(1/2)`, `identity`, and a small SavGol-focused
  branch under the same train-CV-only rule before expanding the full cartesian.
