# AOM staged compact10 CUDA benchmark summary

Date: 2026-06-06

Scope:

- Cohort runner: `benchmarks/cross_binding/run_aom_staged_real_cohort.py`
- Cohort: default diverse11 cohort, `--limit 10`
- Property filter: `--max-features 1200`
- Production selection: train-CV refit only; all OK rows report
  `selection_uses_test_set=False`
- CUDA: `CUDA_VISIBLE_DEVICES=0`,
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`
- Shared profile knobs:
  `--plan compact --max-chains 12 --top-k 12 --refit-top-k 6
  --refit-per-head-top-k 2 --scale-x-grid false,true
  --cuda-pls-parallel-folds --cuda-pls-min-device-features 1`

Artifacts:

| Run | Result CSV | Oracle comparison CSV |
|---|---|---|
| mixed Ridge+PLS | `aom_staged_real_cohort_compact10_cuda_20260606.csv` | `aom_staged_real_cohort_compact10_mixed_cuda_20260606_oracle_compare.csv` |
| Ridge-only | `aom_staged_real_cohort_compact10_ridge_cuda_20260606.csv` | `aom_staged_real_cohort_compact10_ridge_cuda_20260606_oracle_compare.csv` |
| PLS-only | `aom_staged_real_cohort_compact10_pls_cuda_20260606.csv` | `aom_staged_real_cohort_compact10_pls_cuda_20260606_oracle_compare.csv` |

Run summary:

| Run | OK | Skipped | Selected heads | Median fit s | Total fit s | Screen PLS CUDA/host | Refit PLS CUDA/host |
|---|---:|---:|---|---:|---:|---:|---:|
| mixed | 8 | 2 | ridge | 7.01952 | 58.3952 | 960 / 0 | 220 / 0 |
| ridge | 8 | 2 | ridge | 3.06787 | 31.9358 | 0 / 0 | 0 / 0 |
| pls | 8 | 2 | pls | 2.25035 | 54.9493 | 960 / 0 | 450 / 0 |

Oracle comparison, lower is better:

| Run | Baseline | Paired | Target wins | Median ratio | Mean ratio |
|---|---|---:|---:|---:|---:|
| mixed | AOM-PLS oracle | 7 | 1 | 1.03079 | 1.04345 |
| mixed | AOM-Ridge oracle | 8 | 0 | 1.08068 | 1.32063 |
| mixed | TabPFN oracle | 8 | 4 | 0.978527 | 0.978275 |
| ridge | AOM-PLS oracle | 7 | 1 | 1.03079 | 1.04345 |
| ridge | AOM-Ridge oracle | 8 | 0 | 1.08068 | 1.32063 |
| ridge | TabPFN oracle | 8 | 4 | 0.978527 | 0.978275 |
| pls | AOM-PLS oracle | 7 | 0 | 1.22592 | 1.51374 |
| pls | AOM-Ridge oracle | 8 | 0 | 1.21773 | 1.66712 |
| pls | TabPFN oracle | 8 | 1 | 1.27884 | 1.26241 |

Paired target RMSEP:

| Dataset | mixed | ridge | pls |
|---|---:|---:|---:|
| BEEFMARBLING/Beef_Marbling_RandomSplit | 73.45472692579786 | 73.45472692579786 | 77.74474591038131 |
| DIESEL/DIESEL_bp50_246_b-a | 3.379018650442519 | 3.379018650442519 | 6.528871010230355 |
| DIESEL/DIESEL_bp50_246_hla-b | 2.813216233179541 | 2.813216233179541 | 7.30985821262009 |
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | 41.398414151482235 | 41.398414151482235 | 30.13357614622125 |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR | 3.530696563164442 | 3.530696563164442 | 4.19907180296537 |
| MANURE21/All_manure_CaO_SPXY_strat_Manure_type | 7.7580618866024045 | 7.7580618866024045 | 7.998980668389263 |
| MANURE21/All_manure_P2O5_SPXY_strat_Manure_type | 2.5416505080848246 | 2.5416505080848246 | 2.9763922132393708 |
| WOOD_density/WOOD_N_402_Olale | 0.04784775858729744 | 0.04784775858729744 | 0.05496020653956867 |

Interpretation:

- In this compact profile, mixed collapses to Ridge-only: every OK row selects
  `ridge`, and mixed/Ridge RMSEP are identical.
- PLS-only is not competitive on this small cohort, except on the ECOSIS
  Chla+b split where it beats Ridge by a large margin.
- The current compact staged moment workflow remains close to AOM-PLS oracle
  median on the paired subset, but it is still behind AOM-Ridge oracle.
- PLS CUDA routing is working as intended: all PLS screen/refit CV fits in the
  mixed and PLS-only runs use CUDA device counters with zero host PLS CV fits.
