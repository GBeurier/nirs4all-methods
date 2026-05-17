# Cross-binding PLS regression timing

## n = 100, p = 50

| Backend | Language | Tier | Median (ms) | Speedup vs external | OK |
|---|---|---|---:|---:|:---:|
| cpp | C++ | direct | 0.06 | — | ✓ |
| python_tier1 | Python | tier 1 | 0.04 | 9.77x | ✓ |
| python_tier2 | Python | tier 2 | 0.16 | 2.76x | ✓ |
| sklearn | Python | external | 0.43 | 1.00x | ✓ |
| r_tier1 | R | tier 1 | 0.00 | — | ✓ |
| r_tier2 | R | tier 2 | 2.50 | 0.80x | ✓ |
| r_pls | R | external | 2.00 | 1.00x | ✓ |
| matlab_tier1 | MATLAB | tier 1 | 0.06 | 9.71x | ✓ |
| matlab_tier2 | MATLAB | tier 2 | 1.60 | 0.39x | ✓ |
| matlab_pls | MATLAB | external | 0.62 | 1.00x | ✓ |

## n = 500, p = 200

| Backend | Language | Tier | Median (ms) | Speedup vs external | OK |
|---|---|---|---:|---:|:---:|
| cpp | C++ | direct | 1.34 | — | ✓ |
| python_tier1 | Python | tier 1 | 1.02 | 1.47x | ✓ |
| python_tier2 | Python | tier 2 | 0.88 | 1.72x | ✓ |
| sklearn | Python | external | 1.51 | 1.00x | ✓ |
| r_tier1 | R | tier 1 | 1.00 | 15.00x | ✓ |
| r_tier2 | R | tier 2 | 14.00 | 1.07x | ✓ |
| r_pls | R | external | 15.00 | 1.00x | ✓ |
| matlab_tier1 | MATLAB | tier 1 | 0.74 | 3.41x | ✓ |
| matlab_tier2 | MATLAB | tier 2 | 2.97 | 0.85x | ✓ |
| matlab_pls | MATLAB | external | 2.52 | 1.00x | ✓ |

## n = 1000, p = 500

| Backend | Language | Tier | Median (ms) | Speedup vs external | OK |
|---|---|---|---:|---:|:---:|
| cpp | C++ | direct | 6.23 | — | ✓ |
| python_tier1 | Python | tier 1 | 6.15 | 5.23x | ✓ |
| python_tier2 | Python | tier 2 | 7.38 | 4.36x | ✓ |
| sklearn | Python | external | 32.18 | 1.00x | ✓ |
| r_tier1 | R | tier 1 | 9.00 | 8.61x | ✓ |
| r_tier2 | R | tier 2 | 39.50 | 1.96x | ✓ |
| r_pls | R | external | 77.50 | 1.00x | ✓ |
| matlab_tier1 | MATLAB | tier 1 | 5.11 | 5.59x | ✓ |
| matlab_tier2 | MATLAB | tier 2 | 24.30 | 1.17x | ✓ |
| matlab_pls | MATLAB | external | 28.53 | 1.00x | ✓ |

