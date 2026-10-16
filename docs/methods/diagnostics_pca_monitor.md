# `pca_monitor` — n4m.stats.pca_monitor

_Namespace_: **`n4m.stats`** · _Fully-qualified_: `n4m.stats.pca_monitor` · _Catalog id_: `diagnostics.pca_monitor`

## API surface

**C ABI (ABI 2):** [`n4m_stats_pca_monitor`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L409). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Jackson (1991), *A User's Guide to Principal Components*, Wiley, https://doi.org/10.1002/0471725331; Nomikos & MacGregor (1995), *Multivariate SPC charts for monitoring batch processes*, Technometrics 37(1), 41–59, https://doi.org/10.1080/00401706.1995.10485888.

### Mathematical principle

A PCA is fitted on the reference rows (each fit centred on its own mean), keeping the components above the eigenvalue resolution $16\,m\,\epsilon\,\lambda_1$ in every fold and full fit. For a row, $T^2=\sum_a t_a^2/\lambda_a$ and $Q$ is the squared norm of the explicit residual. Reference rows are evaluated out of fold and the limits are type-7 $(1-\alpha)$ quantiles. A monitored row is counted beyond a limit when it exceeds it by more than the tolerance $\tau=16\,m\,\epsilon\,\lambda_1/\lambda_k$ (relative for $T^2$, times the largest squared row norm for $Q$), and as within tolerance otherwise near the limit.

### Appropriate uses

Measuring how much of a new domain (another instrument, campaign or a set of new samples) lies beyond the score and residual ranges of a reference domain, with limits that do not assume normal scores and are not computed on the rows used for the fit.

### Limits and validation

The tolerance is a conservative estimate, not a proven bound; rows within it are numerically undetermined and never reported as decisions. Limits are empirical and coarse with few reference rows; a fraction beyond a limit is descriptive, not a probability of failure. Supported domain: finite values, nonzero $|x|$ in $[2^{-256}, 2^{256}]$, spread at most $2^{32}$ (`N4M_ERR_UNSUPPORTED` otherwise); workloads above 10 billion operations or 50 million cells are refused.

### Implementation

The ABI-2 symbol `n4m_stats_pca_monitor` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` with the shared PCA helper and wrapped in JS/WASM by `statsPcaMonitor`. `parity/fixtures/domain_shift_v1.json` (exact rational inverse covariance and axis-aligned cases, NumPy `linalg.svd`) is checked by `cpp/tests/test_domain_shift.cpp`.

### Sources and provenance

https://doi.org/10.1002/0471725331; https://doi.org/10.1080/00401706.1995.10485888; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/domain_shift_v1.json

## Catalog note

PCA monitoring of rows against a reference PCA: effective k = components above the eigenvalue resolution 16 m eps lambda_1 in every out-of-fold and full fit, Hotelling T2 and Q (explicit residual) of monitored rows, out-of-fold reference T2/Q whose empirical (1-alpha) type-7 quantiles are the limits, counts beyond the limits by more than a conservative tolerance tau = 16 m eps cond and counts within it (numerically undetermined, never decided), centroid shift (full, subspace, residual), centroid T2 and variance traces (each set about its own double-double centre). Supported domain: finite values, nonzero |x| in [2^-256, 2^256], spread <= 2^32. Reference gate: parity/generators/domain_shift_v1.py -> parity/fixtures/domain_shift_v1.json (rational inverse covariance and axis-aligned exact cases, numpy.linalg.svd), cpp/tests/test_domain_shift.cpp.


_See also_: [methods index](index.md).