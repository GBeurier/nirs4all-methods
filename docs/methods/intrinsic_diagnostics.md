# Shared intrinsic diagnostic procedures

Methods 1.3.10 adds five discoverable generic procedures through the existing
ABI 2.20 `n4m_procedure_run` interface. The authoritative input/output contract
is [the versioned JSON](../intrinsic_diagnostics_contract.v1.json). Python, R,
MATLAB/Octave and JS/WASM only translate parameters, matrix layouts and native
named results. Rust consumers use that same generic C ABI. No binding maintains
an algorithm list or implements these numerical methods.

A binary with ABI 2.20 can predate these methods: call `n4m_method_find` or inspect
the manifest before using an ID. Source version, binary capability and registry
publication are separate facts.

| ID | Native behavior |
| --- | --- |
| `diagnostics.spectral_reference_qc` | RMS, SAM in radians, OLS offset/gain, residual RMS; reference is a native per-feature dataset mean or an explicitly supplied frozen reference. |
| `diagnostics.spectral_events_qc` | Row and pooled jump/spike counts, fractions and eligible denominators; physical-axis derivatives are distinct from index differences. Clipping requires declared sensor bounds. Observed-extrema occupancy is a separate descriptive proxy. |
| `diagnostics.spectral_noise_proxy` | Descriptive second-difference MAD proxy, normalization `1.4826/sqrt(6)`, amplitude ratio and `20 log10` ratio. This is neither instrumental SNR nor the existing SG residual proxy. |
| `diagnostics.local_density` | Real native fixed-k LOF with stable distance/index ties; training scores exclude self. Novelty uses the fitted reference neighborhood. The legacy distance-ratio proxy is not called LOF by this protocol. |
| `diagnostics.pca_reference_qc` | Native exact Gram/Jacobi PCA fit on `X_target` only; exposes means, loadings, retained ddof-1 eigenvalues, train/query scores and Q/T² without computing exact full-spectrum moments. |

## Frozen reference PCA

```python
import numpy as np
from n4m.roles import method_class

# X_train and X_test are complete finite float64 matrices. Any imputation,
# scaling and sampling protocol must have been fitted on training data alone.
procedure = method_class("diagnostics.pca_reference_qc")(n_components=5)
report = procedure.run(X_test, X_target=X_train)
assert report["exact_full_moments_status"].tolist() == [6]  # not requested

changed_test = procedure.run(100 * X_test + 7, X_target=X_train)
for key in ("reference_mean", "loadings", "eigenvalues", "reference_scores"):
    np.testing.assert_array_equal(report[key], changed_test[key])
```

The existing `FlexiblePCA.fit/transform` remains the reusable fitted estimator.
Its serialized N4ME state is unchanged, including its calibration means and
loadings. Its fast randomized path is retained. The new descriptive procedure
uses the same fitted representation and the existing native exact Gram/Jacobi
kernels, rather than adding another PCA implementation. It returns the retained
statistics needed for domain scoring while the separate exact-moment endpoint
retains its existing semantics and tests.

PCA centers features and does not standardize them implicitly. Q reconstructs
only resolved retained components. T² divides score squares by the corresponding
retained sample eigenvalues and is unavailable if no component is resolved.
Resolution is `16 * max(n_train, p) * eps * lambda_1`. No p-value, confidence
limit or effective full rank is inferred from truncated eigenvalues. The existing
`n4m_stats_pca_monitor` remains a separate out-of-fold empirical-limit protocol.

The computation is checked before allocation against the existing whole-PCA
budget including the eigensolver iteration ceiling, plus the new query/output
passes: 2.4 billion multiply-add equivalents, 1 GiB buffers and 50 million input
cells. A refusal returns `N4M_ERR_UNSUPPORTED` with an actionable explanation.
This is a work/memory bound, not a browser latency promise.

## Availability, masks and physical meaning

Each row signal exposes finite `<signal>_values` and original zero-based
`<signal>_rows`, plus one `<signal>_status` per input row. Missing values are not
inserted as numeric zeros. Median/p95/count and aggregate status are computed
natively; p95 uses type-7 linear interpolation. No eligible observation means no
value/median/p95 entry, with explicit status/count retained.

For mask-aware procedures, the adapter supplies finite X with invalid coordinates
zero-filled and an explicit row-major validity mask. Reference metrics use common
valid bands; derivatives require adjacent valid pairs/triples and do not bridge
missing bands. This differs deliberately from the legacy Datasets imputation and
fixed denominator protocol. LOF and PCA require complete data and a declared
upstream train-fitted missingness policy.

Physical derivatives require a finite strictly monotonic `axis_values` array.
Descending and irregular spacing are supported with signed differences. The
consumer owns the declared axis unit; feature indices are not wavelengths.
True clipping requires explicit sensor bounds/tolerance. Constant spectra do not
establish sensor saturation simply because they occupy an observed extremum.
Sampling/group IDs, technical versus biological repetitions, population identity,
possible causes and calibrated confidence remain Preflight/Domains protocols.

## Examples in the other bindings

```r
reference <- matrix(c(1, 2, 4, 8), nrow = 1)
report <- n4m_run(n4m_spectral_reference_qc(reference_mode = "supplied"),
                  rbind(reference, 2 * reference + 3), X_target = reference)
```

```javascript
const Procedure = methodClass("diagnostics.spectral_reference_qc");
const report = new Procedure({reference_mode: "supplied"})
    .run(queryMatrix, undefined, {XTarget: referenceMatrix});
```

```matlab
report = n4m.procedure('diagnostics.spectral_reference_qc', X, ...
                       struct('reference_mode', 'supplied'), reference);
% Result row indices retain the native zero-based convention.
```

Independent tests compare OLS, RMS/SAM, masks, smooth/rough spectra, injected
spikes/jumps, sensor bounds, physical axes, noise normalization, genuine training
and novelty LOF, and frozen PCA against NumPy/sklearn. A shared transport fixture
is replayed by native Python, R, WASM and Octave. The exhaustive
[61-function Datasets inventory](../datasets_kernel_reuse_inventory.md) identifies
KEEP/DELEGATE/MIGRATE/REPLACE/DEPRECATED decisions; retaining a distinct legacy or
task/identity protocol is not a claim of numerical equivalence.
