# LVSE and GCU spectral encoders

ABI 2.7 introduces a shared fitted encoder lifecycle in `decomposition.h`.
Numerics live in `core/spectral_encoding.cpp`; bindings only marshal matrices.
These are the NIRS adaptations of the local-subspace and global nonnegative
representations studied with RamanPFN, not a pretrained foundation model.

```python
from n4m.decomposition import LVSE, GCU

local = LVSE(width=32, rank=2, overlap=0.5).fit(X_train)
Z_test = local.transform(X_test)
A, offset = local.export_linear_operator()
# Z_test == X_test @ A.T + offset

global_factors = GCU(rank=16, max_iter=60).fit(X_train)
H_test = global_factors.transform(X_test)
```

LVSE fits training means and, by default, population standard deviations
(floored at 1e-8) channel by channel. Each local centered block is decomposed
using the native exact Gram SVD. Modes with singular values at or below
`4 * sqrt(machine_epsilon) * largest_singular_value` are zeroed, with the
output shape preserved. This avoids amplifying Gram roundoff in nearly
rank-deficient windows; the same threshold applies to tall and wide blocks. The local rank is capped at
`min(n_train, block_width) - 1`, matching the NIRS research implementation;
singleton trailing blocks are skipped. Without overlap, the final block may
be shorter. With overlap, the last full-width window ends at the final channel.
`standardize=False` gives local orthogonal spectral projectors. Standardized
LVSE instead learns a scaled spectral metric. Overlapping projectors add on
their shared channels; their sum is not an orthogonal projector in general.

`snv=True` first applies per-spectrum SNV with an SD floor of 1e-8. This is
not affine in raw spectra, and affine export rejects it. Without SNV, the
export includes channel scaling and the learned offset, not just SVD vectors.

GCU subtracts training channel minima, divides by the training matrix's global
population SD (floored at 1e-8), and clips negative query values to zero.
It fits NNDSVDa-initialized nonnegative factors by cyclic coordinate descent.
Query coordinates use coordinate descent with the fitted basis held fixed.
Rank is capped at `min(n_train, n_channels)-1`; the iteration and tolerance
settings apply to both fitting and query projection. It is non-affine.

The lab prototype uses a randomized SVD and can subsample to 5,000 fit rows.
This native implementation uses exact SVD and **all rows supplied**. Thus it
implements the same representation but does not promise bitwise parity with
the stochastic prototype. Fit any sampling policy on the training fold only.
Rank-deficient eigenspaces may have nonunique bases; compare projectors or
downstream predictions rather than signs of individual SVD modes.

Fit every encoder inside each training fold, including when selecting its
window, rank or overlap. A pipeline that fits LVSE once before internal model
selection leaks validation spectra into the learned representation.

## AOM integration

All four native calibration regressors accept an optional `lvse` sequence:

```python
from n4m.model_selection.aom_calibration import AOMPLSRegressor

model = AOMPLSRegressor(lvse=[(16, 1, 0), (32, 2, 0.5)]).fit(X_train, y_train)
prediction = model.predict(X_test)
```

Each tuple is `(width, rank, overlap[, standardize])`. These terminals augment
the existing fixed chain bank; they do not replace it. The basis is fitted
after the fixed pre-chain using that fold's training rows. Rank variants reuse
the maximal requested basis per window configuration, with caches bounded to
one branch, fold and current pre-chain. FastAOM propagates its covariance and
low-rank sketch through this training-fitted basis. It does not fit bases on
the sketch or validation rows. Final projected coefficients and offsets fold
back through the fixed chain, preserving the existing compact prediction API.

Raw-branch LVSE models are affine in raw inputs. SNV/MSC AOM branches retain
their existing branch-specific prediction transform. GCU is available as a
standalone encoder, not a fixed linear AOM terminal. Global, adaptive,
residual and mixture LVSE variants are research extensions, not implemented
by these local-training operators.

## Bindings and lifetime

- Python: `n4m.decomposition.LVSE`, `GCU`; `fit`, `transform`, `fit_transform`,
  `close`; LVSE additionally exports an affine operator.
- R: `lvse_fit`, `gcu_fit`, `spectral_transform`, `spectral_export_affine`.
- WASM/TypeScript: `SpectralEncoder({kind: "lvse" | "gcu", ...})`, `fit`,
  `transform`, `exportAffine`, `dispose`.
- MATLAB: `n4m.LVSE`, `n4m.GCU`; `fit`, `transform`, `export_linear_operator`,
  `delete` (the MEX gateway owns and releases native handles).
- C/C++: the six `n4m_decomposition_spectral_*` functions. Input and output
  matrices are stride-aware; caller-owned output buffers stay caller-owned.

Standalone encoder handles are process-local, not serialized model files.
Python rejects copying or pickling fitted encoder handles; use sklearn
`clone` to obtain an unfitted estimator. MATLAB handle IDs are transient
(saved/loaded objects require fitting again).
Do not share a handle across concurrent fits. A failed refit preserves its
previous fitted state. AOM's folded coefficients remain independent of encoder
handles and keep the existing estimator serialization behavior.
