# Phase 7b - Python model wrappers + PLS regression benchmark

Status: shipped locally as `phase-7-comprehensive-benchmark` (rolled up
into the Phase 7 tag because 7b–7e ship together).

Goal: extend the ctypes Python binding past lifecycle / AOM-POP so the
benchmark matrix can drive every shipped PLS regression solver. Add a
NumPy zero-copy `p4a_matrix_view_t` constructor and a `Model` wrapper
over `p4a_model_fit` / `p4a_model_predict_alloc` / `p4a_model_get_array`.

Delivered:

- `bindings/python/src/pls4all/_ffi.py` ctypes prototypes for
  `p4a_model_fit`, `p4a_model_destroy`, `p4a_model_predict`,
  `p4a_model_transform`, `p4a_model_predict_alloc`,
  `p4a_model_transform_alloc`, `p4a_model_get_array`,
  `p4a_model_get_n_components/_features/_targets`, plus `p4a_array_*`.
- `bindings/python/src/pls4all/_model.py`: `Model` owning handle with
  `.fit(ctx, cfg, X, y)`, `.predict(ctx, X)`, `.transform(ctx, X)`,
  `.get_array(ctx, kind)`, `n_components/_features/_targets`
  properties. Non-copyable. NumPy zero-copy `MatrixView` constructor.
  `_Array` wrapper that exposes an `np.ndarray` view sharing the
  C-owned buffer plus a `.copy()` helper.
- `bindings/python/src/pls4all/__init__.py` exports `Model` and
  `ModelArrayKind`.
- `benchmarks/runners/pls_regression.py`: matrix runner over the 9
  shipped PLS regression solvers (NIPALS / orthogonal-scores / SIMPLS /
  kernel-algorithm / wide-kernel / SVD / power / randomized-SVD / PCR)
  × (n, p) cells. Smoke set is committed and gated; full matrix is
  parameterized by `--scale full`.
- Committed CSVs at `benchmarks/results/pls_regression/`: 18 smoke
  rows (9 algos × 2 cells), all with `accuracy_pass=True` against
  scikit-learn `PLSRegression`. Bit-for-bit prediction agreement at
  200x100 and 500x100; PCR has a ~1e-6 difference due to PCA-vs-PCR
  reference implementation choice (within tolerance).

Local outcomes:

- pls4all Python is 4–9x faster end-to-end than sklearn on these small
  cells. The kernel-algorithm and wide-kernel solvers are slower than
  the other 7 — they're designed for different shapes (wide n >> p and
  p >> n respectively).
- `python benchmarks/run.py --benchmark pls_regression --check` passes.

Deferred to a later Python tranche:

- sklearn-compatible `BaseEstimator` / `RegressorMixin` /
  `TransformerMixin` wrappers.
- Cross-validation engine ctypes wrappers.
- Variable-selection ctypes wrappers.
