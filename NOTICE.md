# NOTICE — third-party software bundled with nirs4all-methods

This library is licensed under CECILL-2.1 (see `LICENSE`).
Bundled within the source distribution and/or the compiled
shared library are the following third-party components,
each retaining its original license:

- **FITPACK** (BSD-3-Clause) — vendored under `cpp/src/core/common/_vendored/fitpack` from `https://github.com/scipy/scipy/tree/main/scipy/interpolate/fitpack`.
- **Isolation Forest + LOF + Minimum Covariance Determinant** (BSD-3-Clause) — vendored under `cpp/src/core/filters/_vendored` from `https://github.com/scikit-learn/scikit-learn`.

## Documented external references (not vendored)

The parity gate cross-checks numerical outputs against pinned
versions of the libraries below. These are dependencies of the
CI / test environment, not redistribution targets.

- scipy 1.17.1 (BSD-3-Clause)
- scikit-learn 1.8.0 (BSD-3-Clause)
- numpy 1.26.4 (BSD-3-Clause)
- pywt 1.8.0 (MIT)
- pybaselines 1.2.1 (BSD-3-Clause)
- pls (R) >=2.8 (GPL-2.0-or-later)
- ropls (R/Bioconductor) >=1.34 (CECILL-2.0)
- mixOmics (R/Bioconductor) >=6.28 (GPL-3.0)
- plsVarSel (R) any (GPL-2.0)
- ikpls (Python) any (Apache-2.0)
