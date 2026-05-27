# THIRD_PARTY_LICENSES — full attribution for vendored material

## FITPACK  (SPDX: BSD-3-Clause)

**Upstream**: https://github.com/scipy/scipy/tree/main/scipy/interpolate/fitpack
**Vendored at**: `cpp/src/core/common/_vendored/fitpack`
**Notes**: Vendored from SciPy's interpolate/fitpack/. Fortran spline kernels. CECILL-2.1 carrier is compatible because BSD-3 is permissive.

```
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Isolation Forest + LOF + Minimum Covariance Determinant  (SPDX: BSD-3-Clause)

**Upstream**: https://github.com/scikit-learn/scikit-learn
**Vendored at**: `cpp/src/core/filters/_vendored`
**Notes**: Reimplemented from scikit-learn algorithms; no scikit-learn source copied verbatim. Treat as 'derived' under BSD-3 attribution.

```
Vendored algorithm reimplementations — derived from scikit-learn.
================================================================

The C implementations of IsolationForest, Local Outlier Factor (LOF), and
Minimum Covariance Determinant (MCD) under this directory are NOT verbatim
copies of scikit-learn source code; they are independent C reimplementations
written specifically for the n4m / nirs4all-methods library, following the
algorithmic descriptions in:

  - scikit-learn library (BSD-3-Clause)
    https://github.com/scikit-learn/scikit-learn

  - Original references:
    Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. ICDM.
    Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF.
    Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the
        minimum covariance determinant estimator. Technometrics, 41(3).

In recognition of the BSD-3-Clause license that governs scikit-learn (the
algorithmic reference), the following attribution is provided per BSD-3
clause 2:

  This product includes software developed by:
    - scikit-learn developers (https://scikit-learn.org/) — algorithmic
      descriptions of IsolationForest, LOF, and MCD.

The n4m source files in this directory are licensed under CECILL-2.1
(see the repository-level LICENSE file). This NOTICE serves as
clarification that the algorithms are derived from BSD-3-licensed
reference implementations; the C code itself is original.
```


## NumPy random (PCG64 / SeedSequence / Ziggurat / bounded + choice samplers)  (SPDX: BSD-3-Clause)

**Upstream**: https://github.com/numpy/numpy (numpy/random)
**Reproduced in**: `cpp/src/core/common/rng_pcg64.c`, `ziggurat*.{c,h}`,
`cpp/src/core/augmentation/aug_rng_utils.c`
**Notes**: The PCG64 bit generator, the NumPy `SeedSequence` pool expansion,
the Ziggurat standard-normal sampler and its constants, and the bounded-integer
and `choice(replace=False)` samplers reproduce NumPy's `numpy.random`
algorithms and constants for bit-exact parity (verified by the parity
fixtures + `n4m_internal_tests`). They are independent C reimplementations —
no NumPy source is copied verbatim — derived under BSD-3 attribution. The PCG
family is by Melissa O'Neill (https://www.pcg-random.org/).

```
Copyright (c) 2005-2024, NumPy Developers.
Licensed under the BSD-3-Clause license; the algorithms reproduced here are
attributed to the NumPy project. The n4m source files are CECILL-2.1 (see the
repository LICENSE); BSD-3 is permissive and compatible.
```
