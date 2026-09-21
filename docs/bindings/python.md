# Python binding

`n4m` is the full current ctypes binding over **libn4m (ABI 2.5)**. It exposes
ABI introspection, context/config lifecycles, and the method surface through
role subpackages that mirror the `n4m.<role>` namespace. See
`bindings/python/README.md` for installation and loader rules, and the
[ABI 2.0 migration guide](../MIGRATION_ABI2.md) for the old→new mapping.

## Hello-version

```python
import n4m

print(n4m.abi_version())          # (2, 5, 0)

with n4m.Context() as ctx:
    ctx.seed = 42
    assert ctx.seed == 42
```

## Role packages (ABI 2.x)

`n4m.__init__` exposes only metadata/helpers and the role subpackages. Public
classes use plain role names (not `Native*`):

```python
from n4m.estimators.regression.regularized import Ridge, RidgePLS
from n4m.estimators.regression.latent import PLS, PCR
from n4m.transform.scatter import SNV, MSC
from n4m.transform.smoothing import SavitzkyGolay
from n4m.feature_selection.wrapper import CARS
from n4m.model_selection.splitters import KennardStone
from n4m.domain_adaptation.orthogonalization import EPO
from n4m.augmentation.noise import GaussianAdditiveNoise
from n4m.ensemble import AOMRidgeBlender
from n4m.compose.aom_superblock import AOMRidgePLSSuperblock
from n4m.decomposition import FlexiblePCA
```

The estimators/transformers are sklearn-compatible with zero-copy NumPy
`n4m_matrix_view_t` round-trips. The separately shipped `pls4all` package
remains an active slim PLS-focused compatibility subset over the same ABI-2
library. Method pages use `n4m` imports only when their exact current public
role export was verified from source; they do not substitute a `pls4all` route
for a full `n4m` method.

For verified migration tooling, `pls4all.export_linear_predictor_n4mm(...)`
constructs a PREDICT-only N4MM from finite affine coefficients and intercepts.
It does not deserialize a foreign pickle/joblib model or retrain; the caller
must first attest that the source predictor is exactly affine.  The imported
model intentionally does not expose a latent PLS transform.
