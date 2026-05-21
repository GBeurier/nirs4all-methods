# Pls4all.jl

Julia bindings for the `pls4all` C ABI.

## Install from this repository

```julia
using Pkg
Pkg.add(url="https://github.com/GBeurier/pls4all.git",
        subdir="bindings/julia/Pls4all.jl")
```

Until `Pls4all_jll` is registered, point the binding at a local native
library before loading the package:

```julia
ENV["PLS4ALL_LIB"] = "/absolute/path/to/libp4a.so"      # Linux
# ENV["PLS4ALL_LIB"] = "/absolute/path/to/libp4a.dylib" # macOS
# ENV["PLS4ALL_LIB"] = "C:\\path\\to\\p4a.dll"          # Windows

using Pls4all
```

`PLS4ALL_LIB_PATH` is accepted as an alias for compatibility with the
Python binding.

## Usage

```julia
X = randn(50, 5)
y = X[:, 1] .+ 0.5 .* X[:, 2]

model = Pls4all.pls_fit(X, reshape(y, :, 1); n_components=3)
model.coefficients
model.predictions
```

Tables.jl inputs are accepted by the low-level fit helper and the
sklearn-style wrapper. Any table implementing the Tables.jl interface
works, including named tuples of vectors, DataFrames, CSV.File and Arrow
tables:

```julia
tbl = (
    x1 = randn(50),
    x2 = randn(50),
    x3 = randn(50),
)
y = 0.7 .* tbl.x1 .- 0.2 .* tbl.x3

fit = Pls4all.pls_fit(tbl, y; n_components=2)
fit.feature_names

X = Pls4all.table_matrix(tbl; columns=[:x1, :x3])
Y = Pls4all.response_matrix(y)
```

The sklearn-style helper lives in `Pls4all.Sklearn`:

```julia
using Pls4all.Sklearn

reg = PLSRegression(n_components=3)
fit!(reg, tbl, y)
predict(reg, tbl)
score(reg, tbl, y)
```

MLJModelInterface-compatible regressors are exposed at the package top
level:

```julia
using MLJModelInterface

model = Pls4all.PLSRegressor(n_components=3)
fitresult, cache, report = MLJModelInterface.fit(model, 0, tbl, y)
yhat = MLJModelInterface.predict(model, fitresult, tbl)
params = MLJModelInterface.fitted_params(model, fitresult)

pcr = Pls4all.PCRRegressor(n_components=3)
opls = Pls4all.OPLSRegressor(n_components=1)
```

The low-level binding also exposes the full public method surface from
`p4a_method_result_t` through `src/Methods.jl`, including diagnostics,
monitoring, multi-block methods, AOM/POP selection, and variable selectors.

## Test

```bash
cd bindings/julia/Pls4all.jl
PLS4ALL_LIB=/absolute/path/to/libp4a.so julia --project=. -e 'using Pkg; Pkg.test()'
```

The test reuses `bindings/js/test/parity_fixture.json` and checks
coefficients, means, predictions, Tables.jl inputs, MLJModelInterface
regressors, and the method binding surface against the native library.

The Julia binding is also wired into the cross-binding benchmark
orchestrator:

```bash
python benchmarks/cross_binding/orchestrator.py \
  --algorithms sparse_simpls --sizes 20x6 --threads 1 \
  --only julia --n-runs 2 --reference-backends none \
  --libp4a-build dev-release
```

## Registration

The package is intended for Julia General registration from the monorepo
subdirectory:

```text
@JuliaRegistrator register subdir=bindings/julia/Pls4all.jl
```

The current package UUID is `773cca83-73c7-4c8b-9de3-b5f2b2ce6491`.
