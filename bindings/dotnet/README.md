# pls4all — .NET / C# binding

A small .NET 9 class library that wraps libp4a's `p4a_pls_fit_simple`
C ABI via `[DllImport]`. Parity-gated against the shared
cross-binding fixture (`bindings/js/test/parity_fixture.json`) at
machine epsilon (`rmse_rel = 0.0` — bit-exact).

## Layout

```
bindings/dotnet/
├── Pls4all/                 .NET 9 class library
│   ├── Pls4all.csproj
│   └── Pls4all.cs           Public API + P/Invoke declarations
└── TestParity/              Console app for the parity gate
    ├── TestParity.csproj
    └── Program.cs
```

## Build

```bash
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
DOTNET_ROOT=$(dirname $(which dotnet)) \
dotnet build bindings/dotnet/Pls4all/Pls4all.csproj -c Release
```

## Usage

```csharp
using Pls4all;

double[] x = new double[50 * 5]; // row-major n*p
double[] y = new double[50 * 1]; // row-major n*q
// fill x, y ...

var fit = Pls4all.Pls4all.PlsFit(x, y, n: 50, p: 5, q: 1, nComponents: 3);
Console.WriteLine($"version  : {Pls4all.Pls4all.Version()}");
Console.WriteLine($"coefs[0] : {fit.Coefficients[0]}");
```

## Parity gate

```bash
REPO_ROOT=$(pwd) \
DOTNET_ROOT=$(dirname $(which dotnet)) \
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
dotnet run --project bindings/dotnet/TestParity/TestParity.csproj
```

Expected output:

```
Pls4all.Version()        = 0.90.0+abi.1.13.0
Pls4all.AbiVersion()     = 1.13.0
fixture pls4all_version  = 0.90.0+abi.1.13.0
rmse_rel coefficients    = 0.000E+000
rmse_rel x_mean          = 0.000E+000
rmse_rel y_mean          = 0.000E+000
rmse_rel predictions     = 0.000E+000
PARITY GATE PASS
```

## Limitations

* Only the one-shot `PlsFit` is wrapped. A `Predict(model, x)` API
  against a saved model is on the backlog.
* `Span<double>` inputs are pinned via `fixed`; for very large
  matrices, the same `[DllImport]` pattern can be reused with
  pre-pinned arrays from `GC.AllocateUninitializedArray<double>`.
