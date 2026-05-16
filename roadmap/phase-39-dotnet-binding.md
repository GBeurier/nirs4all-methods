# Phase 39 — .NET / C# Binding

**Status**: shipped 2026-05-16 · tag `phase-39-dotnet-binding` ·
release `0.90.0+abi.1.13.0`.

## Goal

Ship a .NET 9 class library wrapping libp4a via `[DllImport]`. Gated
by the shared cross-binding parity fixture. Bit-exact agreement with
the native Python reference is the target.

## Why .NET

* .NET is the lingua franca of enterprise scientific computing —
  Excel add-ins, Power BI custom visuals, Unity-based simulations,
  Avalonia desktop apps. A first-party `[DllImport]` binding lets
  any of these consume libp4a without a separate FFI wrapper.
* P/Invoke is built into the BCL: zero extra dependencies, native
  marshalling, `Span<double>` interop via `fixed`.

## Architecture

```
bindings/dotnet/
├── Pls4all/                 net9.0 class library
│   ├── Pls4all.csproj
│   └── Pls4all.cs           public API + DllImport block
└── TestParity/              console app for the parity gate
    ├── TestParity.csproj
    └── Program.cs
```

Public API:
* `static string Pls4all.Pls4all.Version()` —
  `"0.90.0+abi.1.13.0"`.
* `static (uint, uint, uint) Pls4all.Pls4all.AbiVersion()` —
  `(1, 13, 0)`.
* `static FitResult Pls4all.Pls4all.PlsFit(ReadOnlySpan<double> x,
  ReadOnlySpan<double> y, int n, int p, int q, int nComponents)` —
  one-shot SIMPLS regression returning a `FitResult` record (a
  modern C# record with positional coefficients / xMean / yMean /
  predictions).

`Span<double>` inputs are pinned with `fixed` to hand the raw
pointer to libp4a without copying. Outputs are GC-owned managed
arrays.

## Parity gate

```
$ dotnet run --project bindings/dotnet/TestParity/TestParity.csproj
Pls4all.Version()        = 0.90.0+abi.1.13.0
Pls4all.AbiVersion()     = 1.13.0
fixture pls4all_version  = 0.90.0+abi.1.13.0
rmse_rel coefficients    = 0.000E+000
rmse_rel x_mean          = 0.000E+000
rmse_rel y_mean          = 0.000E+000
rmse_rel predictions     = 0.000E+000
PARITY GATE PASS
```

Same bit-exact result as JNI and Rust — both .NET `double[]` and the
C ABI are IEEE 754 doubles in identical row-major layout.

## Implementation notes

* **System.Text.Json**: used in the parity test for the fixture
  reader. No third-party dependency.
* **`Marshal.PtrToStringUTF8`**: libp4a returns UTF-8 C strings; the
  modern marshalling helper handles this directly. The older
  `Marshal.PtrToStringAnsi` would mis-decode multibyte characters.
* **`AllowUnsafeBlocks=true`**: the P/Invoke signatures use raw
  `double*` to avoid the `[In]/[Out]` marshalling overhead. The
  `unsafe` block scope is tiny (six `fixed` pins around the call).
* **net9.0**: the conda-forge SDK is 9.0.203 only — no 8.0 runtime
  installed. Targeting net8 would require an extra `dotnet-runtime-
  8.0` package. net9 keeps the dependency footprint small.

## ABI surface

No new symbols. Uses `p4a_pls_fit_simple` from ABI 1.13.0.

## Release

* Library version: `0.89.0` → `0.90.0` (additive: new binding).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-39-dotnet-binding`.

## Backlog

* `Pls4all.Predict(model, x)` against a saved model.
* NuGet package metadata is in place; publishing is gated on us
  also shipping prebuilt `libp4a.so/dylib/dll` runtimes for the
  three major platforms.
