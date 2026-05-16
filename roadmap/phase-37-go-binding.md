# Phase 37 — Go (cgo) Binding

**Status**: shipped 2026-05-16 · tag `phase-37-go-binding` ·
release `0.88.0+abi.1.13.0`.

## Goal

Ship a Go binding for libp4a backed by the public
`p4a_pls_fit_simple` C ABI helper, gated by the same cross-binding
parity fixture used by WASM / R / Julia / Octave / JNI. The binding
must agree with the native Python reference to machine epsilon.

## Why Go

* Go's `cgo` bridge is built into the standard toolchain — no
  separate FFI library, no preprocessor branching, no codegen tool.
  A self-contained `go build` produces a single static-ish binary
  that loads `libp4a.so` at runtime.
* Go is the de-facto language for ML infrastructure tooling
  (Kubernetes, Prometheus, OpenTelemetry, gRPC). Shipping a Go
  binding makes libp4a callable from data-plane services.

## Architecture

```
bindings/go/
├── go.mod                       module github.com/GBeurier/pls4all/bindings/go
├── pls4all/pls4all.go           Public API (cgo)
└── cmd/test_parity/main.go      Parity gate
```

The package exposes:

* `Version() string` — `"0.88.0+abi.1.13.0"`.
* `AbiVersion() (int, int, int)` — `(major, minor, patch)`.
* `Fit(x, y []float64, n, p, q, nComponents int) (*FitResult, error)` —
  one-shot SIMPLS regression returning the coefficient slice plus
  centring vectors and in-sample predictions.

Go's `[]float64` is row-major and shares the same memory layout as
C's `double*`, so we hand the raw slice pointer to cgo via
`unsafe.Pointer(&x[0])`. No transpose, no temporary buffer.

## Parity gate

```
$ go run ./cmd/test_parity
pls4all.Version()        = 0.88.0+abi.1.13.0
pls4all.AbiVersion()     = 1.13.0
fixture pls4all_version  = 0.88.0+abi.1.13.0
rmse_rel coefficients    = 2.471e-16
rmse_rel x_mean          = 1.208e-16
rmse_rel y_mean          = 2.200e-16
rmse_rel predictions     = 2.822e-16
PARITY GATE PASS
```

All four arrays are within ~2 ULPs of the native Python reference.
The tiny non-zero residual (vs JNI's exact 0.0) is rounding inside
the Python writer / Go reader of the textual JSON fixture, not in
the libp4a computation itself.

## Implementation notes

* **cgo includes**: the `import "C"` cgo prologue includes
  `pls4all/p4a.h` directly. We pass `CGO_CFLAGS` /
  `CGO_LDFLAGS` at build time (set via environment); avoiding hard-
  coded paths so downstream consumers can install libp4a anywhere.
* **Memory ownership**: outputs are Go-allocated `make([]float64,
  ...)` slices, so the Go garbage collector owns them; the
  C side only writes into the pinned-during-call slice memory and
  returns. No `C.free` or finalizer is needed.
* **Stale `go-cgo` 1.11 package**: conda-forge `go-cgo` is an old
  Go 1.11.3 stub. Installing it before installing modern Go pollutes
  the runtime/ source tree with files (`signal_nacl*.go`) Go 1.26+
  no longer recognizes. The fix is to install only the modern `go`
  package and avoid `go-cgo` entirely.

## ABI surface

No new symbols. Uses `p4a_pls_fit_simple` from ABI 1.13.0.

## Release

* Library version: `0.87.0` → `0.88.0` (additive: new binding).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-37-go-binding`.

## Backlog

* `pls4all.Predict(model, x)` against a saved model.
* `pkg-config` integration so `CGO_CFLAGS` / `CGO_LDFLAGS` can be
  resolved automatically once libp4a ships a `.pc` file.
