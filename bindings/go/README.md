# pls4all — Go (cgo) binding

Go wrapper around libp4a's `p4a_pls_fit_simple` C ABI helper.
Parity-gated against the shared cross-binding fixture
(`bindings/js/test/parity_fixture.json`) at machine epsilon
(`rmse_rel ≈ 2e-16` on all four output arrays).

## Layout

```
bindings/go/
├── go.mod                       module github.com/GBeurier/pls4all/bindings/go
├── pls4all/pls4all.go           Public Go API (cgo)
└── cmd/test_parity/main.go      Parity gate vs shared fixture
```

## Build

cgo needs to know where libp4a's header and shared object live.
The simplest invocation, run from the repo root after the CMake
`dev-release` preset is built:

```bash
export REPO_ROOT=$(pwd)
export CGO_CFLAGS="-I${REPO_ROOT}/cpp/include -I${REPO_ROOT}/build/dev-release/generated"
export CGO_LDFLAGS="-L${REPO_ROOT}/build/dev-release/cpp/src -Wl,-rpath,${REPO_ROOT}/build/dev-release/cpp/src -lp4a"
export LD_LIBRARY_PATH="${REPO_ROOT}/build/dev-release/cpp/src"

cd bindings/go
go build ./...
go run ./cmd/test_parity
```

For downstream consumers, libp4a will typically be installed to a
system prefix and `pkg-config` (or direct `-L/-I` flags) will resolve
the same way.

## Usage

```go
package main

import (
    "fmt"
    "github.com/GBeurier/pls4all/bindings/go/pls4all"
)

func main() {
    fmt.Println("libp4a", pls4all.Version())
    x := make([]float64, 50*5) // row-major n*p
    y := make([]float64, 50*1) // row-major n*q
    // fill x, y...

    fit, err := pls4all.Fit(x, y, 50, 5, 1, 3)
    if err != nil {
        panic(err)
    }
    fmt.Println("coefficients:", fit.Coefficients)
}
```

## Parity gate output

```
pls4all.Version()        = 0.95.0+abi.1.13.0
pls4all.AbiVersion()     = 1.13.0
fixture pls4all_version  = 0.95.0+abi.1.13.0
rmse_rel coefficients    = 2.471e-16
rmse_rel x_mean          = 1.208e-16
rmse_rel y_mean          = 2.200e-16
rmse_rel predictions     = 2.822e-16
PARITY GATE PASS
```

## Limitations

* Only the one-shot fit + in-sample prediction path is wrapped. A
  `Predict(model, x)` against a saved model is on the backlog.
* The binding requires cgo. The package will not build under
  `CGO_ENABLED=0` (which is the default for some Go cross-compile
  targets — set it explicitly).
