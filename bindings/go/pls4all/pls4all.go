// SPDX-License-Identifier: CECILL-2.1

// Package pls4all wraps the libp4a C ABI helper p4a_pls_fit_simple
// for Go consumers. The package is parity-gated against the shared
// cross-binding fixture (bindings/js/test/parity_fixture.json) at
// machine epsilon — see bindings/go/cmd/test_parity.
//
// The binding uses cgo. At build time set CGO_CFLAGS and CGO_LDFLAGS
// (or pkg-config) so that libp4a's headers and shared library are
// reachable:
//
//	CGO_CFLAGS="-I/path/to/cpp/include -I/path/to/build/generated"
//	CGO_LDFLAGS="-L/path/to/build/cpp/src -Wl,-rpath,/path/to/build/cpp/src -lp4a"
//	go build ./...
//
// At run time, libp4a.so must be reachable through LD_LIBRARY_PATH or
// the rpath baked in via -Wl,-rpath.
package pls4all

/*
#include <stdlib.h>
#include "pls4all/p4a.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Version returns the libp4a runtime version, e.g. "0.88.0+abi.1.13.0".
func Version() string {
	return C.GoString(C.p4a_get_version_string())
}

// AbiVersion returns the libp4a ABI {major, minor, patch}.
func AbiVersion() (int, int, int) {
	return int(C.p4a_get_abi_version_major()),
		int(C.p4a_get_abi_version_minor()),
		int(C.p4a_get_abi_version_patch())
}

// FitResult is the bundle returned by Fit. All slices are row-major.
//
//	Coefficients has length P*Q
//	XMean        has length P
//	YMean        has length Q
//	Predictions  has length N*Q
type FitResult struct {
	N            int
	P            int
	Q            int
	NComponents  int
	Coefficients []float64
	XMean        []float64
	YMean        []float64
	Predictions  []float64
}

// Fit fits a SIMPLS PLS regression on the row-major matrices (x, y).
// x must have length n*p; y must have length n*q. The returned
// FitResult owns all output slices.
func Fit(x, y []float64, n, p, q, nComponents int) (*FitResult, error) {
	if x == nil {
		return nil, errors.New("pls4all.Fit: x must not be nil")
	}
	if y == nil {
		return nil, errors.New("pls4all.Fit: y must not be nil")
	}
	if int64(n)*int64(p) != int64(len(x)) {
		return nil, fmt.Errorf(
			"pls4all.Fit: len(x)=%d but n*p=%d", len(x), n*p)
	}
	if int64(n)*int64(q) != int64(len(y)) {
		return nil, fmt.Errorf(
			"pls4all.Fit: len(y)=%d but n*q=%d", len(y), n*q)
	}
	if nComponents < 1 {
		return nil, fmt.Errorf(
			"pls4all.Fit: nComponents must be >= 1, got %d", nComponents)
	}

	coefs := make([]float64, p*q)
	xMean := make([]float64, p)
	yMean := make([]float64, q)
	preds := make([]float64, n*q)

	status := C.p4a_pls_fit_simple(
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&y[0])),
		C.int32_t(n), C.int32_t(p), C.int32_t(q),
		C.int32_t(nComponents),
		(*C.double)(unsafe.Pointer(&coefs[0])),
		(*C.double)(unsafe.Pointer(&xMean[0])),
		(*C.double)(unsafe.Pointer(&yMean[0])),
		(*C.double)(unsafe.Pointer(&preds[0])),
	)
	if status != 0 {
		return nil, fmt.Errorf(
			"pls4all.Fit: p4a_pls_fit_simple failed with status %d",
			int(status))
	}
	return &FitResult{
		N: n, P: p, Q: q, NComponents: nComponents,
		Coefficients: coefs,
		XMean:        xMean,
		YMean:        yMean,
		Predictions:  preds,
	}, nil
}
