// SPDX-License-Identifier: CeCILL-2.1
//
// Cross-binding parity gate for the Go binding. Builds the same
// deterministic (X, Y) the Python generator writes, fits the model
// through cgo, and compares against bindings/js/test/parity_fixture.json.
// Exits non-zero on failure so CI can gate.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/GBeurier/pls4all/bindings/go/pls4all"
)

type fixture struct {
	Version      string    `json:"pls4all_version"`
	N            int       `json:"n"`
	P            int       `json:"p"`
	Q            int       `json:"q"`
	NComponents  int       `json:"n_components"`
	Coefficients []float64 `json:"coefficients"`
	XMean        []float64 `json:"x_mean"`
	YMean        []float64 `json:"y_mean"`
	Predictions  []float64 `json:"predictions"`
}

func rmsRel(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("length mismatch %d vs %d", len(a), len(b)))
	}
	var diff, ref float64
	for i := range a {
		d := a[i] - b[i]
		diff += d * d
		ref += b[i] * b[i]
	}
	denom := math.Sqrt(ref)
	if denom < math.SmallestNonzeroFloat64 {
		denom = math.SmallestNonzeroFloat64
	}
	return math.Sqrt(diff) / denom
}

func main() {
	repoRoot := os.Getenv("REPO_ROOT")
	if repoRoot == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(os.Stderr, "cwd: %v\n", err)
			os.Exit(1)
		}
		repoRoot = cwd
	}
	fixturePath := filepath.Join(
		repoRoot, "bindings", "js", "test", "parity_fixture.json")

	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", fixturePath, err)
		os.Exit(1)
	}
	var fx fixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		fmt.Fprintf(os.Stderr, "parse fixture: %v\n", err)
		os.Exit(1)
	}

	// Build the same deterministic input the Python generator wrote.
	x := make([]float64, fx.N*fx.P)
	y := make([]float64, fx.N*fx.Q)
	for i := 0; i < fx.N; i++ {
		for j := 0; j < fx.P; j++ {
			x[i*fx.P+j] = math.Sin(float64((i+1)*(j+1)) * 0.3)
		}
		y[i*fx.Q+0] = x[i*fx.P+0] +
			0.5*x[i*fx.P+1] -
			0.3*x[i*fx.P+2]
	}

	fit, err := pls4all.Fit(x, y, fx.N, fx.P, fx.Q, fx.NComponents)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fit: %v\n", err)
		os.Exit(1)
	}

	errC := rmsRel(fit.Coefficients, fx.Coefficients)
	errX := rmsRel(fit.XMean, fx.XMean)
	errY := rmsRel(fit.YMean, fx.YMean)
	errP := rmsRel(fit.Predictions, fx.Predictions)

	major, minor, patch := pls4all.AbiVersion()
	fmt.Printf("pls4all.Version()        = %s\n", pls4all.Version())
	fmt.Printf("pls4all.AbiVersion()     = %d.%d.%d\n", major, minor, patch)
	fmt.Printf("fixture pls4all_version  = %s\n", fx.Version)
	fmt.Printf("rmse_rel coefficients    = %.3e\n", errC)
	fmt.Printf("rmse_rel x_mean          = %.3e\n", errX)
	fmt.Printf("rmse_rel y_mean          = %.3e\n", errY)
	fmt.Printf("rmse_rel predictions     = %.3e\n", errP)

	const tol = 1e-12
	if errC > tol || errX > tol || errY > tol || errP > tol {
		fmt.Fprintf(os.Stderr, "PARITY GATE FAIL (tol = %.1e)\n", tol)
		os.Exit(1)
	}
	fmt.Println("PARITY GATE PASS")
}
