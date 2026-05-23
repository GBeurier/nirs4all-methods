// Smoke test for the tier-2 PLSRegression Go wrapper.
//
// Run with the same CGO env vars as the rest of the Go binding:
//
//	CC=gcc CGO_ENABLED=1 \
//	    CGO_CFLAGS="-I$REPO/cpp/include -I$REPO/build/dev-release/generated" \
//	    CGO_LDFLAGS="-L$REPO/build/dev-release/cpp/src -Wl,-rpath,$REPO/build/dev-release/cpp/src -lp4a" \
//	    go run ./bindings/go/cmd/test_sklearn
package main

import (
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/GBeurier/pls4all/bindings/go/pls4all"
)

func main() {
	// Deterministic synthetic regression: y = 2*x[2] - x[5] + 0.5*x[8] + noise.
	rng := rand.New(rand.NewPCG(0, 0))
	const n, p = 80, 10
	x := make([]float64, n*p)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		y[i] = 2.0*x[i*p+2] - x[i*p+5] + 0.5*x[i*p+8] +
			0.05*rng.NormFloat64()
	}

	m := pls4all.NewPLSRegression(5)
	if err := m.Fit(x, y, n, p); err != nil {
		fmt.Println("Fit failed:", err)
		os.Exit(1)
	}

	preds, err := m.Predict(x, n, p)
	if err != nil {
		fmt.Println("Predict failed:", err)
		os.Exit(1)
	}

	r2, _ := m.Score(x, y, n, p)
	fmt.Printf("Fit OK   n_features_in=%d  n_targets=%d\n",
		m.NFeaturesIn, m.NTargets)
	fmt.Printf("R² in-sample = %.4f  (preds len=%d)\n", r2, len(preds))

	// Bit-exact tier1 vs tier2.
	res, err := pls4all.Fit(x, y, n, p, 1, 5)
	if err != nil {
		fmt.Println("tier-1 Fit failed:", err)
		os.Exit(1)
	}
	maxDiff := 0.0
	for i := 0; i < n; i++ {
		d := preds[i] - res.Predictions[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	fmt.Printf("max abs diff tier1 vs tier2 = %g\n", maxDiff)
	if maxDiff > 1e-10 {
		fmt.Println("FAIL: tier1/tier2 disagree")
		os.Exit(1)
	}

	// Shape validation rejection.
	bad := make([]float64, 20)
	if err := m.Fit(bad, y, n, p); err == nil {
		fmt.Println("FAIL: expected shape-validation error")
		os.Exit(1)
	}
	fmt.Println("Shape rejection OK")

	if r2 < 0.9 {
		fmt.Printf("FAIL: expected R² >= 0.9 on synthetic, got %.4f\n", r2)
		os.Exit(1)
	}
	fmt.Println("\n=== ALL PASSED ===")
}
