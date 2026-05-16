// SPDX-License-Identifier: CeCILL-2.1
//
// Idiomatic Go tier-2 wrapper around pls4all.Fit. Mirrors the
// gonum/learn-style estimator surface: Fit / Predict / Score on a
// receiver struct.
//
// Example:
//
//	m := pls4all.NewPLSRegression(5)  // n_components
//	if err := m.Fit(X, y, n, p); err != nil { ... }
//	preds, err := m.Predict(Xnew, len(Xnew)/p, p)
//	r2 := m.Score(Xtest, ytest, ntest, p)

package pls4all

import (
	"errors"
	"fmt"
	"math"
)

// PLSRegression holds the fitted PLS coefficients + preprocessing
// stats, with a Fit/Predict/Score surface matching the gonum / learn
// idiom.
type PLSRegression struct {
	NComponents int

	// Fitted state (populated by Fit).
	Coefficients []float64 // row-major (p × q)
	XMean        []float64 // (p,)
	YMean        []float64 // (q,)
	NFeaturesIn  int
	NTargets     int
}

// NewPLSRegression returns an unfitted PLS regression with the given
// component count.
func NewPLSRegression(nComponents int) *PLSRegression {
	return &PLSRegression{NComponents: nComponents}
}

// Fit trains the model in-place on (X, y). X is row-major (n × p),
// y is row-major (n × q) or a length-n vector when q == 1.
func (m *PLSRegression) Fit(X, y []float64, n, p int) error {
	if n <= 0 || p <= 0 {
		return errors.New("Fit: n and p must be positive")
	}
	if len(X) != n*p {
		return fmt.Errorf("Fit: len(X)=%d must equal n*p=%d", len(X), n*p)
	}
	if len(y) == 0 || len(y)%n != 0 {
		return fmt.Errorf("Fit: len(y)=%d must be a positive multiple of n=%d",
			len(y), n)
	}
	q := len(y) / n
	res, err := Fit(X, y, n, p, q, m.NComponents)
	if err != nil {
		return err
	}
	m.Coefficients = append([]float64(nil), res.Coefficients...)
	m.XMean = append([]float64(nil), res.XMean...)
	m.YMean = append([]float64(nil), res.YMean...)
	m.NFeaturesIn = p
	m.NTargets = q
	return nil
}

// Predict returns predictions for X (row-major, n_pred × p).
// Following the C ABI convention (cpp/src/core/model.cpp::
// fill_prediction): Y_pred = (X - x_mean) @ coef + y_mean.
func (m *PLSRegression) Predict(X []float64, nPred, p int) ([]float64, error) {
	if m.Coefficients == nil {
		return nil, errors.New("Predict: model is not fitted")
	}
	if p != m.NFeaturesIn {
		return nil, errors.New(
			"Predict: feature count mismatch with fitted model")
	}
	preds := make([]float64, nPred*m.NTargets)
	q := m.NTargets
	for i := 0; i < nPred; i++ {
		for t := 0; t < q; t++ {
			sum := m.YMean[t]
			for j := 0; j < p; j++ {
				centered := X[i*p+j] - m.XMean[j]
				sum += centered * m.Coefficients[j*q+t]
			}
			preds[i*q+t] = sum
		}
	}
	return preds, nil
}

// Score returns the R² coefficient of determination on (X, y).
func (m *PLSRegression) Score(X, y []float64, n, p int) (float64, error) {
	preds, err := m.Predict(X, n, p)
	if err != nil {
		return 0, err
	}
	q := m.NTargets
	// uniform_average across targets.
	var r2sum float64
	for t := 0; t < q; t++ {
		var mean float64
		for i := 0; i < n; i++ {
			mean += y[i*q+t]
		}
		mean /= float64(n)
		var ssRes, ssTot float64
		for i := 0; i < n; i++ {
			dr := preds[i*q+t] - y[i*q+t]
			dt := y[i*q+t] - mean
			ssRes += dr * dr
			ssTot += dt * dt
		}
		if ssTot == 0 || math.IsNaN(ssTot) {
			continue
		}
		r2sum += 1 - ssRes/ssTot
	}
	return r2sum / float64(q), nil
}
