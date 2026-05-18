// SPDX-License-Identifier: CECILL-2.1
//
// Idiomatic .NET tier-2 wrapper around Pls4all.Pls4all.Fit. Provides a
// ML.NET-style estimator with Fit / Predict / Score methods on a
// PLSRegression class.
//
// Example:
//
//     var model = new Pls4all.PLSRegression(nComponents: 5);
//     model.Fit(X, y, n, p);
//     var preds = model.Predict(Xnew, nNew, p);
//     double r2 = model.Score(Xtest, yTest, nTest, p);

using System;

namespace Pls4all;

public sealed class PLSRegression
{
    public int NComponents { get; }
    public double[]? Coefficients { get; private set; }
    public double[]? XMean { get; private set; }
    public double[]? YMean { get; private set; }
    public int NFeaturesIn { get; private set; }
    public int NTargets { get; private set; }

    public PLSRegression(int nComponents = 2)
    {
        NComponents = nComponents;
    }

    /// <summary>
    /// Train the model in-place. X is row-major (n × p), y is row-major
    /// (n × q) or length-n when q == 1.
    /// </summary>
    public void Fit(double[] X, double[] y, int n, int p)
    {
        if (n <= 0 || p <= 0)
            throw new ArgumentException("n and p must be positive.");
        if (X.Length != n * p)
            throw new ArgumentException(
                $"X.Length ({X.Length}) must equal n*p ({n * p}).");
        if (y.Length == 0 || y.Length % n != 0)
            throw new ArgumentException(
                $"y.Length ({y.Length}) must be a positive multiple of n ({n}).");
        int q = y.Length / n;
        var res = Pls4all.PlsFit(X, y, n, p, q, NComponents);
        Coefficients = (double[])res.Coefficients.Clone();
        XMean = (double[])res.XMean.Clone();
        YMean = (double[])res.YMean.Clone();
        NFeaturesIn = p;
        NTargets = q;
    }

    /// <summary>
    /// Predict on new X. Y_pred = (X - x_mean) @ coef + y_mean.
    /// </summary>
    public double[] Predict(double[] X, int nPred, int p)
    {
        if (Coefficients is null || XMean is null || YMean is null)
            throw new InvalidOperationException("Model is not fitted.");
        if (p != NFeaturesIn)
            throw new ArgumentException(
                $"X has {p} features but model was fitted with {NFeaturesIn}.");

        int q = NTargets;
        var preds = new double[nPred * q];
        for (int i = 0; i < nPred; i++)
        {
            for (int t = 0; t < q; t++)
            {
                double sum = YMean[t];
                for (int j = 0; j < p; j++)
                {
                    double centered = X[i * p + j] - XMean[j];
                    sum += centered * Coefficients[j * q + t];
                }
                preds[i * q + t] = sum;
            }
        }
        return preds;
    }

    /// <summary>
    /// R² coefficient of determination, uniform-averaged across targets.
    /// </summary>
    public double Score(double[] X, double[] y, int n, int p)
    {
        var preds = Predict(X, n, p);
        int q = NTargets;
        double r2Sum = 0;
        for (int t = 0; t < q; t++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += y[i * q + t];
            mean /= n;
            double ssRes = 0, ssTot = 0;
            for (int i = 0; i < n; i++)
            {
                double dr = preds[i * q + t] - y[i * q + t];
                double dt = y[i * q + t] - mean;
                ssRes += dr * dr;
                ssTot += dt * dt;
            }
            if (ssTot != 0 && !double.IsNaN(ssTot))
                r2Sum += 1 - ssRes / ssTot;
        }
        return r2Sum / q;
    }
}
