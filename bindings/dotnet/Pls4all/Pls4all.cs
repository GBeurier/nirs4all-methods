// SPDX-License-Identifier: CECILL-2.1
//
// .NET / C# binding around libp4a's `p4a_pls_fit_simple` C ABI helper.
// Parity-gated against the shared cross-binding fixture
// (bindings/js/test/parity_fixture.json) at machine epsilon.
//
// Load the matching libp4a.so/dylib/dll alongside the assembly, or
// set LD_LIBRARY_PATH / PATH / DYLD_LIBRARY_PATH before the CLR
// looks up DllImport targets.

using System;
using System.Runtime.InteropServices;

namespace Pls4all;

/// <summary>
/// Static entry points to the pls4all PLS / NIRS engine.
/// </summary>
public static class Pls4all
{
    private const string LibraryName = "p4a";

    [DllImport(LibraryName, EntryPoint = "p4a_get_version_string")]
    private static extern IntPtr p4a_get_version_string_native();

    [DllImport(LibraryName, EntryPoint = "p4a_get_abi_version_major")]
    private static extern uint p4a_get_abi_version_major_native();

    [DllImport(LibraryName, EntryPoint = "p4a_get_abi_version_minor")]
    private static extern uint p4a_get_abi_version_minor_native();

    [DllImport(LibraryName, EntryPoint = "p4a_get_abi_version_patch")]
    private static extern uint p4a_get_abi_version_patch_native();

    [DllImport(LibraryName, EntryPoint = "p4a_pls_fit_simple")]
    private static extern unsafe int p4a_pls_fit_simple_native(
        double* x, double* y,
        int n, int p, int q, int nComponents,
        double* coefficientsOut,
        double* xMeanOut,
        double* yMeanOut,
        double* predictionsOut);

    /// <summary>Returns the libp4a runtime version, e.g. "0.90.0+abi.1.13.0".</summary>
    public static string Version()
    {
        IntPtr raw = p4a_get_version_string_native();
        return raw == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(raw) ?? string.Empty;
    }

    /// <summary>Returns the libp4a ABI (major, minor, patch).</summary>
    public static (uint Major, uint Minor, uint Patch) AbiVersion()
    {
        return (
            p4a_get_abi_version_major_native(),
            p4a_get_abi_version_minor_native(),
            p4a_get_abi_version_patch_native());
    }

    /// <summary>Bundle of arrays returned by <see cref="PlsFit"/>.</summary>
    public sealed record FitResult(
        int N,
        int P,
        int Q,
        int NComponents,
        double[] Coefficients,   // length P*Q (row-major)
        double[] XMean,          // length P
        double[] YMean,          // length Q
        double[] Predictions);   // length N*Q (row-major)

    /// <summary>
    /// Fit a SIMPLS PLS regression on row-major matrices (x, y).
    /// </summary>
    /// <param name="x">Row-major double matrix, length n*p.</param>
    /// <param name="y">Row-major double matrix, length n*q.</param>
    public static FitResult PlsFit(
        ReadOnlySpan<double> x, ReadOnlySpan<double> y,
        int n, int p, int q, int nComponents)
    {
        if (x.Length != (long)n * p)
        {
            throw new ArgumentException(
                $"x length {x.Length} != n*p ({(long)n * p})", nameof(x));
        }
        if (y.Length != (long)n * q)
        {
            throw new ArgumentException(
                $"y length {y.Length} != n*q ({(long)n * q})", nameof(y));
        }
        if (nComponents < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(nComponents), "must be >= 1");
        }

        var coefs = new double[p * q];
        var xMean = new double[p];
        var yMean = new double[q];
        var preds = new double[n * q];

        int status;
        unsafe
        {
            fixed (double* xp = x)
            fixed (double* yp = y)
            fixed (double* cp = coefs)
            fixed (double* xmp = xMean)
            fixed (double* ymp = yMean)
            fixed (double* pp = preds)
            {
                status = p4a_pls_fit_simple_native(
                    xp, yp, n, p, q, nComponents, cp, xmp, ymp, pp);
            }
        }
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"p4a_pls_fit_simple failed with status {status}");
        }
        return new FitResult(n, p, q, nComponents, coefs, xMean, yMean, preds);
    }
}
