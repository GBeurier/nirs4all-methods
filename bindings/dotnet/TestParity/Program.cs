// SPDX-License-Identifier: CECILL-2.1
//
// Cross-binding parity gate for the .NET binding. Builds the same
// deterministic (X, Y) the Python generator writes, fits the model,
// and compares against bindings/js/test/parity_fixture.json. Exits
// non-zero on failure.

using System;
using System.IO;
using System.Text.Json;
using Pls4all;

const double Tolerance = 1e-12;

string repoRoot = Environment.GetEnvironmentVariable("REPO_ROOT")
                   ?? FindRepoRoot()
                   ?? throw new InvalidOperationException(
                       "could not locate repo root containing " +
                       "bindings/js/test/parity_fixture.json");

string fixturePath = Path.Combine(
    repoRoot, "bindings", "js", "test", "parity_fixture.json");
if (!File.Exists(fixturePath))
{
    Console.Error.WriteLine($"Missing fixture: {fixturePath}");
    return 1;
}

string raw = File.ReadAllText(fixturePath);
using JsonDocument doc = JsonDocument.Parse(raw);
JsonElement root = doc.RootElement;

int n = root.GetProperty("n").GetInt32();
int p = root.GetProperty("p").GetInt32();
int q = root.GetProperty("q").GetInt32();
int nComponents = root.GetProperty("n_components").GetInt32();
string fixtureVersion = root.GetProperty("pls4all_version").GetString() ?? string.Empty;
double[] fxCoefs = ParseArray(root, "coefficients");
double[] fxXMean = ParseArray(root, "x_mean");
double[] fxYMean = ParseArray(root, "y_mean");
double[] fxPreds = ParseArray(root, "predictions");

// Build the same deterministic input the Python generator wrote.
var x = new double[n * p];
var y = new double[n * q];
for (int i = 0; i < n; i++)
{
    for (int j = 0; j < p; j++)
    {
        x[i * p + j] = Math.Sin((i + 1) * (j + 1) * 0.3);
    }
    y[i * q + 0] = x[i * p + 0] + 0.5 * x[i * p + 1] - 0.3 * x[i * p + 2];
}

var fit = Pls4all.Pls4all.PlsFit(x, y, n, p, q, nComponents);

double errC = RmsRel(fit.Coefficients, fxCoefs);
double errX = RmsRel(fit.XMean, fxXMean);
double errY = RmsRel(fit.YMean, fxYMean);
double errP = RmsRel(fit.Predictions, fxPreds);

(uint major, uint minor, uint patch) = Pls4all.Pls4all.AbiVersion();
Console.WriteLine($"Pls4all.Version()        = {Pls4all.Pls4all.Version()}");
Console.WriteLine($"Pls4all.AbiVersion()     = {major}.{minor}.{patch}");
Console.WriteLine($"fixture pls4all_version  = {fixtureVersion}");
Console.WriteLine($"rmse_rel coefficients    = {errC:E3}");
Console.WriteLine($"rmse_rel x_mean          = {errX:E3}");
Console.WriteLine($"rmse_rel y_mean          = {errY:E3}");
Console.WriteLine($"rmse_rel predictions     = {errP:E3}");

if (errC > Tolerance || errX > Tolerance || errY > Tolerance || errP > Tolerance)
{
    Console.Error.WriteLine($"PARITY GATE FAIL (tol = {Tolerance:E1})");
    return 1;
}
Console.WriteLine("PARITY GATE PASS");
return 0;

static double[] ParseArray(JsonElement root, string name)
{
    JsonElement arr = root.GetProperty(name);
    int len = arr.GetArrayLength();
    var v = new double[len];
    for (int i = 0; i < len; i++)
    {
        v[i] = arr[i].GetDouble();
    }
    return v;
}

static double RmsRel(double[] a, double[] b)
{
    if (a.Length != b.Length)
    {
        throw new InvalidOperationException(
            $"length mismatch {a.Length} vs {b.Length}");
    }
    double diff = 0.0;
    double reff = 0.0;
    for (int i = 0; i < a.Length; i++)
    {
        double d = a[i] - b[i];
        diff += d * d;
        reff += b[i] * b[i];
    }
    double denom = Math.Max(Math.Sqrt(reff), double.Epsilon);
    return Math.Sqrt(diff) / denom;
}

static string? FindRepoRoot()
{
    string? cur = Directory.GetCurrentDirectory();
    while (cur != null)
    {
        if (File.Exists(Path.Combine(
                cur, "bindings", "js", "test", "parity_fixture.json")))
        {
            return cur;
        }
        cur = Directory.GetParent(cur)?.FullName;
    }
    return null;
}
