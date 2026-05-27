/* SPDX-License-Identifier: CECILL-2.1 */
import io.github.pls4all.Pls4all;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Cross-binding parity gate for the JNI binding. Builds the same
 * deterministic (X, Y) the Python generator writes, calls
 * Pls4all.plsFit(...), then compares against
 * bindings/js/test/parity_fixture.json. Exits with status 1 on
 * failure so CI can gate.
 */
public final class TestParity {

    public static void main(String[] args) throws IOException {
        Path repoRoot = Path.of(System.getProperty("repo.root",
                System.getProperty("user.dir")));
        Path fixturePath = repoRoot
                .resolve("bindings/js/test/parity_fixture.json");

        if (!Files.exists(fixturePath)) {
            System.err.println("Missing fixture: " + fixturePath);
            System.exit(1);
        }

        String raw = Files.readString(fixturePath);
        int n = parseInt(raw, "n");
        int p = parseInt(raw, "p");
        int q = parseInt(raw, "q");
        int nComponents = parseInt(raw, "n_components");
        String fixtureVersion = parseString(raw, "pls4all_version");

        double[] x = new double[n * p];
        double[] y = new double[n * q];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i * p + j] = Math.sin((i + 1) * (j + 1) * 0.3);
            }
            y[i * q + 0] = x[i * p + 0]
                          + 0.5 * x[i * p + 1]
                          - 0.3 * x[i * p + 2];
        }

        Pls4all.FitResult fit = Pls4all.plsFit(x, y, n, p, q, nComponents);

        double[] fxCoefs = parseArray(raw, "coefficients");
        double[] fxXMean = parseArray(raw, "x_mean");
        double[] fxYMean = parseArray(raw, "y_mean");
        double[] fxPreds = parseArray(raw, "predictions");

        double errC = rmsRel(fit.coefficients, fxCoefs);
        double errX = rmsRel(fit.xMean, fxXMean);
        double errY = rmsRel(fit.yMean, fxYMean);
        double errP = rmsRel(fit.predictions, fxPreds);

        System.out.printf("Pls4all.version()        = %s%n", Pls4all.version());
        System.out.printf("fixture pls4all_version  = %s%n", fixtureVersion);
        System.out.printf("rmse_rel coefficients    = %.3e%n", errC);
        System.out.printf("rmse_rel x_mean          = %.3e%n", errX);
        System.out.printf("rmse_rel y_mean          = %.3e%n", errY);
        System.out.printf("rmse_rel predictions     = %.3e%n", errP);

        double tol = 1e-12;
        if (errC > tol || errX > tol || errY > tol || errP > tol) {
            System.err.printf("PARITY GATE FAIL (tol = %.1e)%n", tol);
            System.exit(1);
        }
        System.out.println("PARITY GATE PASS");
    }

    private static double rmsRel(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalStateException(
                "length mismatch " + a.length + " vs " + b.length);
        }
        double diff = 0.0;
        double ref = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            diff += d * d;
            ref += b[i] * b[i];
        }
        return Math.sqrt(diff) / Math.max(Math.sqrt(ref), Double.MIN_NORMAL);
    }

    private static int parseInt(String json, String key) {
        Matcher m = Pattern.compile(
            "\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+)").matcher(json);
        if (!m.find()) {
            throw new IllegalStateException("missing int field: " + key);
        }
        return Integer.parseInt(m.group(1));
    }

    private static String parseString(String json, String key) {
        Matcher m = Pattern.compile(
            "\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]*)\"").matcher(json);
        if (!m.find()) {
            throw new IllegalStateException("missing string field: " + key);
        }
        return m.group(1);
    }

    private static double[] parseArray(String json, String key) {
        Matcher m = Pattern.compile(
            "\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[([^\\]]*)\\]")
            .matcher(json);
        if (!m.find()) {
            throw new IllegalStateException("missing array field: " + key);
        }
        String body = m.group(1).trim();
        if (body.isEmpty()) {
            return new double[0];
        }
        String[] parts = body.split("\\s*,\\s*");
        List<Double> out = new ArrayList<>(parts.length);
        for (String p : parts) {
            String trimmed = p.trim();
            if (!trimmed.isEmpty()) {
                out.add(Double.parseDouble(trimmed));
            }
        }
        double[] arr = new double[out.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = out.get(i);
        }
        return arr;
    }
}
