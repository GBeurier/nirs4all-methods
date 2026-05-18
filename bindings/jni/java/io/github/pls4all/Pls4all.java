/* SPDX-License-Identifier: CECILL-2.1 */
package io.github.pls4all;

/**
 * pls4all — desktop JNI binding around the public C ABI helper
 * {@code p4a_pls_fit_simple}. Loads {@code libp4a_jni} on first use;
 * the {@code -Djava.library.path=...} JVM option (or the
 * {@code PLS4ALL_JNI_LIB} env var pointing at the .so) tells the
 * JVM where the shared library lives.
 *
 * <p>This is the desktop JNI surface. Android NDK packaging is a
 * follow-up phase that wraps the same .so + Java class in an AAR.
 */
public final class Pls4all {

    static {
        String overridePath = System.getenv("PLS4ALL_JNI_LIB");
        if (overridePath != null && !overridePath.isEmpty()) {
            System.load(overridePath);
        } else {
            System.loadLibrary("p4a_jni");
        }
    }

    private Pls4all() {
        /* static-only — keep instances out. */
    }

    /** Returns the libp4a runtime version, e.g. {@code "0.87.0+abi.1.13.0"}. */
    public static native String version();

    /** Returns the libp4a ABI {major, minor, patch}. */
    public static native int[] abiVersion();

    /**
     * Holder for the four arrays returned by {@link #plsFit(double[][], double[][], int)}.
     * All matrices are stored row-major. The shapes are:
     * <ul>
     *   <li>{@code coefficients}: {@code p × q} (flattened length {@code p*q})</li>
     *   <li>{@code xMean}: length {@code p}</li>
     *   <li>{@code yMean}: length {@code q}</li>
     *   <li>{@code predictions}: {@code n × q} (flattened length {@code n*q})</li>
     * </ul>
     */
    public static final class FitResult {
        public final int n;
        public final int p;
        public final int q;
        public final int nComponents;
        public final double[] coefficients;
        public final double[] xMean;
        public final double[] yMean;
        public final double[] predictions;

        public FitResult(int n, int p, int q, int nComponents,
                         double[] coefficients, double[] xMean,
                         double[] yMean, double[] predictions) {
            this.n = n;
            this.p = p;
            this.q = q;
            this.nComponents = nComponents;
            this.coefficients = coefficients;
            this.xMean = xMean;
            this.yMean = yMean;
            this.predictions = predictions;
        }
    }

    /**
     * Fit a SIMPLS PLS regression on (X, Y).
     *
     * @param x           n × p row-major double matrix (flattened length {@code n*p}).
     * @param y           n × q row-major double matrix (flattened length {@code n*q}).
     * @param n           number of samples.
     * @param p           number of features.
     * @param q           number of targets.
     * @param nComponents PLS latent component count.
     * @return            populated {@link FitResult}.
     */
    public static FitResult plsFit(double[] x, double[] y,
                                    int n, int p, int q, int nComponents) {
        if (x == null || y == null) {
            throw new IllegalArgumentException("X and Y must not be null");
        }
        if ((long) n * p != x.length) {
            throw new IllegalArgumentException(
                "X length (" + x.length + ") != n*p (" + (long) n * p + ")");
        }
        if ((long) n * q != y.length) {
            throw new IllegalArgumentException(
                "Y length (" + y.length + ") != n*q (" + (long) n * q + ")");
        }
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be >= 1");
        }

        double[] coefs = new double[p * q];
        double[] xMean = new double[p];
        double[] yMean = new double[q];
        double[] preds = new double[n * q];

        int status = plsFitNative(x, y, n, p, q, nComponents,
                                   coefs, xMean, yMean, preds);
        if (status != 0) {
            throw new RuntimeException(
                "p4a_pls_fit_simple failed with status " + status);
        }
        return new FitResult(n, p, q, nComponents,
                              coefs, xMean, yMean, preds);
    }

    private static native int plsFitNative(double[] x, double[] y,
                                            int n, int p, int q,
                                            int nComponents,
                                            double[] coefficients,
                                            double[] xMean,
                                            double[] yMean,
                                            double[] predictions);
}
