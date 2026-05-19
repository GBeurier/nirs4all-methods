/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SignalTypeDetector reference implementation.
 *
 * Bit-equivalent to `nirs4all.data.signal_type.SignalTypeDetector.detect`:
 *
 *   1. NaN-skipping min, max, mean, std of X.
 *   2. If the data looks preprocessed (centred, standardised, or
 *      derivative-shaped), short-circuit to UNKNOWN with confidence 0.9.
 *   3. Otherwise score each of the five "raw" signal classes from value-
 *      range / mean heuristics.
 *   4. Optionally fold in a water-band cue at 1450/1940/2500 nm.
 *   5. Pick the argmax. Confidence = best / sum(scores). When confidence
 *      falls below `confidence_threshold`, downgrade to UNKNOWN.
 *
 * The C enum values intentionally use the same integer ordering as the
 * public c4a_signal_type_t enum:
 *
 *      0 UNKNOWN, 1 ABSORBANCE, 2 REFLECTANCE, 3 REFLECTANCE_PERCENT,
 *      4 TRANSMITTANCE, 5 TRANSMITTANCE_PERCENT, 6 KUBELKA_MUNK,
 *      7 LOG_1_R, 8 LOG_1_T
 *
 * The reason string is fixed to the 256-byte caller-owned buffer; it never
 * truncates the trailing NUL.
 */

#include "signal_type_detector.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    K_UNKNOWN               = 0,
    K_ABSORBANCE            = 1,
    K_REFLECTANCE           = 2,
    K_REFLECTANCE_PERCENT   = 3,
    K_TRANSMITTANCE         = 4,
    K_TRANSMITTANCE_PERCENT = 5,
};

static const char* kind_to_string(int32_t k) {
    switch (k) {
        case K_UNKNOWN:               return "unknown";
        case K_ABSORBANCE:            return "absorbance";
        case K_REFLECTANCE:           return "reflectance";
        case K_REFLECTANCE_PERCENT:   return "reflectance%";
        case K_TRANSMITTANCE:         return "transmittance";
        case K_TRANSMITTANCE_PERCENT: return "transmittance%";
        default:                       return "unknown";
    }
}

static double score_reflectance_fraction(double mn, double mx, double mean) {
    double s = 0.0;
    if (mn >= 0.0 && mx <= 1.2) {
        s += 0.5;
        if (mean >= 0.1 && mean <= 0.8) s += 0.3;
        if (mx <= 1.0) s += 0.2;
    }
    return s;
}

static double score_reflectance_percent(double mn, double mx, double mean) {
    double s = 0.0;
    if (mn >= 0.0 && mx > 1.5 && mx <= 120.0) {
        s += 0.5;
        if (mean >= 10.0 && mean <= 80.0) s += 0.3;
        if (mx <= 100.0) s += 0.2;
    }
    return s;
}

static double score_transmittance_fraction(double mn, double mx, double mean) {
    double s = 0.0;
    if (mn >= 0.0 && mx <= 1.2) {
        s += 0.4;
        if (mean >= 0.05 && mean <= 0.5) s += 0.2;
    }
    return s;
}

static double score_transmittance_percent(double mn, double mx, double mean) {
    double s = 0.0;
    if (mn >= 0.0 && mx > 1.5 && mx <= 120.0) {
        s += 0.4;
        if (mean >= 5.0 && mean <= 50.0) s += 0.2;
    }
    return s;
}

static double score_absorbance(double mn, double mx, double mean) {
    double s = 0.0;
    if (mn >= -0.5 && mx >= 0.5 && mx <= 5.0) {
        s += 0.4;
        if (mean >= 0.2 && mean <= 2.0) s += 0.3;
        if (mn >= -0.2) s += 0.2;
        if (mx >= 1.0) s += 0.1;
    }
    return s;
}

static int is_preprocessed(double mn, double mx, double mean, double sd) {
    /* Mean-centred. */
    if (fabs(mean) < 0.01 && sd > 0.1) return 1;
    /* SNV / standardised. */
    if (fabs(sd - 1.0) < 0.1 && fabs(mean) < 0.1) return 1;
    /* Derivative-shaped (signed centred). */
    if (mn < -0.5 && mx < 0.5 && fabs(mean) < 0.01) return 1;
    return 0;
}

/* Water-band cue. Returns nudges added to (absorbance, refl, trans) scores. */
static void analyze_water_bands(const double* X, int64_t rows, int64_t cols,
                                 const double* wavelengths, int64_t wl_length,
                                 double* nudge_abs, double* nudge_refl,
                                 double* nudge_trans) {
    *nudge_abs   = 0.0;
    *nudge_refl  = 0.0;
    *nudge_trans = 0.0;
    if (wavelengths == NULL || wl_length != cols || rows <= 0 || cols <= 0) {
        return;
    }
    /* Pre-compute column means (NaN-skipping). */
    double* mean_spectrum = (double*)malloc((size_t)cols * sizeof(double));
    int*    valid_count   = (int*)malloc((size_t)cols * sizeof(int));
    if (mean_spectrum == NULL || valid_count == NULL) {
        free(mean_spectrum);
        free(valid_count);
        return;
    }
    for (int64_t j = 0; j < cols; ++j) {
        mean_spectrum[j] = 0.0;
        valid_count[j]   = 0;
    }
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            const double v = X[(size_t)i * (size_t)cols + (size_t)j];
            if (!isnan(v)) {
                mean_spectrum[j] += v;
                valid_count[j]   += 1;
            }
        }
    }
    for (int64_t j = 0; j < cols; ++j) {
        if (valid_count[j] > 0) {
            mean_spectrum[j] /= (double)valid_count[j];
        }
    }
    free(valid_count);

    /* Range of wavelengths. */
    double wl_min =  INFINITY;
    double wl_max = -INFINITY;
    for (int64_t j = 0; j < cols; ++j) {
        if (wavelengths[j] < wl_min) wl_min = wavelengths[j];
        if (wavelengths[j] > wl_max) wl_max = wavelengths[j];
    }

    const double bands_nm[3] = {1450.0, 1940.0, 2500.0};
    int peak_count = 0;
    int dip_count  = 0;
    const int64_t window = 10;
    for (int b = 0; b < 3; ++b) {
        const double band = bands_nm[b];
        if (band < wl_min || band > wl_max) continue;
        /* Find closest index. */
        int64_t idx = 0;
        double  best_dist = fabs(wavelengths[0] - band);
        for (int64_t j = 1; j < cols; ++j) {
            const double d = fabs(wavelengths[j] - band);
            if (d < best_dist) {
                best_dist = d;
                idx       = j;
            }
        }
        int64_t start = idx - window;
        int64_t end   = idx + window;
        if (start < 0)    start = 0;
        if (end   > cols) end   = cols;
        double local_sum = 0.0;
        int    local_n   = 0;
        for (int64_t j = start; j < end; ++j) {
            if (!isnan(mean_spectrum[j])) {
                local_sum += mean_spectrum[j];
                local_n   += 1;
            }
        }
        const double local_mean =
            (local_n > 0) ? local_sum / (double)local_n : 0.0;
        const double band_value = mean_spectrum[idx];
        if (band_value > local_mean * 1.05) {
            ++peak_count;
        } else if (band_value < local_mean * 0.95) {
            ++dip_count;
        }
    }
    free(mean_spectrum);

    if (peak_count > dip_count) {
        *nudge_abs   =  0.3;
        *nudge_refl  = -0.1;
        *nudge_trans = -0.1;
    } else if (dip_count > peak_count) {
        *nudge_abs   = -0.1;
        *nudge_refl  =  0.2;
        *nudge_trans =  0.2;
    }
}

c4a_status_t c4a_signal_detect_impl(const double* X,
                                     int64_t rows, int64_t cols,
                                     const double* wavelengths,
                                     int64_t wl_length,
                                     double confidence_threshold,
                                     int32_t* type_out,
                                     double* confidence_out,
                                     char* reason_buf) {
    if (type_out == NULL || confidence_out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    *type_out       = K_UNKNOWN;
    *confidence_out = 0.0;
    if (reason_buf != NULL) {
        reason_buf[0] = '\0';
    }

    /* Empty matrix. */
    if (rows == 0 || cols == 0 || X == NULL) {
        *type_out       = K_UNKNOWN;
        *confidence_out = 0.0;
        if (reason_buf != NULL) {
            snprintf(reason_buf, 256, "Empty data");
        }
        return C4A_OK;
    }

    /* Compute NaN-skipping statistics. */
    double sum      = 0.0;
    double sq_sum   = 0.0;
    double mn       =  INFINITY;
    double mx       = -INFINITY;
    int64_t valid_n = 0;
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            const double v = X[(size_t)i * (size_t)cols + (size_t)j];
            if (isnan(v)) continue;
            ++valid_n;
            sum    += v;
            sq_sum += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    if (valid_n == 0) {
        *type_out       = K_UNKNOWN;
        *confidence_out = 0.0;
        if (reason_buf != NULL) {
            snprintf(reason_buf, 256, "All-NaN data");
        }
        return C4A_OK;
    }
    const double mean = sum / (double)valid_n;
    const double var  = (sq_sum / (double)valid_n) - mean * mean;
    const double sd   = (var > 0.0) ? sqrt(var) : 0.0;

    if (is_preprocessed(mn, mx, mean, sd)) {
        *type_out       = K_UNKNOWN;
        *confidence_out = 0.9;
        if (reason_buf != NULL) {
            snprintf(reason_buf, 256,
                     "Data appears preprocessed (centered/normalized)");
        }
        return C4A_OK;
    }

    /* Score the five raw-signal classes. */
    double scores[6] = {0.0};
    scores[K_REFLECTANCE]           = score_reflectance_fraction(mn, mx, mean);
    scores[K_REFLECTANCE_PERCENT]   = score_reflectance_percent(mn, mx, mean);
    scores[K_TRANSMITTANCE]         = score_transmittance_fraction(mn, mx, mean);
    scores[K_TRANSMITTANCE_PERCENT] = score_transmittance_percent(mn, mx, mean);
    scores[K_ABSORBANCE]            = score_absorbance(mn, mx, mean);

    /* Optional water-band cue. */
    double nudge_abs = 0.0, nudge_refl = 0.0, nudge_trans = 0.0;
    if (wavelengths != NULL && wl_length == cols && rows > 0) {
        analyze_water_bands(X, rows, cols, wavelengths, wl_length,
                            &nudge_abs, &nudge_refl, &nudge_trans);
        /* Python weights nudges by 0.2. */
        scores[K_ABSORBANCE]    += nudge_abs   * 0.2;
        scores[K_REFLECTANCE]   += nudge_refl  * 0.2;
        scores[K_TRANSMITTANCE] += nudge_trans * 0.2;
    }

    /* Argmax over the five candidate types. */
    int32_t best     = K_REFLECTANCE;
    double  best_val = scores[K_REFLECTANCE];
    const int32_t cands[5] = {
        K_REFLECTANCE, K_REFLECTANCE_PERCENT,
        K_TRANSMITTANCE, K_TRANSMITTANCE_PERCENT,
        K_ABSORBANCE
    };
    for (int i = 0; i < 5; ++i) {
        if (scores[cands[i]] > best_val) {
            best     = cands[i];
            best_val = scores[cands[i]];
        }
    }
    double total = 0.0;
    for (int i = 0; i < 5; ++i) {
        total += scores[cands[i]];
    }
    const double confidence = (total > 0.0) ? best_val / total : 0.0;

    /* Build the reason string. */
    if (reason_buf != NULL) {
        snprintf(reason_buf, 256,
                 "Range: [%.3f, %.3f] | Mean: %.3f | Detected: %s | "
                 "Confidence: %.1f%%",
                 mn, mx, mean, kind_to_string(best), confidence * 100.0);
    }

    if (confidence < confidence_threshold) {
        *type_out       = K_UNKNOWN;
        *confidence_out = confidence;
        return C4A_OK;
    }
    *type_out       = best;
    *confidence_out = confidence;
    return C4A_OK;
}
