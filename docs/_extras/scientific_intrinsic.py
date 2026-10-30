"""Native intrinsic diagnostics: explicit protocols and source-backed ownership."""
_REPO = "https://github.com/GBeurier/nirs4all-methods/blob/main/"
_SOURCE = _REPO + "cpp/src/core/estimator/intrinsic_diagnostics.cpp"
_TEST = _REPO + "bindings/python/tests/test_intrinsic_diagnostics.py"
_CONTRACT = _REPO + "docs/intrinsic_diagnostics_contract.v1.json"


def entry(title, paper, principle, uses, limits, implementation):
    return dict(title=title, paper=paper, principle=principle, use_cases=uses,
                limitations=limits, implementation=implementation,
                provenance=_SOURCE + "; " + _TEST + "; " + _CONTRACT)


SCIENTIFIC_CONTENT = {
    "diagnostics_spectral_reference_qc": entry(
        "Declared-reference spectral RMS, SAM and affine residuals",
        "This descriptive protocol fixes the reference, valid coordinates, OLS normalization "
        "and finite-output availability explicitly. The least-squares reference definition is "
        "documented by NumPy: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html.",
        "Compute RMS difference and arccos of the normalized inner product on common valid "
        "bands. Regress each spectrum against the reference with an intercept, then report "
        "offset, gain and residual RMS with denominator equal to the eligible coordinate count. "
        "Median and type-7 p95 summaries, including absolute offset and gain-minus-one, remain native.",
        "Comparing spectra against a dataset mean or a supplied calibration-only frozen reference "
        "without silently changing the reference when an evaluation batch changes.",
        "Zero norms make SAM unavailable; a constant reference makes affine coefficients "
        "unidentifiable. Pairwise-valid masks differ from column imputation. These distances "
        "do not identify a physical cause or calibrated anomaly probability.",
        "One generic native procedure over finite X plus explicit masks; supplied reference uses "
        "X_target. Bindings only marshal the existing typed parameters and named result arrays. "
        "Independent OLS, RMS/SAM and layout/mask witnesses qualify the calculation."),
    "diagnostics_spectral_events_qc": entry(
        "Declared-axis derivative events and bounded clipping",
        "A versioned descriptive threshold protocol: native first/second differences, "
        "normal-scaled MAD with a sample-standard-deviation fallback. NumPy defines the "
        "independent difference oracle: https://numpy.org/doc/stable/reference/generated/numpy.diff.html.",
        "An event exceeds threshold times row-local robust scale about the derivative median. "
        "Physical slopes divide by signed axis spacing and second derivatives account for "
        "irregular spacing; index differences are a separate mode. Native pooled counts divide "
        "only by eligible pairs/triples. Sensor clipping compares valid values to declared bounds.",
        "Detecting narrow local irregularities and declared sensor-limit occupancy while "
        "preserving valid-band denominators and distinguishing observed-extrema occupancy.",
        "A derivative event is not proof of detector splice or cosmic-ray causation. Fewer than "
        "three eligible differences cannot establish the row scale. Missing bands are not bridged. "
        "Unknown sensor bounds make clipping unavailable; observed extrema alone do not prove clipping.",
        "Native generic procedure, signed monotonic physical axis and explicit typed masks. "
        "Synthetic offset/gain, injected jumps/spikes, irregular/reversed axes and bounded-sensor "
        "witnesses are independently checked before cross-binding transport parity."),
    "diagnostics_spectral_noise_proxy": entry(
        "Second-difference MAD high-frequency descriptive proxy",
        "A declared noise proxy under an approximately smooth signal, using normal-scaled MAD. "
        "SciPy documents the median absolute deviation definition used by the independent oracle: "
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html.",
        "For each row, median absolute deviation of index-space second differences is multiplied "
        "by 1.4826/sqrt(6). The sqrt(6) removes variance inflation for independent equal-variance "
        "white noise under differencing twice. Native summaries expose mean absolute amplitude, "
        "amplitude/noise and 20log10 of that positive finite ratio. Bandwise ratios divide "
        "each interior feature's absolute valid mean by its across-row second-difference MAD "
        "scale; native outputs retain their finite minimum, median and first minimum index.",
        "Describing high-frequency variation on an explicitly declared index-grid protocol "
        "and comparing that same protocol on identical populations and masks.",
        "Smooth signal structure, correlated/heteroscedastic noise and nonuniform physical "
        "sampling can invalidate an instrumental interpretation. This is not instrument SNR "
        "and is distinct from the existing SG residual/noise-gain estimator. Zero estimated "
        "noise does not create an infinite or arbitrarily large finite ratio.",
        "One mask-aware native procedure with per-row sparse availability and native global "
        "median/noise ratio and bandwise summaries. Independent NumPy reductions validate "
        "normalization, dB, masks and bandwise aggregation; "
        "no MAD or percentile is recomputed by a binding."),
    "diagnostics_local_density": entry(
        "Native fixed-neighborhood Local Outlier Factor",
        "Breunig, Kriegel, Ng and Sander (2000), LOF: Identifying Density-Based Local Outliers. "
        "https://doi.org/10.1145/335191.335388.",
        "Use the existing native LOF training neighborhoods and reachability densities. "
        "Training LOF excludes self; novelty scores use a separately fitted reference "
        "neighborhood. The score is mean neighbor reachability density divided by the row density, "
        "with stable distance/index ties and fixed k capped by reference rows minus one.",
        "Comparing local density under an explicit representation, population and Euclidean "
        "distance protocol; exposing genuine LOF separately from kNN distance-ratio proxies.",
        "Fixed-k tie breaking is not an all-distance-tie expansion. Duplicate or degenerate "
        "density neighborhoods can be undefined and are reported unavailable rather than clean. "
        "Complete data are required; distance over differing masked coordinate subsets is not inferred. "
        "LOF is neither a calibrated anomaly probability nor a physical cause label.",
        "The generic wrapper reuses cpp/src/core/filters/_vendored/lof.c. Only native training "
        "score access was added; the old filter/novelty behavior is unchanged. Independent "
        "sklearn training and novelty witnesses qualify the finite nondegenerate cases."),
    "diagnostics_pca_reference_qc": entry(
        "Exact calibration-reference PCA without full covariance moments",
        "Principal-component projection under sample-covariance normalization (ddof1), with "
        "a declared numerical resolution. The independent full-SVD oracle is documented at "
        "https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html.",
        "Fit means and retained loading vectors on X_target only using the existing native "
        "exact Gram/Jacobi solver. Project calibration and query rows through that same fitted "
        "state. Retained eigenvalues use calibration score sums of squares divided by n_train-1. "
        "Q is squared resolved-subspace residual; T2 sums score squares divided by resolved eigenvalues.",
        "Train-only spectral-domain diagnostics on ordinary wide NIRS matrices, with means, "
        "loadings, eigenvalues and frozen query projection available independently of exact full moments.",
        "No implicit scaling, imputation, confidence limit or causal calibration is provided. "
        "Unresolved components do not contribute to Q/T2; zero resolved rank makes T2 unavailable. "
        "Exact full-spectrum moments remain explicitly not requested, never replaced by truncated "
        "moments. Whole-computation and peak-buffer budgets may refuse a request before allocation.",
        "The new generic procedure reuses FlexiblePCA state and native exact Gram/Jacobi kernels. "
        "Its fast randomized estimator default and serialized state remain unchanged. Independent "
        "SVD plus four real 390–541 by2151 matrices validate retained geometry and frozen state."),
}
