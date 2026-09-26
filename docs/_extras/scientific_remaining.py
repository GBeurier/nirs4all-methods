"""Curated scientific documentation for preprocessing and ABI-2 utility pages.

This module deliberately contains no generic fallback.  Every key names one
generated documentation page and records the behaviour of the implementation
currently shipped by nirs4all-methods.
"""

from __future__ import annotations


_REPO = "https://github.com/GBeurier/nirs4all-methods/blob/main/"


def _entry(
    title: str,
    paper: str,
    principle: str,
    use_cases: str,
    limitations: str,
    implementation: str,
    provenance: str,
) -> dict[str, str]:
    return {
        "title": title,
        "paper": paper,
        "principle": principle,
        "use_cases": use_cases,
        "limitations": limitations,
        "implementation": implementation,
        "provenance": provenance,
    }


SCIENTIFIC_CONTENT: dict[str, dict[str, str]] = {
    "models_pls_pls_regression": _entry(
        "PLS regression with a selectable solver",
        "Wold, Sjöström & Eriksson (2001), *PLS-regression: a basic tool of "
        "chemometrics*, Chemometrics and Intelligent Laboratory Systems 58, 109–130, "
        "https://doi.org/10.1016/S0169-7439(01)00155-1; de Jong (1993), *SIMPLS: an "
        "alternative approach to partial least squares regression*, https://doi.org/10.1016/0169-7439(93)85002-X.",
        "PLS regression extracts $k$ latent components $t_a = X w_a$ whose weights "
        "maximize the covariance between the (centered, optionally scaled) predictors and "
        "responses, deflates, and regresses $Y$ on the scores. The fitted model is the "
        "affine predictor $\\hat Y = \\bar y + (X-\\bar x) B$ with $B = W(P^T W)^{-1} Q^T$; "
        "NIPALS, SIMPLS, kernel, SVD and power solvers compute the same subspace up to "
        "numerical and single-response conventions.",
        "General multivariate calibration of spectra to one or several responses, as the "
        "reference linear model and as a latent-score feature extractor ahead of another "
        "estimator.",
        "The component count must be chosen by validation on held-out data; solvers can "
        "differ at the last digits and SIMPLS deflates only the cross-covariance, so "
        "multi-response scores are not identical across solvers. Scaling choices change "
        "the fitted subspace.",
        "Generic role estimator `models.pls.pls_regression` (regressor and transformer) "
        "over `n4m_model_fit`; state is an N4MM model inside the N4ME format.",
        "https://doi.org/10.1016/S0169-7439(01)00155-1; "
        + _REPO
        + "cpp/src/core/model.cpp",
    ),
    "pp_airpls": _entry(
        "Adaptive iteratively reweighted penalized least squares (airPLS)",
        "Zhang, Chen & Liang (2010), *Baseline correction using adaptive iteratively "
        "reweighted penalized least squares*, Analyst 135, 1138–1146, "
        "https://doi.org/10.1039/B922045C.",
        "For a spectrum $y$, airPLS repeatedly solves a Whittaker problem "
        "$\\min_z \\|W^{1/2}(y-z)\\|_2^2+\\lambda\\|D^2z\\|_2^2$. Points above the "
        "current baseline receive zero weight; negative residuals receive exponentially "
        "increasing weights until their mass is small or the iteration budget is spent.",
        "Removal of smooth fluorescence or background drift when peaks are expected to "
        "lie mainly above the baseline.",
        "The result depends strongly on $\\lambda$ and iteration stopping. Broad, dense, or "
        "negative bands can be mistaken for baseline; it is not a physical scatter model.",
        "`n4m.transform.baseline.AirPLS` wraps the ABI-2 `n4m_transform_airpls_*` "
        "lifecycle; the numerical loop is in `cpp/src/core/preprocessing/baselines/airpls.c`.",
        "https://doi.org/10.1039/B922045C; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/airpls.c",
    ),
    "pp_area": _entry(
        "Area normalization",
        "No single canonical paper: total-area normalization is a direct normalization "
        "rule. The exact sum, absolute-sum, and trapezoidal definitions are specified by "
        "the implementation source.",
        "Each spectrum $x_i$ is divided by a scalar area $a_i$: the signed sum, the sum "
        "of absolute values, or the unit-spacing trapezoidal integral. The implementation "
        "uses $a_i=1$ when $|a_i|<10^{-10}$ to avoid division by a near-zero area.",
        "Comparing spectra on a common integrated-intensity scale when total signal varies "
        "but relative band shape is meaningful.",
        "Signed sums can cancel and all modes couple every wavelength to one scalar. "
        "Normalization removes absolute concentration or path-length information.",
        "`n4m.transform.scatter.AreaNormalization` uses "
        "`n4m_transform_area_normalization_*`; `area_normalization.c` implements all three "
        "row-wise denominators.",
        _REPO + "cpp/src/core/preprocessing/scatter/area_normalization.c",
    ),
    "pp_arpls": _entry(
        "Asymmetrically reweighted penalized least squares (arPLS)",
        "Baek et al. (2015), *Baseline correction using asymmetrically reweighted "
        "penalized least squares smoothing*, Analyst 140, 250–257, "
        "https://doi.org/10.1039/C4AN01061B.",
        "arPLS alternates a Whittaker smoothness solve with a logistic update of weights "
        "estimated from negative residual statistics. Positive peak residuals are "
        "progressively downweighted without requiring a fixed asymmetry parameter.",
        "Automatic baseline removal for spectra with positive peaks and slowly varying "
        "backgrounds, especially when a fixed AsLS asymmetry is hard to choose.",
        "The negative-residual distribution must represent baseline noise. Very broad "
        "bands, negative peaks, or an ill-chosen smoothness penalty bias the estimate.",
        "`n4m.transform.baseline.ArPLS` calls `n4m_transform_arpls_*`; the iterative "
        "reweighting and penalized solver are in `preprocessing/baselines/arpls.c`.",
        "https://doi.org/10.1039/C4AN01061B; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/arpls.c",
    ),
    "pp_asls": _entry(
        "Asymmetric least-squares baseline correction (AsLS)",
        "Eilers & Boelens (2005), *Baseline correction with asymmetric least squares "
        "smoothing*, Leiden University Medical Centre technical report; algorithmic source "
        "also summarized at https://doi.org/10.1039/C4AN01061B.",
        "AsLS alternates minimization of $\\sum_j w_j(y_j-z_j)^2+\\lambda\\|D^2z\\|^2$ "
        "with $w_j=p$ for positive residuals and $w_j=1-p$ otherwise. Subtracting the "
        "smooth $z$ preserves narrow peaks while suppressing background curvature.",
        "Baseline correction for Raman/NIR-like traces when the sign of analyte peaks is "
        "known and a controllable smoothness/asymmetry trade-off is wanted.",
        "Both $\\lambda$ and $p$ require tuning; broad peaks may enter the baseline and the "
        "binary residual weighting can converge slowly near ambiguous points.",
        "`n4m.transform.baseline.AsLS` wraps `n4m_transform_asls_*`; the Whittaker system "
        "and weight updates live in `preprocessing/baselines/asls.c`.",
        _REPO + "cpp/src/core/preprocessing/baselines/asls.c; "
        "https://doi.org/10.1039/C4AN01061B",
    ),
    "pp_baseline": _entry(
        "Training-set column mean centering",
        "No single canonical paper: this is the standard centering operation used by PCA, "
        "PLS, and linear models. The implementation source is the normative definition.",
        "Fit stores $\\mu_j=n^{-1}\\sum_i X_{ij}$ for every wavelength; transform returns "
        "$X_{ij}-\\mu_j$ and inverse transform adds the same training mean. It is feature "
        "centering across samples, not row-wise spectral baseline subtraction.",
        "Preparing calibration matrices for covariance-based models and reproducing the "
        "training coordinate system on validation or prediction data.",
        "It only removes a constant offset per feature and is stateful: fitting before a "
        "data split leaks validation information. It does not correct per-spectrum drift.",
        "`n4m.transform.scaling.BaselineCenter` uses "
        "`n4m_transform_baseline_center_*`; `scaling/baseline.c` stores and reuses the "
        "training column means.",
        _REPO + "cpp/src/core/preprocessing/scaling/baseline.c",
    ),
    "pp_beads": _entry(
        "Baseline estimation and denoising with sparsity (BEADS)",
        "Ning, Selesnick & Duval (2014), *Chromatogram baseline estimation and denoising "
        "using sparsity (BEADS)*, Chemometrics and Intelligent Laboratory Systems 139, "
        "156–167, https://doi.org/10.1016/j.chemolab.2014.09.014.",
        "BEADS decomposes a trace into a slowly varying baseline and a sparse signal by "
        "penalizing signal amplitude and first/second differences while constraining the "
        "baseline through a high-pass filter model. The native solver iterates this sparse "
        "penalized approximation with `lam_0`, `lam_1`, and `lam_2`.",
        "Spectra or chromatograms needing simultaneous smooth-background removal and sparse "
        "peak denoising.",
        "Its sparsity assumptions suit isolated peaks better than broad overlapping bands; "
        "three penalties and the convergence budget need tuning. The native implementation "
        "is an iterative approximation of the published formulation.",
        "`n4m.transform.baseline.BEADS` wraps `n4m_transform_beads_*`; the shipped solver is "
        "`cpp/src/core/preprocessing/baselines/beads.c`.",
        "https://doi.org/10.1016/j.chemolab.2014.09.014; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/beads.c",
    ),
    "pp_cow_align": _entry(
        "Correlation optimized warping (COW)",
        "Nielsen et al. (1998), *Alignment of single and multiple wavelength "
        "chromatographic profiles for chemometric data analysis using correlation optimised "
        "warping*, Journal of Chemometrics 12, 521–538, "
        "https://doi.org/10.1002/(SICI)1099-128X(199811/12)12:6%3C521::AID-CEM515%3E3.0.CO;2-Z.",
        "The reference is divided into fixed intervals. A dynamic program moves internal "
        "sample boundaries within a slack derived from `max_shift`, tries admissible segment "
        "lengths, linearly resamples each candidate to the reference interval width, and "
        "maximizes cumulative centered correlation before backtracking the best warping path.",
        "Correcting local wavelength-axis displacement while preserving a common output "
        "grid, for example before multivariate calibration across instrument sessions.",
        "This compact API couples slack to `max_shift` and fixes other classical COW choices. "
        "Piecewise linear resampling can smooth narrow peaks, short spectra fall back to global "
        "cross-correlation shifting, and a poor reference can induce an incorrect path.",
        "`n4m.transform.alignment.CorrelationOptimizedWarping` uses "
        "`n4m_transform_cow_align_*`; the current algorithm is implemented in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1002/(SICI)1099-128X(199811/12)12:6%3C521::AID-CEM515%3E3.0.CO;2-Z; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_crop": _entry(
        "Wavelength-column cropping",
        "No canonical paper: cropping is an indexing operation, defined here by Python's "
        "half-open slice convention `[start, end)`.",
        "For every row the transform copies columns $j$ satisfying "
        "$\\mathrm{start}\\le j<\\mathrm{end}$. It changes feature count without interpolation "
        "or wavelength-aware lookup.",
        "Restricting a model to a known spectral region or removing detector edge bands.",
        "Indices are positions, not physical wavelengths; grids must already be consistent. "
        "Cropping can discard predictive bands and changes downstream feature alignment.",
        "`n4m.transform.resampling.CropTransformer` wraps `n4m_transform_crop_*`; the copy "
        "and bounds checks are in `preprocessing/resampling/crop.c`.",
        _REPO + "cpp/src/core/preprocessing/resampling/crop.c",
    ),
    "pp_derivate": _entry(
        "Order-$d$ finite-difference derivative",
        "No unique canonical paper: this is repeated forward finite differencing. For the "
        "spectroscopic rationale see Norris & Williams (1984), *Optimization of mathematical "
        "treatments of raw near-infrared signal*.",
        "The operator applies $d$ successive first differences along wavelength and divides "
        "by `delta` at each pass, reducing the output width by `order`. A first difference "
        "removes constant offsets; a second also suppresses linear trends.",
        "Emphasizing slopes or narrow bands when a reduced-width difference representation "
        "is acceptable.",
        "Differencing amplifies high-frequency noise, shortens the feature axis, and assumes "
        "uniform spacing represented by `delta`. It is distinct from the shape-preserving "
        "gradient operators.",
        "`n4m.transform.smoothing.Derivate` calls `n4m_transform_derivative_*`; repeated "
        "difference kernels are in `preprocessing/derivatives/derivate.c`.",
        _REPO + "cpp/src/core/preprocessing/derivatives/derivate.c",
    ),
    "pp_detrend": _entry(
        "Polynomial spectral detrending",
        "Barnes, Dhanoa & Lister (1989), *Standard Normal Variate Transformation and "
        "De-trending of Near-Infrared Diffuse Reflectance Spectra*, Applied Spectroscopy "
        "43, 772–777, https://doi.org/10.1366/0003702894202201.",
        "For each row, least squares fits a polynomial of degree `polyorder` against channel "
        "position and subtracts the fitted curve. Degree one removes offset and slope; higher "
        "degrees remove progressively curved backgrounds.",
        "Removing smooth per-spectrum baseline drift before calibration, often after or "
        "alongside scatter normalization.",
        "A high polynomial degree can remove broad chemical bands or oscillate near edges; "
        "channel positions are treated as equally spaced unless the data were resampled.",
        "`n4m.transform.baseline.Detrend` uses `n4m_transform_detrend_*`; the row-wise normal "
        "equations are implemented in `preprocessing/baselines/detrend.c`.",
        "https://doi.org/10.1366/0003702894202201; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/detrend.c",
    ),
    "pp_direct_standardization": _entry(
        "Direct standardization (DS)",
        "Wang et al. (1991), *Improvement of multivariate calibration through instrument "
        "standardization*, Analytical Chemistry 63, 2750–2756, "
        "https://doi.org/10.1021/ac00023a016.",
        "Using paired source and target spectra, DS fits a global affine map "
        "$X_s B+\\mathbf{1}b\\approx X_t$ by least squares, optionally with ridge "
        "regularization. New source spectra are multiplied by the learned transfer matrix.",
        "Transferring a calibration between instruments or measurement conditions when "
        "representative paired standards are available.",
        "It requires paired samples and assumes one global linear relation. With many "
        "wavelengths the map can overfit unless regularized; extrapolation beyond transfer "
        "standards is unsafe.",
        "`n4m.domain_adaptation.standardization.DirectStandardization` wraps "
        "`n4m_domain_adaptation_direct_standardization_*`; fitting is in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1021/ac00023a016; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_dtw_align": _entry(
        "Dynamic time warping spectral alignment",
        "Sakoe & Chiba (1978), *Dynamic programming algorithm optimization for spoken word "
        "recognition*, IEEE TASSP 26, 43–49, https://doi.org/10.1109/TASSP.1978.1163055.",
        "A dynamic program finds a minimum-cost monotone path through pairwise sample/reference "
        "distances. The path maps local stretches and compressions to the reference grid; "
        "multiple matched input points are aggregated to keep a fixed output length.",
        "Aligning spectra with nonlinear local wavelength shifts that cannot be represented "
        "by one global offset.",
        "Unconstrained DTW can align unrelated bands and is quadratic in spectrum length. "
        "Amplitude differences affect the path, and the result depends on the reference.",
        "`n4m.transform.alignment.DynamicTimeWarpingAlignment` wraps "
        "`n4m_transform_dtw_align_*`; the native path computation is in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1109/TASSP.1978.1163055; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_emsc": _entry(
        "Extended multiplicative scatter correction (EMSC)",
        "Martens & Stark (1991), *Extended multiplicative signal correction and spectral "
        "interference subtraction*, Journal of Pharmaceutical and Biomedical Analysis 9, "
        "625–635, https://doi.org/10.1016/0731-7085(91)80188-F.",
        "Each spectrum is regressed on the fitted mean reference plus polynomial channel "
        "terms through the requested degree. Subtracting polynomial contributions and "
        "dividing by the reference coefficient extends MSC to smooth additive backgrounds.",
        "Correcting multiplicative scatter together with smooth baseline curvature while "
        "preserving the reference-shaped chemical contribution.",
        "The polynomial basis can absorb broad analyte variation, and this implementation "
        "does not accept arbitrary constituent/interferent spectra. Fit the reference on "
        "training data only.",
        "`n4m.transform.scatter.EMSC` uses `n4m_transform_emsc_*`; the training reference and "
        "per-row least-squares correction are in `preprocessing/scatter/emsc.c`.",
        "https://doi.org/10.1016/0731-7085(91)80188-F; "
        + _REPO
        + "cpp/src/core/preprocessing/scatter/emsc.c",
    ),
    "pp_epo": _entry(
        "External parameter orthogonalization (EPO)",
        "Roger, Chauchard & Bellon-Maurel (2003), *EPO-PLS external parameter "
        "orthogonalisation of PLS application to temperature-independent measurement of "
        "sugar content of intact fruits*, Chemometrics and Intelligent Laboratory Systems "
        "66, 191–204, https://doi.org/10.1016/S0169-7439(03)00051-0.",
        "From nuisance-variation spectra, EPO estimates leading singular vectors $P$ of the "
        "external-parameter subspace and projects data with $I-PP^T$. Optional scaling is "
        "learned before the projection and reused at transform time.",
        "Removing measured instrument, temperature, moisture, or batch variation before a "
        "calibration is fitted.",
        "The nuisance experiment must span unwanted variation without confounding analyte "
        "signal. Removing too many components deletes predictive information; EPO is "
        "stateful and must be fit within validation folds.",
        "`n4m.domain_adaptation.orthogonalization.EPO` wraps "
        "`n4m_domain_adaptation_epo_*`; SVD and projection are in "
        "`preprocessing/orthogonalization/epo.c`.",
        "https://doi.org/10.1016/S0169-7439(03)00051-0; "
        + _REPO
        + "cpp/src/core/preprocessing/orthogonalization/epo.c",
    ),
    "pp_fck_static": _entry(
        "Static fractional convolutional-kernel bank",
        "No canonical paper uniquely defines this n4m operator. It is an implementation-specific "
        "bank of fractional convolutional kernels; its generated kernel equation and fixed "
        "$\\sigma=3$ are defined by `fck_kernel.h` and `fck_static.c`.",
        "At construction, one length-`kernel_size` kernel is generated for every Cartesian "
        "pair of fractional order `alpha` and scale. Each row is convolved with all kernels "
        "using nearest-edge extension, and the bands are concatenated into "
        "`n_kernels * n_features` outputs.",
        "Fixed multiscale, fractional-order feature expansion before a linear or sparse model.",
        "Output dimensionality grows multiplicatively with the kernel bank. Edge clamping and "
        "the fixed kernel parameterization may not suit every wavelength grid; it is not a "
        "learned convolutional model.",
        "`n4m.transform.specialized.FCKStaticTransformer` calls "
        "`n4m_transform_fck_static_*`; kernel construction and convolution are in "
        "`preprocessing/specialized/fck_static.c`.",
        _REPO + "cpp/src/core/preprocessing/specialized/fck_static.c; "
        + _REPO
        + "cpp/src/core/common/fck_kernel.h",
    ),
    "pp_first_derivative": _entry(
        "Shape-preserving first numerical derivative",
        "No unique paper defines `numpy.gradient`; the spectroscopic use of derivatives is "
        "discussed by Norris & Williams (1984), *Optimization of mathematical treatments of "
        "raw near-infrared signal*.",
        "The operator reproduces a first `numpy.gradient` pass along each row: centered "
        "differences inside and one-sided formulas of edge order one or two at boundaries, "
        "all divided by `delta`. Output shape equals input shape.",
        "Suppressing constant offsets and sharpening overlapping bands when retaining the "
        "original number of channels is useful.",
        "It amplifies noise and assumes uniform channel spacing. Boundary samples use "
        "different stencils, and `edge_order=2` needs enough wavelengths.",
        "`n4m.transform.smoothing.FirstDerivative` wraps "
        "`n4m_transform_first_derivative_*`; stencils are in "
        "`preprocessing/derivatives/first_derivative.c`.",
        _REPO + "cpp/src/core/preprocessing/derivatives/first_derivative.c; "
        "https://numpy.org/doc/stable/reference/generated/numpy.gradient.html",
    ),
    "pp_flex_pca": _entry(
        "Flexible principal component analysis",
        "Pearson (1901), *On Lines and Planes of Closest Fit to Systems of Points in Space*, "
        "Philosophical Magazine 2, 559–572, https://doi.org/10.1080/14786440109462720.",
        "After column centering, compact SVD gives $X_c=U\\Sigma V^T$ and scores "
        "$T=U_k\\Sigma_k$. `n_components` is interpreted either as an integer count or as a "
        "target cumulative explained-variance fraction.",
        "Unsupervised compression, visualization, noise reduction, or a fixed latent feature "
        "stage before regression.",
        "PCA maximizes variance rather than relevance to a response, is scale-sensitive, and "
        "requires fold-local fitting. A variance threshold may choose different dimensions "
        "across folds.",
        "`n4m.decomposition.FlexiblePCA` wraps `n4m_decomposition_flexible_pca_*`; centering, "
        "SVD, variance selection, and projection are in `flexible_pca.c`.",
        "https://doi.org/10.1080/14786440109462720; "
        + _REPO
        + "cpp/src/core/preprocessing/feature_selection/flexible_pca.c",
    ),
    "pp_flex_svd": _entry(
        "Flexible truncated singular-value decomposition",
        "Eckart & Young (1936), *The approximation of one matrix by another of lower rank*, "
        "Psychometrika 1, 211–218, https://doi.org/10.1007/BF02288367.",
        "Compact SVD factorizes the uncentered input as $X=U\\Sigma V^T$ and emits the first "
        "$k$ coordinates $XV_k$. A parameter at least one is truncated to an integer rank; a "
        "parameter in $(0,1)$ selects the smallest rank reaching that fraction of total "
        "column variance using the projected-component variance ratios.",
        "Low-rank compression when retaining the mean direction is intentional and rank should "
        "be chosen directly or from a variance target.",
        "Uncentered SVD can devote its first direction to mean offset and is scale-sensitive. "
        "The fractional criterion is based on projected variance rather than simply squared "
        "singular-value energy, and fitting must remain inside validation folds.",
        "`n4m.decomposition.FlexibleSVD` calls `n4m_decomposition_flexible_svd_*`; the compact "
        "factorization and transform are in `flexible_svd.c`.",
        "https://doi.org/10.1007/BF02288367; "
        + _REPO
        + "cpp/src/core/preprocessing/feature_selection/flexible_svd.c",
    ),
    "pp_frac_to_pct": _entry(
        "Fraction-to-percent signal conversion",
        "No canonical paper: percent is exactly the conventional unit conversion "
        "$x_{\\%}=100x$.",
        "Every matrix element is multiplied by 100; no statistics are learned and spectral "
        "shape is unchanged.",
        "Converting fractional reflectance or transmittance to percent units for interfaces "
        "or models that explicitly require that scale.",
        "This operation does not infer the signal type and will silently produce wrong units "
        "if the input is already a percent or absorbance value.",
        "`n4m.transform.signal_conversion.FractionToPercent` uses "
        "`n4m_transform_fraction_to_percent_*`; the element-wise kernel is in "
        "`signal_conversion/fraction_to_percent.c`.",
        _REPO
        + "cpp/src/core/preprocessing/signal_conversion/fraction_to_percent.c",
    ),
    "pp_from_absorbance": _entry(
        "Absorbance-to-reflectance/transmittance conversion",
        "Beer (1852), *Bestimmung der Absorption des rothen Lichts in farbigen "
        "Flüssigkeiten*, Annalen der Physik 162, 78–88, "
        "https://doi.org/10.1002/andp.18521620505.",
        "The inverse base-10 absorbance relation is applied element-wise: $R=10^{-A}$ (or "
        "$T=10^{-A}$). With `is_percent`, the result is additionally multiplied by 100.",
        "Returning log-transformed spectra to fractional or percent reflectance/transmittance "
        "units.",
        "The formula cannot distinguish reflectance from transmittance and assumes base-10 "
        "absorbance. Large negative or positive values can overflow or underflow.",
        "`n4m.transform.signal_conversion.FromAbsorbance` wraps "
        "`n4m_transform_from_absorbance_*`; the power transform is in "
        "`signal_conversion/from_absorbance.c`.",
        "https://doi.org/10.1002/andp.18521620505; "
        + _REPO
        + "cpp/src/core/preprocessing/signal_conversion/from_absorbance.c",
    ),
    "pp_gaussian": _entry(
        "Gaussian smoothing and derivative filtering",
        "No single canonical spectroscopic paper: this is discrete Gaussian convolution. "
        "The public reference behaviour is SciPy `gaussian_filter1d`: "
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html.",
        "A truncated discrete Gaussian of standard deviation `sigma` is convolved along each "
        "spectrum; positive `order` uses the corresponding derivative-of-Gaussian kernel. "
        "Boundary samples follow the selected extension mode and `cval`.",
        "Low-pass denoising, or smoothed derivative estimation, with a continuously tunable "
        "spectral scale.",
        "Smoothing broadens narrow bands and derivative orders amplify noise. `sigma` is in "
        "channel units, so unequal wavelength grids require resampling; boundary mode affects "
        "edge bands.",
        "`n4m.transform.smoothing.Gaussian` calls `n4m_transform_gaussian_*`; kernel generation "
        "and boundary handling are in `preprocessing/smoothing/gaussian.c`.",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html; "
        + _REPO
        + "cpp/src/core/preprocessing/smoothing/gaussian.c",
    ),
    "pp_haar": _entry(
        "Single-level Haar discrete wavelet transform",
        "Haar (1910), *Zur Theorie der orthogonalen Funktionensysteme*, Mathematische "
        "Annalen 69, 331–371, https://doi.org/10.1007/BF01456326.",
        "Adjacent sample pairs are projected onto the Haar scaling and wavelet filters, "
        "producing approximation $(x_{2j}+x_{2j+1})/\\sqrt2$ and detail "
        "$(x_{2j}-x_{2j+1})/\\sqrt2$ coefficients under the configured endpoint convention.",
        "Compact separation of coarse spectral shape and one-channel-scale detail.",
        "Haar is discontinuous and shift-sensitive; it can represent smooth bands less "
        "efficiently than longer wavelets. Odd-length boundary behaviour must be considered.",
        "`n4m.transform.wavelet.Haar` wraps `n4m_transform_haar_*`; the single-level native "
        "transform is in `preprocessing/wavelets/haar.c`.",
        "https://doi.org/10.1007/BF01456326; "
        + _REPO
        + "cpp/src/core/preprocessing/wavelets/haar.c",
    ),
    "pp_iasls": _entry(
        "Improved asymmetric least-squares baseline correction (IAsLS)",
        "He et al. (2014), *Baseline correction for Raman spectra using an improved "
        "asymmetric least squares method*, Analytical Methods 6, 4402–4407, "
        "https://doi.org/10.1039/C4AY00068D.",
        "IAsLS augments asymmetric Whittaker fitting with a derivative fidelity penalty "
        "controlled by `lam_1`; it estimates an initial polynomial background, then iterates "
        "asymmetric weights while penalizing the configured baseline difference order.",
        "Baseline correction when ordinary AsLS leaves peak-dependent distortion and a "
        "derivative-aware smoothness term is useful.",
        "The result is sensitive to two penalties, asymmetry, polynomial order, and stopping "
        "criteria. Published IAsLS variants differ, so this page describes the shipped kernel.",
        "`n4m.transform.baseline.IAsLS` uses `n4m_transform_iasls_*`; the exact polynomial "
        "initialization and weighted solve are in `preprocessing/baselines/iasls.c`.",
        "https://doi.org/10.1039/C4AY00068D; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/iasls.c",
    ),
    "pp_icoshift_align": _entry(
        "Interval correlation optimized shifting (icoshift-style)",
        "Savorani, Tomasi & Engelsen (2010), *icoshift: A versatile tool for the rapid "
        "alignment of 1D NMR spectra*, Journal of Magnetic Resonance 202, 190–202, "
        "https://doi.org/10.1016/j.jmr.2009.11.012.",
        "Each fixed interval is compared with the same reference interval over integer lags "
        "within `max_shift`; the lag maximizing the centered cross-product score is applied "
        "independently with edge replication. Output length and interval positions are retained.",
        "Fast correction of piecewise-constant local channel shifts when peak order is stable.",
        "This is a fixed-interval, bounded-shift implementation rather than every option in "
        "the original icoshift software. Independent intervals may create seams and weak "
        "intervals yield unstable correlations.",
        "`n4m.transform.alignment.IcoshiftAlignment` uses "
        "`n4m_transform_icoshift_align_*`; interval search is implemented in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1016/j.jmr.2009.11.012; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_imodpoly": _entry(
        "Improved modified-polynomial baseline correction (IModPoly)",
        "Gan, Ruan & Mo (2006), *Baseline correction by improved iterative polynomial fitting "
        "with automatic threshold*, Chemometrics and Intelligent Laboratory Systems 82, "
        "59–65, https://doi.org/10.1016/j.chemolab.2005.08.009.",
        "A polynomial is repeatedly fit to a working spectrum; points identified as peaks "
        "relative to the current baseline and residual spread are replaced/downweighted "
        "before refitting. IModPoly uses a residual-noise criterion to reduce the systematic "
        "underfit of basic ModPoly.",
        "Automatic subtraction of smooth fluorescence backgrounds under positive Raman-like "
        "bands.",
        "Polynomial order controls bias and edge behaviour; broad or negative peaks violate "
        "the clipping model. Iterative variants differ, so parity claims apply to this kernel.",
        "`n4m.transform.baseline.IModPoly` wraps `n4m_transform_imodpoly_*`; the exact clipping "
        "and convergence rules are in `preprocessing/baselines/imodpoly.c`.",
        "https://doi.org/10.1016/j.chemolab.2005.08.009; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/imodpoly.c",
    ),
    "pp_kbins_disc": _entry(
        "Per-feature integer k-bin discretization",
        "No single canonical paper: this is uniform or empirical-quantile scalar "
        "quantization. Scikit-learn documents the corresponding estimator at "
        "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html.",
        "Fit computes `n_bins + 1` edges independently for every column, either equally "
        "spaced between its extrema or at empirical quantiles. Transform replaces each value "
        "by the integer bin index selected by those stored edges.",
        "Robust coarse encoding for threshold models or exploratory analyses where continuous "
        "amplitude resolution is unnecessary.",
        "Quantization discards within-bin information; quantile edges can collapse with ties "
        "and all edges are training-data dependent. It is rarely appropriate before models "
        "that exploit smooth spectral geometry.",
        "`n4m.transform.resampling.IntegerKBinsDiscretizer` calls "
        "`n4m_transform_kbins_discretizer_*`; fitting and integer encoding are in "
        "`resampling/kbins_discretizer.c`.",
        "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html; "
        + _REPO
        + "cpp/src/core/preprocessing/resampling/kbins_discretizer.c",
    ),
    "pp_kubelka_munk": _entry(
        "Kubelka–Munk remission transform",
        "Kubelka & Munk (1931), *Ein Beitrag zur Optik der Farbanstriche*, Zeitschrift für "
        "Technische Physik 12, 593–601; English translation: "
        "https://doi.org/10.1002/col.5080010404.",
        "Fractional reflectance $R$ is converted element-wise to the infinite-layer remission "
        "function $F(R)=(1-R)^2/(2R)$. Percent input is divided by 100 first and the "
        "denominator is guarded by `epsilon`.",
        "Linearizing diffuse-reflectance measurements of optically thick scattering samples "
        "under the Kubelka–Munk assumptions.",
        "The two-flux, infinite-thickness, homogeneous-sample assumptions often fail. Values "
        "near zero explode, and specular reflection or finite thickness invalidates the "
        "physical interpretation.",
        "`n4m.transform.signal_conversion.KubelkaMunk` wraps "
        "`n4m_transform_kubelka_munk_*`; the guarded formula is in "
        "`signal_conversion/kubelka_munk.c`.",
        "https://doi.org/10.1002/col.5080010404; "
        + _REPO
        + "cpp/src/core/preprocessing/signal_conversion/kubelka_munk.c",
    ),
    "pp_local_centering": _entry(
        "Source-to-target local centering",
        "No unique canonical paper: this is a mean-shift domain-adaptation rule, fully "
        "specified by the implementation.",
        "Fit computes feature means $\\mu_s$ and $\\mu_t$ from source and target calibration "
        "sets. A source row is transferred as $x-\\mu_s+\\mu_t$, matching first moments while "
        "leaving covariance and higher-order structure unchanged.",
        "Correcting simple additive instrument or batch shifts when source and target samples "
        "share aligned wavelength columns.",
        "Only mean shift is corrected. Target means estimated from few or nonrepresentative "
        "samples are noisy, and using evaluation target data during fit causes leakage.",
        "`n4m.transform.scatter.LocalCentering` calls `n4m_transform_local_centering_*`; the "
        "paired-domain state and affine shift are in `cpp/src/c_api/c_api_advanced.cpp`.",
        _REPO + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_localized_msc": _entry(
        "Localized moving-window multiplicative scatter correction",
        "No single paper canonically defines this implementation. It is a moving-window "
        "extension of Geladi, MacDougall & Martens (1985), "
        "https://doi.org/10.1366/0003702854248656.",
        "At each wavelength, a local window is regressed on the corresponding reference "
        "window to obtain an intercept and slope; the center value is corrected by those "
        "local coefficients. The fitted reference is either supplied or learned from the "
        "training mean.",
        "Correcting scatter whose offset or gain changes gradually across the wavelength axis.",
        "Local regressions become unstable in flat windows or with small windows; `eps` only "
        "guards degeneracy. Moving coefficients can distort broad bands and require a stable "
        "reference.",
        "`n4m.transform.scatter.LocalizedMSC` wraps `n4m_transform_localized_msc_*`; the local "
        "regressions are in `cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1366/0003702854248656; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_log": _entry(
        "Element-wise logarithmic transform",
        "No canonical paper: this is the mathematical logarithm with an implementation-defined "
        "offset policy.",
        "Transform evaluates $\\log_b(x+c)$, using the natural logarithm when `base=0`. With "
        "`auto_offset`, fit chooses an offset that moves the training minimum to at least "
        "`min_value`; the same offset is reused on later data.",
        "Compressing right-skewed positive intensities or expressing multiplicative changes on "
        "an additive scale.",
        "Nonpositive shifted values are invalid. A data-derived offset changes interpretation "
        "and must be fit without leakage; logs are not interchangeable with physical "
        "absorbance unless the correct sign and base are used.",
        "`n4m.transform.scaling.LogTransform` uses `n4m_transform_log_transform_*`; offset "
        "fitting and base conversion are in `preprocessing/scaling/log_transform.c`.",
        _REPO + "cpp/src/core/preprocessing/scaling/log_transform.c",
    ),
    "pp_lsnv": _entry(
        "Local standard normal variate (LSNV)",
        "No single canonical paper defines this sliding-window variant. It locally applies "
        "the SNV concept of Barnes, Dhanoa & Lister (1989), "
        "https://doi.org/10.1366/0003702894202201.",
        "For each wavelength, LSNV computes the mean and standard deviation in a centered "
        "window of the same spectrum and standardizes the center sample. Reflect, edge, or "
        "constant padding defines incomplete boundary windows.",
        "Removing wavelength-dependent local offset and scale variation that one global SNV "
        "cannot represent.",
        "It can suppress broad chemical bands and amplify noise where local variance is small. "
        "Window width and padding materially affect the output, especially near edges.",
        "`n4m.transform.scatter.LSNV` calls `n4m_transform_local_snv_*`; rolling statistics "
        "and padding modes are implemented in `preprocessing/scatter/local_snv.c`.",
        "https://doi.org/10.1366/0003702894202201; "
        + _REPO
        + "cpp/src/core/preprocessing/scatter/local_snv.c",
    ),
    "pp_modpoly": _entry(
        "Modified-polynomial baseline correction (ModPoly)",
        "Lieber & Mahadevan-Jansen (2003), *Automated method for subtraction of fluorescence "
        "from biological Raman spectra*, Applied Spectroscopy 57, 1363–1367, "
        "https://doi.org/10.1366/000370203322554518.",
        "A polynomial baseline is fit iteratively. After each fit, samples above the fitted "
        "baseline are clipped to it in the working spectrum, so positive peaks have decreasing "
        "influence on the next polynomial estimate; iteration stops by tolerance or budget.",
        "Subtracting smooth fluorescence from spectra containing mostly positive, relatively "
        "narrow peaks.",
        "Broad peaks can be clipped into the baseline and high polynomial orders are unstable "
        "at edges. Negative bands violate the one-sided clipping assumption.",
        "`n4m.transform.baseline.ModPoly` wraps `n4m_transform_modpoly_*`; iterative clipping "
        "and polynomial fitting are in `preprocessing/baselines/modpoly.c`.",
        "https://doi.org/10.1366/000370203322554518; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/modpoly.c",
    ),
    "pp_msc": _entry(
        "Multiplicative scatter correction (MSC)",
        "Geladi, MacDougall & Martens (1985), *Linearization and Scatter-Correction for "
        "Near-Infrared Reflectance Spectra of Meat*, Applied Spectroscopy 39, 491–500, "
        "https://doi.org/10.1366/0003702854248656.",
        "Fit stores the training mean reference $r$. Each row is fit as "
        "$x_i=a_i+b_i r+e_i$ by ordinary least squares and corrected to "
        "$(x_i-a_i)/b_i$, removing row-wise additive and multiplicative scatter.",
        "Diffuse-reflectance calibration where particle size or path length mainly changes "
        "offset and scale.",
        "MSC assumes a linear relation to a representative reference and becomes unstable when "
        "the fitted slope is near zero. Training-reference estimation must stay within folds, "
        "and chemical variation correlated with scatter can be altered.",
        "`n4m.transform.scatter.MSC` uses `n4m_transform_msc_*`; reference fitting and row-wise "
        "OLS correction are in `preprocessing/scatter/msc.c`.",
        "https://doi.org/10.1366/0003702854248656; "
        + _REPO
        + "cpp/src/core/preprocessing/scatter/msc.c",
    ),
    "pp_normalize": _entry(
        "Column-wise normalization",
        "No canonical paper: the default is column L2 normalization; a non-default feature "
        "range selects column-wise min–max scaling. The source defines this mode switch.",
        "With the default range `(-1, 1)`, every column is divided by "
        "$\\sqrt{\\sum_i X_{ij}^2}$. If either range endpoint is changed, each column is instead "
        "mapped affinely from its observed minimum and maximum to the requested interval.",
        "Equalizing feature magnitudes before algorithms sensitive to Euclidean scale, or "
        "mapping every wavelength to a prescribed numeric range.",
        "The name hides two different operations. Statistics are computed from the matrix "
        "passed to the operation rather than stored as a fitted training state, and zero-norm "
        "or constant columns can produce non-finite values.",
        "`n4m.transform.scaling.Normalize` wraps `n4m_transform_normalize_*`; the exact "
        "default-mode branch is in `preprocessing/scaling/normalize.c`.",
        _REPO + "cpp/src/core/preprocessing/scaling/normalize.c",
    ),
    "pp_norris_williams": _entry(
        "Norris–Williams segment-gap derivative",
        "Norris & Williams (1984), *Optimization of mathematical treatments of raw "
        "near-infrared signal in the measurement of protein in hard red spring wheat*, "
        "Cereal Chemistry 61, 158–165 (bibliographic record: "
        "https://www.cerealsgrains.org/publications/cc/backissues/1984/Documents/61_158.pdf).",
        "Each row is first averaged in non-overlapping or sliding segments of length "
        "`segment`; first or second finite differences are then taken between segment means "
        "separated by `gap`, scaled by `delta`. Smoothing precedes differentiation.",
        "Traditional NIR pretreatment for attenuating noise while removing offsets or linear "
        "background and sharpening broad bands.",
        "Segment and gap are channel counts and assume a uniform grid. The output loses edge "
        "positions, large gaps blur narrow bands, and differentiation still amplifies residual "
        "noise.",
        "`n4m.transform.smoothing.NorrisWilliams` uses "
        "`n4m_transform_norris_williams_*`; segment averaging and gap derivatives are in "
        "`preprocessing/derivatives/norris_williams.c`.",
        _REPO + "cpp/src/core/preprocessing/derivatives/norris_williams.c; "
        "https://www.cerealsgrains.org/publications/cc/backissues/1984/Documents/61_158.pdf",
    ),
    "pp_osc": _entry(
        "Orthogonal signal correction (OSC)",
        "Wold et al. (1998), *Orthogonal signal correction of near-infrared spectra*, "
        "Chemometrics and Intelligent Laboratory Systems 44, 175–185, "
        "https://doi.org/10.1016/S0169-7439(98)00109-9.",
        "OSC extracts directions in $X$ with high variance constrained to be orthogonal to "
        "the response $Y$, then removes their score-loading reconstructions from $X$. The "
        "native fit repeats this deflation for `n_components`, optionally after scaling.",
        "Supervised removal of structured spectral variation known to be unrelated to the "
        "calibration target.",
        "OSC uses the response and therefore must be fit inside every validation fold. If the "
        "calibration set is small or confounded, it can remove transferable analyte signal; "
        "component count needs validation.",
        "`n4m.transform.orthogonalization.OSC` wraps `n4m_transform_osc_*`; supervised "
        "component extraction and deflation are in `preprocessing/orthogonalization/osc.c`.",
        "https://doi.org/10.1016/S0169-7439(98)00109-9; "
        + _REPO
        + "cpp/src/core/preprocessing/orthogonalization/osc.c",
    ),
    "pp_pct_to_frac": _entry(
        "Percent-to-fraction signal conversion",
        "No canonical paper: this is exactly the unit conversion $x=x_{\\%}/100$.",
        "Every element is divided by 100 without estimating statistics or changing spectral "
        "shape.",
        "Converting percent reflectance or transmittance to the fractional scale required by "
        "absorbance and Kubelka–Munk transforms.",
        "The operator does not detect current units; applying it to fractional or absorbance "
        "data silently creates an invalid scale.",
        "`n4m.transform.signal_conversion.PercentToFraction` calls "
        "`n4m_transform_percent_to_fraction_*`; the kernel is "
        "`signal_conversion/percent_to_fraction.c`.",
        _REPO
        + "cpp/src/core/preprocessing/signal_conversion/percent_to_fraction.c",
    ),
    "pp_piecewise_direct_standardization": _entry(
        "Piecewise direct standardization (PDS)",
        "Bouveresse & Massart (1996), *Improvement of the piecewise direct standardization "
        "procedure for the transfer of NIR spectra for multivariate calibration*, "
        "Chemometrics and Intelligent Laboratory Systems 32, 201–213, "
        "https://doi.org/10.1016/0169-7439(95)00074-7.",
        "For target wavelength $j$, PDS fits a local affine regression from a source window "
        "around $j$ to the paired target value at $j$. The fitted local coefficients form a "
        "banded transfer map; optional ridge regularization stabilizes each window solve.",
        "Instrument transfer when wavelength-local response differences make one dense global "
        "DS map unnecessarily flexible.",
        "Requires aligned paired standards. Window width trades locality against conditioning; "
        "edge windows differ in size and local linear maps cannot correct nonlinear detector "
        "effects.",
        "`n4m.domain_adaptation.standardization.PiecewiseDirectStandardization` uses "
        "`n4m_domain_adaptation_piecewise_direct_standardization_*`; window regressions are "
        "implemented in `cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1016/0169-7439(95)00074-7; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_piecewise_msc": _entry(
        "Piecewise multiplicative scatter correction",
        "No single paper canonically defines this fixed-window variant. It applies the MSC "
        "model of Geladi, MacDougall & Martens (1985), "
        "https://doi.org/10.1366/0003702854248656, independently by interval.",
        "The wavelength axis is partitioned into non-overlapping windows. Within each window "
        "and row, local intercept and slope are estimated against the corresponding reference "
        "segment, and every value in that window is corrected by $(x-a)/b$.",
        "Scatter correction when gain and offset differ between broad spectral regions.",
        "Independent windows can create discontinuities and short or flat segments make slopes "
        "unstable. The reference is training-dependent and the method can remove regional "
        "chemical amplitude differences.",
        "`n4m.transform.scatter.PiecewiseMSC` wraps `n4m_transform_piecewise_msc_*`; reference "
        "fitting and segment OLS are in `cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1366/0003702854248656; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_piecewise_snv": _entry(
        "Piecewise standard normal variate",
        "No single paper canonically defines this interval variant. It applies the SNV "
        "normalization of Barnes, Dhanoa & Lister (1989), "
        "https://doi.org/10.1366/0003702894202201, independently by interval.",
        "The axis is partitioned into non-overlapping windows of `window_size`. Within each "
        "window, each row is centered by its local mean and divided by its local standard "
        "deviation using `ddof`, with variance floored by `eps`.",
        "Removing region-dependent scatter while preserving separation between spectral "
        "regions better than a single global SNV statistic.",
        "Window boundaries can cause jumps, and low-variance or narrow intervals amplify "
        "noise. Local normalization may erase broad analyte differences between regions.",
        "`n4m.transform.scatter.PiecewiseSNV` calls `n4m_transform_piecewise_snv_*`; the exact "
        "interval and variance rules are in `cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1366/0003702894202201; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_range_disc": _entry(
        "Fixed-edge range discretization",
        "No canonical paper: this is scalar quantization against user-supplied ordered edges, "
        "equivalent in concept to `numpy.digitize`: "
        "https://numpy.org/doc/stable/reference/generated/numpy.digitize.html.",
        "Every value is replaced by the integer index of the interval delimited by the stored "
        "monotonic numeric edges. Unlike k-bin discretization, no distribution statistics are "
        "learned from $X$.",
        "Applying domain-defined concentration or intensity bands consistently across datasets.",
        "Discretization loses continuous information and results depend on edge inclusion "
        "semantics. The same edges are applied to all columns and may be unsuitable for "
        "wavelengths with different scales.",
        "`n4m.transform.resampling.RangeDiscretizer` wraps "
        "`n4m_transform_range_discretizer_*`; edge validation and bin lookup are in "
        "`resampling/range_discretizer.c`.",
        "https://numpy.org/doc/stable/reference/generated/numpy.digitize.html; "
        + _REPO
        + "cpp/src/core/preprocessing/resampling/range_discretizer.c",
    ),
    "pp_resample": _entry(
        "Fixed-length normalized-axis resampling",
        "No unique canonical paper: this is piecewise-linear interpolation on normalized "
        "sample positions, matching SciPy `interp1d(kind='linear')`: "
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html.",
        "For an input of width $p$ and requested width $q$, source and destination coordinates "
        "are `linspace(0,1,p)` and `linspace(0,1,q)`. Each row is linearly interpolated; equal "
        "width is an exact copy and `-1` is identity.",
        "Making variable-width or differently sampled spectra a common feature length when "
        "physical wavelength coordinates are unavailable.",
        "Normalized positions assume matching endpoints and uniform semantic coverage. The "
        "operator ignores actual wavelengths and linear interpolation can smooth narrow bands.",
        "`n4m.transform.resampling.ResampleTransformer` uses "
        "`n4m_transform_resample_transformer_*`; the linspace and interpolation arithmetic are "
        "in `resampling/resample_transformer.c`.",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html; "
        + _REPO
        + "cpp/src/core/preprocessing/resampling/resample_transformer.c",
    ),
    "pp_resampler": _entry(
        "Wavelength-grid resampler",
        "No single canonical paper: the supported linear, nearest, and not-a-knot cubic "
        "interpolants follow SciPy interpolation semantics documented at "
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html.",
        "Fit stores a strictly increasing source wavelength grid, optional crop indices, and "
        "target bracketing. Transform evaluates each row on configured target wavelengths by "
        "linear, nearest, or not-a-knot cubic interpolation, with explicit fill/extrapolation "
        "rules.",
        "Harmonizing spectra acquired on different physical wavelength grids before transfer "
        "or modeling.",
        "Both grids must be ordered and represent the same physical coordinate. Extrapolation "
        "is poorly constrained; cubic interpolation can overshoot and resampling introduces "
        "correlated errors.",
        "`n4m.transform.resampling.Resampler` wraps `n4m_transform_resampler_*`; cached "
        "brackets and all three interpolation paths are in `resampling/resampler.c`.",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html; "
        + _REPO
        + "cpp/src/core/preprocessing/resampling/resampler.c",
    ),
    "pp_rnv": _entry(
        "Robust normal variate (RNV)",
        "Guo, Wu & Massart (1999), *The robust normal variate transform for pattern "
        "recognition with near-infrared data*, Analytica Chimica Acta 382, 87–103, "
        "https://doi.org/10.1016/S0003-2670(98)00737-5.",
        "For each row, RNV subtracts its median and divides by `k` times its median absolute "
        "deviation, according to the enabled centering/scaling flags. The default "
        "$k=1.4826$ makes MAD consistent for Gaussian scale.",
        "Scatter normalization when isolated spikes or intense bands would destabilize the "
        "mean and standard deviation used by SNV.",
        "MAD is zero for sufficiently flat or tied spectra, requiring guarded behaviour, and "
        "robust row scaling can still erase meaningful absolute intensity. It does not model "
        "wavelength-local scatter.",
        "`n4m.transform.scatter.RNV` uses `n4m_transform_robust_snv_*`; median/MAD selection "
        "and flags are implemented in `preprocessing/scatter/robust_snv.c`.",
        "https://doi.org/10.1016/S0003-2670(98)00737-5; "
        + _REPO
        + "cpp/src/core/preprocessing/scatter/robust_snv.c",
    ),
    "pp_robust_direct_standardization": _entry(
        "Trimmed robust direct standardization",
        "No single canonical paper defines this implementation. It robustifies direct "
        "standardization from Wang et al. (1991), https://doi.org/10.1021/ac00023a016, by "
        "iterative residual trimming.",
        "A global affine DS map is fit on paired spectra. After each fit, row residual norms "
        "are computed and only rows at or below `trim_quantile` are retained for the next fit, "
        "up to `max_iter`; optional ridge regularization remains active.",
        "Instrument transfer with a small proportion of mismatched or corrupted paired "
        "standards.",
        "Hard trimming can discard legitimate domain extremes and the method is not a formal "
        "high-breakdown estimator. It still assumes one global linear map and enough retained "
        "pairs to identify it.",
        "`n4m.domain_adaptation.standardization.RobustDirectStandardization` calls "
        "`n4m_domain_adaptation_robust_direct_standardization_*`; trimming is implemented in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1021/ac00023a016; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp",
    ),
    "pp_rolling_ball": _entry(
        "Rolling-ball morphological baseline correction",
        "Kneen & Annegarn (1996), *Algorithm for fitting XRF, SEM and PIXE X-ray spectra "
        "backgrounds*, Nuclear Instruments and Methods in Physics Research B 109, 209–213, "
        "https://doi.org/10.1016/0168-583X(95)00908-6.",
        "A lower morphological envelope is estimated with a structuring radius "
        "`half_window` (erosion followed by dilation/rolling-ball analogue), optionally "
        "smoothed, and subtracted from the spectrum.",
        "Removing slowly varying positive backgrounds without fitting a global polynomial.",
        "Features wider than the structuring element may enter the baseline, while too large a "
        "window underfits drift. Morphological operations can create edge artifacts and are "
        "not differentiable.",
        "`n4m.transform.baseline.RollingBall` uses `n4m_transform_rolling_ball_*`; the 1-D "
        "morphological envelope is in `preprocessing/baselines/rolling_ball.c`.",
        "https://doi.org/10.1016/0168-583X(95)00908-6; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/rolling_ball.c",
    ),
    "pp_saps": _entry(
        "Score-augmented projection standardization (SAPS)",
        "No canonical publication defines this exact n4m transform. It is an internal "
        "score-augmented linear standardization inspired by projection-based calibration "
        "transfer; the source is the normative specification.",
        "Fit approximates leading source covariance eigenvectors by power iteration, computes "
        "centered source scores, appends `score_weight` times those scores to the original "
        "features, and fits a multi-output affine/ridge map to paired target spectra. Transform "
        "recomputes scores using the stored source mean and loadings before applying that map.",
        "Paired-instrument transfer where a direct linear map benefits from explicit source "
        "latent coordinates.",
        "Power iteration is fixed at 30 steps per deflated component and is not a full robust "
        "PCA solver. It requires paired equal-shape data; component count, score weight, and "
        "ridge penalty can overfit.",
        "`n4m.domain_adaptation.standardization.ScoreAugmentedProjectionStandardization` uses "
        "`n4m_transform_saps_*`; the exact augmentation and OLS/ridge fit are in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        _REPO + "cpp/src/c_api/c_api_advanced.cpp#L356-L496",
    ),
    "pp_savgol": _entry(
        "Savitzky–Golay smoothing and differentiation",
        "Savitzky & Golay (1964), *Smoothing and Differentiation of Data by Simplified Least "
        "Squares Procedures*, Analytical Chemistry 36, 1627–1639, "
        "https://doi.org/10.1021/ac60214a047.",
        "In every odd window, a polynomial of degree `polyorder` is fit by least squares. The "
        "value or derivative of order `deriv` at the center is a precomputed convolution of "
        "the window, scaled by `delta`; `mode` defines edge extension.",
        "Denoising while preserving polynomial peak shape, or estimating smoothed first and "
        "second derivatives.",
        "Window, degree, derivative, and grid spacing must be coherent. Derivatives amplify "
        "noise, large windows merge narrow bands, and edge modes produce different boundary "
        "values.",
        "`n4m.transform.smoothing.SavitzkyGolay` wraps "
        "`n4m_transform_savitzky_golay_*`; coefficient generation and modes are in "
        "`preprocessing/derivatives/savitzky_golay.c`.",
        "https://doi.org/10.1021/ac60214a047; "
        + _REPO
        + "cpp/src/core/preprocessing/derivatives/savitzky_golay.c",
    ),
    "pp_second_derivative": _entry(
        "Shape-preserving second numerical derivative",
        "No unique paper defines two `numpy.gradient` passes; the spectroscopic derivative "
        "rationale follows Norris & Williams (1984), while the numerical reference is "
        "https://numpy.org/doc/stable/reference/generated/numpy.gradient.html.",
        "The implementation applies the shape-preserving first-gradient stencil twice along "
        "each row, using `delta` and the requested boundary `edge_order` at both passes. It "
        "approximates $d^2x/d\\lambda^2$ without dropping edge columns.",
        "Suppressing constant and linear backgrounds and resolving overlapping absorption "
        "bands.",
        "Second differentiation strongly amplifies high-frequency noise and assumes a uniform "
        "grid. Boundary estimates compound one-sided errors and smoothing is usually needed.",
        "`n4m.transform.smoothing.SecondDerivative` calls "
        "`n4m_transform_second_derivative_*`; both gradient passes are in "
        "`preprocessing/derivatives/second_derivative.c`.",
        "https://numpy.org/doc/stable/reference/generated/numpy.gradient.html; "
        + _REPO
        + "cpp/src/core/preprocessing/derivatives/second_derivative.c",
    ),
    "pp_simple_scale": _entry(
        "Column-wise min–max scaling to [0, 1]",
        "No canonical paper: this is the affine min–max transform "
        "$(x_j-\\min x_j)/(\\max x_j-\\min x_j)$.",
        "For each column, the operation computes its minimum and maximum over the supplied "
        "rows and maps the extrema to zero and one, respectively.",
        "Putting wavelength variables on a common bounded numeric scale before distance- or "
        "regularization-sensitive procedures.",
        "Extrema are outlier-sensitive and, in this stateless operation, are recomputed on each "
        "input matrix rather than retained from training. Constant columns require guarded "
        "handling and absolute spectral scale is lost.",
        "`n4m.transform.scaling.SimpleScale` uses `n4m_transform_simple_scale_*`; column extrema "
        "and affine scaling are in `preprocessing/scaling/simple_scale.c`.",
        _REPO + "cpp/src/core/preprocessing/scaling/simple_scale.c",
    ),
    "pp_slope_bias": _entry(
        "Slope-and-bias prediction correction",
        "No single canonical paper defines this two-parameter post-calibration operation. It "
        "is ordinary least-squares bias/slope correction, with source code as the exact "
        "specification.",
        "Given paired source predictions $x$ and target values $y$, fit estimates "
        "$y=ax+b$ by closed-form OLS. Transform returns $ax+b$ for new source predictions; it "
        "acts on a one-dimensional prediction vector, not on spectra.",
        "Correcting a stable linear bias and gain error after moving a calibration between "
        "instruments or populations.",
        "A constant source prediction makes the slope unidentified. The correction cannot fix "
        "nonlinear, heteroscedastic, or wavelength-specific transfer errors and needs paired "
        "target values.",
        "`n4m.domain_adaptation.standardization.SlopeBiasCorrection` wraps "
        "`n4m_domain_adaptation_slope_bias_*`; the closed-form fit is in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        _REPO + "cpp/src/c_api/c_api_advanced.cpp#L499-L523",
    ),
    "pp_snip": _entry(
        "Statistics-sensitive nonlinear iterative peak clipping (SNIP)",
        "Ryan et al. (1988), *SNIP, a statistics-sensitive background treatment for the "
        "quantitative analysis of PIXE spectra in geoscience applications*, Nuclear "
        "Instruments and Methods B 34, 396–402, "
        "https://doi.org/10.1016/0168-583X(88)90063-8.",
        "After a stabilizing transform, iterative half-windows compare each center with a "
        "symmetric neighborhood estimate and clip peaks downward; reversing the transform "
        "yields a slowly varying baseline that is subtracted.",
        "Background removal for spectra dominated by positive peaks with a characteristic "
        "maximum peak width.",
        "`max_half_window` sets which structures are treated as peaks; broad bands can be "
        "removed and boundaries receive fewer symmetric comparisons. Published SNIP variants "
        "differ in transform and window order.",
        "`n4m.transform.baseline.SNIP` uses `n4m_transform_snip_*`; the exact transform and "
        "clipping schedule are in `preprocessing/baselines/snip.c`.",
        "https://doi.org/10.1016/0168-583X(88)90063-8; "
        + _REPO
        + "cpp/src/core/preprocessing/baselines/snip.c",
    ),
    "pp_snv": _entry(
        "Standard normal variate (SNV)",
        "Barnes, Dhanoa & Lister (1989), *Standard Normal Variate Transformation and "
        "De-trending of Near-Infrared Diffuse Reflectance Spectra*, Applied Spectroscopy "
        "43, 772–777, https://doi.org/10.1366/0003702894202201.",
        "Each spectrum is independently centered and scaled: "
        "$x'_i=(x_i-\\bar{x}_i)/s_i$, with optional centering, scaling, and configurable "
        "`ddof`. Row statistics make the transform stateless across samples.",
        "Reducing additive and multiplicative scatter caused by particle size or optical path "
        "variation in diffuse-reflectance spectra.",
        "Near-constant spectra have unstable scale, and SNV removes absolute row amplitude that "
        "may contain analyte information. It does not remove wavelength-dependent baselines.",
        "`n4m.transform.scatter.SNV` wraps `n4m_transform_snv_*`; row mean/variance and flags "
        "are in `preprocessing/scatter/snv.c`.",
        "https://doi.org/10.1366/0003702894202201; "
        + _REPO
        + "cpp/src/core/preprocessing/scatter/snv.c",
    ),
    "pp_to_absorbance": _entry(
        "Reflectance/transmittance-to-absorbance conversion",
        "Beer (1852), *Bestimmung der Absorption des rothen Lichts in farbigen "
        "Flüssigkeiten*, Annalen der Physik 162, 78–88, "
        "https://doi.org/10.1002/andp.18521620505.",
        "Percent inputs are first divided by 100, then the operator computes "
        "$A=-\\log_{10}(\\max(R,\\epsilon))$. With `clip_negative`, negative signals are guarded "
        "before the logarithm according to the implementation contract.",
        "Converting reflectance or transmittance into a log scale closer to additive optical "
        "density before calibration.",
        "The transform assumes correct input units and positive signals. Clipping hides invalid "
        "measurements, and reflectance absorbance is an empirical pseudo-absorbance rather than "
        "a guaranteed Beer–Lambert quantity.",
        "`n4m.transform.signal_conversion.ToAbsorbance` uses "
        "`n4m_transform_to_absorbance_*`; unit handling and guards are in "
        "`signal_conversion/to_absorbance.c`.",
        "https://doi.org/10.1002/andp.18521620505; "
        + _REPO
        + "cpp/src/core/preprocessing/signal_conversion/to_absorbance.c",
    ),
    "pp_vsn": _entry(
        "Variable-sorting normalization (VSN-style weighted SNV)",
        "Rabatel, Marini, Walczak & Roger (2020), *VSN: Variable sorting for normalization*, "
        "Journal of Chemometrics 34, e3164, https://doi.org/10.1002/cem.3164. The shipped "
        "correlation-weighted rule is a compact VSN-style implementation, not a claim of full "
        "paper parity.",
        "Fit computes each wavelength's absolute correlation across samples with the vector of "
        "per-spectrum means, floors correlations by `eps`, and normalizes them to weights. "
        "Transform uses those weights for each row's mean and variance before standardization.",
        "Emphasizing wavelengths associated with global row-level scatter while performing SNV "
        "normalization.",
        "The learned weights can capture chemical or batch structure instead of scatter and "
        "must be fit within folds. This implementation should not be assumed equivalent to "
        "other algorithms also called VSN.",
        "`n4m.transform.scatter.VariableSortingNormalization` wraps `n4m_transform_vsn_*`; "
        "weight learning and weighted standardization are in `cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1002/cem.3164; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp#L560-L635",
    ),
    "pp_wavelet": _entry(
        "Single-level discrete wavelet coefficient transform",
        "Mallat (1989), *A theory for multiresolution signal decomposition: the wavelet "
        "representation*, IEEE TPAMI 11, 674–693, "
        "https://doi.org/10.1109/34.192463.",
        "A one-level analysis filter bank convolves each spectrum with the selected wavelet's "
        "low-pass and high-pass filters and downsamples by two. Approximation coefficients are "
        "concatenated with detail coefficients; the configured extension mode determines edge "
        "samples.",
        "Separating coarse spectral shape from fine-scale detail or supplying wavelet-domain "
        "features to a model.",
        "The coefficients are shift-sensitive and their length/order depend on wavelet family, "
        "boundary mode, and input width. They are not on the original wavelength grid.",
        "`n4m.transform.wavelet.Wavelet` wraps `n4m_transform_wavelet_*`; orchestration is in "
        "`preprocessing/wavelets/wavelet.c` and shared filters are in `wavelet_kernels.c`.",
        "https://doi.org/10.1109/34.192463; "
        + _REPO
        + "cpp/src/core/preprocessing/wavelets/wavelet.c; "
        + _REPO
        + "cpp/src/core/common/wavelet_kernels.c",
    ),
    "pp_wavelet_denoise": _entry(
        "Multilevel wavelet VisuShrink denoising",
        "Donoho & Johnstone (1994), *Ideal spatial adaptation by wavelet shrinkage*, "
        "Biometrika 81, 425–455, https://doi.org/10.1093/biomet/81.3.425.",
        "Each row is decomposed to the requested feasible level. Noise scale is estimated from "
        "the finest detail by MAD/0.6745 or population standard deviation; the universal "
        "threshold $\\sigma\\sqrt{2\\log p}$ is applied by hard or soft shrinkage to every detail "
        "band before inverse reconstruction.",
        "Removing approximately independent high-frequency noise while retaining multiscale "
        "spectral structure.",
        "Universal thresholding can oversmooth weak bands, assumes a noise model, and is "
        "shift-sensitive. Results depend on family, level, boundary mode, estimator, and "
        "hard/soft choice.",
        "`n4m.transform.wavelet.WaveletDenoise` calls `n4m_transform_wavelet_denoise_*`; the "
        "threshold and reconstruction pipeline is in `wavelet_denoise.c`.",
        "https://doi.org/10.1093/biomet/81.3.425; "
        + _REPO
        + "cpp/src/core/preprocessing/wavelets/wavelet_denoise.c",
    ),
    "pp_wavelet_features": _entry(
        "Multilevel wavelet summary features",
        "No single canonical paper defines this exact four-statistic descriptor. It uses the "
        "multiresolution analysis of Mallat (1989), https://doi.org/10.1109/34.192463, with "
        "implementation-specific summaries.",
        "After multilevel DWT, every approximation/detail band contributes four values: mean, "
        "population standard deviation, energy $\\sum c_i^2$, and either normalized-energy "
        "entropy or ten-bin histogram entropy. Output width is $4(L+1)$.",
        "Producing compact, fixed-size multiscale descriptors for classification or regression.",
        "Band summaries discard coefficient position and sign structure beyond the mean. "
        "Entropy definitions are implementation-specific and depend on family, mode, and "
        "feasible decomposition depth.",
        "`n4m.transform.wavelet.WaveletFeatures` uses "
        "`n4m_transform_wavelet_features_*`; all four summaries and entropy modes are in "
        "`preprocessing/wavelets/wavelet_features.c`.",
        "https://doi.org/10.1109/34.192463; "
        + _REPO
        + "cpp/src/core/preprocessing/wavelets/wavelet_features.c",
    ),
    "pp_wavelet_pca": _entry(
        "PCA of multilevel wavelet coefficients",
        "No unique paper defines this composite. It combines Mallat's DWT "
        "(https://doi.org/10.1109/34.192463) with Pearson PCA "
        "(https://doi.org/10.1080/14786440109462720).",
        "Each row's multilevel DWT coefficients are packed into one vector. Fit centers that "
        "coefficient matrix, performs compact SVD, and retains an integer component count or "
        "the smallest count reaching the requested explained-variance fraction; transform "
        "projects centered coefficients onto those loadings.",
        "Joint multiscale denoising and dimensionality reduction when raw wavelet coefficients "
        "are too numerous for the downstream model.",
        "Both DWT and PCA are scale/shift sensitive; interpretation mixes bands and fitting is "
        "training-dependent. A variance threshold can yield changing output dimension across "
        "folds.",
        "`n4m.transform.wavelet.WaveletPCA` wraps `n4m_transform_wavelet_pca_*`; packing, "
        "centering, SVD, selection, and projection are in `wavelet_pca.c`.",
        "https://doi.org/10.1109/34.192463; "
        "https://doi.org/10.1080/14786440109462720; "
        + _REPO
        + "cpp/src/core/preprocessing/wavelets/wavelet_pca.c",
    ),
    "pp_wavelet_svd": _entry(
        "Truncated SVD of multilevel wavelet coefficients",
        "No unique paper defines this composite. It combines Mallat's DWT "
        "(https://doi.org/10.1109/34.192463) with Eckart–Young low-rank approximation "
        "(https://doi.org/10.1007/BF02288367).",
        "Rows are mapped to packed multilevel DWT coefficients and compact SVD is fitted "
        "without the explicit centering used by WaveletPCA. The operator keeps an integer rank "
        "or the smallest rank reaching the requested singular-value energy fraction and emits "
        "right-singular-vector scores.",
        "Low-rank multiscale features when retaining the coefficient-matrix mean direction is "
        "intentional.",
        "Uncentered SVD can spend its first direction on mean offset. The learned basis is "
        "training-dependent, shift-sensitive through the DWT, and output rank may vary for a "
        "fraction threshold.",
        "`n4m.transform.wavelet.WaveletSVD` calls `n4m_transform_wavelet_svd_*`; coefficient "
        "packing and uncentered SVD are in `wavelet_svd.c`.",
        "https://doi.org/10.1109/34.192463; https://doi.org/10.1007/BF02288367; "
        + _REPO
        + "cpp/src/core/preprocessing/wavelets/wavelet_svd.c",
    ),
    "pp_weighted_snv": _entry(
        "Weighted standard normal variate",
        "No single canonical paper defines this exact weighted estimator. It generalizes the "
        "SNV transform of Barnes, Dhanoa & Lister (1989), "
        "https://doi.org/10.1366/0003702894202201.",
        "Fit validates and normalizes supplied nonnegative wavelength weights, or uses uniform "
        "weights. For every row it computes weighted mean and variance, applies the configured "
        "`ddof` correction, floors variance by `eps`, and standardizes all wavelengths with "
        "that weighted location and scale.",
        "Reducing scatter while making trusted or diagnostically useful wavelength regions "
        "dominate the normalization statistics.",
        "Weights require scientific justification and can bias the whole row statistic toward "
        "a small band. Negative/zero-total weights are invalid, and normalization can remove "
        "absolute amplitude information.",
        "`n4m.transform.scatter.WeightedSNV` uses `n4m_transform_weighted_snv_*`; normalized "
        "weights and weighted moments are in `cpp/src/c_api/c_api_advanced.cpp`.",
        "https://doi.org/10.1366/0003702894202201; "
        + _REPO
        + "cpp/src/c_api/c_api_advanced.cpp#L526-L635",
    ),
    "pp_xcorr_align": _entry(
        "Whole-spectrum cross-correlation alignment",
        "No single canonical paper defines this bounded implementation. It applies the "
        "standard discrete cross-correlation lag estimator; the source defines padding and "
        "tie behaviour.",
        "For each row, every integer lag in `[-max_shift,max_shift]` is evaluated by the "
        "centered dot product with the fitted reference. The best lag is applied to the whole "
        "spectrum using edge-replicated samples.",
        "Correcting one global channel offset per spectrum before comparison or calibration.",
        "It cannot model local stretching, and amplitude/shape changes can move the correlation "
        "maximum. Edge replication affects shifted ends; weak or periodic spectra can have "
        "ambiguous lags.",
        "`n4m.transform.alignment.CrossCorrelationAlignment` wraps "
        "`n4m_transform_xcorr_align_*`; reference fitting and lag search are in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        _REPO + "cpp/src/c_api/c_api_advanced.cpp#L798-L900",
    ),
    "interval_generator": _entry(
        "Fixed and overlapping interval expansion",
        "No canonical paper: this is a deterministic interval-construction and column-copy "
        "operation; `interval_fit` is its normative definition.",
        "Fit creates half-open bands of width `interval_size` starting every `step` columns "
        "(`step=interval_size` by default), truncating the last band at the feature count. "
        "Transform concatenates every band, so overlapping columns are deliberately repeated.",
        "Building interval blocks for downstream interval selection or block-wise models, with "
        "optional overlap.",
        "It does not score or select intervals. Overlap increases output dimension and repeats "
        "features, while position indices assume an already aligned common wavelength grid.",
        "`n4m.feature_selection.interval.IntervalGenerator` and `interval_generator` use "
        "`n4m_feature_selection_interval_generator_*`; construction/copying are in "
        "`cpp/src/c_api/c_api_advanced.cpp`.",
        _REPO + "cpp/src/c_api/c_api_advanced.cpp#L1225-L1303",
    ),
    "aom_pop_ridge_global": _entry(
        "AOM Ridge global operator selector",
        "No canonical paper defines this n4m composition. It combines fold-local model "
        "selection over strict-linear preprocessing operators with classical Ridge regression "
        "(Hoerl & Kennard, 1970, https://doi.org/10.1080/00401706.1970.10488634).",
        "For every candidate single operator and positive Ridge alpha, training-fold data fit "
        "the preprocessing and Ridge model and validation RMSE scores the candidate. The best "
        "operator/alpha pair is refit on all calibration data and its linear map is folded back "
        "to input-space coefficients and intercept.",
        "Selecting one deployable linear preprocessing view and regularization strength without "
        "leaking validation folds.",
        "This Python composition excludes nonlinear and row-reference-dependent operators and "
        "is not a fused C ABI grinder. Selection uncertainty can be high when CV scores are "
        "close.",
        "Python-only `n4m.model_selection.aom_search.aom_ridge_global` delegates to the native "
        "`aom_chain_sweep_run` Ridge path and returns `NativeAOMRidgeGlobalRegressor`; there is "
        "no dedicated `ridge_global` C ABI symbol.",
        _REPO + "bindings/python/src/n4m/model_selection/aom_search.py; "
        + _REPO
        + "bindings/python/src/n4m/_impl/native_sweeps.py#L1399-L1455",
    ),
    "aom_pop_ridge_superblock": _entry(
        "AOM Ridge concatenated superblock",
        "No canonical paper defines this exact n4m composition. Ridge follows Hoerl & Kennard "
        "(1970), https://doi.org/10.1080/00401706.1970.10488634; block concatenation and "
        "folding are implementation-specific.",
        "Each strict-linear operator maps the same spectrum to a block; blocks are centered on "
        "training folds and optionally divided by their training RMS, then concatenated. Ridge "
        "alpha is selected fold-locally, the model is refit on all data, and block coefficients "
        "are algebraically folded into one input-space linear predictor.",
        "Combining complementary linear preprocessing views while retaining a compact linear "
        "deployment model.",
        "Correlated blocks duplicate information and can dominate without scaling. Only "
        "strict-linear operators can be folded; reference-dependent/nonlinear branches and a "
        "dedicated native superblock solver are absent.",
        "Python-only `n4m.compose.aom_superblock.aom_ridge_superblock` uses native "
        "`aom_preprocess` and Ridge primitives via `NativeAOMRidgeSuperblockRegressor`; no "
        "dedicated C ABI symbol exists.",
        _REPO + "bindings/python/src/n4m/compose/aom_superblock.py; "
        + _REPO
        + "bindings/python/src/n4m/_impl/native_sweeps.py#L3267-L3363",
    ),
    "aom_pop_ridge_mkl_superblock": _entry(
        "AOM Ridge MKL-light weighted superblock",
        "Kernel-target alignment originates with Cristianini et al. (2002), *On Kernel-Target "
        "Alignment*, NIPS 14, https://proceedings.neurips.cc/paper/2001/hash/1f71e393b3809197ed66df836fe833e5-Abstract.html; "
        "this linear nonnegative weighting scheme is n4m-specific.",
        "Within each training fold, centered operator blocks receive nonnegative weights from "
        "their kernel-target alignment (KTA), the weighted blocks are concatenated, and Ridge "
        "alpha is scored on held-out rows. Full-data KTA weights and Ridge are refit, then the "
        "linear predictor is folded back to original features.",
        "Downweighting uninformative preprocessing views while combining several strict-linear "
        "operator blocks.",
        "This is 'MKL-light', not general multiple-kernel learning: weights use target alignment "
        "rather than joint convex optimization. It excludes nonlinear kernels and requires all "
        "weight estimation inside folds.",
        "Python-only `n4m.compose.aom_superblock.aom_ridge_mkl_superblock` is implemented by "
        "`NativeAOMRidgeMKLSuperblockRegressor` using native preprocessing/Ridge calls; no "
        "dedicated C ABI symbol exists.",
        "https://proceedings.neurips.cc/paper/2001/hash/1f71e393b3809197ed66df836fe833e5-Abstract.html; "
        + _REPO
        + "bindings/python/src/n4m/_impl/native_sweeps.py#L3364-L3477",
    ),
    "aom_pop_ridge_active_superblock": _entry(
        "AOM Ridge active-set superblock",
        "No canonical publication defines this n4m active-superblock heuristic. It combines "
        "response-signature screening with Ridge regression; Ridge follows Hoerl & Kennard "
        "(1970), https://doi.org/10.1080/00401706.1970.10488634.",
        "Inside each alpha-CV training fold, native AOM preprocessing produces strict-linear "
        "operator blocks. Response signatures screen an active subset, only those blocks enter "
        "the Ridge superblock, and the active set is re-estimated on full calibration data "
        "before coefficients are folded to input space.",
        "Reducing a large redundant operator bank before fitting a deployable linear "
        "multi-view Ridge model.",
        "Screening is heuristic and can discard complementary weak blocks; it adds a selection "
        "layer whose stability should be checked across folds. Nonlinear/reference-dependent "
        "operators and a fused C ABI implementation are excluded.",
        "Python-only `n4m.compose.aom_superblock.aom_ridge_active_superblock` is implemented by "
        "`NativeAOMRidgeActiveSuperblockRegressor`; native `aom_preprocess` and Ridge kernels are "
        "called underneath, with no dedicated C ABI symbol.",
        "https://doi.org/10.1080/00401706.1970.10488634; "
        + _REPO
        + "bindings/python/src/n4m/_impl/native_sweeps.py#L3478-L3606",
    ),
    "diagnostics_pls_diagnostics": _entry(
        "PLS model diagnostics: Hotelling T², Q residuals, and DModX",
        "Hotelling (1931), *The Generalization of Student's Ratio*, "
        "https://doi.org/10.1214/aoms/1177732979; Jackson & Mudholkar (1979), *Control "
        "Procedures for Residuals Associated With Principal Component Analysis*, "
        "https://doi.org/10.1080/00401706.1979.10489779; Wold et al. (2001), *PLS-regression: "
        "a basic tool of chemometrics*, https://doi.org/10.1016/S0169-7439(01)00155-1.",
        "Rows are centered by the fitted PLS $X$ mean and projected with the model rotations. "
        "$T^2_i=\\sum_a t_{ia}^2/s_a^2$ measures score-space distance; "
        "$Q_i=\\|x_i-\\hat{x}_i\\|^2$ measures reconstruction residual; DModX scales "
        "$\\sqrt{Q_i}$ by residual degrees of freedom and, when supplied, a reference residual "
        "scale. One call returns all three vectors plus model dimensions.",
        "Diagnosing extrapolation, leverage, and spectral lack of fit for observations evaluated "
        "against one fitted PLS model.",
        "Statistics are model-relative and component-count dependent. T² needs stored training "
        "scores or an explicit reference; DModX scaling is unreliable with too few residual "
        "degrees of freedom. This is distinct from the standalone PCA utilities below.",
        "The C ABI entry is `n4m_metrics_pls_diagnostics_compute`, producing an "
        "`n4m_method_result_t`. The current public Python `n4m.metrics.diagnostics` module does "
        "not yet expose a wrapper; formulas are in `cpp/src/core/pls_diagnostics.cpp`.",
        "https://doi.org/10.1214/aoms/1177732979; "
        "https://doi.org/10.1080/00401706.1979.10489779; "
        + _REPO
        + "cpp/src/core/pls_diagnostics.cpp",
    ),
    "diagnostics_regression_metrics": _entry(
        "Regression and NIR prediction metrics",
        "No single paper defines this API bundle. RPD usage in NIR calibration is reviewed by "
        "Williams & Sobering (1993); RPIQ by Bellon-Maurel et al. (2010), "
        "https://doi.org/10.1016/j.trac.2010.07.005. RMSE, MAE, bias, SEP, R², and NRMSE use "
        "the explicit formulas in the implementation.",
        "For residuals $e_i=\\hat y_i-y_i$, the API computes RMSE, MAE, mean bias, and "
        "population SEP $=\\mathrm{sd}(e)$. It also returns "
        "$RPD=\\mathrm{sd}(y)/SEP$, $RPIQ=IQR(y)/RMSE$, "
        "$R^2=1-SSE/SST$, and $NRMSE=RMSE/\\bar y$; quartiles use linear interpolation.",
        "Reporting predictive error, systematic bias, variance-normalized performance, and "
        "robust range-normalized performance for calibration/validation results.",
        "RPD/RPIQ depend on the response distribution and cannot be compared blindly across "
        "populations. NRMSE divides by the signed response mean: it can be negative when that mean is "
        "negative and its magnitude is unstable near zero (infinite only at exactly zero). RPD/RPIQ "
        "diverge for perfect predictions. This implementation uses population (`ddof=0`) SEP and SD.",
        "Python functions `n4m.metrics.scoring.rmse`, `mae`, `bias`, `sep`, `rpd`, `rpiq`, "
        "`r2`, and `nrmse` wrap the eight "
        "`n4m_metrics_regression_metrics_<name>` ABI-2 symbols; formulas are in "
        "`cpp/src/core/utilities/nirs_metrics.c`.",
        "https://doi.org/10.1016/j.trac.2010.07.005; "
        + _REPO
        + "cpp/src/core/utilities/nirs_metrics.c",
    ),
    "utilities_hotelling_t2": _entry(
        "Standalone PCA Hotelling T² statistic",
        "Hotelling (1931), *The Generalization of Student's Ratio*, Annals of Mathematical "
        "Statistics 2, 360–378, https://doi.org/10.1214/aoms/1177732979.",
        "The input matrix is column-centered and decomposed by compact SVD. For the first $k$ "
        "PCA scores, $T_i^2=\\sum_{j=1}^k t_{ij}^2/\\lambda_j$. The upper control limit is "
        "$k(n-1)(n+1)/[n(n-k)]\\,F_{1-\\alpha}(k,n-k)$, with the F quantile obtained through "
        "regularized incomplete-beta inversion.",
        "Flagging multivariate score-space extremes directly from a calibration matrix when no "
        "PLS model object is involved.",
        "The implementation requires `rows >= cols`, $k< n$, finite dense F64 data, and treats "
        "the same input as both PCA fit and scored set. The finite-sample F limit assumes the "
        "classical approximately normal model.",
        "`n4m.outlier_detection.hotelling_t2` wraps the stateless ABI-2 symbol "
        "`n4m_outlier_detection_hotelling_t2`; SVD, statistic, and UCL are in "
        "`cpp/src/core/utilities/hotelling_t2.c`.",
        "https://doi.org/10.1214/aoms/1177732979; "
        + _REPO
        + "cpp/src/core/utilities/hotelling_t2.c",
    ),
    "utilities_q_residuals": _entry(
        "Standalone PCA Q residuals (squared prediction error)",
        "Jackson & Mudholkar (1979), *Control Procedures for Residuals Associated With "
        "Principal Component Analysis*, Technometrics 21, 341–349, "
        "https://doi.org/10.1080/00401706.1979.10489779.",
        "After column centering and compact SVD, the first $k$ components reconstruct each row. "
        "$Q_i=\\|x_i-\\hat x_i\\|_2^2$. The upper limit uses the Jackson–Mudholkar moment "
        "approximation from residual eigenvalue sums $\\theta_1,\\theta_2,\\theta_3$ and the "
        "standard-normal quantile at $1-\\alpha$.",
        "Detecting observations whose variation lies outside a retained PCA subspace, "
        "complementing score-space Hotelling T².",
        "The implementation requires `rows >= cols` and fits/scores the same matrix. The "
        "Jackson–Mudholkar approximation can be poor for small samples, nonnormal residuals, or "
        "near-degenerate residual eigenvalues.",
        "`n4m.outlier_detection.q_residuals` wraps "
        "`n4m_outlier_detection_q_residuals`; reconstruction and the moment-based UCL are in "
        "`cpp/src/core/utilities/q_residuals.c`.",
        "https://doi.org/10.1080/00401706.1979.10489779; "
        + _REPO
        + "cpp/src/core/utilities/q_residuals.c",
    ),
    "utilities_signal_type_detector": _entry(
        "Heuristic spectral signal-type detector",
        "No canonical scientific paper defines these thresholds. This is an n4m compatibility "
        "heuristic, and `signal_type_detector.c` is the authoritative specification.",
        "NaN-skipping global min, max, mean, and standard deviation first detect centered, "
        "standardized, or derivative-like data and return `UNKNOWN`. Otherwise range/mean rules "
        "score absorbance, fractional/percent reflectance, and fractional/percent transmittance. "
        "Optional wavelength cues near 1450, 1940, and 2500 nm adjust scores; confidence is the "
        "best score divided by the score sum and is thresholded.",
        "Advisory detection of likely raw signal units before choosing absorbance or percentage "
        "conversion in an ingestion workflow.",
        "Reflectance and transmittance ranges overlap, so classification is inherently "
        "ambiguous. Water-band cues assume wavelengths in nm and representative NIR coverage. "
        "The enum contains additional types that this heuristic does not currently score.",
        "`n4m.transform.signal_conversion.signal_type_detector` wraps "
        "`n4m_transform_signal_type_detector`; all thresholds and the 256-byte reason string are "
        "defined in `cpp/src/core/utilities/signal_type_detector.c`.",
        _REPO + "cpp/src/core/utilities/signal_type_detector.c",
    ),
    "utilities_transfer_metrics": _entry(
        "Source–target representation transfer metrics",
        "This bundle has no single canonical paper. Its components include linear CKA "
        "(Kornblith et al., 2019, https://proceedings.mlr.press/v97/kornblith19a.html), "
        "Procrustes analysis (Gower, 1975, https://doi.org/10.1007/BF02291478), and "
        "trustworthiness (Venna & Kaski, 2001, "
        "https://doi.org/10.1016/S0893-6080(01)00150-6).",
        "Source and target are separately centered and reduced by PCA to a common effective "
        "rank. The result contains centroid distance, linear CKA, Grassmann principal-angle "
        "distance, RV coefficient, two-dimensional Procrustes disparity, neighborhood "
        "trustworthiness, a covariance-plus-nearest-cloud spread distance, and source/target "
        "explained-variance ratios.",
        "Quantifying several complementary aspects of dataset shift before or after calibration "
        "transfer.",
        "The nine values have different scales and directions and should not be collapsed "
        "without a declared policy. Grassmann is NaN when feature counts differ; Procrustes and "
        "trustworthiness need enough rows, while spread subsampling depends deterministically "
        "on `seed`.",
        "`n4m.domain_adaptation.metrics.transfer_metrics` wraps "
        "`n4m_domain_adaptation_transfer_metrics_compute`; the PCA/Jacobi and all nine exact "
        "definitions are in `cpp/src/core/utilities/transfer_metrics.c`.",
        "https://proceedings.mlr.press/v97/kornblith19a.html; "
        "https://doi.org/10.1007/BF02291478; "
        "https://doi.org/10.1016/S0893-6080(01)00150-6; "
        + _REPO
        + "cpp/src/core/utilities/transfer_metrics.c",
    ),
}
