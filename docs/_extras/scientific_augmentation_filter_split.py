"""Scientific documentation for augmentation, filters, and splitters.

The keys are generated method-page stems.  Text in this module is deliberately
kept separate from the renderer so that scientific provenance can be reviewed
without changing the documentation pipeline.
"""

from __future__ import annotations


_SRC = "https://github.com/GBeurier/nirs4all-methods/blob/main/"
_BJERRUM = "https://arxiv.org/abs/1710.01927"
_MIXUP = "https://arxiv.org/abs/1710.09412"
_EMSC = "https://doi.org/10.1016/0731-7085(91)80188-F"
_MSC = "https://doi.org/10.1366/0003702854248656"
_KS = "https://doi.org/10.1080/00401706.1969.10490666"
_SPXY = "https://doi.org/10.1016/j.talanta.2005.03.025"
_SPLIT = "https://doi.org/10.1080/00401706.2021.1921037"
_IFOREST = "https://doi.org/10.1109/ICDM.2008.17"
_LOF = "https://doi.org/10.1145/342009.335388"
_MCD = "https://doi.org/10.1080/00401706.1999.10485670"
_KMEANS_PP = "https://dl.acm.org/doi/10.5555/1283383.1283494"


SCIENTIFIC_CONTENT: dict[str, dict[str, str]] = {
    "aug_gaussian_noise": {
        "title": "Gaussian additive noise scaled to each spectrum",
        "paper": (
            "There is no single canonical paper for additive-noise augmentation. "
            f"Bjerrum, Glahder & Skov (2017), *Data Augmentation of Spectral Data for "
            f"CNN Based Deep Chemometrics*, arXiv:1710.01927 ({_BJERRUM}), is a "
            "spectroscopy-specific precedent for perturbing spectra during training."
        ),
        "principle": (
            "For row $i$, the implementation computes the population standard deviation "
            "$s_i=\\operatorname{std}(X_{i,:},\\mathrm{ddof}=0)$ and returns "
            "$X'_{ij}=X_{ij}+\\sigma s_i Z_{ij}$ with independent "
            "$Z_{ij}\\sim\\mathcal N(0,1)$. Thus `sigma` is a fraction of each spectrum's "
            "own spread; it is not an absolute noise standard deviation. `seed` or the "
            "supplied PCG64 stream fixes the draw sequence."
        ),
        "use_cases": (
            "Training-time robustness to approximately white detector or read noise when "
            "noise magnitude should follow the dynamic range of each spectrum."
        ),
        "limitations": (
            "Noise is independent across wavelengths and samples, with no smoothing or "
            "wavelength-dependent variance. Constant rows have $s_i=0$ and are unchanged; "
            "do not describe this operator as fixed-variance IID noise."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.noise.GaussianAdditiveNoise`; ABI 2 family "
            "`n4m_augmentation_gaussian_noise_{create,apply,destroy}`. The formula is in "
            "`cpp/src/core/augmentation/noise/gaussian_noise.c`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/noise/gaussian_noise.h; " + _BJERRUM,
    },
    "aug_multiplicative_noise": {
        "title": "Per-spectrum multiplicative gain noise",
        "paper": (
            "No publication uniquely defines this implementation. It belongs to the family "
            f"of spectral offset/slope/gain augmentation discussed by Bjerrum et al. (2017), "
            f"arXiv:1710.01927 ({_BJERRUM})."
        ),
        "principle": (
            "One draw is made per row: $g_i=1+\\sigma_g Z_i$, "
            "$Z_i\\sim\\mathcal N(0,1)$, then $X'_{ij}=g_iX_{ij}$ for every wavelength. "
            "`sigma_gain` controls relative gain variation, so all channels in a spectrum "
            "move together."
        ),
        "use_cases": "Simulating sample-wise optical gain, concentration scale, or path-length changes.",
        "limitations": (
            "The public operator implements per-sample gain only: it does not draw an "
            "independent gain at every wavelength and does not constrain $g_i$ to be positive. "
            "Large `sigma_gain` can invert a spectrum."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.noise.MultiplicativeNoise`; ABI 2 family "
            "`n4m_augmentation_multiplicative_noise_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/noise/multiplicative_noise.h; " + _BJERRUM,
    },
    "aug_spike_noise": {
        "title": "Sparse impulsive spike injection",
        "paper": (
            "No canonical paper defines this exact spike simulator. It is an internal "
            "artifact model; the implementation source is the normative specification."
        ),
        "principle": (
            "For each row, an integer count $K_i$ is drawn uniformly from "
            "[`n_spikes_min`, `n_spikes_max`]. Candidate channel indices and amplitudes "
            "$a\\sim U(a_{min},a_{max})$ are drawn, indices are sorted and deduplicated, "
            "and $X'_{i,j}=X_{i,j}+a$ at the retained locations. The deterministic draw "
            "order and the deduplication quirk are part of parity."
        ),
        "use_cases": "Stress-testing models against isolated cosmic-ray, electronic, or acquisition spikes.",
        "limitations": (
            "Coincident candidate indices reduce the realized spike count. Spikes affect one "
            "channel each and therefore do not model finite-width peaks or correlated glitches."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.noise.SpikeNoise`; ABI 2 family "
            "`n4m_augmentation_spike_noise_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/noise/spike_noise.h",
    },
    "aug_hetero_noise": {
        "title": "Signal-dependent heteroscedastic noise",
        "paper": (
            "No single publication defines this affine noise law. It is an internal "
            "measurement-noise heuristic documented by the native implementation."
        ),
        "principle": (
            "At cell $(i,j)$ the noise scale is "
            "$s_{ij}=b+c|X_{ij}|$, where $b$ is `noise_base` and $c$ is "
            "`noise_signal_dep`; the output is $X'_{ij}=X_{ij}+s_{ij}Z_{ij}$ with "
            "$Z_{ij}\\sim\\mathcal N(0,1)$."
        ),
        "use_cases": "Simulating instruments whose absolute noise increases with signal magnitude.",
        "limitations": (
            "The affine variance law is phenomenological, independent across channels, and "
            "can yield a negative scale if parameters are chosen outside their intended "
            "non-negative range. It is not a calibrated photon-counting model."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.noise.HeteroscedasticNoiseAugmenter`; ABI 2 "
            "family `n4m_augmentation_hetero_noise_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/noise/hetero_noise.h",
    },
    "aug_linear_drift": {
        "title": "Random affine baseline drift",
        "paper": (
            "No unique paper defines this augmenter. Additive offset and slope perturbation "
            f"is a spectroscopy augmentation pattern used by Bjerrum et al. (2017), "
            f"arXiv:1710.01927 ({_BJERRUM})."
        ),
        "principle": (
            "For spectrum $i$, draw $a_i\\sim U(a_{min},a_{max})$ and "
            "$b_i\\sim U(b_{min},b_{max})$, then set "
            "$X'_{ij}=X_{ij}+a_i+b_i(j-\\bar j)$. Centering the implicit channel index "
            "makes `offset_*` the drift at the spectral midpoint and `slope_*` the "
            "per-index gradient."
        ),
        "use_cases": "Robustness to baseline displacement and linear tilt between acquisitions.",
        "limitations": (
            "The ABI uses channel index, not physical wavelength. Consequently the same "
            "slope parameter has a different physical meaning after resampling or cropping."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.drift.LinearBaselineDrift`; ABI 2 family "
            "`n4m_augmentation_linear_drift_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/drift/linear_drift.h; " + _BJERRUM,
    },
    "aug_poly_drift": {
        "title": "Random polynomial baseline drift",
        "paper": (
            "No canonical publication specifies these random coefficient ranges. The method "
            "is a controlled extension of offset/slope spectral augmentation."
        ),
        "principle": (
            "Channels are mapped to $t_j\\in[-1,1]$. For every order "
            "$k=0,\\ldots,d$, a coefficient $c_{ik}$ is sampled uniformly between "
            "`coeff_min[k]` and `coeff_max[k]`, and "
            "$X'_{ij}=X_{ij}+\\sum_{k=0}^{d}c_{ik}t_j^k$. `degree` sets curvature order; "
            "the coefficient arrays must contain `degree + 1` bounds."
        ),
        "use_cases": "Generating smooth, low-frequency baseline shapes beyond an affine tilt.",
        "limitations": (
            "Polynomial drift is a numerical nuisance model rather than a physical scatter "
            "model. High degrees can oscillate and extrapolation meaning changes with the "
            "number or ordering of channels."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.drift.PolynomialBaselineDrift`; ABI 2 family "
            "`n4m_augmentation_poly_drift_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/drift/poly_drift.h",
    },
    "aug_path_length": {
        "title": "Multiplicative path-length perturbation",
        "paper": (
            "There is no paper canonical to this random simulator. Its physical motivation "
            "is the multiplicative path-length/scatter term treated by Martens & Stark "
            f"(1991), DOI 10.1016/0731-7085(91)80188-F ({_EMSC})."
        ),
        "principle": (
            "For row $i$, draw $L_i=1+sZ_i$ with `path_length_std` $s$ and "
            "$Z_i\\sim\\mathcal N(0,1)$, clamp $L_i$ below by `min_path_length`, and "
            "return $X'_{ij}=L_iX_{ij}$."
        ),
        "use_cases": "Training against moderate sample-thickness or optical path-length variability.",
        "limitations": (
            "Every wavelength receives the same factor, so wavelength-dependent scattering "
            "and additive baselines are absent. The lower clamp makes the factor distribution "
            "non-Gaussian when variability is large."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.drift.PathLengthAugmenter`; ABI 2 family "
            "`n4m_augmentation_path_length_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/drift/path_length.h; " + _EMSC,
    },
    "aug_wavelength_shift": {
        "title": "Rigid wavelength-axis shift",
        "paper": (
            "No unique paper defines this interpolation augmenter. It is an internal model "
            "of wavelength-registration uncertainty."
        ),
        "principle": (
            "For each spectrum draw $s_i\\sim U(\\text{shift_lo},\\text{shift_hi})$ and "
            "evaluate $X'_i(\\lambda_j)=X_i(\\lambda_j-s_i)$ by linear interpolation. "
            "Outside the measured interval, endpoint values are held as in `numpy.interp`."
        ),
        "use_cases": "Robustness to small calibration offsets in the wavelength axis.",
        "limitations": (
            "Interpolation smooths sharp features and clamps beyond the endpoints. If no "
            "wavelength array is supplied, shift units are channel indices rather than nm."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.wavelength.WavelengthShift`; ABI 2 family "
            "`n4m_augmentation_wavelength_shift_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/wavelength/wavelength_shift.h",
    },
    "aug_wavelength_stretch": {
        "title": "Wavelength-axis stretch about its center",
        "paper": (
            "No canonical publication specifies this augmenter; it is a wavelength-scale "
            "calibration heuristic whose exact interpolation is defined by the source."
        ),
        "principle": (
            "With axis center $\\bar\\lambda$, draw "
            "$f_i\\sim U(\\text{stretch_lo},\\text{stretch_hi})$ and evaluate "
            "$X'_i(\\lambda_j)=X_i(\\bar\\lambda+(\\lambda_j-\\bar\\lambda)/f_i)$ "
            "using linear interpolation. Factors above one broaden the coordinate scale."
        ),
        "use_cases": "Simulating small wavelength-scale expansion or compression across instruments.",
        "limitations": (
            "The deformation is globally affine and cannot represent local calibration error. "
            "Endpoint clamping and the implicit unit grid apply as for wavelength shift."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.wavelength.WavelengthStretch`; ABI 2 family "
            "`n4m_augmentation_wavelength_stretch_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/wavelength/wavelength_stretch.h",
    },
    "aug_local_warp": {
        "title": "Piecewise-linear local wavelength warp",
        "paper": (
            "No paper is canonical for this exact local-warp heuristic. The native source, "
            "rather than the historical cubic Python variant, defines the shipped behavior."
        ),
        "principle": (
            "Choose `n_control_points` equally spaced wavelengths, draw their shifts from "
            "$U(-m,m)$ where $m$ is `max_shift`, linearly interpolate the shift field "
            "$s_i(\\lambda)$, and compute "
            "$X'_i(\\lambda)=X_i(\\lambda-s_i(\\lambda))$ by linear interpolation."
        ),
        "use_cases": "Robustness to smooth, non-uniform wavelength calibration distortions.",
        "limitations": (
            "The current C implementation uses piecewise-linear control interpolation; it "
            "deliberately differs from the older cubic `splrep/splev` oracle. Large shifts can "
            "fold the query grid or create endpoint plateaus."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.wavelength.LocalWarpAugmenter`; ABI 2 family "
            "`n4m_augmentation_local_warp_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/wavelength/local_warp.h",
    },
    "aug_band_mask": {
        "title": "Random contiguous-band masking",
        "paper": (
            "No single spectroscopy paper defines this operator. It is analogous to structured "
            "feature dropout, with exact band sampling defined by the native source."
        ),
        "principle": (
            "For each spectrum, draw a band count in `n_bands_range`; for each band draw a "
            "center and integer width in `band_width_range`. In `zero` mode set the half-open "
            "slice to zero; in `interp` mode replace it by the line joining the nearest edge "
            "values. Overlapping bands are applied sequentially."
        ),
        "use_cases": "Testing whether a model survives missing or corrupted contiguous wavelength regions.",
        "limitations": (
            "Zero is not a neutral absorbance for every representation, while interpolation "
            "can create unrealistically straight segments. Edge clipping and band overlap mean "
            "realized masked width may be below the requested sum."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.spectral.BandMasking`; ABI 2 family "
            "`n4m_augmentation_band_mask_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/spectral/band_mask.h",
    },
    "aug_band_perturb": {
        "title": "Local band gain-and-offset perturbation",
        "paper": (
            "No canonical paper specifies this bandwise affine perturbation. It is a native "
            "spectral corruption heuristic related to offset/gain augmentation."
        ),
        "principle": (
            "For each row and each of `n_bands`, draw a center, width $w$, gain $g$ and "
            "offset $a$. Within the clipped band, update values sequentially as "
            "$X'_{ij}=gX_{ij}+a$. Parameter ranges `bw_*`, `gain_*`, and `offset_*` control "
            "extent and amplitude."
        ),
        "use_cases": "Robustness to wavelength-local sensitivity and baseline changes.",
        "limitations": (
            "Band boundaries are abrupt and overlapping bands compound in draw order. The "
            "operator exposes per-sample variation only, not one shared perturbation per batch."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.spectral.BandPerturbationAugmenter`; ABI 2 "
            "family `n4m_augmentation_band_perturb_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/spectral/band_perturb.h",
    },
    "aug_channel_dropout": {
        "title": "Independent spectral-channel dropout",
        "paper": (
            "No canonical NIR publication defines this cellwise dropout. The exact mask and "
            "replacement rules are implementation-defined."
        ),
        "principle": (
            "Each cell is marked independently with probability `dropout_prob`. Marked values "
            "are either set to zero or linearly interpolated from surviving channel indices in "
            "that row. The latter uses endpoint values beyond the first or last survivor."
        ),
        "use_cases": "Simulating sporadic dead pixels/channels and discouraging reliance on single wavelengths.",
        "limitations": (
            "Real sensor failures are often persistent or contiguous rather than IID. If too "
            "few channels survive, interpolation becomes poorly informative; zero mode is "
            "representation-dependent."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.spectral.ChannelDropout`; ABI 2 family "
            "`n4m_augmentation_channel_dropout_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/spectral/channel_dropout.h",
    },
    "aug_gauss_jitter": {
        "title": "Random Gaussian smoothing jitter",
        "paper": (
            "No unique paper defines randomizing the smoothing width. The operator is a "
            "stochastic Gaussian convolution whose source defines boundary behavior."
        ),
        "principle": (
            "For each spectrum draw $s_i\\sim U(\\text{sigma_lo},\\text{sigma_hi})$, build "
            "the normalized odd-width kernel $k(t)\\propto\\exp[-t^2/(2s_i^2)]$, and convolve "
            "along wavelength with reflect padding. `kernel_width` truncates the Gaussian."
        ),
        "use_cases": "Robustness to small variations in spectral resolution or smoothing strength.",
        "limitations": (
            "This is smoothing, despite the name “jitter”; it adds no random residual noise. "
            "Finite kernel width and reflected boundaries affect edge bands."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.spectral.GaussianJitter`; ABI 2 family "
            "`n4m_augmentation_gauss_jitter_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/spectral/gauss_jitter.h",
    },
    "aug_local_clip": {
        "title": "Random local 90th-percentile clipping",
        "paper": (
            "No canonical paper specifies this saturation heuristic. Its fixed 90th-percentile "
            "rule is defined by the native implementation."
        ),
        "principle": (
            "For every row, sample `n_regions` centers and widths. In each resulting slice, "
            "compute its linearly interpolated 90th percentile $q_{0.9}$ and replace each "
            "value by $\\min(X_{ij},q_{0.9})$. Overlapping regions operate on the already "
            "modified row."
        ),
        "use_cases": "Simulating local high-end detector saturation or peak truncation.",
        "limitations": (
            "Only positive excursions are clipped, the 90% level is fixed, and the threshold "
            "depends on the randomly selected region. It is not a detector transfer function."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.spectral.LocalClip`; ABI 2 family "
            "`n4m_augmentation_local_clip_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/spectral/local_clip.h",
    },
    "aug_magnitude_warp": {
        "title": "Smooth multiplicative magnitude warp",
        "paper": (
            "No paper is canonical for this exact spectral warp. The current native linear "
            "interpolation path is the normative algorithm."
        ),
        "principle": (
            "At equally spaced control wavelengths draw gains from "
            "$U(\\text{gain_lo},\\text{gain_hi})$, linearly interpolate a gain field "
            "$g_i(\\lambda)$, and return $X'_i(\\lambda)=g_i(\\lambda)X_i(\\lambda)$. "
            "`n_control_points` controls the field's spatial frequency."
        ),
        "use_cases": "Simulating smooth wavelength-dependent sensitivity or multiplicative scatter drift.",
        "limitations": (
            "The C path is piecewise linear and intentionally differs from the historical "
            "cubic-spline oracle. Gains are unconstrained beyond user bounds and may cross zero."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.spectral.MagnitudeWarp`; ABI 2 family "
            "`n4m_augmentation_magnitude_warp_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/spectral/magnitude_warp.h",
    },
    "aug_unsharp_mask": {
        "title": "Unsharp spectral masking",
        "paper": (
            "Unsharp masking is a classical signal/image sharpening construction, but no "
            "single NIR paper defines this stochastic parameterization; source code is normative."
        ),
        "principle": (
            "Build a Gaussian-smoothed row $S_i=G_{\\sigma,w}*X_i$, draw "
            "$a_i\\sim U(\\text{amount_lo},\\text{amount_hi})$, and compute "
            "$X'_i=X_i+a_i(X_i-S_i)$. `sigma` and odd `kernel_width` set the low-pass scale; "
            "`amount_*` controls sharpening."
        ),
        "use_cases": "Varying apparent spectral resolution and sensitivity to narrow bands during training.",
        "limitations": (
            "Sharpening amplifies high-frequency noise and can create overshoot. Reflect padding "
            "changes behavior near the first and last wavelengths."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.spectral.UnsharpMask`; ABI 2 family "
            "`n4m_augmentation_unsharp_mask_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/spectral/unsharp_mask.h",
    },
    "aug_mixup": {
        "title": "Within-batch convex mixup of spectra",
        "paper": (
            f"Zhang, Cisse, Dauphin & Lopez-Paz (2018), *mixup: Beyond Empirical Risk "
            f"Minimization*, ICLR, arXiv:1710.09412 ({_MIXUP})."
        ),
        "principle": (
            "A random permutation $\\pi$ pairs rows and independent weights are drawn as "
            "$\\lambda_i\\sim\\operatorname{Beta}(\\alpha,\\alpha)$. The operator returns "
            "$X'_i=\\lambda_iX_i+(1-\\lambda_i)X_{\\pi(i)}$. `alpha` below one favors "
            "near-endpoint mixtures; larger values concentrate around one half."
        ),
        "use_cases": "Regularizing models by filling linear neighborhoods between observed spectra.",
        "limitations": (
            "This transformer returns only mixed X. It does not mix target labels, although "
            "the original mixup method requires the same convex combination of y; callers "
            "must preserve label consistency themselves. Pairing is restricted to the batch."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.mixup.MixupAugmenter`; ABI 2 family "
            "`n4m_augmentation_mixup_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/mixup/mixup.h; " + _MIXUP,
    },
    "aug_local_mixup": {
        "title": "Nearest-neighbor constrained mixup",
        "paper": (
            f"This is an internal local variant of Zhang et al.'s mixup (2018), "
            f"arXiv:1710.09412 ({_MIXUP}); no separate canonical paper defines the "
            "specific k-nearest-neighbor rule used here."
        ),
        "principle": (
            "Compute exact Euclidean neighbors in X. For each row choose uniformly among its "
            "`k_neighbors` nearest non-self rows, draw "
            "$\\lambda_i\\sim\\operatorname{Beta}(\\alpha,\\alpha)$, and form "
            "$X'_i=\\lambda_iX_i+(1-\\lambda_i)X_{n(i)}$."
        ),
        "use_cases": "Interpolating locally within a spectral manifold while avoiding arbitrary distant pairs.",
        "limitations": (
            "Euclidean distance is scale- and preprocessing-dependent. Neighborhood search is "
            "quadratic in sample count, and, as for global mixup, y is not returned or mixed."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.mixup.LocalMixupAugmenter`; ABI 2 family "
            "`n4m_augmentation_local_mixup_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/mixup/local_mixup.h; " + _MIXUP,
    },
    "aug_random_x_op": {
        "title": "Independent random elementwise arithmetic",
        "paper": (
            "No canonical paper defines this operator. It is a deliberately simple internal "
            "perturbation baseline; the native source is the algorithmic specification."
        ),
        "principle": (
            "Draw one operand $u_{ij}\\sim U(r_{min},r_{max})$ per cell and apply the selected "
            "`op_kind`: $X'_{ij}=X_{ij}u_{ij}$, $X_{ij}+u_{ij}$, or $X_{ij}-u_{ij}$. Results "
            "are clipped to the finite float32 range even though computation uses doubles."
        ),
        "use_cases": "A generic sensitivity baseline for small unstructured amplitude perturbations.",
        "limitations": (
            "Independent per-channel operands usually lack spectroscopic smoothness. Float32 "
            "clipping is a parity behavior, not an instrument saturation model."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.mixup.RandomXOperation`; ABI 2 family "
            "`n4m_augmentation_random_x_op_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/random/random_x_op.h",
    },
    "aug_rotate_translate": {
        "title": "Random hinged rotation-and-translation pattern",
        "paper": (
            "No canonical publication defines this piecewise-linear augmenter. Its exact "
            "hinge construction and scaling are internal nirs4all behavior."
        ),
        "principle": (
            "On a normalized channel axis, the engine samples a hinge and constructs two "
            "linear slopes meeting there. The resulting rotation/translation pattern is "
            "scaled by each row's standard deviation and by `p_range`/`y_factor`, then added "
            "to the spectrum."
        ),
        "use_cases": "Perturbing global tilt and offset while allowing different left/right slopes.",
        "limitations": (
            "This is a geometric heuristic rather than a literal coordinate rotation. Its "
            "amplitude vanishes for constant rows and depends on the row's scale."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.mixup.RotateTranslateAugmenter`; ABI 2 family "
            "`n4m_augmentation_rotate_translate_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/random/rotate_translate.h",
    },
    "aug_scatter_sim": {
        "title": "MSC-style affine scatter simulation",
        "paper": (
            f"Geladi, MacDougall & Martens (1985), *Linearization and Scatter-Correction for "
            f"Near-Infrared Reflectance Spectra of Meat*, Applied Spectroscopy 39, 491–500, "
            f"DOI 10.1366/0003702854248656 ({_MSC}). The augmenter simulates the affine "
            "effects that MSC is designed to remove."
        ),
        "principle": (
            "For each spectrum draw $a_i\\sim U(a_{low},a_{high})$ and "
            "$b_i\\sim U(b_{low},b_{high})$, then compute $X'_i=a_i+b_iX_i$. This is the "
            "forward affine scatter model; it does not estimate or correct coefficients."
        ),
        "use_cases": "Training robustness to additive baseline and multiplicative scatter variability.",
        "limitations": (
            "Only wavelength-independent offset and gain are represented. The operator ignores "
            "wavelength input and does not implement a global-mean reference branch."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.scattering.ScatterSimulationMSC`; ABI 2 family "
            "`n4m_augmentation_scatter_sim_msc_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/scattering/scatter_sim_msc.h; " + _MSC,
    },
    "aug_particle_size": {
        "title": "Particle-size and path-length scatter heuristic",
        "paper": (
            "No single paper defines this simulator. It is physically motivated by diffuse "
            "scattering but uses an explicit empirical power law documented in the source, "
            "not a Mie or Kubelka–Munk solver."
        ),
        "principle": (
            "Draw a particle size $d_i$ from the configured uniform range or a normal law "
            "clipped to 5–500 µm. With $r_i=d_i/d_0$ and "
            "$w(\\lambda)=\\operatorname{clip}(\\lambda/1500,0.1,10)^{-e}$, add the "
            "zero-mean baseline $s(r_i^{-1/2}-1)w(\\lambda)$. Optionally multiply by "
            "$\\operatorname{clip}[1+p\\log r_i,0.7,1.5]$ and add smoothed Gaussian scatter noise."
        ),
        "use_cases": "Qualitative robustness studies for powders with particle-size and path-length variation.",
        "limitations": (
            "The coefficients and power law are heuristic; wavelengths are required and assumed "
            "compatible with the 1500-unit scale. It should not be used for quantitative optical simulation."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.scattering.ParticleSizeAugmenter`; ABI 2 family "
            "`n4m_augmentation_particle_size_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/scattering/particle_size.h",
    },
    "aug_emsc_distort": {
        "title": "Random EMSC-like affine and polynomial distortion",
        "paper": (
            f"Martens & Stark (1991), *Extended multiplicative signal correction and spectral "
            f"interference subtraction*, Journal of Pharmaceutical and Biomedical Analysis 9, "
            f"625–635, DOI 10.1016/0731-7085(91)80188-F ({_EMSC}). This operator generates "
            "EMSC-shaped distortions; it does not perform EMSC correction."
        ),
        "principle": (
            "Normalize wavelength to $t\\in[-1,1]$. Draw clipped multiplicative $b_i$ and an "
            "additive $a_i$ whose mean is correlated with the standardized $b_i$ deviation. "
            "For orders $k=1..q$, draw $c_{ik}\\sim N(0,s^2/k)$ and return "
            "$X'_i=a_i+b_iX_i+\\sum_k c_{ik}t^k$."
        ),
        "use_cases": "Training against correlated additive, multiplicative, and smooth baseline scatter effects.",
        "limitations": (
            "Ranges are converted to mean and one-quarter-range standard deviations, then clipped; "
            "they are not uniform draws. Required wavelengths must span a nonzero range."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.scattering.EMSCDistortionAugmenter`; ABI 2 family "
            "`n4m_augmentation_emsc_distort_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/scattering/emsc_distort.h; " + _EMSC,
    },
    "aug_batch_effect": {
        "title": "Offset, slope, and gain batch effects",
        "paper": (
            "No canonical publication defines this three-component simulator. It is an internal "
            "instrument/session perturbation model related to affine spectral augmentation."
        ),
        "principle": (
            "On centered normalized wavelength $t_j$, draw "
            "$a\\sim N(0,\\sigma_a^2)$, $b\\sim N(0,\\sigma_b^2)$ and "
            "$g\\sim N(1,\\sigma_g^2)$, then $X'_{ij}=gX_{ij}+a+bt_j$. "
            "`variation_scope=0` draws a tuple per row; scope 1 shares one tuple across the batch."
        ),
        "use_cases": "Simulating acquisition-session or instrument-to-instrument affine shifts.",
        "limitations": (
            "Zero defaults make this a no-op until standard deviations are set. A shared batch "
            "draw models one batch only and does not attach batch labels."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.scattering.BatchEffectAugmenter`; ABI 2 family "
            "`n4m_augmentation_batch_effect_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/scattering/batch_effect.h",
    },
    "aug_instrument_broaden": {
        "title": "Gaussian instrumental broadening from FWHM",
        "paper": (
            "No single paper defines this implementation; Gaussian line-spread convolution is "
            "a standard instrumental-resolution model, with exact kernel construction specified in source."
        ),
        "principle": (
            "Convert full width at half maximum to channel standard deviation as "
            "$\\sigma_p=\\mathrm{FWHM}/[2\\sqrt{2\\ln2}\\,\\Delta\\lambda]$, where "
            "$\\Delta\\lambda$ is the median wavelength step (or 1 without wavelengths), and "
            "Gaussian-convolve each row. A fixed FWHM or uniform range may be used per row or batch."
        ),
        "use_cases": "Testing transfer across instruments with different spectral resolution.",
        "limitations": (
            "The line-spread function is assumed Gaussian and stationary. Irregular wavelength "
            "grids are summarized by one median step, and broadening cannot sharpen a low-resolution spectrum."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.scattering.InstrumentalBroadeningAugmenter`; ABI 2 "
            "family `n4m_augmentation_instrument_broaden_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/scattering/instrument_broaden.h",
    },
    "aug_dead_band": {
        "title": "Random noisy detector dead bands",
        "paper": (
            "No canonical paper defines this simulator. The number, width, probability, and "
            "replacement-noise rules are implementation-specific."
        ),
        "principle": (
            "With `probability`, select `n_bands`; sample each integer width and start, and replace "
            "the slice by independent $N(0,\\text{noise_std}^2)$ values. Per-sample scope draws "
            "locations for each row; batch scope shares locations but still draws independent row noise."
        ),
        "use_cases": "Stress-testing models against failed or unusable contiguous detector regions.",
        "limitations": (
            "Replacement is centered at zero rather than the local baseline. At the default "
            "`probability=0`, the operator is a no-op. Overlapping bands may reduce affected coverage."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.scattering.DeadBandAugmenter`; ABI 2 family "
            "`n4m_augmentation_dead_band_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/scattering/dead_band.h",
    },
    "aug_temperature": {
        "title": "Region-specific temperature perturbation heuristic",
        "paper": (
            "No publication canonically defines the six coefficient presets used here. This is "
            "an internal phenomenological model; source coefficients and regions are normative."
        ),
        "principle": (
            "Use fixed `temperature_delta` or draw one from `temp_low`–`temp_high`. In six built-in "
            "O–H/C–H/N–H/water regions, sigmoid windows blend coefficient-scaled wavelength shift, "
            "intensity change, and Gaussian broadening. With `region_specific=False`, averages of "
            "those coefficients act over the full axis."
        ),
        "use_cases": "Qualitative robustness experiments for temperature-sensitive NIR band positions and shapes.",
        "limitations": (
            "The coefficients are presets, not fitted thermodynamic parameters. Physical wavelengths "
            "are required; data outside the encoded regions receive little or averaged effect. "
            "Changes below 0.01 degree-equivalent are ignored."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.instrument.TemperatureAugmenter`; ABI 2 family "
            "`n4m_augmentation_temperature_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/environmental/temperature.h",
    },
    "aug_moisture": {
        "title": "Water-activity and moisture-band perturbation",
        "paper": (
            "No canonical paper specifies this combined sigmoid/free-bound-water model. It is "
            "an internal heuristic with source-defined 1435 and 1930 wavelength bands."
        ),
        "principle": (
            "Set water activity $a_w=\\operatorname{clip}(a_{ref}+\\Delta a,0,1)$, with fixed "
            "or uniformly drawn $\\Delta a$. A sigmoid allocates free versus bound water. Bound "
            "fraction shifts windows around 1435 and 1930; optional intensity adds Gaussian band "
            "profiles scaled by `moisture_content/0.10 - 1` and the row mean magnitude."
        ),
        "use_cases": "Sensitivity studies for water-band displacement and amplitude changes in moist materials.",
        "limitations": (
            "Band centers assume compatible wavelength units and coverage. The model is qualitative, "
            "not a calibration from water activity or moisture percentage to absorbance."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.instrument.MoistureAugmenter`; ABI 2 family "
            "`n4m_augmentation_moisture_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/environmental/moisture.h",
    },
    "aug_detector_rolloff": {
        "title": "Detector sensitivity roll-off artifact",
        "paper": (
            "No single paper defines the five detector presets in this implementation. They are "
            "literature-inspired internal curves; the enum and native source are the auditable specification."
        ),
        "principle": (
            "The selected InGaAs, extended InGaAs, PbS, silicon CCD, or generic NIR preset defines "
            "an optimal range, roll-off rate, and minimum sensitivity. Falling sensitivity toward "
            "the edges amplifies random noise and, when enabled, adds a small baseline distortion; "
            "`effect_strength` and `noise_amplification` scale these terms."
        ),
        "use_cases": "Robustness testing near detector range limits or across detector technologies.",
        "limitations": (
            "Presets are not manufacturer response calibrations and require a meaningful wavelength axis. "
            "They should not be used to identify a detector or predict its signal-to-noise ratio."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.instrument.DetectorRollOffAugmenter`; ABI 2 family "
            "`n4m_augmentation_detector_rolloff_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/edge_artifacts/detector_rolloff.h",
    },
    "aug_stray_light": {
        "title": "Stray-light distortion in transmittance space",
        "paper": (
            "No canonical publication defines this edge-enhanced profile. The physical mixing equation "
            "is standard, while its wavelength profile and optional truncation are source-defined."
        ),
        "principle": (
            "Convert absorbance $A$ to transmittance $T=10^{-A}$, form an edge-enhanced stray fraction "
            "$s(\\lambda)$, compute $T_{obs}=(T+s)/(1+s)$, and return "
            "$A'=-\\log_{10}T_{obs}$. `edge_width` and `edge_enhancement` shape $s$; optional peak "
            "truncation models loss of high absorbance."
        ),
        "use_cases": "Testing nonlinear absorbance compression caused by out-of-band or internal stray light.",
        "limitations": (
            "Input is assumed to be base-10 absorbance. Applying the formula directly to reflectance, "
            "transmittance, derivatives, or standardized data is not physically meaningful."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.instrument.StrayLightAugmenter`; ABI 2 family "
            "`n4m_augmentation_stray_light_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/edge_artifacts/stray_light.h",
    },
    "aug_edge_curve": {
        "title": "Smooth detector-edge curvature",
        "paper": (
            "No canonical paper defines the smile/frown/asymmetric templates. They are internal "
            "edge-response heuristics with formulas fixed by the native source."
        ),
        "principle": (
            "A normalized wavelength coordinate drives a smooth edge-focused curve. `curvature_type` "
            "selects random, smile, frown, or asymmetric shape; `curvature_strength` sets amplitude, "
            "`asymmetry` differentiates left and right, and `edge_focus` concentrates the effect toward edges."
        ),
        "use_cases": "Robustness to smooth baseline curvature near the boundaries of an instrument's range.",
        "limitations": (
            "Template curvature is phenomenological and depends on the supplied wavelength span. "
            "It does not estimate optical smile or detector geometry from metadata."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.instrument.EdgeCurvatureAugmenter`; ABI 2 family "
            "`n4m_augmentation_edge_curvature_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/edge_artifacts/edge_curvature.h",
    },
    "aug_truncated_peak": {
        "title": "Off-range Gaussian peak tails",
        "paper": (
            "No canonical paper defines this artifact generator. It is an internal model of bands "
            "whose centers lie just outside the measured interval."
        ),
        "principle": (
            "With `peak_probability` per sample, choose an enabled left or right edge, draw amplitude "
            "and width from their configured ranges, place a Gaussian center beyond that edge, and add "
            "the in-range tail to the spectrum."
        ),
        "use_cases": "Testing sensitivity to partially observed absorption bands at cropped spectral edges.",
        "limitations": (
            "Only Gaussian positive tails from enabled edges are represented. Width and amplitude units "
            "must match the wavelength axis and data scale; overlapping real bands are not inferred."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.instrument.TruncatedPeakAugmenter`; ABI 2 family "
            "`n4m_augmentation_truncated_peak_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/edge_artifacts/truncated_peak.h",
    },
    "aug_edge_artifacts": {
        "title": "Configured pipeline of four edge artifacts",
        "paper": (
            "No paper defines this composite. It is an implementation-defined pipeline combining "
            "truncated peaks, curvature, stray light, and detector roll-off."
        ),
        "principle": (
            "`enabled_flags` selects four suboperators. Each receives a deterministic child PCG64 stream "
            "and source-defined defaults scaled by `overall_strength`. Enabled stages run in this exact "
            "order: truncated peaks, edge curvature, stray light, detector roll-off, with every stage "
            "consuming the previous stage's output."
        ),
        "use_cases": "Combined stress tests of several boundary artifacts with one reproducible operator.",
        "limitations": (
            "Composition is noncommutative and `overall_strength` scales suboperators differently; it is "
            "not a universal scalar severity. Suboperator defaults are fixed rather than individually exposed."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.instrument.EdgeArtifactsAugmenter`; ABI 2 family "
            "`n4m_augmentation_edge_artifacts_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/c_api/c_api_augmenters_edge_splines_random.cpp#L426",
    },
    "aug_spline_smooth": {
        "title": "Deterministic natural-cubic spline smoothing",
        "paper": (
            "Reinsch (1967), *Smoothing by spline functions*, Numerische Mathematik 10, 177–183, "
            "DOI 10.1007/BF02162161 (https://doi.org/10.1007/BF02162161), provides the classical "
            "smoothing-spline basis; the exact fixed smoothing choice here is implementation-specific."
        ),
        "principle": (
            "Fit a natural cubic smoothing spline independently to each row on the channel grid and "
            "evaluate it on that same grid. The shipped parity target corresponds to a fixed smoothing "
            "budget $s=1/p$ for $p$ channels; the RNG and `seed` are accepted by the common wrapper but unused."
        ),
        "use_cases": "Creating a smoothed view of spectra to reduce high-frequency variation during training.",
        "limitations": (
            "This operator is deterministic and therefore does not enlarge a dataset with multiple draws. "
            "A fixed $1/p$ smoothing choice may be inappropriate across data scales."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.splines.SplineSmoothingAugmenter`; ABI 2 family "
            "`n4m_augmentation_spline_smoothing_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/splines/spline_smoothing.h; https://doi.org/10.1007/BF02162161",
    },
    "aug_spline_x_perturb": {
        "title": "Spline-based x-axis perturbation",
        "paper": (
            "No canonical paper defines this random x-grid perturbation. The native implementation is a "
            "parity-oriented analogue of SciPy spline resampling."
        ),
        "principle": (
            "Fit a not-a-knot cubic B-spline to each row. Set the number of equally spaced perturbation "
            "anchors to $\\max[2,\\operatorname{round}((p+4)d)]$, where d is `perturbation_density`; "
            "draw an offset at each anchor from `perturbation_range_min` to `perturbation_range_max`, "
            "linearly interpolate those offsets onto the spline knot vector, perturb the knots, and "
            "evaluate the altered spline on the original channel grid."
        ),
        "use_cases": "Robustness to sparse wavelength-registration errors with smooth interpolation between points.",
        "limitations": (
            "Only `spline_degree=3` is accepted by the native constructor. Perturbed knots can become poorly "
            "ordered at excessive ranges; cubic evaluation can overshoot, and channel coordinates rather than "
            "physical wavelengths drive this API."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.splines.SplineXPerturbationAugmenter`; ABI 2 family "
            "`n4m_augmentation_spline_x_perturbations_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/splines/spline_x_perturbations.h",
    },
    "aug_spline_y_perturb": {
        "title": "Smooth additive spline perturbation",
        "paper": (
            "No publication canonically defines this control-point noise augmenter. Its random-control "
            "construction is specified by the native source."
        ),
        "principle": (
            "Let $v=\\max(X)$ times `perturbation_intensity` and draw one batch-shared baseline from "
            "$U(-v,v)$. For every row, draw y offsets at `spline_points` controls (default $p/2$, minimum 4) "
            "from the baseline-shifted interval, fit a not-a-knot cubic curve on [0,p], evaluate it on "
            "channels 0 through p-1, and add that curve to the row."
        ),
        "use_cases": "Generating smooth additive baseline fluctuations instead of channelwise white noise.",
        "limitations": (
            "Intensity is tied to the global maximum, not absolute magnitude; all-negative data therefore invert "
            "the intended interval. Many controls can approach high-frequency noise, while few reduce the effect "
            "to broad drift. The last anchor lies at p, one step beyond the final queried channel p-1."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.splines.SplineYPerturbationAugmenter`; ABI 2 family "
            "`n4m_augmentation_spline_y_perturbations_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/splines/spline_y_perturbations.h",
    },
    "aug_spline_x_simplify": {
        "title": "Cubic-spline reconstruction from sparse x controls",
        "paper": (
            "No canonical paper defines this augmentation heuristic. It uses a classical interpolating "
            "B-spline but the control subset and parity rules are implementation-specific."
        ),
        "principle": (
            "Keep `spline_points` channel/value controls (default $p/4$), selected uniformly or by "
            "PCG64 sampling without replacement. Fit an interpolating not-a-knot cubic B-spline through "
            "those controls and evaluate it at all original channels."
        ),
        "use_cases": "Testing robustness to reduced effective spectral resolution and sparse sampling.",
        "limitations": (
            "Fine features between controls are irrecoverable and cubic interpolation may overshoot. "
            "This x variant does not apply the curve variant's final `unique` step in uniform mode."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.splines.SplineXSimplificationAugmenter`; ABI 2 family "
            "`n4m_augmentation_spline_x_simplification_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/splines/spline_simplify_common.h",
    },
    "aug_spline_curve_simplify": {
        "title": "Curve-control cubic-spline simplification",
        "paper": (
            "No canonical paper defines this augmenter. It shares the internal interpolating B-spline "
            "engine with x simplification and exists for historical parity."
        ),
        "principle": (
            "Reduce each spectrum to `spline_points` controls (default $p/4$), fit a not-a-knot cubic "
            "interpolant, and reconstruct the dense curve. Random mode is identical to x simplification; "
            "in uniform mode this variant additionally deduplicates rounded linspace indices."
        ),
        "use_cases": "Parity with the historical curve-simplification operator and sparse-curve stress tests.",
        "limitations": (
            "The distinction from x simplification is only the uniform-index deduplication rule. It is "
            "not a geometry-aware arc-length simplifier despite the name."
        ),
        "implementation": (
            "Python role API `n4m.augmentation.splines.SplineCurveSimplificationAugmenter`; ABI 2 family "
            "`n4m_augmentation_spline_curve_simplification_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/augmentation/splines/spline_curve_simplification.h",
    },
    "filter_variance": {
        "title": "Population-variance feature filter",
        "paper": (
            "Variance thresholding is a basic descriptive-statistics filter with no single "
            "canonical method paper. The shipped scoring and tie behavior are defined by the ABI implementation."
        ),
        "principle": (
            "At fit time, score channel $j$ by the population variance "
            "$v_j=n^{-1}\\sum_i(X_{ij}-\\bar X_j)^2$. With no `top_k`, retain channels satisfying "
            "$v_j>\\text{threshold}$ (strict inequality). With `top_k`, retain the k largest scores; "
            "stable ascending ranking makes original channel order the tie-break before the retained indices are emitted."
        ),
        "use_cases": "Removing constant or nearly constant wavelengths before modeling.",
        "limitations": (
            "Variance depends on units and preprocessing and says nothing about association with y. The selector "
            "does not expose a Boolean support mask through the Python API, only transformed columns."
        ),
        "implementation": (
            "Python role API `n4m.feature_selection.filter.VarianceFilter`; ABI 2 family "
            "`n4m_feature_selection_variance_{create,fit,transform,output_cols,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/c_api/c_api_advanced.cpp#L1138",
    },
    "filter_correlation": {
        "title": "Absolute Pearson-correlation feature filter",
        "paper": (
            "There is no single canonical feature-selection paper for this use of Pearson correlation. "
            "The absolute-score threshold and top-k policy are implementation-specific."
        ),
        "principle": (
            "For each channel, compute the centered Pearson coefficient "
            "$r_j=\\sum_i(x_{ij}-\\bar x_j)(y_i-\\bar y)/"
            "[\\sqrt{\\sum_i(x_{ij}-\\bar x_j)^2}\\sqrt{\\sum_i(y_i-\\bar y)^2}]$ and score "
            "it by $|r_j|$. Zero-denominator features score zero. Retain scores strictly above `threshold` "
            "or the `top_k` largest."
        ),
        "use_cases": "Fast supervised screening of wavelengths with a strong marginal linear relation to one target.",
        "limitations": (
            "It is univariate, linear, and ignores redundancy between selected wavelengths. Absolute value removes "
            "sign information; constant y makes every score zero; using it before cross-validation would leak targets."
        ),
        "implementation": (
            "Python role API `n4m.feature_selection.filter.CorrelationFilter`; ABI 2 family "
            "`n4m_feature_selection_correlation_{create,fit,transform,output_cols,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/c_api/c_api_advanced.cpp#L1138",
    },
    "filter_y_outlier": {
        "title": "Univariate target outlier filter",
        "paper": (
            "No single paper defines this four-rule facade. IQR fences, z scores, percentiles, and scaled MAD "
            "are classical robust/descriptive rules; the exact NumPy-compatible percentile interpolation is source-defined."
        ),
        "principle": (
            "Fit one interval on y: IQR uses $[Q_1-tIQR,Q_3+tIQR]$; z score uses "
            "$[\\mu-t\\sigma,\\mu+t\\sigma]$; percentile uses configured quantiles; MAD uses "
            "$[m-t(1.4826\\,MAD),m+t(1.4826\\,MAD)]$. Apply keeps values inside the learned bounds "
            "and returns mask plus counts. Quantiles use linear interpolation."
        ),
        "use_cases": "Auditable removal or flagging of extreme reference values before fitting a calibration model.",
        "limitations": (
            "Filtering y narrows the calibration domain and can bias validation. Z score assumes a meaningful mean/scale; "
            "IQR and MAD can degenerate on tied values. Fit bounds only on the training partition."
        ),
        "implementation": (
            "Python role API `n4m.outlier_detection.YOutlierFilter`; ABI 2 family "
            "`n4m_outlier_detection_y_outlier_{create,fit,apply,is_fitted,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/filters/y_outlier.h",
    },
    "filter_leverage": {
        "title": "Hat-matrix or PCA-score leverage filter",
        "paper": (
            "Rousseeuw & van Zomeren (1990), *Unmasking Multivariate Outliers and Leverage Points*, "
            "JASA 85, 633–639, DOI 10.1080/01621459.1990.10474920 "
            "(https://doi.org/10.1080/01621459.1990.10474920), provides the leverage/outlier context; "
            "the fallback and thresholds here are implementation-specific."
        ),
        "principle": (
            "Hat mode computes $h_i=x_i^T(X^TX+10^{-10}I)^{-1}x_i$ after optional centering. "
            "PCA mode computes leverage in retained score space; `n_components<=0` selects "
            "$\\min(n-1,p,50)$. Unless `absolute_threshold` in (0,1) is supplied, the cutoff is "
            "`threshold_multiplier` times mean training leverage. Rows at or below the cutoff are kept."
        ),
        "use_cases": "Detecting calibration samples geometrically remote from the fitted X design space.",
        "limitations": (
            "When rows are not greater than columns, requested hat mode silently uses PCA. Leverage measures influence "
            "geometry, not spectral quality or response error, and thresholds are training-distribution dependent."
        ),
        "implementation": (
            "Python role API `n4m.outlier_detection.HighLeverageFilter`; ABI 2 family "
            "`n4m_outlier_detection_high_leverage_{create,fit,apply,threshold,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/filters/high_leverage.h; https://doi.org/10.1080/01621459.1990.10474920",
    },
    "filter_quality": {
        "title": "Rule-based row-level spectral quality filter",
        "paper": (
            "No canonical paper defines this conjunction of data-quality checks. It is an explicit internal "
            "quality-control rule set whose thresholds are user policy."
        ),
        "principle": (
            "A row is kept only if every enabled rule passes: NaN fraction $\\le$ `max_nan_ratio`; no infinity "
            "when `check_inf`; exact-zero fraction $\\le$ `max_zero_ratio`; NaN-aware population variance "
            "$\\ge$ `min_variance`; and optional finite extrema within `min_value`/`max_value`. Fit is an idempotent no-op."
        ),
        "use_cases": "Rejecting empty, flat, non-finite, saturated, or out-of-range acquisitions before modeling.",
        "limitations": (
            "Thresholds are scale- and representation-specific. NaNs count as nonzero for the zero-ratio rule, and a "
            "row can pass these syntactic checks while remaining chemically implausible."
        ),
        "implementation": (
            "Python role API `n4m.outlier_detection.SpectralQualityFilter`; ABI 2 family "
            "`n4m_outlier_detection_spectral_quality_{create,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/filters/spectral_quality.h",
    },
    "filter_x_outlier": {
        "title": "Multistrategy X-space outlier filter",
        "paper": (
            f"The facade combines distinct methods: Liu, Ting & Zhou (2008), Isolation Forest, DOI "
            f"10.1109/ICDM.2008.17 ({_IFOREST}); Breunig et al. (2000), LOF, DOI "
            f"10.1145/342009.335388 ({_LOF}); and Rousseeuw & Van Driessen (1999), FAST-MCD, DOI "
            f"10.1080/00401706.1999.10485670 ({_MCD}). Mahalanobis and PCA Q/T² options are classical."
        ),
        "principle": (
            "`method` selects empirical Mahalanobis distance, simplified FAST-MCD robust Mahalanobis, PCA residual "
            "$Q_i=\\|x_i-\\hat x_i\\|^2$, PCA-score Hotelling-like $T_i^2$, isolation-forest mean path score, "
            "or LOF with $k=\\min(20,n-1)$. Defaults use a chi-square-derived distance cutoff for Mahalanobis, "
            "training 95th percentiles for PCA scores, and the `contamination` quantile for forest/LOF."
        ),
        "use_cases": "Choosing a global, robust, subspace, isolation, or local-density detector under one mask API.",
        "limitations": (
            "The native robust covariance, isolation forest, and LOF are independent vendored implementations, not calls "
            "to scikit-learn and not guaranteed numerically identical. Distance methods are sensitive to scaling; "
            "contamination fixes an expected fraction rather than an absolute scientific criterion."
        ),
        "implementation": (
            "Python role API `n4m.outlier_detection.XOutlierFilter`; ABI 2 family "
            "`n4m_outlier_detection_x_outlier_{create,fit,apply,destroy}`; vendored kernels live under "
            "`cpp/src/core/filters/_vendored/`."
        ),
        "provenance": (
            _SRC + "cpp/src/core/filters/x_outlier.h; " + _IFOREST + "; " + _LOF + "; " + _MCD
        ),
    },
    "filter_composite": {
        "title": "Boolean composition of sample keep masks",
        "paper": (
            "No paper canonically defines this composition wrapper. Boolean aggregation of precomputed masks is "
            "an implementation utility, not a new outlier-detection algorithm."
        ),
        "principle": (
            "Each child filter computes a binary keep mask. Mode `all` keeps row i iff every child keeps it "
            "($m_i=\\bigwedge_km_{ki}$); mode `any` keeps it iff at least one child does "
            "($m_i=\\bigvee_km_{ki}$). An empty child list keeps every row."
        ),
        "use_cases": "Combining independent leverage and spectral-quality policies into one reproducible decision.",
        "limitations": (
            "The current Python wrapper accepts only `HighLeverageFilter` and `SpectralQualityFilter` children. "
            "“any” is permissive for keep masks, whereas “all” is restrictive; this is easy to invert mentally."
        ),
        "implementation": (
            "Python role API `n4m.outlier_detection.CompositeFilter`; ABI 2 family "
            "`n4m_outlier_detection_composite_{create,add_leverage,add_quality,apply,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/filters/composite.h",
    },
    "split_kennard_stone": {
        "title": "Kennard–Stone maximin calibration split",
        "paper": (
            f"Kennard & Stone (1969), *Computer Aided Design of Experiments*, Technometrics 11, 137–148, "
            f"DOI 10.1080/00401706.1969.10490666 ({_KS})."
        ),
        "principle": (
            "Compute all Euclidean distances in X, initialize the training set with the globally farthest pair, then "
            "repeatedly add the unselected sample maximizing its minimum distance to the selected set. For n samples, "
            "`test_size` gives $n_{test}=\\lceil n\\,test\\_size\\rceil$ and the remaining maximin points are training."
        ),
        "use_cases": "Constructing a calibration set that covers X-space while reserving the complement for testing.",
        "limitations": (
            "It is deterministic, quadratic in memory/time for distances, sensitive to feature scaling, and ignores y. "
            "The test complement is not an IID random sample."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.KennardStone`; binding class `KennardStoneSplitter`; "
            "ABI 2 family `n4m_model_selection_kennard_stone_{create,split,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/kennard_stone.h; " + _KS,
    },
    "split_spxy": {
        "title": "SPXY joint X–Y maximin split",
        "paper": (
            f"Galvão et al. (2005), *A method for calibration and validation subset partitioning*, Talanta 67, "
            f"736–740, DOI 10.1016/j.talanta.2005.03.025 ({_SPXY})."
        ),
        "principle": (
            "Compute Euclidean pairwise matrices $D_X$ and $D_Y$, normalize each by its maximum, then form "
            "$D=D_X/\\max D_X+D_Y/\\max D_Y$. Apply the Kennard–Stone farthest-pair then maximin sequence to D. "
            "Y may have one or several columns."
        ),
        "use_cases": "Calibration/validation partitioning that covers both spectra and reference-value space.",
        "limitations": (
            "It uses all y values to design the split, so it is unsuitable for a final blind test and can be optimistic. "
            "Euclidean geometry remains scale-sensitive within multicolumn X or Y, and full distances are quadratic. "
            "In the current Python binding, univariate y must be passed explicitly as shape `(n_samples, 1)`; a 1-D "
            "array is promoted to one row and fails the X/Y row check."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.SPXY`; binding class `SPXYSplitter`; ABI 2 family "
            "`n4m_model_selection_spxy_{create,split,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/spxy.h; " + _SPXY,
    },
    "split_spxy_fold": {
        "title": "Alternating maximin SPXY K-fold assignment",
        "paper": (
            f"This is an internal K-fold extension of Galvão et al.'s SPXY criterion (2005), DOI "
            f"10.1016/j.talanta.2005.03.025 ({_SPXY}); the alternating fold assignment itself is not defined by that paper."
        ),
        "principle": (
            "Build normalized X-plus-Y pairwise distances (or X alone). The engine seeds each fold with one of the "
            "`n_splits` samples having the largest mean distance, then cycles through folds. On each turn it assigns "
            "to that fold the remaining sample whose minimum distance to members already in that same fold is largest, "
            "up to `ceil(n/n_splits)`. Each requested fold returns that assignment as test and its complement as train."
        ),
        "use_cases": "Deterministic folds intended to spread X/Y diversity across validation partitions.",
        "limitations": (
            "The public string aliases `y_metric='euclidean'` and `'mahalanobis'` both map to the same Euclidean-Y code; "
            "the latter name does not invoke a Mahalanobis kernel. `x_only` is pure Kennard–Stone. Using y in fold design "
            "is supervised resampling and must remain inside the training data."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.SPXYFold`; binding class `SPXYFoldSplitter`; ABI 2 family "
            "`n4m_model_selection_spxy_fold_{create,n_splits,split_fold,destroy}`."
        ),
        "provenance": _SRC + "bindings/python/src/n4m/_impl/splitters.py#L146; " + _SRC + "cpp/src/core/splitters/spxy_fold.h; " + _SPXY,
    },
    "split_spxy_g_fold": {
        "title": "Group-preserving SPXY K-fold assignment",
        "paper": (
            f"This is an internal grouped extension of SPXY (Galvão et al., 2005, DOI "
            f"10.1016/j.talanta.2005.03.025, {_SPXY}); no canonical paper defines its representative aggregation."
        ),
        "principle": (
            "Aggregate every integer-labeled group columnwise by mean or median in X and Y. Run the same alternating "
            "per-fold maximin SPXY assignment on group representatives, then expand each group assignment back to all member rows."
        ),
        "use_cases": "Preventing replicates, batches, subjects, or lots from crossing train/test folds while spreading representatives.",
        "limitations": (
            "Only int64 group labels cross the C ABI. A mean or median representative can hide within-group heterogeneity; "
            "fold sample counts may be unbalanced. As in SPXYFold, `'mahalanobis'` currently aliases Euclidean Y distance."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.SPXYGroupFold`; binding class `SPXYGroupFoldSplitter`; ABI 2 "
            "family `n4m_model_selection_spxy_g_fold_{create,n_splits,split_fold,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/spxy_g_fold.h; " + _SRC + "bindings/python/src/n4m/_impl/splitters.py#L202; " + _SPXY,
    },
    "split_kmeans": {
        "title": "K-means++ representative-sample split",
        "paper": (
            f"Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*, SODA, "
            f"1027–1035 ({_KMEANS_PP}). The surrounding representative split is implementation-specific."
        ),
        "principle": (
            "Set k to the requested training count, initialize k centroids with seeded k-means++, and run Lloyd iterations "
            "up to `max_iter`. Select the nearest observed sample to each centroid, deduplicate those indices, and use the "
            "sorted complement as test."
        ),
        "use_cases": "Selecting observed spectra near cluster centers as a representative training subset.",
        "limitations": (
            "Deduplication can yield fewer training samples than requested when centroids choose the same row. Results depend "
            "on scaling and seed; the native PCG64 implementation is not scikit-learn KMeans."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.KMeans`; binding class `KMeansSplitter`; ABI 2 family "
            "`n4m_model_selection_kmeans_{create,split,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/kmeans.h; " + _KMEANS_PP,
    },
    "split_kbins_stratified": {
        "title": "Continuous-target binned stratified split",
        "paper": (
            "No single paper defines this composition of discretization and stratified shuffle split. The parity target is "
            "the scikit-learn `KBinsDiscretizer` plus `StratifiedShuffleSplit` behavior documented at "
            "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html."
        ),
        "principle": (
            "Discretize y into `n_bins` by equal-width (`uniform`) or linearly interpolated equal-frequency (`quantile`) edges. "
            "Allocate class-wise train/test counts with sklearn's approximate-mode rule, shuffle members and final outputs "
            "with legacy MT19937 seeded by `seed`, and choose $n_{test}=\\lceil n\\,test\\_size\\rceil$."
        ),
        "use_cases": "Preserving an approximate continuous-target distribution in one random train/test split.",
        "limitations": (
            "Sparse or repeated y values can collapse bins or leave too few members for stratification. Output index order is "
            "shuffled, not sorted, and only y drives the split. The current Python binding requires y with shape "
            "`(n_samples, 1)`; its generic 1-D promotion produces shape `(1, n_samples)` and the ABI rejects it."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.KBinsStratified`; binding class `KBinsStratifiedSplitter`; ABI 2 "
            "family `n4m_model_selection_kbins_stratified_{create,split,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/kbins_stratified.h; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html",
    },
    "split_binned_strat_group_kfold": {
        "title": "Binned, group-preserving round-robin folds",
        "paper": (
            "No canonical paper defines this simplified grouped heuristic. It is inspired by binned stratification and "
            "`StratifiedGroupKFold`, but the native source explicitly documents a different assignment algorithm."
        ),
        "principle": (
            "Bin continuous y by uniform or quantile edges. Label each group using the bin of its first encountered sample; "
            "within each bin, optionally PCG64-shuffle groups and assign them round-robin to `n_splits`. Expand group fold "
            "labels to samples, guaranteeing group integrity."
        ),
        "use_cases": "Approximate target stratification when all rows from a group must stay in one fold.",
        "limitations": (
            "This is not sklearn's constraint-aware `StratifiedGroupKFold`: a group's first sample alone determines its bin. "
            "Heterogeneous or unequal-size groups can produce poor class balance and sample-count imbalance. The current "
            "Python binding also requires y as `(n_samples, 1)`, not a 1-D vector."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.BinnedStratifiedGroupKFold`; binding class "
            "`BinnedStratifiedGroupKFoldSplitter`; ABI 2 family "
            "`n4m_model_selection_binned_strat_group_kfold_{create,n_splits,split_fold,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/binned_strat_group_kfold.h",
    },
    "split_systematic_circular": {
        "title": "Systematic circular sampling over sorted targets",
        "paper": (
            "No canonical paper defines this exact rotate-and-systematically-sample implementation. It is an internal "
            "systematic sampling heuristic whose seeded offset and rounding rules are source-defined."
        ),
        "principle": (
            "Sort sample indices by y, draw a seeded circular offset, rotate that order, and choose `n_train` positions at "
            "approximately equal spacing $step=n/n_{train}$ using rounded $step\\,i$. The remaining rotated positions form "
            "the test set; the engine sorts both final index arrays for stable output."
        ),
        "use_cases": "Spreading calibration samples across the ordered response range without bin boundaries.",
        "limitations": (
            "It uses y and therefore is inappropriate for a blind final test. Periodic ordering and rounding can create "
            "structure; ties are resolved by sorting behavior, and multicolumn y is rejected. The current Python binding "
            "requires y as `(n_samples, 1)` because its generic 1-D promotion creates the wrong orientation."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.SystematicCircular`; binding class "
            "`SystematicCircularSplitter`; ABI 2 family "
            "`n4m_model_selection_systematic_circular_{create,split,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/systematic_circular.h",
    },
    "split_split_splitter": {
        "title": "SPlit sequential data-twinning split",
        "paper": (
            f"Joseph & Vakayil (2021/2022), *SPlit: An Optimal Method for Data Splitting*, Technometrics 64, "
            f"166–176, DOI 10.1080/00401706.2021.1921037 ({_SPLIT}). The shipped routine is the paper-inspired "
            "sequential nearest-neighbor algorithm documented in source."
        ),
        "principle": (
            "Drop constant columns and z-score the rest. Let $r=\\lfloor1/test\\_size\\rfloor$ and choose a seeded start. "
            "Repeatedly take the r nearest active samples to the current point, place the closest representative in the "
            "smaller twin (test), deactivate that neighborhood, and choose the next current point from the remaining set "
            "near the farthest deactivated member. Training is the ascending complement."
        ),
        "use_cases": "Producing model-independent train/test twins intended to have similar multivariate distributions.",
        "limitations": (
            "The class name is `SPlitSplitter`/role alias `DataTwinning`, but the native algorithm implements the SPlit-style "
            "sequential routine, not the later full Twinning package. The reciprocal conversion means arbitrary "
            "`test_size` values are quantized through integer r; seed changes the start."
        ),
        "implementation": (
            "Python role API `n4m.model_selection.splitters.DataTwinning`; binding class `SPlitSplitter`; ABI 2 family "
            "`n4m_model_selection_data_twinning_{create,split,destroy}`."
        ),
        "provenance": _SRC + "cpp/src/core/splitters/split_splitter.h; " + _SPLIT,
    },
}
