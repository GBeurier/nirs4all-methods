r"""Scientific bibliographies for the pls4all method catalogue.

Each entry is a dict with:
  - group:   taxonomy key (core, sparse, ensemble, ...).
  - title:   long-form method name.
  - paper:   primary bibliographic reference(s).
  - principle: 2-4 paragraph mathematical / geometric explanation,
               with the relevant equations inline in MyST $…$ form.
  - implementation: pls4all kernel + reference library + caveats.

Style guide:
  * Write for a chemometrician / data scientist, not a beginner.
  * State *why* the method exists vs the previous one in its family.
  * Use $\mathbf{X}, \mathbf{Y}, \mathbf{T}, \mathbf{P}, \mathbf{W}$
    consistently. Bold capitals for matrices, lower-case for vectors.
  * Include complexity ($O(\cdot)$) where relevant.
  * Cite by author + year + journal, not just "Wold 2001".

Re-ordered by usage frequency: core PLS variants first, then sparse,
ensemble, robust, nonlinear, multi-block, calibration transfer,
classification, missing data, regularised, diagnostics, selectors.
"""
from __future__ import annotations


BIBLIOGRAPHY: dict[str, dict] = {

    # ================================================================
    #   CORE PLS FAMILY
    # ================================================================

    "pls": {
        "group": "core",
        "title": "PLS regression (SIMPLS)",
        "paper": (
            "de Jong, S. (1993). *SIMPLS: an alternative approach to "
            "partial least squares regression*. Chemometrics and "
            "Intelligent Laboratory Systems 18(3), 251–263."
        ),
        "principle": (
            "Partial Least Squares regression seeks a set of latent "
            "directions in the predictor space that maximise the "
            "*covariance* with the response, in contrast to PCA which "
            "maximises only the variance of $\\mathbf{X}$.\n\n"
            "Given centred $\\mathbf{X} \\in \\mathbb{R}^{n\\times p}$ "
            "and $\\mathbf{Y} \\in \\mathbb{R}^{n\\times q}$, the first "
            "PLS component is the unit-norm direction "
            "$\\mathbf{w}_1$ maximising $\\operatorname{Cov}(\\mathbf{X}\\mathbf{w}_1, \\mathbf{Y})$. "
            "Closed form: $\\mathbf{w}_1 \\propto \\mathbf{X}^{\\top}\\mathbf{Y}$ "
            "(or its dominant left singular vector when $q>1$). "
            "Subsequent components are extracted from the deflated "
            "residual matrix so the resulting scores "
            "$\\mathbf{T} = \\mathbf{X}\\mathbf{W}$ are orthogonal.\n\n"
            "**SIMPLS** (de Jong 1993) is algebraically equivalent to "
            "NIPALS but computes the loading weights directly from the "
            "cross-product $\\mathbf{S} = \\mathbf{X}^{\\top}\\mathbf{Y}$ "
            "without re-deflating $\\mathbf{X}$ at each step. This "
            "avoids accumulating floating-point error from iterative "
            "deflation and runs in roughly half the time of NIPALS for "
            "the same number of components. SIMPLS is the variant "
            "exposed by MATLAB's `plsregress`.\n\n"
            "Once $k$ latent scores have been extracted the regression "
            "coefficients are reconstructed as "
            "$\\mathbf{B} = \\mathbf{W}(\\mathbf{P}^{\\top}\\mathbf{W})^{-1}\\mathbf{Q}^{\\top}$, "
            "where $\\mathbf{P}, \\mathbf{Q}$ are the X- and Y-loadings. "
            "Predictions on new $\\mathbf{X}^{\\star}$ follow "
            "$\\hat{\\mathbf{Y}} = \\mathbf{X}^{\\star}\\mathbf{B} + \\bar{\\mathbf{y}}$. "
            "The choice of $k$ trades bias and variance: use "
            "cross-validated PRESS or the one-SE rule of Hastie et al. "
            "(2009) to select it."
        ),
        "implementation": (
            "Dispatched through `Algorithm.PLS_REGRESSION` + "
            "`Solver.SIMPLS` in libp4a (the `p4a_pls_fit` C entry "
            "point). The same `Model.fit` / `Model.predict` surface is "
            "used by every binding. NIPALS, SVD, power-iteration, "
            "randomised-SVD, orthogonal-scores, kernel and wide-kernel "
            "solver variants are all available — see the "
            "`Solver` enum."
        ),
    },

    "pcr": {
        "group": "core",
        "title": "Principal Components Regression",
        "paper": (
            "Massy, W. F. (1965). *Principal Components Regression in "
            "Exploratory Statistical Research*. JASA 60(309), 234–256."
        ),
        "principle": (
            "PCR sidesteps the multicollinearity of $\\mathbf{X}$ by "
            "regressing on its orthogonal principal-component scores "
            "rather than on the raw columns. The factorisation "
            "$\\mathbf{X} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^{\\top}$ "
            "(SVD) yields scores "
            "$\\mathbf{T}_k = \\mathbf{U}_k\\boldsymbol{\\Sigma}_k$ "
            "for the top $k$ components, and the regression "
            "$\\mathbf{Y} = \\mathbf{T}_k\\mathbf{Q}_k + \\mathbf{E}$ is "
            "fit by ordinary least squares.\n\n"
            "Unlike PLS, PCR is **unsupervised in its dimensionality "
            "reduction**: the first $k$ directions maximise the variance "
            "of $\\mathbf{X}$ regardless of how relevant they are to "
            "$\\mathbf{Y}$. This makes PCR a useful baseline for "
            "diagnosing whether the predictive directions in a "
            "calibration set really do coincide with the high-variance "
            "directions (in which case PCR ≈ PLS) or not (in which case "
            "PLS is strictly preferable at the same $k$).\n\n"
            "Coefficients in the original feature scale are recovered "
            "as $\\mathbf{B} = \\mathbf{V}_k \\boldsymbol{\\Sigma}_k^{-1} \\mathbf{T}_k^{\\top}\\mathbf{Y}$. "
            "Total cost is dominated by the partial SVD: "
            "$O(np\\min(n,p))$ for a full decomposition, or "
            "$O(npk)$ with a truncated method (Lanczos, randomised SVD)."
        ),
        "implementation": (
            "`Algorithm.PCR` + `Solver.SVD` in libp4a. Reference "
            "implementations are scikit-learn's "
            "`Pipeline(PCA(n_components=k), LinearRegression())` and "
            "R `pls::pcr`."
        ),
    },

    "opls": {
        "group": "core",
        "title": "Orthogonal PLS (OPLS)",
        "paper": (
            "Trygg, J. & Wold, S. (2002). *Orthogonal projections to "
            "latent structures (O-PLS)*. Journal of Chemometrics "
            "16(3), 119–128."
        ),
        "principle": (
            "OPLS rotates the standard PLS latent space so that a "
            "single direction captures all $\\mathbf{Y}$-correlated "
            "variation while the remaining components capture "
            "$\\mathbf{Y}$-orthogonal structural variation in "
            "$\\mathbf{X}$. The resulting decomposition "
            "$\\mathbf{X} = \\mathbf{t}_p\\mathbf{p}_p^{\\top} + "
            "\\mathbf{T}_o\\mathbf{P}_o^{\\top} + \\mathbf{E}$ "
            "separates the **predictive component** $\\mathbf{t}_p$ "
            "from the orthogonal block $\\mathbf{T}_o$, which absorbs "
            "spectroscopic baselines, scatter and other nuisance "
            "factors that confound interpretation of the predictive "
            "loading.\n\n"
            "Numerically OPLS proceeds by NIPALS-deflating "
            "$\\mathbf{X}$ against directions orthogonal to "
            "$\\mathbf{X}^{\\top}\\mathbf{y}$ before each new "
            "predictive component is extracted. Predictions are "
            "identical to those of a one-component PLS on the "
            "orthogonal-corrected $\\mathbf{X}$; the value is in **the "
            "interpretation of the loadings**, not in better "
            "predictions per se.\n\n"
            "OPLS shines in metabolomics and process spectroscopy "
            "where the spectra carry strong systematic but "
            "non-predictive variation; in those settings the "
            "single-vector predictive loading is far easier to relate "
            "to biology / chemistry than a multi-component PLS "
            "loading matrix."
        ),
        "implementation": (
            "`Algorithm.OPLS` + `Solver.NIPALS` + "
            "`Deflation.ORTHOGONAL`. Reference: Bioconductor "
            "`ropls::opls`. Note: orthogonal-component ordering and "
            "the criterion that stops orthogonal extraction differ "
            "between implementations — exact bit parity is not "
            "expected, but RMSE-rel parity within ~1e-3 is."
        ),
    },

    "cppls": {
        "group": "core",
        "title": "Powered PLS (Indahl 2005)",
        "paper": (
            "Indahl, U. G. (2005). *A twist to partial least squares "
            "regression*. Journal of Chemometrics 19(1), 32–44."
        ),
        "principle": (
            "Powered PLS introduces a single hyperparameter "
            "$\\gamma \\in [0, 1]$ that morphs the loading-weight "
            "definition between PCA ($\\gamma=0$) and PLS "
            "($\\gamma=1$). Concretely, the loading weight is "
            "$\\mathbf{w} \\propto \\operatorname{sign}(\\mathbf{X}^{\\top}\\mathbf{y}) \\odot |\\mathbf{X}^{\\top}\\mathbf{y}|^{\\gamma}$ "
            "where $\\odot$ is the element-wise product.\n\n"
            "When $\\mathbf{X}$ carries weakly informative columns "
            "alongside strongly informative ones, raising the "
            "covariance to a power $\\gamma < 1$ tempers the influence "
            "of the dominant columns and produces a more uniform "
            "weighting, which empirically improves CV-RMSE on "
            "spectra with sharp absorption peaks dominating the "
            "covariance. The implementation reduces to standard "
            "SIMPLS at $\\gamma=1$.\n\n"
            "**Important nomenclature caveat:** R's `pls::cppls` "
            "(Liland 2009) implements *Canonical Powered PLS*, a "
            "completely different algorithm that orthogonalises blocks "
            "of $\\mathbf{Y}$ before powering. The pls4all "
            "implementation matches Indahl 2005, **not** Liland 2009. "
            "The benchmark widens the parity tolerance for this method "
            "to surface the divergence as a drift rather than treat "
            "it as a bug."
        ),
        "implementation": (
            "`p4a_cppls_fit` (MethodResult entry point). No widely "
            "installable Python reference; R `pls::cppls 2.8.5` is the "
            "closest installable analogue but implements Liland 2009 "
            "rather than Indahl 2005."
        ),
    },

    "recursive_pls": {
        "group": "core",
        "title": "Recursive (moving-window) PLS",
        "paper": (
            "Helland, K., Berntsen, H. E., Borgen, O. S. & Martens, H. "
            "(1992). *Recursive algorithm for partial least squares "
            "regression*. Chemometrics and Intelligent Laboratory "
            "Systems 14(1–3), 129–137."
        ),
        "principle": (
            "Process-analytical instruments produce streams of "
            "spectra under drifting conditions (changing humidity, "
            "instrument warm-up, fouling). Recursive PLS maintains a "
            "fitted model that **adapts as new samples arrive** by "
            "re-fitting on a sliding window of the most recent "
            "$w$ samples.\n\n"
            "At time step $t$, the model is fit on "
            "$\\{(\\mathbf{x}_{t-w+1}, y_{t-w+1}), \\ldots, "
            "(\\mathbf{x}_t, y_t)\\}$ and applied to incoming "
            "$\\mathbf{x}_{t+1}$. Computational cost is "
            "$O(wpk)$ per step. The window width $w$ controls a "
            "stability/adaptability trade-off: short windows track "
            "drift aggressively but are noisier; long windows are "
            "stable but lag.\n\n"
            "More sophisticated recursive variants (Qin 1998) use "
            "exponential forgetting factors instead of a hard window. "
            "pls4all's variant uses the hard-window form for "
            "deterministic parity with R `pls` rolling refits."
        ),
        "implementation": (
            "`p4a_recursive_pls_run` (returns predictions only — "
            "no global coefficient export, since the model changes per "
            "step). The Python sklearn wrapper is an in-sample-only "
            "estimator."
        ),
    },

    # ================================================================
    #   SPARSE PLS FAMILY
    # ================================================================

    "sparse_simpls": {
        "group": "sparse",
        "title": "Sparse SIMPLS (Chun & Keleş 2010)",
        "paper": (
            "Chun, H. & Keleş, S. (2010). *Sparse partial least "
            "squares regression for simultaneous dimension reduction "
            "and variable selection*. JRSS B 72(1), 3–25."
        ),
        "principle": (
            "Sparse PLS adds a soft-thresholding step to each SIMPLS "
            "loading weight so that the latent direction is supported "
            "on only a small subset of features. Mathematically, after "
            "the un-thresholded weight $\\mathbf{w}$ is computed, we "
            "solve "
            "$\\mathbf{w}^{\\star} = \\arg\\min_{\\|\\mathbf{c}\\|=1} "
            "\\|\\mathbf{c} - \\mathbf{w}\\|_2^2 + "
            "\\lambda \\|\\mathbf{c}\\|_1$, "
            "which has the closed-form soft-threshold solution "
            "$c_j = \\operatorname{sign}(w_j)\\,(|w_j| - \\lambda/2)_+$ "
            "followed by re-normalisation.\n\n"
            "The penalty $\\lambda$ controls sparsity: small $\\lambda$ "
            "approaches standard PLS, large $\\lambda$ zeroes most "
            "weights. In high-dimensional ($p \\gg n$) spectroscopy or "
            "omics data, sparse PLS simultaneously builds the latent "
            "predictive direction *and* selects the variables that "
            "support it — a much cleaner story than running PLS then "
            "thresholding coefficients post-hoc.\n\n"
            "The Chun & Keleş formulation differs subtly from the "
            "earlier Lê Cao 2008 sPLS (used in mixOmics): Chun & Keleş "
            "threshold the un-deflated weight while Lê Cao threshold "
            "the deflated weight at each iteration. pls4all implements "
            "the Chun & Keleş formulation."
        ),
        "implementation": (
            "`p4a_sparse_simpls_fit`. Reference: CRAN `spls 2.3.2` "
            "(Chun & Keleş authors). No widely installable Python port "
            "exists with this exact normalisation convention."
        ),
    },

    "group_sparse_pls": {
        "group": "sparse",
        "title": "Group-sparse PLS (Liquet 2016)",
        "paper": (
            "Liquet, B., de Micheaux, P. L., Hejblum, B. P. & "
            "Thiébaut, R. (2016). *Group and sparse group partial "
            "least squares approaches applied in genomics context*. "
            "Bioinformatics 32(1), 35–42."
        ),
        "principle": (
            "When features partition into known groups — gene "
            "pathways, spectroscopic bands, biological assays — "
            "group-sparse PLS forces entire groups in or out together "
            "via a group-lasso penalty: "
            "$\\mathcal{P}(\\mathbf{w}) = \\sum_g \\sqrt{|g|}\\,"
            "\\|\\mathbf{w}_g\\|_2$, "
            "where $\\mathbf{w}_g$ is the sub-vector of weights "
            "belonging to group $g$ and $|g|$ is its size. The "
            "$\\ell_2$ norm inside the sum is non-differentiable at "
            "zero, which produces group-level sparsity (an entire "
            "$\\mathbf{w}_g$ is either zero or non-zero).\n\n"
            "Compared to plain sparse PLS, this gives a much more "
            "interpretable model when groups have biological meaning "
            "and avoids the situation where one or two members of a "
            "co-regulated cluster get selected while the rest don't.\n\n"
            "Required input: a `group_assignment` vector mapping each "
            "feature to a group id."
        ),
        "implementation": (
            "`p4a_group_sparse_pls_fit`. Reference: CRAN `sgPLS 1.8.1`."
        ),
    },

    "fused_sparse_pls": {
        "group": "sparse",
        "title": "Fused-sparse PLS",
        "paper": (
            "Tibshirani, R., Saunders, M., Rosset, S., Zhu, J. & "
            "Knight, K. (2005). *Sparsity and smoothness via the fused "
            "lasso*. JRSS B 67(1), 91–108. — generalised to PLS "
            "loadings."
        ),
        "principle": (
            "The fused lasso penalty combines L1 sparsity with an "
            "L1 penalty on first differences of neighbouring "
            "coefficients: "
            "$\\mathcal{P}(\\mathbf{w}) = "
            "\\lambda_1 \\|\\mathbf{w}\\|_1 + "
            "\\lambda_2 \\sum_{j=1}^{p-1}|w_{j+1} - w_j|$. "
            "For spectroscopic data this is uniquely useful: the "
            "second term *encourages neighbouring wavelengths to "
            "share a coefficient*, which produces piecewise-constant "
            "loadings reflecting the underlying band structure of the "
            "spectrum rather than the choppy sample-noise pattern that "
            "plain L1 alone produces.\n\n"
            "Tuning two penalties simultaneously is non-trivial: "
            "$\\lambda_1$ controls overall sparsity, "
            "$\\lambda_2$ controls within-band smoothness. Plotting the "
            "selected wavelength bands as a function of $\\lambda_2$ "
            "at fixed sparsity is a common diagnostic.\n\n"
            "Computationally the fused-lasso step is a 1-D total "
            "variation denoising problem with a closed-form taut-string "
            "solution in $O(p)$; pls4all uses Condat's algorithm."
        ),
        "implementation": (
            "`p4a_fused_sparse_pls_fit`. No widely installable "
            "reference library — `sgPLS` covers group-sparse but not "
            "fused-sparse. Documented as `paper_only` in the registry."
        ),
    },

    "sparse_pls_da": {
        "group": "sparse",
        "title": "Sparse PLS-DA (Lê Cao 2008)",
        "paper": (
            "Lê Cao, K.-A., Rossouw, D., Robert-Granié, C. & "
            "Besse, P. (2008). *A sparse PLS for variable selection "
            "when integrating omics data*. Statistical Applications in "
            "Genetics and Molecular Biology 7(1)."
        ),
        "principle": (
            "Discriminant variant of sparse PLS. Encode class labels "
            "$y \\in \\{0, 1, \\ldots, C-1\\}$ as a one-hot matrix "
            "$\\mathbf{Y} \\in \\{0, 1\\}^{n \\times C}$, fit a sparse "
            "PLS regression on it, then assign new samples to the class "
            "with the largest predicted score. The L1 penalty selects a "
            "discriminative subset of features along each latent "
            "direction.\n\n"
            "In high-dimensional biomarker discovery (microarray, "
            "MALDI-TOF, NIR food classification) sparse PLS-DA is a "
            "standard since it simultaneously builds the discriminant "
            "and shortlists the candidate markers in a single "
            "regularised fit. Class probabilities follow from a "
            "softmax over the predicted score columns."
        ),
        "implementation": (
            "`p4a_sparse_pls_da_fit`. Reference: Bioconductor "
            "`mixOmics::splsda`."
        ),
    },

    # ================================================================
    #   ENSEMBLE METHODS
    # ================================================================

    "bagging_pls": {
        "group": "ensemble",
        "title": "Bagging PLS",
        "paper": (
            "Breiman, L. (1996). *Bagging predictors*. Machine "
            "Learning 24(2), 123–140. — adapted for PLS by various "
            "chemometric authors."
        ),
        "principle": (
            "Bootstrap aggregating draws $B$ bootstrap samples "
            "$\\{(\\mathbf{X}^{(b)}, \\mathbf{y}^{(b)})\\}_{b=1}^{B}$ "
            "from the calibration set (sampling with replacement, "
            "$n$ rows each), fits a PLS model on each, and averages "
            "the predictions: "
            "$\\hat{y}_{\\mathrm{bag}}(\\mathbf{x}) = "
            "\\frac{1}{B}\\sum_b \\hat{y}^{(b)}(\\mathbf{x})$.\n\n"
            "PLS is a high-bias / low-variance learner, so bagging "
            "rarely beats a single well-tuned PLS in pure RMSE. Its "
            "real value is **inferential**: the bootstrap distribution "
            "of coefficients gives non-parametric standard errors and "
            "confidence intervals that are otherwise inaccessible. The "
            "per-bag $\\mathbf{B}^{(b)}$ matrices form an empirical "
            "distribution from which posterior intervals on each "
            "feature's contribution can be read off.\n\n"
            "Computational cost: $B$ times a single fit, embarrassingly "
            "parallel. Use $B \\in [50, 500]$ depending on how stable "
            "the CIs need to be."
        ),
        "implementation": (
            "`p4a_bagging_pls_fit`. Reference: CRAN `enpls 6.1.1`."
        ),
    },

    "boosting_pls": {
        "group": "ensemble",
        "title": "Boosting PLS",
        "paper": (
            "Friedman, J. H. (2001). *Greedy function approximation: "
            "a gradient boosting machine*. Annals of Statistics 29(5), "
            "1189–1232. — adapted for PLS as a base learner."
        ),
        "principle": (
            "Gradient boosting builds an additive predictor "
            "$F_M(\\mathbf{x}) = \\sum_{m=1}^M \\eta\\, h_m(\\mathbf{x})$ "
            "where each weak learner $h_m$ is fit on the negative "
            "gradient (the residuals, for squared-error loss) of the "
            "current ensemble. With PLS as the weak learner, each "
            "$h_m$ is a small ($k$-component) PLS fitted on the "
            "pseudo-response "
            "$r_i^{(m)} = y_i - F_{m-1}(\\mathbf{x}_i)$.\n\n"
            "The learning rate $\\eta$ (typically 0.05–0.1) and the "
            "number of boosting iterations $M$ are the key "
            "hyperparameters; their product roughly controls the "
            "effective number of latent dimensions explored. Because "
            "boosting reduces bias, it can recover non-linear "
            "$Y$–$X$ relationships even with linear PLS base "
            "learners — at the cost of much higher computational cost "
            "than a single PLS."
        ),
        "implementation": (
            "`p4a_boosting_pls_fit`. Reference: CRAN `mboost::glmboost` "
            "with a PLS base learner (`mboost 2.9.11`)."
        ),
    },

    "random_subspace_pls": {
        "group": "ensemble",
        "title": "Random-subspace PLS",
        "paper": (
            "Ho, T. K. (1998). *The random subspace method for "
            "constructing decision forests*. IEEE TPAMI 20(8), "
            "832–844. — adapted for PLS regressors."
        ),
        "principle": (
            "Each ensemble member fits PLS on a random subset of "
            "$m \\ll p$ features. Compared to bagging (which "
            "randomises rows), random subspaces randomise **columns**, "
            "which is a much stronger variance-reduction mechanism for "
            "high-dimensional collinear data like NIR spectra: "
            "different subsets pick up different bands of the spectrum, "
            "and averaging across them smooths out band-specific noise.\n\n"
            "Variance per member is higher than a full-feature PLS "
            "(less information per fit), but the ensemble average "
            "outperforms a single fit when the underlying truth is "
            "spread across many weakly-correlated features. Choosing "
            "$m \\approx \\sqrt{p}$ is a Breiman-style default; for "
            "spectra a more informed choice respects band widths.\n\n"
            "Note that prediction on a new sample requires evaluating "
            "every member on **its own subset** of features, so the "
            "feature-index map must be stored per member."
        ),
        "implementation": "`p4a_random_subspace_pls_fit`.",
    },

    # ================================================================
    #   ROBUST / WEIGHTED
    # ================================================================

    "weighted_pls": {
        "group": "robust",
        "title": "Sample-weighted PLS",
        "paper": (
            "Martens, H. & Næs, T. (1989). *Multivariate Calibration*. "
            "Wiley. §4.5 'Weighted regression for non-i.i.d. errors'."
        ),
        "principle": (
            "When the residual variance is not constant across samples "
            "— typical when calibration spectra are aggregated across "
            "instruments, sites or operators — a weighted least "
            "squares fit can dramatically improve generalisation. "
            "Weighted PLS prescales centred rows by $\\sqrt{w_i}$ "
            "before extracting the SIMPLS components: "
            "$\\tilde{\\mathbf{X}} = \\operatorname{diag}(\\sqrt{w})\\,"
            "\\mathbf{X}_c, \\quad "
            "\\tilde{\\mathbf{Y}} = \\operatorname{diag}(\\sqrt{w})\\,"
            "\\mathbf{Y}_c$, "
            "then runs vanilla SIMPLS on $(\\tilde{\\mathbf{X}}, "
            "\\tilde{\\mathbf{Y}})$.\n\n"
            "Weights $w_i > 0$ encode any known per-sample reliability: "
            "inverse residual variance from a previous fit, instrument "
            "noise estimates, sample replicate counts. The weighted "
            "fit is mathematically equivalent to running standard PLS "
            "on a duplicated dataset where each row appears $w_i$ "
            "times.\n\n"
            "This is a building block for robust PLS (IRLS over a "
            "weighted fit) and for incorporating known measurement "
            "noise into the calibration."
        ),
        "implementation": (
            "`p4a_weighted_pls_fit` (in-sample only — no global "
            "coefficient export, since the weighted fit's "
            "$\\bar{\\mathbf{x}}, \\bar{\\mathbf{y}}$ depend on the "
            "weights). Python reference: sklearn `PLSRegression` on "
            "the prescaled matrices."
        ),
    },

    "robust_pls": {
        "group": "robust",
        "title": "Robust PLS (Partial Robust M-regression)",
        "paper": (
            "Serneels, S., Croux, C., Filzmoser, P. & Van Espen, "
            "P. J. (2005). *Partial Robust M-Regression*. Chemometrics "
            "and Intelligent Laboratory Systems 79(1–2), 55–64."
        ),
        "principle": (
            "Robust PLS performs a sequence of weighted PLS fits where "
            "the weights $w_i$ are reduced for samples with large "
            "current residuals — the iteratively-reweighted "
            "least-squares (IRLS) recipe.\n\n"
            "At each iteration, the weight of sample $i$ is set from "
            "Huber's $\\psi$-function applied to the standardised "
            "residual: "
            "$w_i = \\psi(r_i / s) / r_i$ where "
            "$\\psi(z) = z$ for $|z| \\le k$ and "
            "$\\psi(z) = k\\,\\operatorname{sign}(z)$ otherwise. "
            "$k = 1.345$ gives 95 % asymptotic efficiency at the "
            "Gaussian. The robust scale $s$ is typically the MAD "
            "of the residuals.\n\n"
            "Convergence is rapid: 3–5 iterations typically suffice. "
            "Robust PLS down-weights — rather than removes — outliers, "
            "which is desirable when outlier-ness is a continuous "
            "concept (mild spectral artefacts) rather than binary "
            "(broken samples).\n\n"
            "Compared to median-PLS variants, the M-regression form "
            "preserves the analytic structure of SIMPLS and offers "
            "smooth weighting; it also generalises to leverage-based "
            "weights (PRM with x-weights)."
        ),
        "implementation": (
            "`p4a_robust_pls_fit`. Reference: CRAN `chemometrics::prm` "
            "(Serneels et al. authors). The exact weight schedule and "
            "scale estimator differ from `prm` so RMSE-rel parity is "
            "widened to ~2.0 to flag presence rather than enforce "
            "exact agreement."
        ),
    },

    "ridge_pls": {
        "group": "regularized",
        "title": "Ridge-augmented PLS",
        "paper": (
            "Hoerl, A. E. & Kennard, R. W. (1970). *Ridge regression: "
            "biased estimation for nonorthogonal problems*. "
            "Technometrics 12(1), 55–67. — combined with PLS via "
            "Tikhonov regularisation of the inner regression."
        ),
        "principle": (
            "When the number of components $k$ approaches the "
            "rank of $\\mathbf{X}$, the inner regression of $\\mathbf{Y}$ "
            "on the PLS scores becomes ill-conditioned. Ridge-augmented "
            "PLS adds an L2 penalty to that inner regression: "
            "$\\hat{\\mathbf{Q}} = "
            "(\\mathbf{T}^{\\top}\\mathbf{T} + \\lambda \\mathbf{I})^{-1}"
            "\\mathbf{T}^{\\top}\\mathbf{Y}$, "
            "yielding a shrinkage-stabilised coefficient matrix.\n\n"
            "Setting $\\lambda$ from cross-validation on a "
            "logarithmic grid is the standard procedure. The combined "
            "method is more forgiving than pure PLS to a slightly "
            "over-specified $k$: pure PLS over-fits hard at "
            "$k > k_{\\mathrm{opt}}$ while ridge-augmented degrades "
            "smoothly. Conceptually it is a continuous interpolation "
            "between PLS ($\\lambda=0$) and a heavily-regularised "
            "low-rank ridge regression in latent space.\n\n"
            "When $\\lambda$ is set per component via the SVD spectrum "
            "of $\\mathbf{T}$, ridge PLS is closely related to "
            "Krylov-subspace PCR with shrinkage."
        ),
        "implementation": (
            "`p4a_ridge_pls_fit` (in-sample only). No widely "
            "installable reference for this exact formulation; the "
            "test compares against an sklearn `PLSRegression` + manual "
            "Tikhonov inner regression."
        ),
    },

    "continuum_regression": {
        "group": "nonlinear",
        "title": "Continuum Regression (Stone & Brooks 1990)",
        "paper": (
            "Stone, M. & Brooks, R. J. (1990). *Continuum regression: "
            "cross-validated sequentially constructed prediction "
            "embracing ordinary least squares, partial least squares "
            "and principal components regression*. JRSS B 52(2), "
            "237–269."
        ),
        "principle": (
            "Continuum regression introduces a single shape parameter "
            "$\\tau \\in [0, 1]$ that selects the loading-weight "
            "criterion: "
            "$\\mathbf{w} \\propto \\operatorname{Cov}(\\mathbf{X}\\mathbf{w}, \\mathbf{y})^{\\tau} \\cdot \\operatorname{Var}(\\mathbf{X}\\mathbf{w})^{1-\\tau}$. "
            "Special cases: $\\tau = 0$ gives PCR (variance-maximising), "
            "$\\tau = 1/2$ gives PLS (covariance-maximising), $\\tau = 1$ "
            "gives OLS (correlation-maximising, in the appropriate "
            "limit).\n\n"
            "Cross-validating $\\tau$ on a fine grid often improves "
            "RMSE over the discrete PLS / PCR choices — the optimum is "
            "rarely exactly at $\\tau = 0.5$. Stone & Brooks' original "
            "treatment also cross-validates the number of components "
            "$k$ jointly with $\\tau$, producing a 2-D grid.\n\n"
            "Implementation note: numerically stable continuum "
            "regression uses the parameterised power method on the "
            "matrix $\\mathbf{X}^{\\top}\\mathbf{y}\\mathbf{y}^{\\top}\\mathbf{X} / (\\mathbf{X}^{\\top}\\mathbf{X})^{1-\\tau}$, "
            "which avoids forming the rank-1 outer product explicitly "
            "and is what pls4all uses."
        ),
        "implementation": "`p4a_continuum_regression_fit` (in-sample only).",
    },

    # ================================================================
    #   NONLINEAR / LOCAL
    # ================================================================

    "kernel_pls_rbf": {
        "group": "nonlinear",
        "title": "Kernel PLS (Rosipal & Trejo 2001)",
        "paper": (
            "Rosipal, R. & Trejo, L. J. (2001). *Kernel partial least "
            "squares regression in reproducing kernel Hilbert space*. "
            "Journal of Machine Learning Research 2, 97–123."
        ),
        "principle": (
            "Kernel PLS runs the NIPALS PLS procedure entirely in the "
            "feature space of a Mercer kernel "
            "$k(\\mathbf{x}, \\mathbf{x}') = \\langle \\phi(\\mathbf{x}), "
            "\\phi(\\mathbf{x}') \\rangle$ without ever forming "
            "$\\phi$ explicitly. The kernel matrix "
            "$\\mathbf{K}_{ij} = k(\\mathbf{x}_i, \\mathbf{x}_j) \\in "
            "\\mathbb{R}^{n \\times n}$ replaces $\\mathbf{X}\\mathbf{X}^{\\top}$ "
            "in the NIPALS recursion and the score matrix is built "
            "directly from $\\mathbf{K}$.\n\n"
            "The RBF kernel "
            "$k(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma \\|\\mathbf{x} - \\mathbf{x}'\\|^2)$ "
            "is the standard choice for non-linear PLS: it captures "
            "smooth non-linear relationships between $\\mathbf{X}$ and "
            "$y$ at the cost of a single bandwidth hyperparameter "
            "$\\gamma$. Other kernels (polynomial, sigmoid) are available "
            "via the `kernel_type` enum.\n\n"
            "Memory scales as $O(n^2)$ which is the binding constraint "
            "for kernel PLS on spectroscopy datasets; subsampling "
            "(Nyström) or random Fourier features are the standard "
            "scale-up strategies but are not currently exposed."
        ),
        "implementation": (
            "`p4a_kernel_pls_fit`. Predict-on-new-X is currently "
            "marked in-sample-only in the Python `sklearn` wrapper "
            "because the C ABI does not yet export the kernel-centring "
            "constants required to handle a fresh test point. The "
            "tier-1 entry point will refit on (X_train ∪ X_test) on "
            "demand."
        ),
    },

    "lw_pls": {
        "group": "nonlinear",
        "title": "Locally-Weighted PLS (LW-PLS)",
        "paper": (
            "Centner, V. & Massart, D. L. (1998). *Optimisation in "
            "locally weighted regression*. Analytical Chemistry 70(19), "
            "4206–4211."
        ),
        "principle": (
            "Instead of fitting a single global PLS, LW-PLS refits a "
            "*per-prediction-point* local PLS using only the $k$-nearest "
            "calibration samples (in $\\mathbf{X}$-space distance). "
            "This adapts the model to the local geometry around each "
            "query point and is effective on calibration sets that "
            "span heterogeneous regimes (e.g. a single instrument "
            "calibrated across several product classes).\n\n"
            "The neighbourhood weight typically combines distance "
            "(Gaussian or tricube kernel on the Euclidean / Mahalanobis "
            "distance) with the inverse residual variance from a "
            "preliminary global fit. The local PLS uses few components "
            "(typically 2–4) because the neighbourhood is small.\n\n"
            "Prediction cost is $O(n)$ for the neighbour search plus "
            "$O(k_{\\mathrm{nn}} \\cdot p \\cdot k_{\\mathrm{pls}})$ for "
            "the local fit, per query. KD-tree / ball-tree indices "
            "accelerate the neighbour search; pls4all uses an "
            "exhaustive scan because $p \\gg n$ defeats most spatial "
            "indices for NIR data anyway."
        ),
        "implementation": (
            "`p4a_lw_pls_fit`. Reference: sanctioned git-pinned port "
            "`nirs4all.operators.models.sklearn.lwpls`."
        ),
    },

    "gpr_pls": {
        "group": "nonlinear",
        "title": "Gaussian Process on PLS scores",
        "paper": (
            "Bishop, C. M. (2006). *Pattern Recognition and Machine "
            "Learning*, §6.4 (Gaussian Processes). — combined with a "
            "preliminary PLS dimensionality reduction for spectroscopy."
        ),
        "principle": (
            "Spectroscopic data are too high-dimensional for a direct "
            "Gaussian Process: GP inference is $O(n^3)$ in samples but "
            "the *kernel quality* degrades rapidly when "
            "$p$ exceeds a few hundred — most pairwise distances become "
            "near-identical, the kernel matrix loses contrast and the "
            "GP under-fits.\n\n"
            "GPR-PLS solves this by first projecting "
            "$\\mathbf{X} \\to \\mathbf{T} = \\mathbf{X}\\mathbf{W}$ "
            "into a low-dimensional PLS latent space and **then** "
            "training a GP on $\\mathbf{T}$. The latent space "
            "preserves the variance most relevant to $y$, the GP "
            "captures smooth non-linear residual structure, and the "
            "kernel matrix is well-conditioned because pairwise "
            "distances in $\\mathbb{R}^k$ remain informative.\n\n"
            "Default kernel: RBF with length scale $\\ell$ and "
            "amplitude $\\sigma_f^2$, plus an isotropic noise "
            "variance $\\sigma_n^2$. Marginal-likelihood maximisation "
            "selects the three hyperparameters; pls4all uses a "
            "fixed-iteration L-BFGS pass to keep the cost bounded "
            "per cell."
        ),
        "implementation": (
            "`p4a_gpr_pls_fit`. Reference: sklearn "
            "`GaussianProcessRegressor` with an RBF kernel applied to "
            "the score matrix from a separate sklearn `PLSRegression`."
        ),
    },

    # ================================================================
    #   MULTI-BLOCK
    # ================================================================

    "mb_pls": {
        "group": "multi-block",
        "title": "Multi-block PLS (Westerhuis 1998)",
        "paper": (
            "Westerhuis, J. A., Kourti, T. & MacGregor, J. F. (1998). "
            "*Analysis of multiblock and hierarchical PCA and PLS "
            "models*. Journal of Chemometrics 12(5), 301–321."
        ),
        "principle": (
            "When predictors come from several distinct sources — "
            "NIR, MIR, Raman, process tags, lab assays — concatenating "
            "them into one wide matrix lets the block with the most "
            "variance dominate. Multi-block PLS instead **block-scales** "
            "each $\\mathbf{X}_b$ so blocks contribute proportionally "
            "to their information content rather than their "
            "dimensionality.\n\n"
            "Formally, each block is centred and autoscaled, then "
            "scaled by $1 / \\sqrt{p_b}$ so its total variance is "
            "unit-normalised. PLS then runs on the concatenated "
            "$[\\mathbf{X}_1, \\ldots, \\mathbf{X}_B]$ with optional "
            "per-block weights. Block-level *importance* statistics "
            "(block-VIP, block-RMSE) are recovered from the loadings "
            "by restriction to each block's columns.\n\n"
            "Compared to plain concatenation, MB-PLS gives "
            "interpretable per-block contributions and is the "
            "standard approach in process spectroscopy."
        ),
        "implementation": (
            "`p4a_mb_pls_fit` — requires a `block_sizes` integer "
            "vector summing to $p$. The C ABI materialises the "
            "intercept directly (no separate $\\bar{\\mathbf{y}}$ "
            "key) because the block scaling changes the centring "
            "semantics. Reference: sanctioned git-pinned port "
            "`nirs4all.operators.models.sklearn.mbpls`."
        ),
    },

    "mir_pls": {
        "group": "multi-block",
        "title": "MIR-PLS (Mid-InfraRed PLS, regularised)",
        "paper": (
            "Sjöblom, J., Svensson, O., Josefson, M., Kullberg, H. "
            "& Wold, S. (1998). *An evaluation of orthogonal signal "
            "correction applied to calibration transfer of near "
            "infrared spectra*. Chemometrics and Intelligent "
            "Laboratory Systems 44(1–2), 229–244. — adapted for MIR "
            "regularisation conventions."
        ),
        "principle": (
            "MIR-PLS is a PLS variant tuned to mid-infrared "
            "spectroscopy conventions: it operates in absorbance "
            "space, uses a different ridge constant on the inner "
            "regression, and exposes a coefficient-export path that "
            "matches the mid-IR community's expectations (sign and "
            "ordering of loadings agree with `plsregress` rather than "
            "`pls::plsr`).\n\n"
            "Algorithmically it is a SIMPLS variant with a fixed "
            "ridge constant in the inner regression and an alternate "
            "intercept derivation. The differences from plain PLS are "
            "numerical, not algorithmic — the resulting predictions "
            "are within FP noise of plain PLS for well-conditioned "
            "inputs but more stable on the highly-correlated columns "
            "typical of MIR FTIR spectra."
        ),
        "implementation": (
            "`p4a_mir_pls_fit`. No widely installable library "
            "reference; treated as `paper_only` in the registry."
        ),
    },

    "so_pls": {
        "group": "multi-block",
        "title": "Sequential and Orthogonalised PLS (SO-PLS)",
        "paper": (
            "Næs, T., Tomic, O., Mevik, B.-H. & Martens, H. (2011). "
            "*Path modelling by sequential PLS regression*. Journal of "
            "Chemometrics 25(1), 28–40."
        ),
        "principle": (
            "SO-PLS extends MB-PLS by sequentially processing blocks "
            "in a user-specified order, orthogonalising each "
            "subsequent block against the scores extracted from the "
            "previous ones. Concretely: fit PLS on block 1, extract "
            "scores $\\mathbf{T}_1$; orthogonalise block 2 to "
            "$\\mathbf{T}_1$, fit PLS on the residual; and so on.\n\n"
            "This makes each block's contribution **additive and "
            "interpretable** — the unique variance in block $b$ that "
            "is predictive of $y$ given everything in blocks "
            "$1, \\ldots, b-1$. Compared to MB-PLS (which fits all "
            "blocks simultaneously), SO-PLS encodes a domain hypothesis "
            "about which block is causally upstream of which.\n\n"
            "Tuning: a per-block $k_b$ (number of components) plus the "
            "block order. The order matters; permuting blocks changes "
            "the attribution. Cross-validating the per-block $k_b$ is "
            "the standard procedure."
        ),
        "implementation": (
            "`p4a_so_pls_fit` — requires "
            "`n_components_per_block` and `block_sizes`. Reference: "
            "CRAN `multiblock 0.8.10`."
        ),
    },

    "on_pls": {
        "group": "multi-block",
        "title": "OnPLS (Orthogonal N-block PLS)",
        "paper": (
            "Löfstedt, T. & Trygg, J. (2011). *OnPLS — a novel "
            "multiblock method for the modelling of predictive and "
            "orthogonal variation*. Journal of Chemometrics 25(8), "
            "441–455."
        ),
        "principle": (
            "OnPLS generalises OPLS to multiple blocks: it decomposes "
            "the joint structure into a globally predictive component "
            "shared by all blocks plus block-unique orthogonal "
            "components per block. This separates 'integrated' "
            "biology / chemistry information from block-specific "
            "noise.\n\n"
            "The procedure iteratively refines a joint component by "
            "alternating projections and orthogonalisations across "
            "blocks. Compared to SO-PLS — which is asymmetric in "
            "block order — OnPLS is **symmetric**: no causal "
            "directionality is implied between blocks. This is the "
            "right choice when blocks are observation modalities of "
            "the same underlying process (e.g. transcriptomics + "
            "metabolomics + proteomics on the same biological samples).\n\n"
            "The CRAN `OnPLS` package was archived in 2024 so the "
            "Python implementation is vendored at "
            "`bindings/python/vendor/OnPLS/` to remove the dependency."
        ),
        "implementation": (
            "`p4a_on_pls_fit` — requires `n_joint`, "
            "`n_unique_per_block`, `block_sizes`. The CRAN OnPLS "
            "package is archived; pls4all carries an in-tree vendored "
            "port for the parity reference."
        ),
    },

    "rosa": {
        "group": "multi-block",
        "title": "ROSA (Response-Oriented Sequential Alternation)",
        "paper": (
            "Liland, K. H. & Næs, T. (2016). *Response-oriented "
            "sequential alternation: a fast multiblock regression "
            "algorithm*. Journal of Chemometrics 30(11), 651–662."
        ),
        "principle": (
            "ROSA is a forward-greedy multi-block PLS: at each "
            "component extraction step, it tries one new component "
            "from **every block** and keeps the block whose new "
            "component most reduces residual variance in "
            "$\\mathbf{y}$. The component sequence is therefore "
            "data-driven rather than pre-specified by the user.\n\n"
            "This auto-attribution makes ROSA a strong default when "
            "the analyst has no prior on which block matters most: "
            "the block budget is allocated dynamically. ROSA is also "
            "much faster than SO-PLS for large block counts because "
            "it adds one component per iteration rather than refitting "
            "an inner PLS per block.\n\n"
            "Output includes the **block-attribution vector** — the "
            "sequence of block indices selected — which is the main "
            "interpretive artefact: it tells you which block "
            "contributed which component, in order."
        ),
        "implementation": (
            "`p4a_rosa_fit`. Reference: CRAN `multiblock 0.8.10`."
        ),
    },

    "n_pls": {
        "group": "multi-block",
        "title": "N-way PLS (Trilinear PLS, Bro 1996)",
        "paper": (
            "Bro, R. (1996). *Multiway calibration. Multilinear PLS*. "
            "Journal of Chemometrics 10(1), 47–61."
        ),
        "principle": (
            "When the predictor is naturally a tensor "
            "$\\mathcal{X} \\in \\mathbb{R}^{n \\times J \\times K}$ — "
            "e.g. excitation × emission fluorescence spectra, or "
            "wavelength × time chromatographic data — N-way PLS "
            "decomposes $\\mathcal{X}$ into a sum of rank-one tensor "
            "components: "
            "$\\mathcal{X} \\approx \\sum_{a=1}^{k} \\mathbf{t}_a "
            "\\circ \\mathbf{w}_a^{J} \\circ \\mathbf{w}_a^{K}$, "
            "where $\\circ$ is the outer product. The score vectors "
            "$\\mathbf{t}_a$ are computed to maximise covariance with "
            "$\\mathbf{y}$, the loading vectors "
            "$\\mathbf{w}_a^{J}, \\mathbf{w}_a^{K}$ live in the "
            "respective tensor modes.\n\n"
            "Compared to unfolding the tensor and running standard "
            "PLS, N-PLS respects the multilinear structure of the "
            "data and produces interpretable per-mode loadings. The "
            "computational cost is comparable to standard PLS — "
            "matrix-vector products in each mode rather than one "
            "large matrix-vector product.\n\n"
            "Note that pls4all takes the tensor as a **flattened** "
            "matrix plus `mode_j` and `mode_k` shape parameters; the "
            "kernel reshapes internally."
        ),
        "implementation": (
            "`p4a_n_pls_fit`. Reference: Python `tensorly 0.9.0` "
            "(`tensorly.regression.tucker_regression`) and Bro's "
            "original MATLAB code."
        ),
    },

    "o2pls": {
        "group": "multi-block",
        "title": "O2-PLS (two-way orthogonal)",
        "paper": (
            "Trygg, J. & Wold, S. (2003). *O2-PLS, a two-block (X–Y) "
            "latent variable regression method with an integral OSC "
            "filter*. Journal of Chemometrics 17(1), 53–64."
        ),
        "principle": (
            "O2-PLS extends OPLS symmetrically to both $\\mathbf{X}$ "
            "and $\\mathbf{Y}$: it decomposes each block into a joint "
            "predictive component plus block-orthogonal components. "
            "Unlike OPLS, which is asymmetric ($\\mathbf{Y}$ drives "
            "the decomposition of $\\mathbf{X}$), O2-PLS treats both "
            "matrices as observation blocks of equal status.\n\n"
            "Required hyperparameters: $n_{\\mathrm{pred}}$ (joint "
            "components), $n_{\\mathrm{X,ortho}}$ "
            "(X-unique orthogonal), $n_{\\mathrm{Y,ortho}}$ "
            "(Y-unique orthogonal). Choosing all three by "
            "cross-validation is a 3-D grid which can be expensive; "
            "a common compromise fixes the orthogonal counts at 1 and "
            "tunes only $n_{\\mathrm{pred}}$.\n\n"
            "O2-PLS is dominant in metabolomics ↔ transcriptomics "
            "integration where the analyst wants to disentangle "
            "platform-specific orthogonal variation from biology that "
            "is consistent across platforms."
        ),
        "implementation": (
            "`p4a_o2pls_fit`. Reference: CRAN `OmicsPLS 2.1.0`."
        ),
    },

    # ================================================================
    #   CALIBRATION TRANSFER
    # ================================================================

    "di_pls": {
        "group": "calibration-transfer",
        "title": "Domain-Invariant PLS (di-PLS)",
        "paper": (
            "Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S. "
            "& Moser, B. A. (2018). *Domain-invariant partial-least-"
            "squares regression*. Analytical Chemistry 90(11), "
            "6693–6701."
        ),
        "principle": (
            "Calibration transfer methods reconcile spectra acquired "
            "on different instruments or under different "
            "environmental conditions. di-PLS does this by augmenting "
            "the PLS objective with a domain-discrepancy penalty: "
            "$\\mathcal{L}(\\mathbf{w}) = "
            "-\\operatorname{Cov}(\\mathbf{X}_s\\mathbf{w}, \\mathbf{y}_s)^2 + "
            "\\lambda \\,\\mathrm{MMD}^2(\\mathbf{X}_s\\mathbf{w}, \\mathbf{X}_t\\mathbf{w})$, "
            "where $(\\mathbf{X}_s, \\mathbf{y}_s)$ is a labelled "
            "source domain, $\\mathbf{X}_t$ is an unlabelled target "
            "domain and MMD is the maximum mean discrepancy.\n\n"
            "Minimising $\\mathcal{L}$ produces latent directions "
            "$\\mathbf{w}$ that simultaneously **predict $y$ in the "
            "source** and have **matched distributions across "
            "domains**. The model is therefore robust to drift between "
            "calibration and prediction sets without requiring labels "
            "on the target domain.\n\n"
            "Computational cost is dominated by the MMD term, which "
            "is $O((n_s + n_t)^2)$ in a naive implementation; pls4all "
            "uses a linear-kernel MMD which reduces this to "
            "$O((n_s + n_t) p)$.\n\n"
            "$\\lambda$ controls the bias–transferability trade-off: "
            "$\\lambda = 0$ recovers vanilla PLS on the source, large "
            "$\\lambda$ shrinks toward a domain-aligned but "
            "potentially under-predictive model."
        ),
        "implementation": (
            "`p4a_di_pls_fit` — requires `X_target` at fit time. "
            "Reference: Python `diPLSlib.models.DIPLS` "
            "(Nikzad-Langerodi authors). The pls4all variant matches "
            "diPLSlib's `rescale='Target'` source-centred default."
        ),
    },

    "ds": {
        "group": "calibration-transfer",
        "title": "Direct Standardisation",
        "paper": (
            "Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). "
            "*Multivariate instrument standardisation*. Analytical "
            "Chemistry 63(23), 2750–2756."
        ),
        "principle": (
            "Direct Standardisation (DS) learns a **single global "
            "transfer matrix** $\\mathbf{F}$ that maps secondary-"
            "instrument spectra back onto the primary instrument: "
            "$\\hat{\\mathbf{X}}_{\\mathrm{primary}} = \\mathbf{X}_{\\mathrm{secondary}} \\mathbf{F}$. "
            "The matrix is fit by least squares on a small set of "
            "transfer-standard samples measured on both instruments, "
            "typically with a Tikhonov ridge for stability.\n\n"
            "DS is the simplest member of the calibration-transfer "
            "family and works well when the inter-instrument response "
            "is approximately linear and stationary — i.e. when "
            "instrument differences amount to a constant linear "
            "transformation that is invariant across the spectrum. "
            "When the transformation varies across wavelength bands "
            "(common in dispersive vs FT spectrometers) the global "
            "transfer matrix produces band-mixing artefacts and PDS "
            "should be used instead.\n\n"
            "A canonical workflow: fit DS on $\\le 30$ transfer "
            "standards, apply $\\mathbf{F}$ to all subsequent "
            "secondary-instrument spectra, then use a single PLS "
            "model fit only on primary data."
        ),
        "implementation": (
            "`p4a_ds_fit` (TransformerMixin in tier 2). "
            "Reference: R `chemometrics::stdize`."
        ),
    },

    "pds": {
        "group": "calibration-transfer",
        "title": "Piecewise Direct Standardisation",
        "paper": (
            "Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). "
            "*Multivariate instrument standardization*. Analytical "
            "Chemistry 63(23), 2750–2756. "
            "https://doi.org/10.1021/ac00023a016 — same paper as `ds`; "
            "PDS is introduced in §3 (piecewise local regression with "
            "a sliding window of width 2w+1)."
        ),
        "principle": (
            "PDS generalises DS by allowing the transfer to vary "
            "across wavelength bands. For each wavelength $j$ on the "
            "primary instrument, a local regression maps a window of "
            "$\\pm w$ wavelengths from the secondary instrument: "
            "$\\hat{x}_{\\mathrm{primary}, j} = \\mathbf{x}_{\\mathrm{secondary}, j-w:j+w} \\cdot \\mathbf{f}_j$. "
            "The full transfer matrix is then **banded**: only "
            "$\\pm w$ off-diagonal columns per row are non-zero.\n\n"
            "PDS handles wavelength-dependent inter-instrument "
            "behaviour — wavelength-axis drift, resolution differences, "
            "detector non-linearities — that DS cannot. The window "
            "half-width $w$ controls the locality: $w=0$ recovers a "
            "diagonal-only transfer, $w \\to p/2$ recovers DS.\n\n"
            "PDS is the de-facto standard in NIR / FT-IR calibration "
            "transfer; the `prospectr` R package's implementation is "
            "considered canonical."
        ),
        "implementation": (
            "`p4a_pds_fit` (TransformerMixin in tier 2). "
            "Reference: R `prospectr::pds`. Note: pls4all applies the "
            "transpose convention so that `transform(X_secondary)` "
            "returns the standardised primary-instrument estimate."
        ),
    },

    "ecr": {
        "group": "calibration-transfer",
        "title": "ECR — Elastic Component Regression",
        "paper": (
            "Liu, Y., Zhang, B. & Hu, J. (2013). *Elastic Component "
            "Regression*. Chemometrics and Intelligent Laboratory "
            "Systems 124, 73–79. — adapted in pls4all as a "
            "continuum/elastic blend."
        ),
        "principle": (
            "ECR interpolates between PCR and PLS via a single "
            "parameter $\\alpha \\in [0, 1]$ that mixes the two "
            "loading-weight criteria. The latent direction is "
            "$\\mathbf{w} \\propto (1-\\alpha)\\mathbf{X}^{\\top}\\mathbf{X}\\mathbf{w} "
            "+ \\alpha \\mathbf{X}^{\\top}\\mathbf{y}$, "
            "which recovers PCR at $\\alpha = 0$ (the leading "
            "eigenvector of $\\mathbf{X}^{\\top}\\mathbf{X}$) and PLS "
            "at $\\alpha = 1$ (proportional to $\\mathbf{X}^{\\top}\\mathbf{y}$). "
            "Intermediate $\\alpha$ blends variance and covariance "
            "criteria; the optimum is typically located by "
            "cross-validation.\n\n"
            "ECR is closely related to continuum regression with a "
            "different parameterisation, and in practice serves a "
            "similar purpose: when neither PCR nor PLS dominates "
            "RMSE on a given dataset, an interpolating method often "
            "wins by a small margin and offers a smooth tunable "
            "spectrum."
        ),
        "implementation": (
            "`p4a_ecr_fit`. No widely installable reference; treated "
            "as `paper_only` in the registry."
        ),
    },

    # ================================================================
    #   CLASSIFICATION & GLM
    # ================================================================

    "pls_lda": {
        "group": "classification",
        "title": "PLS-LDA",
        "paper": (
            "Barker, M. & Rayens, W. (2003). *Partial least squares "
            "for discrimination*. Journal of Chemometrics 17(3), "
            "166–173."
        ),
        "principle": (
            "PLS-LDA is a two-stage classifier: first project "
            "$\\mathbf{X}$ into the PLS latent space using one-hot "
            "encoded class labels as $\\mathbf{Y}$, then fit Linear "
            "Discriminant Analysis on the resulting scores "
            "$\\mathbf{T} = \\mathbf{X}\\mathbf{W}$.\n\n"
            "LDA in the latent space is well-conditioned (the score "
            "matrix has $k \\ll p$ columns by construction), and the "
            "PLS projection has already aligned the latent axes with "
            "the class separation direction. This is more robust than "
            "applying LDA directly to high-dimensional $\\mathbf{X}$, "
            "where the within-class covariance is singular.\n\n"
            "The decision boundary is **linear in the latent space** "
            "(and therefore also linear in the original feature space "
            "via $\\mathbf{W}$). For non-linear class boundaries use "
            "PLS-QDA or PLS-logistic."
        ),
        "implementation": (
            "`p4a_pls_lda_fit`. The reference is composite "
            "(sklearn `PLSRegression` + sklearn "
            "`LinearDiscriminantAnalysis`); no library exposes a "
            "single PLS-LDA call."
        ),
    },

    "pls_qda": {
        "group": "classification",
        "title": "PLS-QDA",
        "paper": (
            "Pérez-Enciso, M. & Tenenhaus, M. (2003). *Prediction of "
            "clinical outcome with microarray data: a partial least "
            "squares discriminant analysis (PLS-DA) approach*. Human "
            "Genetics 112(5–6), 581–592."
        ),
        "principle": (
            "Replace LDA with QDA in the second stage of PLS-LDA: "
            "instead of assuming a shared covariance across classes, "
            "fit a per-class covariance "
            "$\\boldsymbol{\\Sigma}_c$ on the latent scores. The "
            "resulting decision rule "
            "$\\hat{c}(\\mathbf{x}) = \\arg\\min_c (\\mathbf{t}(\\mathbf{x}) - \\boldsymbol{\\mu}_c)^{\\top} "
            "\\boldsymbol{\\Sigma}_c^{-1} (\\mathbf{t}(\\mathbf{x}) - \\boldsymbol{\\mu}_c) + "
            "\\log|\\boldsymbol{\\Sigma}_c|$ "
            "is **quadratic** in the latent scores.\n\n"
            "QDA needs at least $k + 1$ samples per class to estimate "
            "$\\boldsymbol{\\Sigma}_c$ stably, but otherwise gives "
            "more flexible decision boundaries than LDA. Worth trying "
            "whenever the LDA boundary visibly under-fits in a "
            "2-D latent score plot.\n\n"
            "Class probabilities follow from the Mahalanobis "
            "distance via the Bayes rule with uniform priors (or "
            "user-supplied priors)."
        ),
        "implementation": (
            "`p4a_pls_qda_fit`. Reference: composite PLSRegression + "
            "sklearn `QuadraticDiscriminantAnalysis` on the scores."
        ),
    },

    "pls_logistic": {
        "group": "classification",
        "title": "PLS-logistic regression",
        "paper": (
            "Bastien, P., Esposito Vinzi, V. & Tenenhaus, M. (2005). "
            "*PLS generalised linear regression*. Computational "
            "Statistics & Data Analysis 48(1), 17–46."
        ),
        "principle": (
            "Iteratively-reweighted-least-squares PLS with a "
            "logit link function. At each iteration the current "
            "predictor is converted to a working response via "
            "$z_i = \\eta_i + (y_i - p_i) / (p_i(1 - p_i))$ where "
            "$p_i = 1/(1 + e^{-\\eta_i})$, a PLS fit is run on "
            "$(\\mathbf{X}, \\mathbf{z})$ with weights "
            "$p_i(1 - p_i)$, and the linear predictor is updated.\n\n"
            "This is the natural extension of PLS to binary / "
            "multinomial classification when class probabilities "
            "(rather than hard labels or class scores) are needed, "
            "and it generalises smoothly to GLM families beyond "
            "Bernoulli (Poisson — see `pls_glm`). The multinomial "
            "case extends to $K$ classes via a softmax link.\n\n"
            "Convergence is typically reached in 5–10 IRLS "
            "iterations. The Bastien et al. variant is closely "
            "related to Marx 1996's *Iteratively Reweighted PLS* but "
            "differs in the deflation convention."
        ),
        "implementation": (
            "`p4a_pls_logistic_fit` (in-sample only). Reference: "
            "R `plsRglm 1.7.0`."
        ),
    },

    "pls_glm": {
        "group": "classification",
        "title": "PLS-GLM (Generalised Linear Model PLS)",
        "paper": (
            "Marx, B. D. (1996). *Iteratively reweighted partial "
            "least squares estimation for generalized linear "
            "regression*. Technometrics 38(4), 374–381."
        ),
        "principle": (
            "PLS-GLM generalises PLS-logistic to any GLM family. The "
            "IRLS recipe is identical — derive a working response "
            "from the current linear predictor, fit PLS with the "
            "GLM weights, iterate — but the link function varies: "
            "identity for Gaussian, log for Poisson, logit for "
            "Bernoulli/binomial.\n\n"
            "pls4all currently supports Gaussian and Poisson families "
            "(controlled by the `poisson` flag). The Poisson case is "
            "useful for count regression on spectroscopy data where "
            "the response is an integer abundance (cell counts, "
            "particle counts) rather than a continuous concentration.\n\n"
            "Compared to running a vanilla PLS on $\\log(y+1)$, the "
            "true Poisson formulation correctly handles the "
            "mean–variance relationship and is less biased for low "
            "counts."
        ),
        "implementation": (
            "`p4a_pls_glm_fit`. Reference: R `plsRglm 1.7.0`."
        ),
    },

    "pls_cox": {
        "group": "classification",
        "title": "PLS-Cox (survival regression)",
        "paper": (
            "Bastien, P., Bertrand, F., Meyer, N. & "
            "Maumy-Bertrand, M. (2015). *Deviance residuals-based "
            "sparse PLS and sparse kernel PLS regression for "
            "censored data*. Bioinformatics 31(3), 397–404."
        ),
        "principle": (
            "Cox proportional-hazards regression with PLS-based "
            "dimensionality reduction. The Cox model "
            "$\\lambda(t \\mid \\mathbf{x}) = \\lambda_0(t)\\exp(\\mathbf{x}^{\\top}\\boldsymbol{\\beta})$ "
            "is degenerate in $p \\gg n$ because the partial "
            "likelihood loses identifiability. PLS-Cox replaces the "
            "$p$-dimensional $\\boldsymbol{\\beta}$ with a "
            "$k$-dimensional latent representation by extracting "
            "PLS scores from the **deviance residuals** of a null "
            "Cox model.\n\n"
            "Required inputs are survival times and event indicators "
            "(0 = censored, 1 = event observed). The output is a "
            "fitted Cox model on the latent scores; risk scores for "
            "new samples are computed by first projecting them into "
            "the latent space and then evaluating "
            "$\\mathbf{t}^{\\top}\\boldsymbol{\\beta}$.\n\n"
            "This is the canonical method in high-dimensional "
            "biomarker survival studies (RNA-seq, MALDI-TOF) where a "
            "direct Cox model is infeasible."
        ),
        "implementation": (
            "`p4a_pls_cox_fit`. Reference: R `plsRcox 1.8.2`."
        ),
    },

    # ================================================================
    #   MISSING DATA
    # ================================================================

    "missing_aware_nipals": {
        "group": "missing",
        "title": "Missing-aware NIPALS",
        "paper": (
            "Walczak, B. & Massart, D. L. (2001). *Dealing with "
            "missing data: part I & II*. Chemometrics and Intelligent "
            "Laboratory Systems 58(1), 15–27 & 29–42. — applied to "
            "the NIPALS PLS algorithm."
        ),
        "principle": (
            "Standard PLS cannot ingest `NaN` values: the matrix "
            "operations propagate NaN. Missing-aware NIPALS treats "
            "missing entries as **excluded from the inner products** "
            "in the iterative loading-weight calculation. Concretely, "
            "the score "
            "$t_i = \\sum_{j \\in \\mathcal{O}_i} x_{ij} w_j$ "
            "and the loading "
            "$p_j = \\sum_{i \\in \\mathcal{O}_j} x_{ij} t_i / \\sum_{i \\in \\mathcal{O}_j} t_i^2$ "
            "are computed only over the observed indices "
            "$\\mathcal{O}_{i}, \\mathcal{O}_{j}$.\n\n"
            "Convergence is slower than vanilla NIPALS and the result "
            "is sensitive to the missingness pattern, but the "
            "alternative (delete rows with any NaN or impute by "
            "column means) usually performs much worse on "
            "spectroscopic data. Acceptable up to ~10 % missing entries; "
            "beyond that low-rank matrix completion (softImpute) is a "
            "better front-end.\n\n"
            "Note that missing-aware NIPALS produces a regression "
            "model that can still predict on **complete** new "
            "$\\mathbf{x}$; only the training step tolerates missing "
            "data."
        ),
        "implementation": (
            "`p4a_missing_aware_nipals_fit`. Reference: R "
            "`softImpute 1.4.3` for the imputation step; the PLS "
            "fitting itself is in-tree."
        ),
    },

    # ================================================================
    #   DIAGNOSTICS & MONITORING
    # ================================================================

    "pls_diagnostic_t2": {
        "group": "diagnostic",
        "title": "Hotelling T² score",
        "paper": (
            "Hotelling, H. (1931). *The generalization of Student's "
            "ratio*. Annals of Mathematical Statistics 2(3), 360–378. "
            "— applied to PLS scores by MacGregor & Kourti 1995."
        ),
        "principle": (
            "Hotelling T² measures how unusual a sample is **within "
            "the latent score space**: "
            "$T_i^2 = \\mathbf{t}_i^{\\top} \\boldsymbol{\\Lambda}^{-1} \\mathbf{t}_i$ "
            "where $\\boldsymbol{\\Lambda}$ is the diagonal matrix of "
            "score variances. Under multivariate normality of the "
            "scores, "
            "$\\frac{n(n-k)}{k(n^2-1)} T^2 \\sim F_{k, n-k}$, "
            "giving an exact upper control limit at any "
            "$\\alpha$.\n\n"
            "T² complements the Q residual (next entry): Q measures "
            "the **distance to the model** (variation in "
            "$\\mathbf{X}$ that the latent space does not explain), "
            "while T² measures the **distance within the model** "
            "(unusual score combination on otherwise well-explained "
            "samples). Joint T²/Q monitoring catches both kinds of "
            "out-of-control points.\n\n"
            "Reported per-sample as a 1-D vector aligned with the "
            "rows of the input."
        ),
        "implementation": (
            "`p4a_pls_diagnostics_compute` with stat='t2'. "
            "Reference: R `mdatools 0.15.0` (Kucheryavskiy)."
        ),
    },

    "pls_diagnostic_q": {
        "group": "diagnostic",
        "title": "Q residual (squared prediction error)",
        "paper": (
            "Jackson, J. E. & Mudholkar, G. S. (1979). *Control "
            "procedures for residuals associated with principal "
            "component analysis*. Technometrics 21(3), 341–349."
        ),
        "principle": (
            "Q (also called SPE — Squared Prediction Error) is the "
            "sum of squared residuals between $\\mathbf{x}$ and its "
            "PLS reconstruction "
            "$\\hat{\\mathbf{x}} = \\mathbf{T}\\mathbf{P}^{\\top}$: "
            "$Q_i = \\|\\mathbf{x}_i - \\mathbf{t}_i \\mathbf{P}^{\\top}\\|_2^2 = "
            "\\sum_j (x_{ij} - \\hat{x}_{ij})^2$. "
            "It measures the part of $\\mathbf{x}$ that lies "
            "**orthogonal** to the latent space — variation in the "
            "predictor that the model could not capture.\n\n"
            "Under the assumption of Gaussian residuals, Jackson & "
            "Mudholkar (1979) derived a parametric upper control "
            "limit. High Q with low T² typically signals a sample "
            "with a fundamentally different spectral fingerprint "
            "from the calibration set (e.g. contamination, "
            "instrument failure); low Q with high T² signals an "
            "extreme combination of otherwise normal features.\n\n"
            "Reported per-sample as a 1-D vector aligned with the "
            "rows of the input."
        ),
        "implementation": (
            "`p4a_pls_diagnostics_compute` with stat='q'. "
            "Reference: R `mdatools 0.15.0`."
        ),
    },

    "pls_diagnostic_dmodx": {
        "group": "diagnostic",
        "title": "DModX (distance to the model in X)",
        "paper": (
            "Eriksson, L., Byrne, T., Johansson, E., Trygg, J. & "
            "Vikström, C. (2013). *Multi- and Megavariate Data "
            "Analysis. Basic Principles and Applications*, 3rd ed., "
            "Umetrics Academy, §4.7."
        ),
        "principle": (
            "DModX is a **normalised** version of Q that accounts for "
            "the model's residual degrees of freedom: "
            "$\\mathrm{DModX}_i = \\sqrt{Q_i \\,/\\, (p - k - 1) \\,\\cdot\\, "
            "(n - k - 1) / \\bar{Q}_{\\mathrm{cal}}}$ "
            "where $\\bar{Q}_{\\mathrm{cal}}$ is the average Q on the "
            "calibration set.\n\n"
            "DModX is the diagnostic of choice in the SIMCA software "
            "(Umetrics) and is commonly used because the "
            "normalisation produces a unit-less quantity directly "
            "comparable across models with different $p$, $n$, $k$. "
            "Critical thresholds follow the F-distribution: DModX > 2 "
            "is the heuristic outlier cutoff in practice.\n\n"
            "Conceptually equivalent to Q for monitoring purposes, "
            "but the normalisation is what makes it portable."
        ),
        "implementation": (
            "`p4a_pls_diagnostics_compute` with stat='dmodx'."
        ),
    },

    "pls_monitoring": {
        "group": "diagnostic",
        "title": "PLS monitoring (T² + Q with control limits)",
        "paper": (
            "Kourti, T. & MacGregor, J. F. (1996). *Multivariate "
            "SPC methods for process and product monitoring and "
            "control*. Journal of Quality Technology 28(4), "
            "409–428."
        ),
        "principle": (
            "Combine T² and Q with parametric control limits to "
            "obtain a 2-D monitoring chart for online process control. "
            "Samples are classified as in-control if both statistics "
            "fall below their respective limits; otherwise an alarm "
            "is raised. The two statistics are statistically nearly "
            "independent (T² lives in the latent space, Q in its "
            "orthogonal complement), so joint alarms reflect "
            "compound failures.\n\n"
            "pls4all's monitoring routine returns, for each sample, "
            "the T² and Q values, their control-limit ratios, and a "
            "boolean alarm flag. Limits are derived from the "
            "calibration distribution: F-quantile for T², "
            "Jackson–Mudholkar normal approximation for Q.\n\n"
            "Used as the back-end of a process SPC dashboard or as a "
            "test-set sanity check before deploying a PLS model in "
            "production."
        ),
        "implementation": (
            "`p4a_pls_monitoring_run` — returns a dict with alarm "
            "vectors."
        ),
    },

    "approximate_press": {
        "group": "diagnostic",
        "title": "Approximate PRESS (leave-one-out by hat-matrix)",
        "paper": (
            "Allen, D. M. (1974). *The relationship between variable "
            "selection and data augmentation and a method for "
            "prediction*. Technometrics 16(1), 125–127."
        ),
        "principle": (
            "Exact leave-one-out cross-validation requires $n$ refits "
            "and costs $O(n^2 p k)$. The approximate PRESS uses the "
            "hat-matrix shortcut: the leave-one-out residual for "
            "sample $i$ is approximately "
            "$r_i^{(-i)} \\approx r_i / (1 - h_{ii})$, "
            "where $h_{ii}$ is the diagonal of the hat matrix "
            "$\\mathbf{H} = \\mathbf{T}(\\mathbf{T}^{\\top}\\mathbf{T})^{-1}\\mathbf{T}^{\\top}$. "
            "Total PRESS is $\\sum_i (r_i^{(-i)})^2$, computed in "
            "$O(n p k)$ from a single fit.\n\n"
            "The approximation is exact for OLS and tight for PLS as "
            "long as no single $h_{ii}$ approaches 1 (a high-leverage "
            "outlier). Use exact LOO when the approximate PRESS "
            "diverges from the cross-validated RMSEP by more than "
            "a few percent.\n\n"
            "Drives the one-SE rule for selecting the component "
            "count $k$."
        ),
        "implementation": (
            "`p4a_approximate_press_compute`. Returns a length-"
            "$(k_{\\max}+1)$ vector indexed by component count."
        ),
    },

    "one_se_rule": {
        "group": "diagnostic",
        "title": "One-SE rule for component selection",
        "paper": (
            "Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The "
            "Elements of Statistical Learning*, 2nd ed., Springer, "
            "§7.10."
        ),
        "principle": (
            "Cross-validated RMSE as a function of $k$ is typically "
            "U-shaped with a relatively flat minimum. Picking the "
            "absolute minimum $k^{\\star}$ can over-fit because it "
            "exploits sampling noise. The one-SE rule instead picks "
            "the **smallest** $k$ whose CV-RMSE is within one "
            "standard error of $\\mathrm{RMSE}(k^{\\star})$.\n\n"
            "This yields a more parsimonious model with negligible "
            "predictive cost — the smaller-$k$ alternative is "
            "indistinguishable from the optimum within the noise of "
            "the CV estimate. The rule is non-parametric (no "
            "assumption about the CV-RMSE distribution) and is the "
            "standard practice in regularised regression "
            "(`glmnet`, `pls::pls`).\n\n"
            "Inputs: a fold × component RMSE matrix from "
            "cross-validation. Output: an integer component count."
        ),
        "implementation": (
            "`p4a_one_se_rule_compute`. Returns an integer."
        ),
    },

    "aom_preprocess": {
        "group": "diagnostic",
        "title": "AOM (Adaptive Operator Mixture) preprocessing bank",
        "paper": (
            "Beurier, G., Reiter, R., Noûs, C., Rouan, L. & Cornet, D. "
            "(2026). *Reframing preprocessing selection as model-"
            "internal calibration in near-infrared spectroscopy: a "
            "large-scale benchmark of operator-adaptive PLS and Ridge "
            "models*. arXiv:2605.13587. "
            "https://arxiv.org/abs/2605.13587 — introduces operator-"
            "adaptive PLS (AOM-PLS / POP-PLS) and the bench against 50+ "
            "NIRS datasets that the git-pinned oracle "
            "`nirs4all.operators.models.sklearn.aom_pls` is calibrated "
            "against."
        ),
        "principle": (
            "AOM is a **soft preprocessing ensemble**: instead of "
            "committing to a single spectroscopic preprocessing "
            "(SNV, MSC, Savitzky–Golay derivative, …), AOM operates "
            "a bank of $M$ candidate operators in parallel and "
            "combines their outputs via gating. With equal weights "
            "(soft gating) the output is the average preprocessed "
            "spectrum; with first-operator gating (hard) it is a "
            "deterministic choice driven by a downstream criterion "
            "(typically SIMPLS CV-RMSE).\n\n"
            "The motivation is that **no single preprocessing is best "
            "on all calibrations** — even within a single project, "
            "different wavelength regions favour different "
            "transforms. AOM lets the downstream PLS adapt rather "
            "than forcing the analyst to pre-commit, at the cost of "
            "$M$× the preprocessing work.\n\n"
            "AOM-PLS family methods (AOM-SIMPLS, POP-PLS) use this "
            "primitive as their building block. Phase 6a–6e of the "
            "pls4all roadmap implement the bench-parity strict-linear "
            "tranche covering identity, detrending, "
            "Savitzky–Golay, finite-difference, Norris–Williams, "
            "Whittaker smoothing and FCK operators."
        ),
        "implementation": (
            "`p4a_aom_preprocess_fit`. Reference: git-pinned oracle "
            "`nirs4all.operators.models.sklearn.aom_pls` "
            "(sanctioned exception)."
        ),
    },

    # ================================================================
    #   VARIABLE SELECTORS (28 methods)
    # ================================================================

    "variable_select_vip": {
        "group": "selector",
        "title": "VIP (Variable Importance in Projection)",
        "paper": (
            "Wold, S., Sjöström, M. & Eriksson, L. (2001). *PLS-"
            "regression: a basic tool of chemometrics*. Chemometrics "
            "and Intelligent Laboratory Systems 58(2), 109–130."
        ),
        "principle": (
            "VIP scores quantify each feature's contribution across "
            "all $k$ latent components of a PLS model, weighted by "
            "how much each component explains of $\\mathbf{y}$: "
            "$\\mathrm{VIP}_j = \\sqrt{\\frac{p}{\\mathrm{SSY}} \\sum_{a=1}^{k} w_{ja}^2 \\, \\mathrm{SSY}_a}$, "
            "where $w_{ja}$ is the loading weight of feature $j$ in "
            "component $a$ and $\\mathrm{SSY}_a$ is the explained "
            "sum of squares of $\\mathbf{y}$ in component $a$.\n\n"
            "The normalisation guarantees "
            "$\\sum_j \\mathrm{VIP}_j^2 = p$, so the heuristic "
            "$\\mathrm{VIP}_j > 1$ identifies features contributing "
            "more than their fair share. VIP is the workhorse of "
            "spectroscopic variable selection — simple, deterministic, "
            "fast, and well understood."
        ),
        "implementation": (
            "`p4a_variable_select_rank` with metric=VIP. "
            "Reference: R `plsVarSel 0.10.0`."
        ),
    },

    "variable_select_coef": {
        "group": "selector",
        "title": "Coefficient-magnitude selection",
        "paper": (
            "Martens, H. & Næs, T. (1989). *Multivariate "
            "Calibration*, §5. — the simplest ranking baseline."
        ),
        "principle": (
            "Rank features by the absolute magnitude of their PLS "
            "regression coefficient $|b_j|$ in the original feature "
            "scale. Pick the top-$k$ as the selected subset.\n\n"
            "This is the simplest possible PLS variable selector. It "
            "is biased — features with large variance get smaller "
            "coefficients for the same predictive effect — so it "
            "should usually be applied after autoscaling to remove "
            "the variance-induced bias. Once autoscaled, $|b_j|$ "
            "ranks features by their **standardised partial effect "
            "on $y$**, which is statistically meaningful.\n\n"
            "Useful as a sanity-check baseline against more "
            "sophisticated selectors. If a complex method does not "
            "beat coefficient-magnitude selection, it is probably "
            "over-engineered."
        ),
        "implementation": (
            "`p4a_variable_select_rank` with metric=COEF."
        ),
    },

    "variable_select_sr": {
        "group": "selector",
        "title": "Selectivity Ratio",
        "paper": (
            "Rajalahti, T., Arneberg, R., Berven, F. S., Myhr, K.-M., "
            "Ulvik, R. J. & Kvalheim, O. M. (2009). *Biomarker "
            "discovery in mass spectral profiles by means of "
            "selectivity ratio plot*. Chemometrics and Intelligent "
            "Laboratory Systems 95(1), 35–48."
        ),
        "principle": (
            "Selectivity Ratio (SR) measures the relative explained-"
            "to-residual variance of each feature along the **target-"
            "projected** PLS direction: "
            "$\\mathrm{SR}_j = \\mathrm{Var}(\\hat{x}_j) / \\mathrm{Var}(x_j - \\hat{x}_j)$, "
            "where $\\hat{x}_j$ is the projection of feature $j$ "
            "onto the target-projected loading vector "
            "$\\mathbf{p}_{\\mathrm{tp}}$ (a single direction in "
            "$\\mathbf{X}$ space that captures all $\\mathbf{Y}$-"
            "correlated variation).\n\n"
            "High SR means a feature's variance is dominated by its "
            "$y$-correlated part; low SR means the feature's variance "
            "is mostly orthogonal to $y$ (noise / interferent / "
            "matrix). SR therefore separates predictive features from "
            "structurally-correlated nuisance features.\n\n"
            "Unlike VIP, SR works with a single direction (the "
            "target projection), which means it scales gracefully to "
            "very many components and is interpretable as a "
            "univariate diagnostic per feature."
        ),
        "implementation": (
            "`p4a_variable_select_rank` with metric=SR."
        ),
    },

    "spa_select": {
        "group": "selector",
        "title": "SPA — Successive Projections Algorithm",
        "paper": (
            "Araújo, M. C. U., Saldanha, T. C. B., Galvão, R. K. H., "
            "Yoneyama, T., Chame, H. C. & Visani, V. (2001). *The "
            "successive projections algorithm for variable selection "
            "in spectroscopic multicomponent analysis*. Chemometrics "
            "and Intelligent Laboratory Systems 57(2), 65–73."
        ),
        "principle": (
            "SPA is a greedy forward selector that seeks **minimally "
            "collinear** features. Starting from the feature with "
            "largest coefficient (or user-supplied seed), at each "
            "step it adds the feature whose direction is **maximally "
            "orthogonal** to all previously-selected features. "
            "Formally, at step $m$ it picks "
            "$j^{\\star} = \\arg\\max_j \\| \\mathbf{P}_{S_{m-1}}^{\\perp} \\mathbf{x}_j \\|_2$, "
            "where $\\mathbf{P}_{S}^{\\perp}$ projects onto the "
            "orthogonal complement of the span of the already-"
            "selected features.\n\n"
            "This produces a feature subset that is well-conditioned "
            "for downstream regression: the inverse "
            "$(\\mathbf{X}_S^{\\top}\\mathbf{X}_S)^{-1}$ has small "
            "condition number by construction. SPA is therefore "
            "particularly effective when followed by MLR (or PLS "
            "with very few components) on the selected subset.\n\n"
            "Stops at a user-specified top-$k$. Computational cost: "
            "$O(k\\, n\\, p)$."
        ),
        "implementation": "`p4a_spa_select`. Reference: R `plsVarSel`.",
    },

    "stability_select": {
        "group": "selector",
        "title": "MC-UVE (Monte-Carlo coefficient stability)",
        "paper": (
            "Cai, W., Li, Y. & Shao, X. (2008). *A variable selection "
            "method based on uninformative variable elimination for "
            "multivariate calibration of near-infrared spectra*. "
            "Chemometrics and Intelligent Laboratory Systems 90(2), "
            "188–194."
        ),
        "principle": (
            "MC-UVE evaluates the **stability** of each feature's "
            "PLS coefficient across Monte-Carlo subsamples of the "
            "calibration set: "
            "$\\mathrm{stab}_j = |\\bar{b}_j| / s(b_j)$, where "
            "$\\bar{b}_j$ and $s(b_j)$ are the mean and standard "
            "deviation of $b_j$ across $B$ bootstrap fits. Features "
            "with high stability ratio are reliably predictive; those "
            "with low ratio are noise-driven and discarded.\n\n"
            "Conceptually a univariate analogue of stability "
            "selection (Meinshausen & Bühlmann 2010). The interaction "
            "with collinearity in the spectrum is benign: collinear "
            "features tend to share the contribution across "
            "bootstraps in a stable way, so their joint stability is "
            "high.\n\n"
            "Typical Monte-Carlo budget: $B = 50$–$200$ subsamples, "
            "each at 80 % of the calibration size."
        ),
        "implementation": (
            "`p4a_stability_select`. Reference: R `plsVarSel`."
        ),
    },

    "uve_select": {
        "group": "selector",
        "title": "UVE — Uninformative Variable Elimination",
        "paper": (
            "Centner, V., Massart, D. L., de Noord, O. E., de Jong, "
            "S., Vandeginste, B. M. & Sterna, C. (1996). *Elimination "
            "of uninformative variables for multivariate calibration*. "
            "Analytical Chemistry 68(21), 3851–3858."
        ),
        "principle": (
            "UVE introduces a clever threshold: augment $\\mathbf{X}$ "
            "with $p$ columns of **deterministic artificial noise** "
            "(unit-variance Gaussian, fixed seed), fit a PLS on the "
            "augmented "
            "$[\\mathbf{X}, \\mathbf{X}_{\\mathrm{noise}}]$, then "
            "compute MC-UVE stability ratios for *all* columns. The "
            "noise columns establish a baseline distribution of "
            "stabilities under the null hypothesis 'this column "
            "contributes nothing'. Any real feature whose stability "
            "falls below the maximum stability of the noise columns "
            "is eliminated.\n\n"
            "The threshold is therefore data-adaptive — it tightens "
            "automatically as $n$ grows (noise stabilities concentrate) "
            "and relaxes when the SNR is low. UVE is the canonical "
            "starting point for any chemometrics paper proposing a "
            "new selector; everything since is benchmarked against it."
        ),
        "implementation": (
            "`p4a_uve_select`. Reference: R `plsVarSel`."
        ),
    },

    "cars_select": {
        "group": "selector",
        "title": "CARS — Competitive Adaptive Reweighted Sampling",
        "paper": (
            "Li, H., Liang, Y., Xu, Q. & Cao, D. (2009). *Key "
            "wavelengths screening using competitive adaptive "
            "reweighted sampling method for multivariate calibration*. "
            "Analytica Chimica Acta 648(1), 77–84."
        ),
        "principle": (
            "CARS is one of the most widely-used spectroscopic "
            "variable selectors. It runs $M$ iterations of: "
            "(1) draw a Monte-Carlo subsample, "
            "(2) fit PLS, "
            "(3) compute coefficient weights "
            "$w_j = |b_j| / \\sum |b_j|$, "
            "(4) keep a shrinking fraction of features ranked by "
            "weighted competitive sampling — features compete "
            "stochastically with probability proportional to $w_j$.\n\n"
            "The retention fraction shrinks **exponentially**: "
            "$r_m = \\exp(-\\mu(m - 1))$ with $\\mu$ chosen so that "
            "two features survive at the final iteration. The "
            "iteration whose surviving subset minimises CV-RMSE is "
            "returned.\n\n"
            "CARS combines deterministic exponential decay with "
            "stochastic competition; the latter prevents premature "
            "elimination of correlated features. Practically very "
            "robust to noise."
        ),
        "implementation": (
            "`p4a_cars_select`. Reference: R `enpls 6.1.1` ("
            "`enpls.fs(method='mc')` is the closest analogue)."
        ),
    },

    "random_frog_select": {
        "group": "selector",
        "title": "Random Frog",
        "paper": (
            "Li, H., Xu, Q. & Liang, Y. (2012). *Random frog: an "
            "efficient reversible jump Markov chain Monte Carlo-like "
            "approach for variable selection*. Analytica Chimica Acta "
            "740, 20–26."
        ),
        "principle": (
            "Random Frog runs a random walk over feature subsets: at "
            "each step it proposes a transition to a neighbouring "
            "subset (add / remove / swap a feature) and accepts the "
            "transition with a Metropolis-style probability based on "
            "the improvement in CV-RMSE. Features that appear "
            "frequently in the visited subsets are deemed important.\n\n"
            "Output: the **inclusion frequency** vector — fraction "
            "of iterations in which each feature was selected. Sort "
            "by frequency and take the top-$k$ for the final subset.\n\n"
            "Random Frog is sample-efficient compared to GA-PLS (no "
            "population of full subsets to maintain) but slower to "
            "mix on very high-dimensional data. Recommended on "
            "spectra of moderate size (a few hundred wavelengths)."
        ),
        "implementation": "`p4a_random_frog_select`.",
    },

    "scars_select": {
        "group": "selector",
        "title": "SCARS — Stability-CARS",
        "paper": (
            "Zheng, K., Li, Q., Wang, J., Geng, J., Cao, P., Sui, T., "
            "Wang, X. & Du, Y. (2012). *Stability competitive adaptive "
            "reweighted sampling (SCARS) and its applications to "
            "multivariate calibration of NIR spectra*. Chemometrics "
            "and Intelligent Laboratory Systems 112, 48–54."
        ),
        "principle": (
            "Replace CARS's coefficient-magnitude weights with **"
            "coefficient-stability** weights: "
            "$w_j = |\\bar{b}_j| / s(b_j)$ from the bootstrap "
            "distribution. Stability-weighted retention is more "
            "robust to spurious high-magnitude coefficients caused "
            "by particular bootstrap subsamples.\n\n"
            "Otherwise identical to CARS: exponential decay schedule "
            "and stochastic competition. SCARS typically improves "
            "CARS on datasets with strong baseline drift or where "
            "a few high-leverage samples dominate the coefficient "
            "estimates."
        ),
        "implementation": "`p4a_scars_select`.",
    },

    "ga_select": {
        "group": "selector",
        "title": "GA-PLS — Genetic Algorithm variable selection",
        "paper": (
            "Leardi, R. (2000). *Application of genetic algorithm–"
            "PLS for feature selection in spectral data sets*. "
            "Journal of Chemometrics 14(5–6), 643–655."
        ),
        "principle": (
            "Wrap a binary genetic algorithm around PLS CV-RMSE. "
            "Each chromosome is a $p$-bit binary mask encoding which "
            "features to include; fitness is "
            "$-\\mathrm{CV\\text{-}RMSE}$ from PLS on the masked "
            "subset; standard GA operators (single-point crossover, "
            "bit-flip mutation, elitism) drive the population.\n\n"
            "Cost is high — every fitness evaluation is a full PLS "
            "fit — but GA-PLS handles non-convex fitness landscapes "
            "(non-additive interactions between selected features) "
            "that greedy methods miss. Recommended population sizes: "
            "30–100; generations: 100–500.\n\n"
            "Stochastic by construction: results vary across RNG "
            "seeds. For deterministic comparisons against this "
            "selector the benchmark widens the parity tolerance and "
            "fixes the seed; in production use a small ensemble of "
            "GA runs and take the consensus."
        ),
        "implementation": (
            "`p4a_ga_select`. Reference: R `plsVarSel`."
        ),
    },

    "pso_select": {
        "group": "selector",
        "title": "PSO-PLS — Particle Swarm Optimisation",
        "paper": (
            "Kennedy, J. & Eberhart, R. (1995). *Particle swarm "
            "optimization*. IEEE ICNN 1995, vol. 4, 1942–1948. — "
            "binary PSO variant used for variable selection."
        ),
        "principle": (
            "Binary PSO maintains a swarm of particles where each "
            "particle's position is a $p$-bit feature mask. Velocity "
            "updates blend the particle's personal best, the swarm's "
            "global best, and an inertia term; positions are "
            "stochastically rounded to 0/1 via a sigmoid.\n\n"
            "Compared to GA-PLS, PSO converges faster on smooth "
            "fitness landscapes but is more susceptible to premature "
            "convergence on multi-modal ones. The two methods are "
            "complementary: PSO for quick reconnaissance, GA for "
            "final polish.\n\n"
            "Recommended swarm size: 20–50. The fitness is again "
            "PLS CV-RMSE on the masked subset."
        ),
        "implementation": (
            "`p4a_pso_select`. Reference: Python `pyswarms` for the "
            "PSO core, wrapped against PLS CV-RMSE."
        ),
    },

    "vissa_select": {
        "group": "selector",
        "title": "VISSA — Variable Iterative Space-Shrinkage",
        "paper": (
            "Deng, B. C., Yun, Y. H., Liang, Y. Z. & Yi, L. Z. (2014). "
            "*A new strategy to prevent over-fitting in partial least "
            "squares models based on model population analysis*. "
            "Analytica Chimica Acta 880, 32–41."
        ),
        "principle": (
            "VISSA evaluates a **population of random subsets** of "
            "the same size, refines the population by selecting the "
            "best by CV-RMSE, and iteratively shrinks the search "
            "space toward features that survive in many high-"
            "performing subsets. Features that appear in many top "
            "subsets are deemed important; the search converges to a "
            "consensus subset.\n\n"
            "Different from CARS in that the search space is "
            "shrunken **by consensus over a population** rather than "
            "by exponential decay over iterations. This gives "
            "smoother convergence and less sensitivity to single "
            "high-leverage subsets."
        ),
        "implementation": "`p4a_vissa_select`.",
    },

    "shaving_select": {
        "group": "selector",
        "title": "Shaving (recursive elimination)",
        "paper": (
            "Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. "
            "(2012). *A review of variable selection methods in "
            "partial least squares regression*. Chemometrics and "
            "Intelligent Laboratory Systems 118, 62–69 (§3.2 "
            "Shaving)."
        ),
        "principle": (
            "Shaving recursively eliminates a fraction "
            "$\\rho \\in (0, 1)$ of the lowest-scoring features at "
            "each step, refits PLS, and tracks the CV-RMSE of the "
            "shrinking subset. The subset with the lowest recorded "
            "CV-RMSE across the whole shaving trajectory is "
            "returned.\n\n"
            "Compared to backward variable elimination (BVE — see "
            "next), shaving removes a **batch** of features per "
            "step instead of one, which is faster ($O(\\log p)$ steps "
            "vs $O(p)$) but more aggressive — a single bad shave "
            "removes many useful features irrecoverably. Recommended "
            "$\\rho \\le 0.2$ to keep shave granularity reasonable."
        ),
        "implementation": "`p4a_shaving_select`.",
    },

    "bve_select": {
        "group": "selector",
        "title": "BVE — Backward Variable Elimination",
        "paper": (
            "Forina, M., Casolino, M. C. & Pizarro Millán, C. (2004). "
            "*Iterative predictor weighting (IPW) PLS: a technique "
            "for the elimination of useless predictors in regression "
            "problems*. Journal of Chemometrics 18(2), 105–112 (§2)."
        ),
        "principle": (
            "Greedy backward elimination: at each step, evaluate "
            "every possible **one-variable removal** by CV-RMSE and "
            "drop the feature whose removal hurts least (or helps "
            "most). Continue until either a target subset size is "
            "reached or removal starts to hurt CV-RMSE materially.\n\n"
            "Cost: $O(p^2 \\cdot \\mathrm{CV})$ — quadratic in the "
            "number of features, since each step evaluates $\\sim p$ "
            "candidate removals. For $p \\le 200$ this is tractable; "
            "for larger spectra the shaving variant is preferred.\n\n"
            "Strength: BVE is essentially exhaustive at each step, so "
            "it cannot be tricked by collinearity the way SPA can. "
            "Weakness: very expensive on full NIR spectra."
        ),
        "implementation": "`p4a_bve_select`.",
    },

    "rep_select": {
        "group": "selector",
        "title": "REP — Recursive Elimination of Predictors",
        "paper": (
            "Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). "
            "*A review of variable selection methods in partial least "
            "squares regression*. Chemometrics and Intelligent Laboratory "
            "Systems 118, 62–69. "
            "https://doi.org/10.1016/j.chemolab.2012.07.010 — same review "
            "as `shaving_select`; §3.3 *Recursive elimination* introduces "
            "the fixed-count variant implemented here."
        ),
        "principle": (
            "REP removes a **fixed count** of features per recursive "
            "step (rather than a fraction as in shaving). At each "
            "step, sort features by absolute coefficient score, "
            "remove the $m$ lowest, refit, record CV-RMSE. Return "
            "the subset with lowest CV-RMSE across all retained "
            "trajectories.\n\n"
            "Useful when the analyst wants control over total "
            "iteration count: with $m$ features removed per step, "
            "the process terminates in $\\lceil p / m \\rceil$ "
            "iterations. Same intent as shaving but with linear "
            "instead of geometric decay."
        ),
        "implementation": "`p4a_rep_select`.",
    },

    "ipw_select": {
        "group": "selector",
        "title": "IPW — Iterative Predictor Weighting",
        "paper": (
            "Forina, M., Casolino, C. & Pizarro Millán, C. (1999). "
            "*Iterative predictor weighting (IPW) PLS: a technique for "
            "the elimination of useless predictors in regression "
            "problems*. Journal of Chemometrics 13(2), 165–184. "
            "https://doi.org/10.1002/(SICI)1099-128X(199903/04)13:2<165::AID-CEM535>3.0.CO;2-Y"
        ),
        "principle": (
            "IPW iteratively re-weights features in $\\mathbf{X}$ by "
            "their importance, refits PLS on the re-weighted data, "
            "and tracks the score path. Weights are derived from "
            "coefficient magnitude after each fit; the iteration "
            "converges to a stable importance ranking.\n\n"
            "Compared to single-fit coefficient ranking, IPW's "
            "iterative refinement gives more stable rankings when "
            "the calibration set is small or noisy. Exposes both the "
            "score path (for diagnostic) and the weight path (for "
            "interpretation)."
        ),
        "implementation": "`p4a_ipw_select`.",
    },

    "st_select": {
        "group": "selector",
        "title": "ST-PLS — Score Threshold selection",
        "paper": (
            "Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). "
            "*A review of variable selection methods in partial least "
            "squares regression*. Chemometrics and Intelligent Laboratory "
            "Systems 118, 62–69. "
            "https://doi.org/10.1016/j.chemolab.2012.07.010 — same review "
            "as `shaving_select`; §3.4 *Score-threshold methods* covers "
            "the deterministic-threshold family implemented here."
        ),
        "principle": (
            "Apply deterministic thresholds on the standardised "
            "coefficient (or VIP) scores: keep features whose "
            "absolute score exceeds the threshold $\\tau$, with a "
            "minimum-retained fallback to avoid the empty selection. "
            "The benchmark scans a grid of thresholds and returns "
            "the subset with lowest CV-RMSE.\n\n"
            "Conceptually similar to UVE but uses absolute thresholds "
            "rather than noise-baseline-relative ones. Less elegant "
            "but cheaper since no augmented noise matrix is needed."
        ),
        "implementation": "`p4a_st_select`.",
    },

    "interval_select": {
        "group": "selector",
        "title": "iPLS — Interval PLS (moving-window)",
        "paper": (
            "Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., "
            "Munck, L. & Engelsen, S. B. (2000). *Interval partial "
            "least-squares regression (iPLS): a comparative chemometric "
            "study with an example from near-infrared spectroscopy*. "
            "Applied Spectroscopy 54(3), 413–419."
        ),
        "principle": (
            "Slide a fixed-width window of $w$ consecutive "
            "wavelengths across the spectrum, fit PLS on each window "
            "alone, evaluate by CV-RMSE. The window with the lowest "
            "CV-RMSE is returned as the selected interval.\n\n"
            "iPLS is the simplest **interval** selector — it returns "
            "a single contiguous band rather than scattered "
            "wavelengths. The output is therefore directly "
            "interpretable as a spectroscopic feature (functional "
            "group, electronic transition, …). For multi-band selection "
            "use biPLS or siPLS.\n\n"
            "The window width $w$ is the main tunable; cross-validating "
            "$w$ jointly with the window position is the standard "
            "extension."
        ),
        "implementation": (
            "`p4a_interval_select`. Reference: R `plsVarSel`."
        ),
    },

    "bipls_select": {
        "group": "selector",
        "title": "biPLS — Backward Interval PLS",
        "paper": (
            "Leardi, R. & Nørgaard, L. (2004). *Sequential application "
            "of backward interval partial least squares and genetic "
            "algorithms for the selection of relevant spectral "
            "regions*. Journal of Chemometrics 18(11), 486–497."
        ),
        "principle": (
            "Start with the spectrum partitioned into $I$ equal "
            "intervals (typically 10–40). At each iteration, remove "
            "the interval whose removal **least** hurts CV-RMSE — i.e. "
            "the least informative interval. Iterate until removing "
            "any further interval materially worsens performance.\n\n"
            "Returns a multi-band subset with each band aligned to the "
            "original equal-partition grid. The discrete structure "
            "makes biPLS robust to noise (no fine-grained fishing) "
            "and easy to interpret (each retained interval is a "
            "spectroscopic region of contiguous wavelengths).\n\n"
            "Commonly chained with GA-PLS as a coarse-to-fine "
            "pipeline (Leardi & Nørgaard 2004): biPLS narrows the "
            "candidate intervals, GA-PLS does the within-interval "
            "feature selection."
        ),
        "implementation": (
            "`p4a_bipls_select`. Reference: R `plsVarSel`."
        ),
    },

    "sipls_select": {
        "group": "selector",
        "title": "siPLS — Synergy Interval PLS",
        "paper": (
            "Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., "
            "Munck, L. & Engelsen, S. B. (2000). *Interval partial "
            "least-squares regression (iPLS): a comparative chemometric "
            "study with an example from near-infrared spectroscopy*. "
            "Applied Spectroscopy 54(3), 413–419 — same paper as "
            "`interval_select`; siPLS is the synergy-combinations "
            "extension proposed in §3."
        ),
        "principle": (
            "Exhaustively score every combination of $m$ fixed-size "
            "intervals (out of $I$ total) by CV-RMSE; return the "
            "best combination. This captures **synergy** — pairs (or "
            "triples) of intervals that work well together even if "
            "neither alone is the best single interval.\n\n"
            "Combinatorial cost is $\\binom{I}{m}$ which is "
            "manageable only for small $m$ (typically $m \\le 4$ "
            "with $I \\le 20$, giving up to ~5000 combinations). For "
            "larger search spaces use biPLS or interval-GA.\n\n"
            "siPLS is the natural extension of iPLS when single-"
            "interval performance is unsatisfactory: it explicitly "
            "looks for complementary bands."
        ),
        "implementation": (
            "`p4a_sipls_select`. Reference: R `plsVarSel`."
        ),
    },

    "t2_select": {
        "group": "selector",
        "title": "Hotelling T² loading selection",
        "paper": (
            "Mehmood, T. (2016). *Hotelling T² based variable selection "
            "in partial least squares regression*. Chemometrics and "
            "Intelligent Laboratory Systems 154, 23–28. "
            "https://doi.org/10.1016/j.chemolab.2016.03.020 — proposes "
            "T²-PLS, the loading-weights-level Hotelling T² selector. "
            "See also Wold, Sjöström & Eriksson (2001), Chemometrics "
            "and Intelligent Laboratory Systems 58(2), 109–130 §6.2 "
            "for the original T²-vs-VIP discussion in PLS."
        ),
        "principle": (
            "Apply Hotelling T² to the **loading weights** rather "
            "than the scores: features with loading vectors of "
            "large T² are deemed important. Threshold via the "
            "F-distribution upper control limit at a user-specified "
            "$\\alpha$, with a top-$k$ fallback to avoid empty "
            "selections.\n\n"
            "Distinct from sample-level T² monitoring (see "
            "`pls_diagnostic_t2`) — here T² acts as a multivariate "
            "feature ranker that respects correlation structure "
            "across components. Useful when the loadings have strong "
            "between-component structure and per-component "
            "VIP under-counts contributions spread across multiple "
            "components."
        ),
        "implementation": "`p4a_t2_select`.",
    },

    "wvc_select": {
        "group": "selector",
        "title": "WVC — Weighted Variable Contribution",
        "paper": (
            "Andries, J. P. M. & Vander Heyden, Y. (2011). *Improved "
            "variable reduction in partial least squares modelling "
            "based on predictive-property-ranked variables and "
            "adaptation of partial least squares complexity*. "
            "Analytica Chimica Acta 705(1–2), 292–305."
        ),
        "principle": (
            "WVC builds a **normalised weighted-variable-contribution** "
            "score from the SVD of the PLS components. Each feature's "
            "contribution to each component is weighted by the "
            "component's singular value (importance), then summed "
            "and normalised. Sort by WVC, take the top-$k$.\n\n"
            "Compared to VIP, WVC uses SVD-based weighting which "
            "downweights components dominated by noise; this gives "
            "more stable rankings when $k$ is over-specified."
        ),
        "implementation": "`p4a_wvc_select`.",
    },

    "wvc_threshold_select": {
        "group": "selector",
        "title": "WVC-threshold selection",
        "paper": (
            "Andries, J. P. M. & Vander Heyden, Y. (2011). *Improved "
            "variable reduction in partial least squares modelling "
            "based on predictive-property-ranked variables and "
            "adaptation of partial least squares complexity*. "
            "Analytica Chimica Acta 705(1–2), 292–305. "
            "https://doi.org/10.1016/j.aca.2011.06.037 — same paper as "
            "`wvc_select`; introduces both the top-$k$ ranking and the "
            "threshold / factor-of-mean rules used here."
        ),
        "principle": (
            "Apply fixed-threshold and factor-of-mean rules over "
            "WVC scores, with a minimum-selected fallback. Two "
            "selection rules are evaluated and the lowest-CV-RMSE "
            "one returned: "
            "(1) WVC > absolute threshold $\\tau$, or "
            "(2) WVC > $f \\cdot \\overline{\\mathrm{WVC}}$ "
            "(factor-of-mean).\n\n"
            "The factor-of-mean rule is dataset-adaptive; the "
            "absolute rule is more conservative. The minimum-selected "
            "fallback (e.g. retain at least 10 features) prevents "
            "empty selections on flat-WVC datasets."
        ),
        "implementation": "`p4a_wvc_threshold_select`.",
    },

    "emcuve_select": {
        "group": "selector",
        "title": "EMCUVE — Ensemble MC-UVE",
        "paper": (
            "Han, Q.-J., Wu, H.-L., Cai, C.-B., Xu, L. & Yu, R.-Q. "
            "(2008). *An ensemble of Monte Carlo uninformative variable "
            "elimination for wavelength selection*. Analytica Chimica "
            "Acta 612(2), 121–125. "
            "https://doi.org/10.1016/j.aca.2008.02.032 — extends the "
            "MC-UVE procedure of Cai et al. (2008) (`stability_select`) "
            "by aggregating independent MC-UVE rounds through a vote "
            "rule."
        ),
        "principle": (
            "Run multiple independent MC-UVE rounds with different "
            "seeds, threshold each independently, then **vote** "
            "across rounds: a feature is selected if it survives "
            "thresholding in a majority of rounds. Robust against "
            "single-round instability caused by particular bootstrap "
            "samples.\n\n"
            "The voting rule has a free parameter (majority "
            "threshold); the default of $\\lceil R/2 \\rceil$ is the "
            "median-style majority. Stricter thresholds give smaller "
            "but more reliable subsets."
        ),
        "implementation": "`p4a_emcuve_select`.",
    },

    "randomization_select": {
        "group": "selector",
        "title": "Randomisation test (Y-permutation)",
        "paper": (
            "Westad, F. & Martens, H. (2000). *Variable selection in "
            "near infrared spectroscopy based on significance testing "
            "in partial least squares regression*. JNIRS 8(2), "
            "117–124."
        ),
        "principle": (
            "Compute the observed PLS coefficient magnitudes "
            "$|b_j^{\\mathrm{obs}}|$, then permute $\\mathbf{y}$ "
            "$M$ times, refit PLS each time, and collect "
            "$|b_j^{(m)}|$. The empirical p-value of feature $j$ is "
            "$p_j = \\frac{1 + \\#\\{m : |b_j^{(m)}| \\ge |b_j^{\\mathrm{obs}}|\\}}{1 + M}$. "
            "Retain features with $p_j < \\alpha$.\n\n"
            "Y-permutation is the gold standard for **null-"
            "calibrated** significance testing in PLS — no "
            "distributional assumptions, no asymptotic approximations. "
            "Cost is $M$× a fit but trivially parallelisable.\n\n"
            "Critically, Y-permutation tests the joint hypothesis "
            "'feature $j$ contributes to $y$'; multiple-testing "
            "correction (Benjamini-Hochberg) is recommended for "
            "$p \\gg 100$."
        ),
        "implementation": "`p4a_randomization_select`.",
    },

    "iriv_select": {
        "group": "selector",
        "title": "IRIV — Iteratively Retaining Informative Variables",
        "paper": (
            "Yun, Y. H., Wang, W. T., Tan, M. L., Liang, Y. Z., Li, "
            "H. D., Cao, D. S., Lu, H. M. & Xu, Q. S. (2014). *A "
            "strategy that iteratively retains informative variables "
            "for selecting optimal variable subset in multivariate "
            "calibration*. Analytica Chimica Acta 807, 36–43."
        ),
        "principle": (
            "IRIV classifies each variable into four categories at "
            "each iteration: **strongly informative**, **weakly "
            "informative**, **uninformative**, **interfering**. The "
            "first two are retained, the last two eliminated. "
            "Iteration continues until no further interfering "
            "variables remain.\n\n"
            "Categories are determined by Monte-Carlo subset analysis "
            "with a permutation-based reference distribution: each "
            "variable's CV-RMSE contribution distribution is compared "
            "against the distribution under random subset inclusion. "
            "This four-way classification is more nuanced than "
            "single-threshold methods and tends to handle correlated "
            "predictors well (correlated features can both be "
            "'weakly informative')."
        ),
        "implementation": "`p4a_iriv_select`.",
    },

    "irf_select": {
        "group": "selector",
        "title": "IRF — Iterative Random Forest",
        "paper": (
            "Basu, S., Kumbier, K., Brown, J. B. & Yu, B. (2018). "
            "*Iterative random forests to discover predictive and "
            "stable high-order interactions*. Proceedings of the "
            "National Academy of Sciences 115(8), 1943–1948."
        ),
        "principle": (
            "IRF iteratively re-weights random forest feature "
            "importances and refits. At each iteration, features with "
            "high feature-importance get oversampled in the bootstrap "
            "of the next forest; the loop converges to a stable "
            "ranking of features by their **interaction-aware** "
            "importance.\n\n"
            "Adapted for PLS prediction: the IRF importance ranking "
            "is used to select the top-$k$ features, then PLS is fit "
            "on the selected subset. The RF importance is "
            "non-linear so this catches predictive features that "
            "interact rather than contributing additively — typically "
            "missed by linear selectors like VIP."
        ),
        "implementation": "`p4a_irf_select`.",
    },

    "vip_spa_select": {
        "group": "selector",
        "title": "VIP-seeded SPA",
        "paper": (
            "Hybrid heuristic combining VIP ranking and the "
            "Successive Projections Algorithm. See registry notes; "
            "no single canonical paper."
        ),
        "principle": (
            "Use VIP scores to **seed** SPA's projection-orthogonal "
            "forward selection. SPA starts with the highest-VIP "
            "feature rather than the highest-coefficient one, then "
            "proceeds with the standard projection step. This biases "
            "SPA toward $y$-correlated seed features while preserving "
            "SPA's collinearity-minimising selection of subsequent "
            "features.\n\n"
            "In practice this tends to outperform plain SPA on "
            "datasets where the first SPA seed is well-known to be "
            "noise-dominated (some real-world NIR datasets) but "
            "VIP correctly flags a different region as predictive."
        ),
        "implementation": "`p4a_vip_spa_select`.",
    },
}
