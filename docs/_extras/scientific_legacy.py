"""Verified scientific overlay for the 73 legacy method catalogue entries.

``methods_bibliography.py`` is kept as an historical source.  This module is
the rendered, auditable layer: every entry identifies an intended use, an
implementation-specific limitation, and the source file used to verify it.
"""

from __future__ import annotations

from methods_bibliography import BIBLIOGRAPHY as _LEGACY


_REPOSITORY = "https://github.com/GBeurier/nirs4all-methods/blob/main/"


def _src(path: str) -> str:
    return f"Current implementation: [{path}]({_REPOSITORY}{path})."


# These annotations deliberately describe the shipped implementation rather
# than silently substituting the canonical algorithm suggested by its name.
_ANNOTATIONS: dict[str, tuple[str, str, str]] = {
    "pls": (
        "Linear multivariate calibration when predictors are numerous, collinear, and response-linked latent directions are useful.",
        "Predictions remain linear in X; component count and preprocessing must be validated without leaking validation samples.",
        _src("cpp/src/core/model.cpp"),
    ),
    "pcr": (
        "Baseline regression after unsupervised variance compression, especially when comparison with response-guided PLS is informative.",
        "High-variance PCs need not predict Y, so PCR can discard low-variance predictive directions.",
        _src("cpp/src/core/model.cpp"),
    ),
    "opls": (
        "Separating X variation correlated with Y from orthogonal spectral variation for interpretation and linear prediction.",
        "Orthogonal filtering is supervised and must be fitted inside each validation fold; it does not make the final predictor nonlinear.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "cppls": (
        "Classification or multivariate calibration where Canonical Powered PLS response weighting is useful.",
        "In the default path gamma is recorded but does not alter the fit; only the legacy SIMPLS path applies a gamma-dependent column rescaling.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "recursive_pls": (
        "Updating a linear calibration as new batches arrive without retaining and refitting the complete historical data set.",
        "Forgetting-factor updates can drift or forget rare regimes; sample order and batch composition affect the result.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "sparse_simpls": (
        "Joint linear prediction and variable screening when a compact wavelength set is required.",
        "The selected set depends on eta and component count; the default Chun--Keles-style path differs from the optional legacy thresholding path.",
        _src("cpp/src/core/model.cpp"),
    ),
    "group_sparse_pls": (
        "Exploratory group selection with predefined wavelength groups when post-fit coefficient shrinkage is acceptable.",
        "The penalty shrinks predictive coefficient groups and affects predictions, but does not refit latent directions or implement sgPLS::gPLS. Its units depend on raw-X and target scaling; select it using held-out data.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "fused_sparse_pls": (
        "Exploratory smoothing of adjacent regression coefficients for ordered spectral variables.",
        "This is four fixed neighbour-averaging passes, not a fused-lasso/total-variation optimizer; l1_lambda uses post-SIMPLS group shrinkage.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "sparse_pls_da": (
        "Sparse multiclass discrimination with one-hot responses and an interpretable wavelength subset.",
        "The prediction surface emits an argmax class and one-hot output, not calibrated softmax probabilities.",
        _src("cpp/src/c_api/c_api_method_result.cpp"),
    ),
    "bagging_pls": (
        "Reducing variance by averaging PLS models fitted to bootstrap samples.",
        "Bootstrap averaging does not correct systematic bias, and training cost grows with the number of estimators.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "boosting_pls": (
        "Stagewise fitting of linear PLS base learners to residuals when shrinkage across rounds improves stability.",
        "The summed coefficient vectors still define a linear predictor; this implementation cannot recover general nonlinear X--Y relations.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "random_subspace_pls": (
        "Ensembling PLS fits across random wavelength subsets to reduce dependence on any one correlated region.",
        "Small subspaces can omit important bands and large ensembles increase fitting cost; the random seed affects the selected subsets.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "weighted_pls": (
        "Calibration with known nonnegative sample reliabilities, frequency weights, or deliberate regime reweighting.",
        "Weights change the estimand and require substantive justification; zero or extreme weights reduce effective sample size.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "robust_pls": (
        "Calibration containing moderate response or leverage contamination where robust residual weighting is justified.",
        "The default PRM-style path uses Fair weights and reinterprets huber_k as its tuning constant; Huber IRLS is available only in legacy mode.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "ridge_pls": (
        "Stabilizing latent calibration when the predictor cross-product is ill-conditioned.",
        "The ridge penalty is applied by augmenting X and Y before PLS, so it changes latent directions as well as coefficient shrinkage.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "continuum_regression": (
        "Exploring the continuum between covariance-driven PLS and inverse-covariance/OLS-like directions.",
        "Current tau uses PLS at 0 and full-rank OLS-like weighting at 1; it has no PCR endpoint and differs from other continuum-regression parameterizations.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "kernel_pls_rbf": (
        "Nonlinear calibration when similarity in a chosen kernel space is scientifically plausible.",
        "Prediction needs all training rows, costs O(n_train*n_new), and training stores an O(n_train^2) Gram matrix; kernel parameters require nested validation.",
        _src("cpp/src/core/kernel_pls.cpp"),
    ),
    "lw_pls": (
        "Local nonlinear calibration where each query is well represented by nearby calibration samples.",
        "A separate weighted model is fitted per query; distance scaling, bandwidth, and poor neighbourhood coverage can dominate predictions.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "gpr_pls": (
        "Gaussian-process regression on a low-dimensional PLS representation when smooth nonlinear residual structure is expected.",
        "length_scale and noise_level are fixed inputs, amplitude is one, and no marginal-likelihood optimizer is run; seed is currently unused.",
        _src("cpp/src/core/gpr_pls.cpp"),
    ),
    "mb_pls": (
        "Joint calibration from explicitly defined predictor blocks such as instruments, spectral ranges, or sensor families.",
        "Block scaling determines influence; concatenation-compatible outputs should not be interpreted as proving block-specific causality.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "mir_pls": (
        "Experimental inverse-regression calibration in which a SIMPLS map from Y to X is algebraically inverted.",
        "This is an implementation-specific regularized pseudoinverse, not a standard mid-infrared preprocessing method or canonical MIR-PLS algorithm.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "so_pls": (
        "Sequential modelling of ordered data blocks after orthogonalizing each new block against previous block scores.",
        "Results depend on block order and component allocation; validation must repeat the complete sequential procedure.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "on_pls": (
        "Multi-block exploratory modelling that separates globally joint, locally joint, and unique variation.",
        "Component interpretation is sensitive to block scaling and rank choices; predictive claims require external validation.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "rosa": (
        "Response-oriented sequential extraction across blocks when each component should be assigned to a contributing block.",
        "Greedy block selection and block scaling can change the extraction order and the resulting interpretation.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "n_pls": (
        "Regression of multiway arrays while preserving tensor modes instead of unfolding all variables into one matrix.",
        "Mode ranks and scaling must be chosen carefully; separable tensor components can underfit nonseparable structure.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "o2pls": (
        "Two-block integration that separates joint predictive covariance from block-specific structured variation.",
        "Joint and orthogonal ranks are not identified automatically and may be unstable in small samples.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "di_pls": (
        "Adapting a linear calibration across labelled source and target domains with different predictor distributions.",
        "Domain-alignment penalties can remove predictive variation; target information and tuning must remain inside validation folds.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "ds": (
        "Transferring spectra from a secondary instrument to a reference instrument using paired standards.",
        "Requires representative paired transfer samples and a stable linear relation between instruments.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "pds": (
        "Local-window instrument standardization when wavelength-specific transfer maps are more plausible than one global map.",
        "Window width and paired-standard coverage control extrapolation; boundary wavelengths have less contextual support.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "ecr": (
        "Supervised dimension reduction that explicitly blends X variance and X--Y covariance before regression.",
        "alpha changes the eigensystem and must be validated; endpoint interpretations assume the implemented scaling and sequential deflation.",
        _src("cpp/src/core/ecr.cpp"),
    ),
    "pls_lda": (
        "Multiclass classification using supervised PLS scores followed by a shared-covariance Gaussian discriminant.",
        "LDA covariance assumptions apply in score space; score extraction and class imbalance must be handled inside validation.",
        _src("cpp/src/core/pls_lda.cpp"),
    ),
    "pls_qda": (
        "Classification on PLS scores when class-specific covariance matrices are supported by enough samples.",
        "Class priors are empirical in the current implementation; class-specific covariance estimates can be singular or unstable for small classes.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "pls_logistic": (
        "Multiclass probabilistic classification with a linear softmax model on a fixed PLS score representation.",
        "PLS scores are fitted once before Newton optimization rather than recomputed from each IRLS working response; ridge and component count need validation.",
        _src("cpp/src/core/pls_logistic.cpp"),
    ),
    "pls_glm": (
        "Compatibility experiments requiring the public PLS-GLM entry point with continuous responses.",
        "Current code only centers Y and runs one SIMPLS fit; the Poisson flag is stored but does not change fitting, and no link or IRLS is implemented.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "pls_cox": (
        "Exploratory survival ranking through the shipped log-time pseudo-response approximation.",
        "This is not a Cox partial-likelihood fit: censored log-times are replaced by an event-time mean before SIMPLS, then coefficients are negated.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "missing_aware_nipals": (
        "Latent regression when predictor matrices contain scattered missing entries and complete-case deletion is undesirable.",
        "Pairwise updates can use different samples for different products, and convergence deteriorates with structured or extensive missingness.",
        _src("cpp/src/core/model.cpp"),
    ),
    "pls_diagnostic_t2": (
        "Flagging observations with unusual leverage in the fitted PLS score space.",
        "T-squared detects score-space leverage rather than large residuals; thresholds require an appropriate reference population.",
        _src("cpp/src/core/pls_diagnostics.cpp"),
    ),
    "pls_diagnostic_q": (
        "Flagging spectra poorly reconstructed by the retained latent X subspace.",
        "Q residuals do not detect all high-leverage points and are sensitive to preprocessing and component count.",
        _src("cpp/src/core/pls_diagnostics.cpp"),
    ),
    "pls_diagnostic_dmodx": (
        "Standardizing X reconstruction distance for model-population screening.",
        "The implementation uses p-a degrees of freedom and a reference residual scale; without reference data sigma is fixed to one.",
        _src("cpp/src/core/pls_diagnostics.cpp"),
    ),
    "pls_monitoring": (
        "Process or instrument monitoring with paired score-leverage and X-residual alarms.",
        "Current control limits are empirical reference quantiles, so alpha is resolution-limited in small reference sets and no F/Jackson--Mudholkar model is fitted.",
        _src("cpp/src/core/pls_monitoring.cpp"),
    ),
    "approximate_press": (
        "Selecting PLS component count from leave-one-out prediction error on small data sets.",
        "The default performs n refits for every component count and is expensive; legacy mode uses a constant k/n inflation, not individual hat diagonals.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "one_se_rule": (
        "Choosing a parsimonious component count whose cross-validation error is statistically indistinguishable from the minimum.",
        "The result depends on fold construction and on how the standard error is computed across genuinely independent validation units.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "aom_preprocess": (
        "Automated comparison of candidate spectral preprocessing chains under an explicit validation protocol.",
        "Searching many chains increases selection bias; every fitted transform and chain choice must be confined to training folds.",
        _src("cpp/src/core/aom_preprocessing.cpp"),
    ),
    "aom_pls": (
        "Joint automated choice of preprocessing and PLS complexity for a defined calibration task.",
        "The selected pipeline is conditional on the candidate grid and validation split and needs an untouched external assessment.",
        _src("cpp/src/core/aom_preprocessing.cpp"),
    ),
    "pop_pls": (
        "Population-based search over PLS model configurations when exhaustive enumeration is impractical.",
        "Stochastic search has no guarantee of the global optimum and must evaluate candidates on leakage-free validation data.",
        _src("cpp/src/core/extra_pls.cpp"),
    ),
    "variable_select_vip": (
        "Ranking wavelengths by their aggregate contribution to response-oriented PLS components.",
        "VIP is model-dependent and correlated variables can share or exchange importance; a threshold of one is heuristic.",
        _src("cpp/src/core/variable_selection.cpp"),
    ),
    "variable_select_coef": (
        "Screening variables by fitted PLS coefficient magnitude when a compact linear predictor is desired.",
        "Coefficient magnitude depends on variable scaling and is unstable among correlated wavelengths.",
        _src("cpp/src/core/variable_selection.cpp"),
    ),
    "variable_select_sr": (
        "Ranking variables by selectivity ratio to favour explained target-related variance over residual variance.",
        "Scores depend on the fitted latent model and preprocessing; collinearity can distribute signal over neighbouring bands.",
        _src("cpp/src/core/variable_selection.cpp"),
    ),
    "spa_select": (
        "Selecting minimally collinear wavelengths for compact spectrometer or interpretable calibration designs.",
        "SPA's projection criterion is only indirectly response-aware and the starting variable/target subset size can change the result.",
        _src("cpp/src/core/spa_selection.cpp"),
    ),
    "stability_select": (
        "Retaining wavelengths repeatedly selected under subsampling to quantify selection reproducibility.",
        "Frequency thresholds do not by themselves provide error control unless the formal stability-selection assumptions and sampling design hold.",
        _src("cpp/src/core/stability_selection.cpp"),
    ),
    "uve_select": (
        "Eliminating variables whose PLS coefficient stability is no better than injected noise variables.",
        "Results depend on resampling, noise construction, and scaling; UVE must be rerun inside each validation fold.",
        _src("cpp/src/core/uve_selection.cpp"),
    ),
    "cars_select": (
        "Competitive wavelength reduction through repeated Monte Carlo PLS fits and adaptive elimination.",
        "CARS is stochastic and optimizes a validation criterion over many subsets, so nested or external validation is required.",
        _src("cpp/src/core/cars_selection.cpp"),
    ),
    "random_frog_select": (
        "Sampling variable subsets to estimate inclusion probabilities under a PLS validation objective.",
        "Finite-chain inclusion frequencies depend on initialization, proposal settings, and seed; they are not Bayesian posterior probabilities.",
        _src("cpp/src/core/random_frog_selection.cpp"),
    ),
    "scars_select": (
        "Sequential competitive elimination of weak wavelengths across repeated PLS evaluations.",
        "Aggressive elimination is irreversible within a run and can discard correlated alternatives; validation reuse creates optimism.",
        _src("cpp/src/core/scars_selection.cpp"),
    ),
    "ga_select": (
        "Combinatorial wavelength-subset search with crossover and mutation under a PLS fitness score.",
        "The stochastic search can converge prematurely and heavily reuses the fitness data; population size and seed affect results.",
        _src("cpp/src/core/ga_selection.cpp"),
    ),
    "pso_select": (
        "Population-based exploration of binary wavelength masks under a predictive fitness function.",
        "Binary PSO is a heuristic without global-optimum guarantees and can overfit a repeatedly queried validation set.",
        _src("cpp/src/core/pso_selection.cpp"),
    ),
    "vissa_select": (
        "Iteratively concentrating weighted sampling on wavelengths associated with better PLS subsets.",
        "Sampling probabilities are path-dependent and stochastic; reported subsets need independent assessment.",
        _src("cpp/src/core/vissa_selection.cpp"),
    ),
    "shaving_select": (
        "Backward elimination of low-importance wavelengths to trace a nested sequence of compact PLS models.",
        "Removed variables cannot re-enter and correlated importance rankings are unstable; subset choice can overfit CV.",
        _src("cpp/src/core/shaving_selection.cpp"),
    ),
    "bve_select": (
        "Backward variable elimination using a PLS-derived relevance score.",
        "Greedy removal ignores interactions among discarded variables and requires nested validation of the stopping point.",
        _src("cpp/src/core/bve_selection.cpp"),
    ),
    "rep_select": (
        "Iterative elimination based on regression-coefficient-derived variable importance for parsimonious calibration.",
        "Coefficient rankings inherit scaling and collinearity sensitivity, and repeated validation queries can bias model choice.",
        _src("cpp/src/core/rep_selection.cpp"),
    ),
    "ipw_select": (
        "Iteratively reweighting predictors from PLS coefficients to concentrate a model on consistently influential wavelengths.",
        "The update is heuristic and can amplify early ranking errors; convergence does not establish causal relevance.",
        _src("cpp/src/core/ipw_selection.cpp"),
    ),
    "st_select": (
        "Removing wavelengths with small stability-normalized PLS coefficients.",
        "Reliability estimates depend on the resampling design, and the ratio becomes unstable when coefficient variability is poorly estimated.",
        _src("cpp/src/core/st_selection.cpp"),
    ),
    "interval_select": (
        "Selecting contiguous spectral intervals when bands are expected to carry joint chemical information.",
        "The answer depends on interval partition and width; coarse boundaries can split or dilute informative peaks.",
        _src("cpp/src/core/interval_selection.cpp"),
    ),
    "bipls_select": (
        "Backward elimination of predefined spectral intervals using PLS validation error.",
        "Greedy deletion cannot recover removed intervals and repeated CV comparison can produce optimistic subset estimates.",
        _src("cpp/src/core/bipls_selection.cpp"),
    ),
    "sipls_select": (
        "Evaluating combinations of a small number of spectral intervals for interpretable regional models.",
        "Combination count grows rapidly and results depend on the initial interval grid and validation split.",
        _src("cpp/src/core/sipls_selection.cpp"),
    ),
    "t2_select": (
        "Screening variables by their contribution to leverage in the implemented PLS weight space.",
        "The cutoff uses a Beta quantile with an implementation-specific weight-based statistic, not the archived F-limit/top-k algorithm; min_selected may add variables.",
        _src("cpp/src/core/t2_selection.cpp"),
    ),
    "wvc_select": (
        "Ranking wavelengths by the shipped weighted variable-contribution score across sequential PLS1 components.",
        "The implementation uses PLS weights, score energy, and residual energy; it performs no SVD and should not be interpreted as a singular-value criterion.",
        _src("cpp/src/core/wvc_selection.cpp"),
    ),
    "wvc_threshold_select": (
        "Applying an explicit cutoff to WVC scores for a compact selected set.",
        "One cutoff max(score_threshold, threshold_factor*mean_score) is used, with a min_selected fallback; no competing-rule CV selection is performed.",
        _src("cpp/src/core/wvc_selection.cpp"),
    ),
    "emcuve_select": (
        "Combining Monte Carlo coefficient stability with UVE-style elimination for wavelength screening.",
        "Noise-variable and resampling choices set the effective cutoff; stochastic stability is not evidence of mechanistic relevance.",
        _src("cpp/src/core/emcuve_selection.cpp"),
    ),
    "randomization_select": (
        "Permutation testing of individual wavelength association under a specified exchangeability null.",
        "Current selection uses p <= alpha without multiplicity correction; Benjamini--Hochberg is only a recommendation, not implemented.",
        _src("cpp/src/core/randomization_selection.cpp"),
    ),
    "iriv_select": (
        "Iteratively comparing variable subsets to classify informative and interfering wavelengths.",
        "Repeated random subset tests are stochastic and computationally expensive; significance labels depend on the sampled combinations.",
        _src("cpp/src/core/iriv_selection.cpp"),
    ),
    "irf_select": (
        "Searching contiguous wavelength intervals with random-frog proposals and a PLS cross-validation objective.",
        "IRF here means Interval Random Frog: it uses no random forest or interaction-aware tree importance, and depends on interval width and chain mixing.",
        _src("cpp/src/core/irf_selection.cpp"),
    ),
    "vip_spa_select": (
        "Seeding a low-collinearity SPA wavelength sequence from a response-aware VIP ranking.",
        "The hybrid is heuristic with no single canonical estimator; VIP and SPA tuning must be repeated within validation folds.",
        _src("cpp/src/core/vip_spa_selection.cpp"),
    ),
}


_OVERRIDES: dict[str, dict[str, str]] = {'approximate_press': {'implementation': 'Cost scales with n refits per component count. The '
                                         'compatibility-only legacy branch avoids those refits '
                                         'with a constant k/n residual inflation; it does not '
                                         'calculate observation-specific leverage h_ii.',
                       'paper': 'Allen, D. M. (1974). *The relationship between variable selection '
                                'and data augmentation and a method for prediction*. Technometrics '
                                '16(1), 125–127. Verified primary link: '
                                '[https://doi.org/10.1080/00401706.1974.10489157](https://doi.org/10.1080/00401706.1974.10489157).',
                       'principle': 'The default path computes prediction residual error sums of '
                                    'squares by leaving out each observation, refitting PLS, and '
                                    'predicting the held-out row for every requested component '
                                    'count. This is direct LOO PRESS rather than a hat-matrix '
                                    'shortcut.',
                       'title': 'PRESS by leave-one-out refitting'},
 'bagging_pls': {'paper': 'Breiman, L. (1996). *Bagging predictors*. Machine Learning 24(2), '
                          '123–140. — adapted for PLS by various chemometric authors. Verified '
                          'primary link: '
                          '[https://doi.org/10.1023/A:1018054314350](https://doi.org/10.1023/A:1018054314350).'},
 'bipls_select': {'paper': 'Leardi, R. & Nørgaard, L. (2004). *Sequential application of backward '
                           'interval partial least squares and genetic algorithms for the '
                           'selection of relevant spectral regions*. Journal of Chemometrics '
                           '18(11), 486–497. Verified primary link: '
                           '[https://doi.org/10.1002/cem.893](https://doi.org/10.1002/cem.893).'},
 'boosting_pls': {'implementation': 'The implementation stores the accumulated coefficients and '
                                    'intercept. Use kernel or explicitly nonlinear features when '
                                    'nonlinear response structure is required.',
                  'paper': 'Friedman, J. H. (2001). *Greedy function approximation: a gradient '
                           'boosting machine*. Annals of Statistics 29(5), 1189–1232. — adapted '
                           'for PLS as a base learner. Verified primary link: '
                           '[https://doi.org/10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).',
                  'principle': 'Each round fits a linear PLS model to the current residual and '
                               'adds its coefficient matrix, scaled by the learning rate, to the '
                               'ensemble coefficient matrix. Algebraically, a sum of these base '
                               'predictors is still one linear map from X to Y. Boosting can '
                               'regularize the stagewise fit but does not introduce nonlinear '
                               'features.'},
 'bve_select': {'implementation': '`cpp/src/core/bve_selection.cpp` evaluates the CV RMSE of every '
                                  'one-variable removal at each of the exact `n_steps`, records '
                                  'all generated subsets, then returns the global RMSE minimum. '
                                  "The registry's `plsVarSel::bve_pls` reference documents the "
                                  'canonical comparison; it does not establish identical behavior '
                                  'for every compatibility path.',
                'limitations': 'The greedy path is not globally optimal. Correlated predictors can '
                               'exchange predictive signal, making the removal order and the '
                               'selected trajectory step unstable; repeat the whole elimination '
                               'inside an outer validation protocol rather than treating one CV '
                               'minimum as unbiased evidence.',
                'paper': 'Mehmood, T. et al. (2011). *A Partial Least Squares based algorithm for '
                         'parsimonious variable selection*. Algorithms for Molecular Biology 6, '
                         '27. DOI '
                         '[10.1186/1748-7188-6-27](https://doi.org/10.1186/1748-7188-6-27).',
                'principle': 'Starting from the full variable set, the routine performs exactly '
                             '`n_steps` backward-elimination rounds. At every round it scores '
                             'every available one-variable removal by CV RMSE, removes the '
                             'candidate with the smallest RMSE, and stores the resulting subset. '
                             'It finally returns the stored step whose CV RMSE is smallest over '
                             'the complete trajectory; it does not stop when a single removal '
                             'increases RMSE.'},
 'cars_select': {'paper': 'Li, H., Liang, Y., Xu, Q. & Cao, D. (2009). *Key wavelengths screening '
                          'using competitive adaptive reweighted sampling method for multivariate '
                          'calibration*. Analytica Chimica Acta 648(1), 77–84. Verified primary '
                          'link: '
                          '[https://doi.org/10.1016/j.aca.2009.06.046](https://doi.org/10.1016/j.aca.2009.06.046).'},
 'continuum_regression': {'implementation': 'The default eigensystem path implements this tau '
                                            'convention. The optional legacy NIPALS solver uses a '
                                            'different score-rescaling heuristic, so solver and '
                                            'tau must be reported together.',
                          'paper': 'Stone, M. & Brooks, R. J. (1990). *Continuum regression: '
                                   'Cross-validated sequentially constructed prediction embracing '
                                   'ordinary least squares, partial least squares and principal '
                                   'components regression*. Journal of the Royal Statistical '
                                   'Society: Series B 52(2), 237--269. DOI '
                                   '[10.1111/j.2517-6161.1990.tb01786.x](https://doi.org/10.1111/j.2517-6161.1990.tb01786.x).',
                          'principle': 'For the current parameter tau, each direction is '
                                       'proportional to $(\\mathbf X^T\\mathbf X)^{-\\tau}\\mathbf '
                                       'X^T\\mathbf y$, equivalently emphasizing squared '
                                       'covariance with a $\\operatorname{Var}(\\mathbf '
                                       'Xw)^{-2\\tau}$ factor. Thus tau=0 is PLS-like and, at full '
                                       'rank, tau=1 is OLS-like. The implementation has no PCR '
                                       'endpoint.'},
 'cppls': {'implementation': 'The default Canonical Powered PLS path records gamma but does not '
                             'currently use it in component extraction. Only the explicitly '
                             'selected legacy SIMPLS branch applies gamma as a '
                             'column-standard-deviation power rescaling; it still does not '
                             'implement the formerly documented closed form.',
           'paper': 'Indahl, U. G., Liland, K. H. & Næs, T. (2009). *Canonical partial least '
                    'squares---a unified PLS approach to classification and regression problems*. '
                    'Journal of Chemometrics 23(9--10), 495--504. DOI '
                    '[10.1002/cem.1243](https://doi.org/10.1002/cem.1243).',
           'principle': 'Canonical Powered PLS combines PLS covariance information with a '
                        'canonical-correlation criterion so that multivariate or class-coded '
                        'responses can guide a common latent space. In the current default '
                        'implementation, components are extracted from the centred X--Y '
                        'cross-product and X is deflated after each score. This is the CPPLS '
                        'family implemented by R `pls::cppls`, rather than the unrelated 2005 '
                        '$\\operatorname{sign}(X^Ty)|X^Ty|^\\gamma$ formula formerly shown here.',
           'title': 'Canonical Powered PLS (CPPLS)'},
 'di_pls': {'paper': 'Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S. & Moser, B. A. '
                     '(2018). *Domain-invariant partial-least-squares regression*. Analytical '
                     'Chemistry 90(11), 6693–6701. Verified primary link: '
                     '[https://doi.org/10.1021/acs.analchem.8b00498](https://doi.org/10.1021/acs.analchem.8b00498).'},
 'ds': {'paper': 'Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). *Multivariate instrument '
                 'standardisation*. Analytical Chemistry 63(23), 2750–2756. Verified primary link: '
                 '[https://doi.org/10.1021/ac00023a016](https://doi.org/10.1021/ac00023a016).'},
 'ecr': {'implementation': 'The earlier scalar formula mixing $w^TX^TXw$ with $w^TX^TY$ was '
                           'dimensionally incomplete for multivariate Y. The matrix H above is the '
                           'authoritative implemented criterion.',
         'paper': 'Li, H.-D., Liang, Y.-Z. & Xu, Q.-S. (2010). *Uncover the path from PCR to PLS '
                  'via elastic component regression*. Chemometrics and Intelligent Laboratory '
                  'Systems 104(2), 341--346. DOI '
                  '[10.1016/j.chemolab.2010.08.003](https://doi.org/10.1016/j.chemolab.2010.08.003).',
         'principle': 'For each component the implementation forms $\\mathbf H=(1-\\alpha)\\mathbf '
                      'X^T\\mathbf X+\\alpha(\\mathbf X^T\\mathbf Y)(\\mathbf X^T\\mathbf Y)^T$, '
                      'takes its dominant eigenvector, computes the corresponding scores and '
                      'loadings, and sequentially deflates X and Y. alpha therefore interpolates '
                      'the implemented variance and squared-covariance criteria.'},
 'fused_sparse_pls': {'implementation': '`l1_lambda` shrinks the full predictive coefficient matrix '
                                        'as one group before four neighbour-averaging passes. '
                                        '`fusion_lambda` controls a smoothing heuristic, not a '
                                        'converged fused-lasso solution.',
                      'paper': 'Implementation-specific heuristic inspired by fused sparsity; no '
                               'canonical fused-lasso solver is implemented.',
                      'principle': 'The routine delegates to the current group-sparse path with '
                                   'all variables in one group and then applies four fixed passes '
                                   'of neighbour averaging to adjacent coefficient rows. It '
                                   'neither minimizes an L1 plus total-variation objective nor '
                                   'calls the Condat fused-lasso algorithm described in the '
                                   'historical text.'},
 'ga_select': {'paper': 'Leardi, R. (2000). *Application of genetic algorithm–PLS for feature '
                        'selection in spectral data sets*. Journal of Chemometrics 14(5–6), '
                        '643–655. Verified primary link: '
                        '[https://doi.org/10.1002/1099-128X(200009/12)14:5/6%3C643::AID-CEM621%3E3.0.CO;2-E](https://doi.org/10.1002/1099-128X(200009/12)14:5/6%3C643::AID-CEM621%3E3.0.CO;2-E).'},
 'gpr_pls': {'implementation': 'No marginal-likelihood search or L-BFGS hyperparameter '
                               'optimization is run. The seed is currently unused, so the '
                               'documented fit is deterministic for fixed inputs and parameters.',
             'paper': 'Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §6.4 '
                      '(Gaussian Processes). — combined with a preliminary PLS dimensionality '
                      'reduction for spectroscopy. Verified primary link: '
                      '[https://link.springer.com/book/9780387310732](https://link.springer.com/book/9780387310732).',
             'principle': 'The method first obtains PLS scores and performs Gaussian-process '
                          'regression in that reduced space with a unit-amplitude kernel, '
                          'caller-supplied length scale, and caller-supplied diagonal noise level. '
                          'Posterior means follow the usual kernel solve for those fixed '
                          'hyperparameters.'},
 'group_sparse_pls': {'implementation': '`group_lambda` applies group-lasso proximal shrinkage '
                                        'to the returned SIMPLS predictive coefficients. This '
                                        'changes predictions but is not a latent-direction refit '
                                        'or a numerical implementation of sgPLS::gPLS.',
                      'paper': 'Liquet, B., de Micheaux, P. L., Hejblum, B. P. & Thiébaut, R. '
                               '(2016). *Group and sparse group partial least squares approaches '
                               'applied in genomics context*. Bioinformatics 32(1), 35–42. '
                               'Verified primary link: '
                               '[https://doi.org/10.1093/bioinformatics/btv535](https://doi.org/10.1093/bioinformatics/btv535).',
                      'principle': 'The routine fits ordinary SIMPLS, then applies the proximal '
                                   'map `B_g <- max(0, 1 - group_lambda / ||B_g||_F) B_g` '
                                   'to each group of predictive coefficient rows. Entire groups '
                                   'can be zeroed, but latent directions are not refitted.'},
 'interval_select': {'paper': 'Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., Munck, L. & '
                              'Engelsen, S. B. (2000). *Interval partial least-squares regression '
                              '(iPLS): a comparative chemometric study with an example from '
                              'near-infrared spectroscopy*. Applied Spectroscopy 54(3), 413–419. '
                              'Verified primary link: '
                              '[https://doi.org/10.1366/0003702001949500](https://doi.org/10.1366/0003702001949500).'},
 'irf_select': {'implementation': '`cpp/src/core/irf_selection.cpp` contains no decision trees or '
                                  'random-forest feature importance. Interval width, proposal '
                                  'count, seed, and chain mixing determine the frequency '
                                  'estimates.',
                'paper': 'Yun, Y.-H. et al. (2013). *An efficient method of wavelength interval '
                         'selection based on random frog for multivariate spectral calibration*. '
                         'Spectrochimica Acta Part A 111, 31--36. DOI '
                         '[10.1016/j.saa.2013.03.083](https://doi.org/10.1016/j.saa.2013.03.083). '
                         'The shipped libPLS-compatible path may differ in details; it is not the '
                         'random-forest IRF of Basu et al.',
                'principle': 'IRF partitions the ordered variables into sliding intervals, uses '
                             'absolute PLS coefficients to score them, and proposes random changes '
                             'to the active interval subset. Cross-validated RMSE accepts or '
                             'rejects proposals and selection frequencies summarize the sampled '
                             'chain.',
                'title': 'Interval Random Frog (IRF) selection'},
 'iriv_select': {'paper': 'Yun, Y. H., Wang, W. T., Tan, M. L., Liang, Y. Z., Li, H. D., Cao, D. '
                          'S., Lu, H. M. & Xu, Q. S. (2014). *A strategy that iteratively retains '
                          'informative variables for selecting optimal variable subset in '
                          'multivariate calibration*. Analytica Chimica Acta 807, 36–43. Verified '
                          'primary link: '
                          '[https://doi.org/10.1016/j.aca.2013.11.032](https://doi.org/10.1016/j.aca.2013.11.032).'},
 'kernel_pls_rbf': {'implementation': 'Out-of-sample prediction is implemented and requires the '
                                      'stored training predictors. Its memory is $O(n_{train}^2)$ '
                                      'for the Gram matrix and prediction kernel work is '
                                      '$O(n_{train}n_{new})$.',
                    'paper': 'Rosipal, R. & Trejo, L. J. (2001). *Kernel partial least squares '
                             'regression in reproducing kernel Hilbert space*. Journal of Machine '
                             'Learning Research 2, 97–123. Verified primary link: '
                             '[https://www.jmlr.org/papers/v2/rosipal01a.html](https://www.jmlr.org/papers/v2/rosipal01a.html).',
                    'principle': 'Kernel PLS extracts response-oriented components from a centred '
                                 'training Gram matrix. For new rows the implementation evaluates '
                                 'their kernel against every stored training row, centres that '
                                 'cross-kernel with the saved training row and grand means, and '
                                 'applies the learned dual coefficients. RBF, polynomial, and '
                                 'sigmoid kernels are available through the kernel configuration.'},
 'lw_pls': {'paper': 'Centner, V. & Massart, D. L. (1998). *Optimization in locally weighted '
                     'regression*. Analytical Chemistry 70(19), 4206--4211. DOI '
                     '[10.1021/ac980208r](https://doi.org/10.1021/ac980208r).'},
 'mb_pls': {'paper': 'Westerhuis, J. A., Kourti, T. & MacGregor, J. F. (1998). *Analysis of '
                     'multiblock and hierarchical PCA and PLS models*. Journal of Chemometrics '
                     '12(5), 301–321. Verified primary link: '
                     '[https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5%3C301::AID-CEM515%3E3.0.CO;2-S](https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5%3C301::AID-CEM515%3E3.0.CO;2-S).'},
 'mir_pls': {'implementation': 'The pseudoinverse regularizer is fixed at $10^{-6}$. Treat this as '
                               'a compact library-specific variant and validate it directly rather '
                               'than attributing properties of unrelated MIR or OSC algorithms.',
             'paper': 'Implementation-specific inverse-regression construction; no canonical paper '
                      'is claimed.',
             'principle': 'This routine fits a SIMPLS mapping from Y to X and then converts that '
                          'map into a predictor from X to Y with the regularized right '
                          'pseudoinverse $\\mathbf A^T(\\mathbf A\\mathbf A^T+10^{-6}\\mathbf '
                          'I)^{-1}$. The name does not denote a fixed mid-infrared absorbance '
                          'transform or OSC procedure.',
             'title': 'MIR-PLS inverse-regression variant'},
 'missing_aware_nipals': {'paper': 'Walczak, B. & Massart, D. L. (2001). *Dealing with missing '
                                   'data: Part I and Part II*. Chemometrics and Intelligent '
                                   'Laboratory Systems 58(1), 15--27 and 29--42. DOIs '
                                   '[10.1016/S0169-7439(01)00131-9](https://doi.org/10.1016/S0169-7439(01)00131-9) '
                                   'and '
                                   '[10.1016/S0169-7439(01)00132-0](https://doi.org/10.1016/S0169-7439(01)00132-0).'},
 'n_pls': {'paper': 'Bro, R. (1996). *Multiway calibration. Multilinear PLS*. Journal of '
                    'Chemometrics 10(1), 47–61. Verified primary link: '
                    '[https://doi.org/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C](https://doi.org/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C).'},
 'o2pls': {'paper': 'Trygg, J. & Wold, S. (2003). *O2-PLS, a two-block (X–Y) latent variable '
                    'regression method with an integral OSC filter*. Journal of Chemometrics '
                    '17(1), 53–64. Verified primary link: '
                    '[https://doi.org/10.1002/cem.775](https://doi.org/10.1002/cem.775).'},
 'on_pls': {'paper': 'Löfstedt, T. & Trygg, J. (2011). *OnPLS — a novel multiblock method for the '
                     'modelling of predictive and orthogonal variation*. Journal of Chemometrics '
                     '25(8), 441–455. Verified primary link: '
                     '[https://doi.org/10.1002/cem.1388](https://doi.org/10.1002/cem.1388).'},
 'one_se_rule': {'paper': 'Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of '
                          'Statistical Learning*, 2nd ed., Springer, §7.10. Verified primary link: '
                          '[https://doi.org/10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7).'},
 'opls': {'paper': 'Trygg, J. & Wold, S. (2002). *Orthogonal projections to latent structures '
                   '(O-PLS)*. Journal of Chemometrics 16(3), 119--128. DOI '
                   '[10.1002/cem.695](https://doi.org/10.1002/cem.695).'},
 'pcr': {'implementation': "The current PCR path uses the library's SVD/eigendecomposition "
                           'machinery and returns a linear coefficient matrix with training means '
                           'for prediction. The retained singular values must be nonzero.',
         'paper': 'Massy, W. F. (1965). *Principal components regression in exploratory '
                  'statistical research*. Journal of the American Statistical Association 60(309), '
                  '234--256. DOI '
                  '[10.1080/01621459.1965.10480787](https://doi.org/10.1080/01621459.1965.10480787).',
         'principle': 'PCR first computes a truncated singular-value decomposition of the centred '
                      'predictor matrix, $\\mathbf X=\\mathbf U\\boldsymbol\\Sigma\\mathbf '
                      'V^\\top$, and regresses Y on the first $k$ scores $\\mathbf T_k=\\mathbf '
                      'U_k\\boldsymbol\\Sigma_k$. The coefficient matrix is therefore $\\mathbf '
                      'B=\\mathbf V_k\\boldsymbol\\Sigma_k^{-1}\\mathbf U_k^\\top\\mathbf '
                      'Y=\\mathbf V_k\\boldsymbol\\Sigma_k^{-2}\\mathbf T_k^\\top\\mathbf Y$. '
                      'Components are ordered by predictor variance without using Y, which is the '
                      'defining contrast with PLS.'},
 'pls': {'paper': 'de Jong, S. (1993). *SIMPLS: an alternative approach to partial least squares '
                  'regression*. Chemometrics and Intelligent Laboratory Systems 18(3), 251--263. '
                  'DOI '
                  '[10.1016/0169-7439(93)85002-X](https://doi.org/10.1016/0169-7439(93)85002-X).'},
 'pls_cox': {'implementation': 'No Cox partial likelihood or deviance-residual iteration is '
                               'optimized. The mean imputation for censored times can bias risk '
                               'scores; treat results as an exploratory library-specific '
                               'approximation.',
             'paper': 'Implementation-specific approximation inspired by PLS survival modelling; '
                      'it is not a Cox partial-likelihood estimator.',
             'principle': 'The implementation takes log event times, replaces each censored '
                          'log-time with the mean log-time among events, fits SIMPLS to that '
                          'pseudo-response, and negates its coefficients to form a risk score. A '
                          'Breslow-like cumulative hazard is then assembled from event times and '
                          'those scores.',
             'title': 'PLS survival pseudo-response approximation'},
 'pls_diagnostic_dmodx': {'implementation': 'Valid degrees of freedom require p>a and, for '
                                            'reference scaling, n_ref>a+1. With no reference '
                                            'residuals the implementation sets sigma to one rather '
                                            'than estimating a model-population scale.',
                          'paper': 'Eriksson, L., Byrne, T., Johansson, E., Trygg, J. & Vikström, '
                                   'C. (2013). *Multi- and Megavariate Data Analysis. Basic '
                                   'Principles and Applications*, 3rd ed., Umetrics Academy, §4.7. '
                                   'Verified primary catalogue: '
                                   '[https://search.worldcat.org/title/891362497](https://search.worldcat.org/title/891362497).',
                          'principle': "Let Q be an observation's squared X reconstruction "
                                       'residual and let a be the number of components. The '
                                       'implementation reports $\\sqrt{Q/(p-a)}/\\sigma$, where '
                                       '$\\sigma=\\sqrt{\\sum Q_{ref}/((p-a)(n_{ref}-a-1))}$ when '
                                       'a reference set is supplied.'},
 'pls_diagnostic_q': {'paper': 'Jackson, J. E. & Mudholkar, G. S. (1979). *Control procedures for '
                               'residuals associated with principal component analysis*. '
                               'Technometrics 21(3), 341–349. Verified primary link: '
                               '[https://doi.org/10.1080/00401706.1979.10489779](https://doi.org/10.1080/00401706.1979.10489779).'},
 'pls_diagnostic_t2': {'paper': "Hotelling, H. (1931). *The generalization of Student's ratio*. "
                                'Annals of Mathematical Statistics 2(3), 360--378. DOI '
                                '[10.1214/aoms/1177732979](https://doi.org/10.1214/aoms/1177732979).'},
 'pls_glm': {'implementation': 'Outputs must be interpreted as linear PLS predictions. In '
                               'particular, they are not constrained positive and are not fitted '
                               'by a Poisson likelihood; use of this entry point for count '
                               'inference would be misleading.',
             'paper': 'No canonical paper validates the shipped SIMPLS compatibility path as a '
                      'Poisson/generalized-linear model; it is not attributed to a GLM solver.',
             'principle': 'The shipped routine centres Y once and fits ordinary SIMPLS. It does '
                          'not construct GLM working responses, apply an inverse link, or iterate '
                          'reweighted least squares. The stored Poisson flag does not alter the '
                          'numerical fit.',
             'title': 'PLS-GLM compatibility entry point'},
 'pls_lda': {'paper': 'Barker, M. & Rayens, W. (2003). *Partial least squares for discrimination*. '
                      'Journal of Chemometrics 17(3), 166--173. DOI '
                      '[10.1002/cem.785](https://doi.org/10.1002/cem.785).'},
 'pls_logistic': {'implementation': 'Probabilities are produced by the multinomial softmax head in '
                                    '`cpp/src/core/pls_logistic.cpp`. Component count, '
                                    'regularization, and convergence settings all affect '
                                    'calibration.',
                  'paper': 'Bastien, P., Esposito Vinzi, V. & Tenenhaus, M. (2005). *PLS '
                           'generalised linear regression*. Computational Statistics & Data '
                           'Analysis 48(1), 17–46. Verified primary link: '
                           '[https://doi.org/10.1016/j.csda.2004.02.005](https://doi.org/10.1016/j.csda.2004.02.005).',
                  'principle': 'The implementation extracts a fixed PLS score matrix, then fits a '
                               'ridge-regularized multinomial logistic model on those scores with '
                               'Newton updates and a softmax likelihood. Unlike iterative PLS-GLR '
                               'variants, it does not recompute PLS directions from an IRLS '
                               'working response.'},
 'pls_monitoring': {'implementation': 'No parametric F approximation for T-squared and no '
                                      'Jackson--Mudholkar Q approximation is evaluated in this '
                                      'path. Reference-set size and representativeness directly '
                                      'determine the empirical limits.',
                    'paper': 'Kourti, T. & MacGregor, J. F. (1996). *Multivariate SPC methods for '
                             'process and product monitoring and control*. Journal of Quality '
                             'Technology 28(4), 409–428. Verified primary link: '
                             '[https://doi.org/10.1080/00224065.1996.11979699](https://doi.org/10.1080/00224065.1996.11979699).',
                    'principle': 'The routine computes T-squared and Q statistics for the '
                                 'reference observations, sorts each empirical distribution, and '
                                 'chooses its $(1-\\alpha)$ sample quantile as the corresponding '
                                 'control limit. New observations are flagged when either '
                                 'statistic exceeds that stored limit.'},
 'pls_qda': {'implementation': 'Current priors are observed class proportions; there is no '
                               'uniform-prior or caller-supplied-prior option on this path. '
                               'Covariance regularization remains necessary when class samples are '
                               'few.',
             'paper': 'Pérez-Enciso, M. & Tenenhaus, M. (2003). *Prediction of clinical outcome '
                      'with microarray data: a partial least squares discriminant analysis '
                      '(PLS-DA) approach*. Human Genetics 112, 581--592. DOI '
                      '[10.1007/s00439-003-0921-9](https://doi.org/10.1007/s00439-003-0921-9).',
             'principle': 'PLS first supplies a low-dimensional supervised score space. QDA then '
                          'estimates a separate mean and regularized covariance for each class and '
                          'compares Gaussian discriminant scores including the empirical log class '
                          'prior.'},
 'pso_select': {'paper': 'Kennedy, J. & Eberhart, R. (1995). *Particle swarm optimization*. IEEE '
                         'ICNN 1995, vol. 4, 1942–1948. — binary PSO variant used for variable '
                         'selection. Verified primary link: '
                         '[https://doi.org/10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).'},
 'random_frog_select': {'paper': 'Li, H., Xu, Q. & Liang, Y. (2012). *Random frog: an efficient '
                                 'reversible jump Markov chain Monte Carlo-like approach for '
                                 'variable selection*. Analytica Chimica Acta 740, 20–26. Verified '
                                 'primary link: '
                                 '[https://doi.org/10.1016/j.aca.2012.06.031](https://doi.org/10.1016/j.aca.2012.06.031).'},
 'random_subspace_pls': {'paper': 'Ho, T. K. (1998). *The random subspace method for constructing '
                                  'decision forests*. IEEE TPAMI 20(8), 832–844. — adapted for PLS '
                                  'regressors. Verified primary link: '
                                  '[https://doi.org/10.1109/34.709601](https://doi.org/10.1109/34.709601).'},
 'randomization_select': {'implementation': 'The selector applies the unadjusted per-variable '
                                            'cutoff `p <= alpha`. False-discovery-rate control '
                                            'such as Benjamini--Hochberg is scientifically '
                                            'advisable for many simultaneous tests but is not '
                                            'implemented here.',
                          'paper': 'Westad, F. & Martens, H. (2000). *Variable selection in near '
                                   'infrared spectroscopy based on significance testing in partial '
                                   'least squares regression*. JNIRS 8(2), 117–124. Verified '
                                   'primary link: '
                                   '[https://doi.org/10.1255/jnirs.271](https://doi.org/10.1255/jnirs.271).',
                          'principle': 'For each variable, response permutations form a null '
                                       'distribution of its association score. The Monte Carlo '
                                       'p-value includes the usual finite-permutation correction, '
                                       'and the implementation retains variables whose p-value is '
                                       'less than or equal to alpha.'},
 'recursive_pls': {'paper': 'Helland, K., Berntsen, H. E., Borgen, O. S. & Martens, H. (1992). '
                            '*Recursive algorithm for partial least squares regression*. '
                            'Chemometrics and Intelligent Laboratory Systems 14(1--3), 129--137. '
                            'DOI '
                            '[10.1016/0169-7439(92)80098-O](https://doi.org/10.1016/0169-7439(92)80098-O).'},
 'ridge_pls': {'implementation': 'This augmented-data construction is not equivalent to applying '
                                 '$(\\mathbf T^T\\mathbf T+\\lambda\\mathbf I)^{-1}\\mathbf '
                                 'T^T\\mathbf Y$ after an unchanged PLS score extraction. lambda '
                                 'must be tuned for this implementation.',
               'paper': 'Hoerl, A. E. & Kennard, R. W. (1970). *Ridge regression: biased '
                        'estimation for nonorthogonal problems*. Technometrics 12(1), 55–67. — '
                        'combined with PLS via Tikhonov regularisation of the inner regression. '
                        'Verified primary link: '
                        '[https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).',
               'principle': 'The current method augments the centred training problem with '
                            '$\\sqrt{\\lambda}\\mathbf I_p$ rows in X and matching zero rows in Y, '
                            'then fits PLS to that augmented problem. Consequently the penalty can '
                            'alter the latent directions as well as shrink the final regression '
                            'coefficients.'},
 'robust_pls': {'implementation': 'In the default path the public `huber_k` value is interpreted '
                                  'as the Fair tuning constant for parity with the referenced '
                                  'implementation. The separately gated legacy branch performs the '
                                  'Huber IRLS algorithm described by older documentation.',
                'paper': 'Serneels, S., Croux, C., Filzmoser, P. & Van Espen, P. J. (2005). '
                         '*Partial robust M-regression*. Chemometrics and Intelligent Laboratory '
                         'Systems 79(1--2), 55--64. DOI '
                         '[10.1016/j.chemolab.2005.04.007](https://doi.org/10.1016/j.chemolab.2005.04.007).',
                'principle': 'The default robust path follows a partial robust M-regression style '
                             'iteration: residual and leverage information update Fair robustness '
                             'weights before weighted latent fits. Fair weights decay smoothly '
                             'rather than clipping residual influence at the Huber threshold.'},
 'rosa': {'paper': 'Liland, K. H., Næs, T. & Indahl, U. G. (2016). *ROSA---a fast extension of '
                   'partial least squares regression for multiblock data analysis*. Journal of '
                   'Chemometrics 30(11), 651--662. DOI '
                   '[10.1002/cem.2824](https://doi.org/10.1002/cem.2824).'},
 'scars_select': {'paper': 'Zheng, K., Li, Q., Wang, J., Geng, J., Cao, P., Sui, T., Wang, X. & '
                           'Du, Y. (2012). *Stability competitive adaptive reweighted sampling '
                           '(SCARS) and its applications to multivariate calibration of NIR '
                           'spectra*. Chemometrics and Intelligent Laboratory Systems 112, 48–54. '
                           'Verified primary link: '
                           '[https://doi.org/10.1016/j.chemolab.2012.01.002](https://doi.org/10.1016/j.chemolab.2012.01.002).'},
 'shaving_select': {'paper': 'Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). *A review '
                             'of variable selection methods in partial least squares regression*. '
                             'Chemometrics and Intelligent Laboratory Systems 118, 62–69 (§3.2 '
                             'Shaving). Verified primary link: '
                             '[https://doi.org/10.1016/j.chemolab.2012.07.010](https://doi.org/10.1016/j.chemolab.2012.07.010).'},
 'sipls_select': {'paper': 'Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., Munck, L. & '
                           'Engelsen, S. B. (2000). *Interval partial least-squares regression '
                           '(iPLS): a comparative chemometric study with an example from '
                           'near-infrared spectroscopy*. Applied Spectroscopy 54(3), 413–419 — '
                           'same paper as `interval_select`; siPLS is the synergy-combinations '
                           'extension proposed in §3. Verified primary link: '
                           '[https://doi.org/10.1366/0003702001949500](https://doi.org/10.1366/0003702001949500).'},
 'so_pls': {'paper': 'Næs, T., Tomic, O., Mevik, B.-H. & Martens, H. (2011). *Path modelling by '
                     'sequential PLS regression*. Journal of Chemometrics 25(1), 28–40. Verified '
                     'primary link: '
                     '[https://doi.org/10.1002/cem.1357](https://doi.org/10.1002/cem.1357).'},
 'spa_select': {'paper': 'Araújo, M. C. U., Saldanha, T. C. B., Galvão, R. K. H., Yoneyama, T., '
                         'Chame, H. C. & Visani, V. (2001). *The successive projections algorithm '
                         'for variable selection in spectroscopic multicomponent analysis*. '
                         'Chemometrics and Intelligent Laboratory Systems 57(2), 65–73. Verified '
                         'primary link: '
                         '[https://doi.org/10.1016/S0169-7439(01)00119-8](https://doi.org/10.1016/S0169-7439(01)00119-8).'},
 'sparse_pls_da': {'implementation': 'The C result wrapper performs argmax over class scores and '
                                     'emits zero/one entries. Probability calibration, softmax '
                                     'conversion, and rejection thresholds are not part of this '
                                     'method.',
                   'paper': 'Lê Cao, K.-A., Rossouw, D., Robert-Granié, C. & Besse, P. (2008). *A '
                            'sparse PLS for variable selection when integrating omics data*. '
                            'Statistical Applications in Genetics and Molecular Biology 7(1). '
                            'Verified primary link: '
                            '[https://doi.org/10.2202/1544-6115.1390](https://doi.org/10.2202/1544-6115.1390).',
                   'principle': 'Class labels are encoded as one-hot response columns and a sparse '
                                'PLS regression model produces one score per class. Prediction '
                                'selects the largest score and writes a one-hot class vector. This '
                                'supplies a discriminant rule but does not normalize the scores '
                                'into probabilities.'},
 'sparse_simpls': {'implementation': 'The standard branch is implemented in '
                                     '`cpp/src/core/model.cpp`. Setting the legacy compatibility '
                                     'switch selects the older per-component absolute-threshold '
                                     'algorithm, so results from the two branches are not '
                                     'interchangeable.',
                   'paper': 'Chun, H. & Keleş, S. (2010). *Sparse partial least squares regression '
                            'for simultaneous dimension reduction and variable selection*. Journal '
                            'of the Royal Statistical Society: Series B 72(1), 3--25. DOI '
                            '[10.1111/j.1467-9868.2009.00723.x](https://doi.org/10.1111/j.1467-9868.2009.00723.x).',
                   'principle': 'The default sparse path follows the Chun--Keles/R `spls` pattern. '
                                'At each step a cross-covariance direction Z is soft-thresholded '
                                'at $\\eta\\max_j|Z_j|$; nonzero variables are accumulated, '
                                'ordinary SIMPLS is refitted on that active union, and Y is '
                                'deflated for the next component. The final model is therefore a '
                                'SIMPLS refit on the selected columns, not merely a permanently '
                                'thresholded weight vector.'},
 'stability_select': {'implementation': '`cpp/src/core/stability_selection.cpp` computes sample '
                                        'mean and sample standard deviation of each fitted '
                                        'coefficient over `plan.folds`, takes the maximum '
                                        'target-specific ratio, then stable-sorts it. It is not a '
                                        'bootstrap stability-selection implementation.',
                      'limitations': 'This routine returns exactly `top_k` features and supplies '
                                     'no selection frequency or error-control threshold. '
                                     'Correlated wavelengths can exchange signal across folds, '
                                     'inflating coefficient variation and changing the rank; the '
                                     'validation plan must be nested inside model assessment.',
                      'paper': 'Cai, W., Li, Y. & Shao, X. (2008). *A variable selection method '
                               'based on uninformative variable elimination for multivariate '
                               'calibration of near-infrared spectra*. Chemometrics and '
                               'Intelligent Laboratory Systems 90(2), 188–194. Verified primary '
                               'link: '
                               '[https://doi.org/10.1016/j.chemolab.2007.10.001](https://doi.org/10.1016/j.chemolab.2007.10.001).',
                      'principle': 'For every supplied validation fold, the routine fits the '
                                   "configured model on that fold's training rows and stores its "
                                   'coefficients. For each feature and target it calculates '
                                   '$|\\bar b|/s_b$; the feature score is the largest of those '
                                   'ratios across targets, and the `top_k` largest scores are '
                                   'returned. The supplied folds, rather than bootstrap resampling '
                                   'or a frequency threshold, define the repeats.',
                      'use_cases': 'Ranking a fixed number of wavelengths by coefficient '
                                   'signal-to-fold-variation when a validation plan is already '
                                   'available.'},
 't2_select': {'implementation': 'This is not the F control-limit plus top-k procedure formerly '
                                 'documented. Beta-quantile parameter validity and the fallback '
                                 'selected count depend on p and a; alpha selection and its CV '
                                 'must be performed inside an outer validation loop to avoid '
                                 'selection leakage.',
               'principle': 'The routine derives a variable statistic from PLS weight '
                            'contributions and compares it with a cutoff based on a Beta quantile '
                            '$B_{1-\\alpha}(a/2,(p-a-1)/2)$ and factor $(p-1)^2/p$. Variables '
                            'above the cutoff are selected; the highest statistics fill any '
                            '`min_selected` shortfall. It repeats this process for each supplied '
                            'alpha, evaluates a PLS cross-validation fit for every resulting '
                            'subset, and reports both `selected_indices_best_error` (smallest CV '
                            'RMSE) and `selected_indices_min_set` (fewest variables, then CV-RMSE '
                            'tie-break).'},
 'uve_select': {'paper': 'Centner, V., Massart, D. L., de Noord, O. E., de Jong, S., Vandeginste, '
                         'B. M. & Sterna, C. (1996). *Elimination of uninformative variables for '
                         'multivariate calibration*. Analytical Chemistry 68(21), 3851–3858. '
                         'Verified primary link: '
                         '[https://doi.org/10.1021/ac960321m](https://doi.org/10.1021/ac960321m).'},
 'variable_select_coef': {'paper': 'Martens, H. & Næs, T. (1989). *Multivariate Calibration*, §5. '
                                   '— the simplest ranking baseline. Verified primary catalogue: '
                                   '[https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282](https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282).'},
 'variable_select_sr': {'paper': 'Rajalahti, T., Arneberg, R., Berven, F. S., Myhr, K.-M., Ulvik, '
                                 'R. J. & Kvalheim, O. M. (2009). *Biomarker discovery in mass '
                                 'spectral profiles by means of selectivity ratio plot*. '
                                 'Chemometrics and Intelligent Laboratory Systems 95(1), 35–48. '
                                 'Verified primary link: '
                                 '[https://doi.org/10.1016/j.chemolab.2008.08.004](https://doi.org/10.1016/j.chemolab.2008.08.004).'},
 'variable_select_vip': {'paper': 'Wold, S., Sjöström, M. & Eriksson, L. (2001). *PLS-regression: '
                                  'a basic tool of chemometrics*. Chemometrics and Intelligent '
                                  'Laboratory Systems 58(2), 109–130. Verified primary link: '
                                  '[https://doi.org/10.1016/S0169-7439(01)00155-1](https://doi.org/10.1016/S0169-7439(01)00155-1).'},
 'vissa_select': {'paper': 'Deng, B. C., Yun, Y. H., Liang, Y. Z. & Yi, L. Z. (2015). *A new '
                           'strategy to prevent over-fitting in partial least squares models based '
                           'on model population analysis*. Analytica Chimica Acta 880, 32--41. DOI '
                           '[10.1016/j.aca.2015.04.045](https://doi.org/10.1016/j.aca.2015.04.045).'},
 'weighted_pls': {'implementation': 'The fit returns a replayable $p\\times q$ global coefficient '
                                    'matrix together with X and Y means, and the C API packs '
                                    'coefficients, means, and predictions for ordinary '
                                    'out-of-sample use. The historical statement that no global '
                                    'coefficient export is available is incorrect.',
                  'paper': 'Martens, H. & Næs, T. (1989). *Multivariate Calibration*. Wiley. §4.5 '
                           "'Weighted regression for non-i.i.d. errors'. Verified primary "
                           'catalogue: '
                           '[https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282](https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282).',
                  'principle': 'Weighted PLS applies nonnegative row weights when estimating '
                               "centres and the latent regression. This changes each observation's "
                               'contribution while retaining one global linear coefficient '
                               'matrix.'},
 'wvc_select': {'implementation': '`cpp/src/core/wvc_selection.cpp` implements the PLS1 '
                                  'weight/score/loading loop and its cumulative WVC formula. Treat '
                                  'the ranking as an implementation-specific component '
                                  'contribution; the threshold selector is a separate method with '
                                  'its own cutoff rule.',
                'paper': 'Andries, J. P. M. & Vander Heyden, Y. (2011). *Improved variable '
                         'reduction in partial least squares modelling based on '
                         'predictive-property-ranked variables and adaptation of partial least '
                         'squares complexity*. Analytica Chimica Acta 705(1–2), 292–305. Verified '
                         'primary link: '
                         '[https://doi.org/10.1016/j.aca.2011.06.037](https://doi.org/10.1016/j.aca.2011.06.037).',
                'principle': 'The code runs sequential PLS1/NIPALS: for each component it '
                             'normalizes $w=X^\\top y$, forms $t=Xw$, estimates loadings, and '
                             'accumulates score/residual energy terms. For feature $j$, WVC is '
                             '$\\sqrt{p\\sum_a ee_a w_{ja}^2 ss_a / \\sum_a ee_a ss_a}$, '
                             'optionally normalized before returning the `top_k` scores. No '
                             'singular-value decomposition or cross-validation occurs in this '
                             'selector.'},
 'wvc_threshold_select': {'implementation': 'The routine does not compare two thresholding rules '
                                            'by cross-validation. Any predictive validation of the '
                                            'chosen cutoff must be performed by the caller.',
                          'principle': 'After WVC scores are computed, the implementation uses one '
                                       'cutoff: the larger of `score_threshold` and '
                                       '`threshold_factor` times the mean score. Variables meeting '
                                       'it are retained, with the largest scores added if fewer '
                                       'than `min_selected` survive.'}}

_FIELDS = (
    "title",
    "paper",
    "principle",
    "use_cases",
    "limitations",
    "implementation",
    "provenance",
)

def _build_content() -> dict[str, dict[str, str]]:
    if set(_ANNOTATIONS) != set(_LEGACY):
        missing = sorted(set(_LEGACY) - set(_ANNOTATIONS))
        extra = sorted(set(_ANNOTATIONS) - set(_LEGACY))
        raise RuntimeError(
            f"scientific legacy key mismatch: missing={missing}, extra={extra}"
        )

    content: dict[str, dict[str, str]] = {}
    for stem, historical in _LEGACY.items():
        use_cases, limitations, provenance = _ANNOTATIONS[stem]
        entry = {
            "title": str(historical.get("title") or stem.replace("_", " ").title()),
            "paper": str(
                historical.get("paper") or "No single canonical paper identified."
            ),
            "principle": str(
                historical.get("principle")
                or "See the verified implementation provenance."
            ),
            "use_cases": use_cases,
            "limitations": limitations,
            "implementation": str(
                historical.get("implementation")
                or "See the verified implementation provenance."
            ),
            "provenance": provenance,
        }
        # One explicit key per method; no duplicate dictionary literals or
        # post-merge layers can silently discard a scientific correction.
        for field, value in _OVERRIDES.get(stem, {}).items():
            entry[field] = value
        if tuple(entry) != _FIELDS or any(
            not value.strip() for value in entry.values()
        ):
            raise RuntimeError(f"invalid scientific legacy record: {stem}")
        content[stem] = entry
    return content


SCIENTIFIC_CONTENT = _build_content()

# Compatibility name for the integration branch while the generator migrates
# from overlays to uniform seven-field content modules.
SCIENTIFIC_OVERLAY = SCIENTIFIC_CONTENT
