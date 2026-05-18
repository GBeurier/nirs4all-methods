"""One-line descriptions for every parameter exposed by the chemometrics4all
sklearn-style wrappers. Consumed by `docs/_extras/build_methods.py` to fill
the `Notes` column of the generated method-page parameter tables.

Style:
  * One concise sentence per parameter. Long mathematical explanations
    belong in the `principle` block of `methods_bibliography.py`, not here.
  * Generic descriptions live in `PARAM_DOCS`. When a parameter name has
    distinct meanings across methods (e.g. `gamma` for the kernel-PLS RBF
    width vs CPPLS' tuning power, `alpha` for elastic-net mixing vs a
    significance level), add a method-specific override in
    `METHOD_PARAM_OVERRIDES`.
"""

from __future__ import annotations


# Generic, method-agnostic parameter descriptions.
PARAM_DOCS: dict[str, str] = {
    # ---- Shared PLS core knobs (libc4a Config) ----
    "n_components": "Number of latent components extracted (k).",
    "solver": "Inner algorithm: 'nipals', 'simpls', 'svd', 'kernel', 'orthogonal-scores', 'power', 'randomized-svd', 'wide-kernel'.",
    "center_x": "Subtract the column mean of X before fitting.",
    "scale_x": "Standardize X columns to unit variance before fitting.",
    "center_y": "Subtract the column mean of y before fitting.",
    "scale_y": "Standardize y columns to unit variance before fitting.",
    "tol": "Convergence tolerance for iterative solvers (NIPALS / power-iteration).",
    "max_iter": "Maximum iterations for iterative solvers.",
    "store_scores": "If True, keep the latent score matrix (`x_scores_`) after fit.",

    # ---- Selector framework ----
    "top_k": "Number of features to retain.",
    "n_folds": "Number of cross-validation folds used inside the selector.",
    "seed": "Random seed for reproducible sampling/initialization.",
    "n_iterations": "Number of selection iterations or Monte-Carlo passes.",
    "min_features": "Minimum number of variables the selector is allowed to keep (defaults to `n_components`).",
    "min_selected": "Lower bound on the surviving feature count after thresholding.",
    "normalize": "Normalize per-variable scores to sum to one before ranking.",

    # ---- Ensembles ----
    "n_estimators": "Number of base PLS sub-models in the ensemble.",
    "learning_rate": "Boosting shrinkage applied to each successive base learner.",
    "features_per_subspace": "Number of features randomly drawn per random-subspace base learner.",

    # ---- Multi-block / 3-way ----
    "block_sizes": "Sequence of contiguous block widths defining the X-block partition (columns of X).",
    "n_components_per_block": "Per-block latent-component budget (one int per block).",
    "mode_j": "Length of the second mode (J) of the 3-way input tensor.",
    "mode_k": "Length of the third mode (K) of the 3-way input tensor.",
    "n_predictive": "Number of joint (predictive) components shared by X and Y in O2-PLS.",
    "n_x_orthogonal": "Number of X-orthogonal components (Y-irrelevant structure in X).",
    "n_y_orthogonal": "Number of Y-orthogonal components (X-irrelevant structure in Y).",

    # ---- Robust / regularised / sparse PLS ----
    "huber_k": "Huber threshold (in residual-stdev units) controlling IRLS reweighting; smaller = more robust.",
    "max_irls_iter": "Maximum IRLS reweighting iterations.",
    "ridge_lambda": "L2 (ridge) penalty added to the SIMPLS augmented system.",
    "tau": "Continuum mixing parameter in [0, 1]; 0 ≈ PLS, 1 ≈ whitened OLS / PCR-like.",
    "sparsity_lambda": "L1 soft-threshold magnitude applied to the PLS weight vectors.",
    "l1_lambda": "L1 sparsity penalty for fused-sparse PLS.",
    "fusion_lambda": "Fused-lasso penalty enforcing smoothness between adjacent coefficients.",
    "group_assignment": "Integer array assigning each feature to a group (length n_features).",
    "group_lambda": "L1 penalty applied at the group level (group-sparse PLS).",
    "di_lambda": "Domain-invariance penalty weight balancing covariance alignment vs response fit.",

    # ---- Locally-weighted / nearest-neighbours ----
    "n_neighbors": "Number of training neighbours used for each local prediction (LW-PLS).",
    "window_size": "Length of the moving window for recursive / interval-random-frog models.",

    # ---- Kernels (kernel-PLS) ----
    "kernel_type": "Kernel family: 0=linear, 1=RBF, 2=polynomial, 3=sigmoid.",
    "coef0": "Independent term in polynomial / sigmoid kernels.",
    "degree": "Degree of the polynomial kernel (ignored otherwise).",

    # ---- GLM / Poisson ----
    "poisson": "If True, fit a Poisson-deviance PLS-GLM (default Gaussian link).",

    # ---- Random frog / interval-random-frog ----
    "initial_size": "Starting subset size for the random-frog chain.",
    "min_size": "Minimum allowed subset size during the random-frog Markov chain.",
    "max_size": "Maximum allowed subset size during the random-frog Markov chain.",
    "initial_intervals": "Number of seed intervals for the interval-random-frog walk.",

    # ---- SCARS / sample fraction ----
    "sample_fraction": "Fraction of samples drawn per Monte-Carlo replicate.",

    # ---- Genetic algorithm ----
    "n_generations": "Number of GA generations to evolve.",
    "population_size": "Number of candidate feature subsets per generation.",
    "max_features": "Upper bound on the GA chromosome cardinality (defaults to all features).",
    "mutation_rate": "Per-bit mutation probability applied to GA chromosomes.",

    # ---- Particle swarm ----
    "n_swarm": "Number of particles in the binary PSO swarm.",
    "w": "PSO inertia weight on the previous-velocity term.",
    "c1": "PSO cognitive (personal-best) acceleration coefficient.",
    "c2": "PSO social (global-best) acceleration coefficient.",
    "v_max": "Velocity clipping bound for binary PSO.",

    # ---- VISSA ----
    "n_submodels": "Number of bootstrap sub-models drawn per VISSA iteration.",
    "ratio_kept": "Fraction of top-scoring features retained at each VISSA shrinkage step.",
    "threshold": "Inclusion-probability cut-off below which features are dropped.",
    "floor_probability": "Lower bound applied to per-feature inclusion probabilities to avoid premature pruning.",

    # ---- Shaving / BVE / REP ----
    "n_steps": "Number of elimination passes performed.",
    "shave_fraction": "Fraction of the worst-ranked variables removed at each shaving step.",
    "remove_count": "Number of variables removed per REP step.",

    # ---- IPW ----
    "damping": "Exponential moving-average factor mixing previous and current weights in IPW.",
    "weight_floor": "Lower bound applied to per-feature weights to prevent zero-trapping.",

    # ---- ST / T² / WVC thresholding ----
    "thresholds": "Sequence of soft-threshold values to sweep; the most aggressive surviving subset is kept.",
    "alpha_thresholds": "Sequence of significance levels (α) defining the T² acceptance regions to sweep.",
    "score_threshold": "Absolute lower bound on WVC scores for retention.",
    "threshold_factor": "Multiplier applied to the mean WVC score to define the dynamic threshold.",

    # ---- Interval selectors ----
    "interval_width": "Width (in variables) of each contiguous spectral interval.",
    "step": "Stride between consecutive forward-iPLS intervals.",
    "min_intervals": "Minimum number of intervals retained by biPLS backward elimination.",
    "combination_size": "Number of intervals combined into the synergistic siPLS subset.",

    # ---- UVE / EMCUVE ----
    "noise_features": "Number of artificial noise variables appended to X for the UVE threshold.",
    "noise_seed": "Seed for the appended noise variables.",
    "n_ensembles": "Number of UVE replicates aggregated by majority vote.",
    "vote_threshold": "Minimum vote fraction required to retain a variable in EMCUVE.",

    # ---- Randomization ----
    "n_permutations": "Number of Y-permutations used to build the null distribution.",
    "randomization_seed": "Seed for the permutation generator.",

    # ---- IRIV ----
    "max_rounds": "Maximum rounds of strongly/weakly informative reclassification.",

    # ---- VIPSPA ----
    "vip_threshold": "Minimum VIP score required to enter the SPA candidate pool.",

    # ---- Calibration transfer ----
    "window_half_width": "Half-width (in channels) of the PDS local regression window.",
}


# Method-specific overrides for parameters whose meaning is context-dependent.
METHOD_PARAM_OVERRIDES: dict[tuple[str, str], str] = {
    # CPPLS: γ in Indahl's twist controls the trade-off between covariance
    # (γ→0) and correlation (γ→1) when selecting loading weights.
    ("cppls", "gamma"): "Covariance/correlation mixing exponent (0 → covariance-maximizing PLS, 1 → correlation-maximizing).",
    # Kernel PLS RBF: kernel bandwidth.
    ("kernel_pls_rbf", "gamma"): "RBF kernel bandwidth γ (with K(x, x') = exp(-γ‖x − x'‖²)).",
    # ECR: elastic-net mixing (interpreted as the L2/L1 trade-off in the C kernel).
    ("ecr", "alpha"): "Elastic-net mixing weight (0 = pure L2, 1 = pure L1) applied to the PLS coefficient path.",
    # Randomization test: significance level for the permutation p-value cut.
    ("randomization_select", "alpha"): "Significance level for the permutation-based variable retention test.",
}


def lookup(method: str, param: str) -> str:
    """Return the description for `param` in the context of `method`, or
    an empty string when nothing is registered."""
    override = METHOD_PARAM_OVERRIDES.get((method, param))
    if override:
        return override
    return PARAM_DOCS.get(param, "")


__all__ = ["PARAM_DOCS", "METHOD_PARAM_OVERRIDES", "lookup"]
