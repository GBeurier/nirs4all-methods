"""Reviewed comparison-block ranking and supplied-variance effect synthesis."""

_REPO = "https://github.com/GBeurier/nirs4all-methods/blob/main/"
_SOURCE = _REPO + "cpp/src/core/ranking/ranking.cpp"
_HEADER = _REPO + "cpp/include/n4m/ranking.h"
_TEST = _REPO + "bindings/python/tests/test_ranking.py"
_FIXTURES = _REPO + "parity/fixtures/ranking_reference_v1.json"

SCIENTIFIC_CONTENT: dict[str, dict[str, str]] = {
    "model_selection_fanova": {
        "title": "Forest functional ANOVA under a declared uniform product measure",
        "paper": (
            "Hutter, Hoos and Leyton-Brown (2014), An Efficient Approach for Assessing "
            "Hyperparameter Importance, ICML, PMLR 32(1), 754-762, "
            "https://proceedings.mlr.press/v32/hutter14.html."
        ),
        "principle": (
            "A fresh bootstrap regression forest provides a surrogate surface. Under the declared "
            "independent uniform product measure, each tree's main variance is Var(E[f|x_j]); "
            "a pair variance is Var(E[f|x_j,x_k]) minus both main variances. Exact integration "
            "enumerates split intervals weighted by length and categorical codes weighted by 1/K. "
            "Component fractions normalize each informative tree by its own total variance before "
            "averaging. The remainder contains all unreported interaction orders."
        ),
        "use_cases": (
            "Descriptive surrogate attribution of observed pipeline, component or hyperparameter "
            "responses when the host can define valid finite continuous domains, category codes "
            "and a scientifically meaningful independent product measure."
        ),
        "limitations": (
            "Correlated or conditional parameters can make the product measure include unsupported "
            "combinations. Fractions describe the fitted surrogate and are not causal effects. "
            "Tree sample SD is variability across normalized trees, not an SE or confidence interval. "
            "Constant trees are excluded and counted; constant responses and noninformative forests "
            "withhold attribution. Training RMSE is descriptive; optional validation RMSE and R-squared "
            "do not confer a generalization guarantee. The host owns matching and dependence policy."
        ),
        "implementation": (
            "An optional compile-time C++17 backend vendors only the reviewed BSD-3-Clause header "
            "closure of automl/random_forest_run at 1ba17650e734edd3ea536efc91d32815c4874e72, "
            "including cereal BSD-3-Clause and RapidJSON MIT headers and notices. No Python fanova "
            "code is used. The native adapter independently implements the decomposition, preserves "
            "strides and output atomicity, bounds tree and query resources, and retains backend-unavailable "
            "C ABI stubs by default. Tests include additive, XOR, category, constant, repeated-fit, "
            "holdout, stride, seed, invalid-input and budget cases."
        ),
        "provenance": "; ".join((
            _REPO + "cpp/src/core/component_analysis/fanova.cpp",
            _REPO + "cpp/src/c_api/c_api_fanova.cpp",
            _REPO + "cpp/include/n4m/fanova.h",
            _REPO + "cpp/tests/test_fanova.cpp",
            _REPO + "cpp/backends/rfr/README.md",
            _REPO + "docs/FANOVA.md",
        )),
    },
    "model_selection_ranking": {
        "title": "Comparable-block rankings with explicit connectivity and approximation",
        "paper": (
            "Weng and Lin (2011), A Bayesian Approximation Method for Online Ranking, "
            "Journal of Machine Learning Research 12, 267-300: "
            "https://www.jmlr.org/papers/v12/weng11a.html. The two-player Gaussian "
            "moment update follows Microsoft's TrueSkill1 description: "
            "https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/. "
            "Friedman's available complete-block approximation is independently documented at "
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html."
        ),
        "principle": (
            "Rows are matched task/dataset/split comparison blocks; columns are arbitrary candidates. "
            "Only jointly present columns produce a pair outcome, with ties inside an absolute metric "
            "tolerance. Elo updates a rating by K times observed outcome minus its base-10 logistic "
            "win probability. Batch Bradley-Terry maximizes the pairwise logistic log likelihood minus "
            "an L2 penalty, with ties counted as half wins. TrueSkill1 applies the two-player truncated "
            "Gaussian mean/variance update; Weng-Lin applies Algorithm 1 to two singleton teams, "
            "with gamma_i=sigma_i/c. Available-case mean ranks average within-block tied ranks. "
            "For k candidates and N complete blocks, the Friedman statistic is "
            r"$12\sum_j(R_j-N(k+1)/2)^2/[Nk(k+1)C]$, where R_j is a complete-block rank sum and "
            r"$C=1-\sum(t^3-t)/[N(k^3-k)]$ is the tie correction. Its reported p-value is the "
            "chi-square upper tail with k-1 degrees of freedom."
        ),
        "use_cases": (
            "Descriptive comparison of pipelines or host-defined model, preprocessing, parameter and "
            "split candidates. Shared candidates connect incomplete dataset panels: A/B on one dataset "
            "and B/C on another can support a transitive connected rating without a direct A/C row. "
            "The host retains identities, matching policy and dependence assumptions."
        ),
        "limitations": (
            "This is not a causal component attribution or a TrueSkill2, team or multiplayer factor-graph "
            "implementation. Replaying all pairs from a multi-candidate block creates correlated evidence; "
            "Bayesian posterior SDs do not account for those correlations or repeated folds. Online ratings "
            "depend on supplied row/pair order. Global ranks are unavailable for disconnected graphs or "
            "nonconverged Bradley-Terry fits. Mean ranks use every observed block while Friedman uses "
            "complete blocks only; its asymptotic p-value requires independent blocks and adequate sample "
            "size. Fewer than two complete blocks/candidates or all complete-block ties make it unavailable."
        ),
        "implementation": (
            "The dependency-free C++17 kernel owns all numerical updates. ABI 2.21 exposes initialized "
            "versioned options, strided F64 scores, a required explicit 0/1 presence mask and caller-owned "
            "candidate/summary buffers. Python and actual JS/WASM wrappers marshal this ABI; they do not "
            "reimplement ranking. Unsupported variants are refused, masked nonfinite values are allowed, "
            "and failed calls preserve output buffers. Resource/offset checks bound supported requests. "
            "Coverage, connectivity, win/tie/loss counts, convergence and uncertainty kind accompany scores. "
            "Pinned trueskill 0.4.5, openskill 6.2.0 and SciPy 1.18.1 fixtures independently qualify the "
            "implemented variants; R/MATLAB ranking wrappers are not claimed."
        ),
        "provenance": "; ".join((
            _SOURCE, _HEADER, _TEST, _FIXTURES,
            _REPO + "docs/ranking_contract.v1.json",
            _REPO + "parity/generate_ranking_fixtures.py",
            _REPO + "bindings/js/test/run_ranking.mjs",
        )),
    },
    "model_selection_relative_gain": {
        "title": "Paired positive-loss relative gains and supplied-variance random effects",
        "paper": (
            "DerSimonian and Laird (1986), Meta-analysis in clinical trials, Controlled Clinical Trials "
            "7, 177-188, https://doi.org/10.1016/0197-2456(86)90046-2. The inverse-variance "
            "random-effects assumptions and heterogeneity quantities are described in the Cochrane "
            "Handbook, Chapter 10: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10."
        ),
        "principle": (
            "For each explicitly present pair of strictly positive losses, define the relative improvement "
            "$e_i=(b_i-c_i)/b_i$. The arithmetic mean and sample SD describe the paired effects. "
            "Optional caller-supplied sampling variances v_i define fixed weights w_i=1/v_i and "
            r"$Q=\sum_i w_i(e_i-\bar e_w)^2$. For m studies, the DerSimonian-Laird estimate is "
            r"$\tau^2=\max(0,(Q-m+1)/(\sum w_i-\sum w_i^2/\sum w_i))$. The pooled effect uses "
            r"random-effects weights $u_i=1/(v_i+\tau^2)$, with standard error $1/\sqrt{\sum u_i}$. "
            "I-squared is the nonnegative fraction (Q-m+1)/Q when Q is positive, otherwise zero."
        ),
        "use_cases": (
            "Comparing a declared baseline and candidate through matched positive losses; optionally "
            "synthesizing independent study-level relative effects when their sampling variances, "
            "including paired covariance, are supplied in the same relative-effect units."
        ),
        "limitations": (
            "Losses must be finite and strictly positive on present rows. Raw-loss variances and fold "
            "spread are not valid substitutes for relative-effect sampling variances. Repeated or "
            "overlapping folds do not become independent studies by passing a variance vector. "
            "The sample SD is descriptive dispersion, not a standard error. Without variances or with "
            "fewer than two effects, pooled effect, SE, tau-squared, Q and I-squared are unavailable; "
            "no confidence interval is fabricated. Finite extremes outside representable intermediate "
            "or output ranges are explicitly refused rather than silently discarding positive weights."
        ),
        "implementation": (
            "The same dependency-free native ranking kernel supplies the n4m_ranking_relative_gain_v1 "
            "caller-buffer C ABI and Python paired_relative_gain wrapper. Explicit 0/1 masks control "
            "absence; core validation checks positive losses and supplied variances. Positive prefix "
            "cross-products evaluate the DL denominator in linear time without subtractive cancellation, "
            "and scaled/extended intermediates retain supported variance ranges. Standardized residuals "
            "and separately square-rooted SE factors avoid avoidable underflow. Tests use exact rational "
            "DL examples, dominant-weight and subnormal-variance counterexamples, and actual WASM C ABI "
            "execution. No standalone R, MATLAB or JS relative-gain facade is claimed."
        ),
        "provenance": "; ".join((
            _SOURCE, _HEADER, _TEST,
            _REPO + "cpp/tests/test_ranking.cpp",
            _REPO + "bindings/js/test/run_ranking.mjs",
            _REPO + "docs/RANKING.md",
        )),
    },
}

SCIENTIFIC_CONTENT["model_selection_normalize_loss"] = {
    "title": "Dimensionless loss against an explicit matched baseline",
    "paper": "The relative-baseline scale is defined by the native protocol at https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/component_analysis.h; this arithmetic utility does not claim a new statistical estimator.",
    "principle": "For finite nonnegative loss L and finite strictly positive baseline B, return L/B. Zero loss is valid; zero baseline and overflowing ratios are refused.",
    "use_cases": "Place losses with dataset-dependent units on a dimensionless relative-baseline scale before a separately specified component analysis.",
    "limitations": "Matching, baseline selection and dependence assumptions are host responsibilities. Ratios do not remove baseline uncertainty or establish comparable difficulty, causal effects or independent repeated folds.",
    "implementation": "ABI 2.22 computes ratios natively into a temporary buffer and commits all outputs only after validation. Python and JS/WASM translate buffers without calculating ratios. Input/output aliases are supported; errors leave outputs unchanged.",
    "provenance": "; ".join((_REPO + "cpp/src/c_api/c_api_component_analysis.cpp", _REPO + "cpp/include/n4m/component_analysis.h", _REPO + "bindings/python/tests/test_component_analysis.py")),
}

SCIENTIFIC_CONTENT["model_selection_mixed_effects"] = {
    "title": "Gaussian ML/REML with explicit fixed designs and random-intercept factors",
    "paper": (
        "Bates, Maechler, Bolker and Walker (2015), Fitting Linear Mixed-Effects Models "
        "Using lme4, Journal of Statistical Software 67(1): https://doi.org/10.18637/jss.v067.i01. "
        "Pinned independent Rust implementation: https://github.com/bbuchsbaum/mixeff-rs/tree/"
        "696e5e74623301280142e73cbb6bf7f2650647d7. Development oracle: "
        "https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html."
    ),
    "principle": (
        "Fit y=X beta+sum_j Z_j b_j+epsilon with independent zero-mean Gaussian random "
        "intercepts b_j and IID residual. ML/REML profile the fixed effects and residual "
        "scale while the pinned native TrustBQ optimizer searches nonnegative relative "
        "covariance parameters under an explicit evaluation bound. Beta and conditional "
        "plug-in covariance preserve the caller's complete X order; variance components "
        "preserve group order, residual first. SVD checks fixed rank. A covariance trace "
        "Gram in the residual space orthogonal to X rejects structurally confounded "
        "variance directions without allocating an n-by-n covariance matrix."
    ),
    "use_cases": (
        "Conditionally adjusted descriptive model/preprocessing parameter contrasts when "
        "the host supplies a defensible fixed design, metric scale, dataset and actual "
        "evaluation-block grouping. Crossed/nested intercept factors are supported; "
        "nested IDs must include parent context. Candidate comparisons require explicit "
        "reference coding and are not a universal causal component ranking."
    ),
    "caveats": (
        "Only Gaussian identity-link random-intercept LMMs are exposed. The caller adds "
        "any intercept and all contrasts. Rank-deficient, constant/perfectly explained, "
        "variance-confounded and nonconverged fits withhold all estimates with typed "
        "diagnostics. Fitted zero-variance boundaries remain flagged. Optimizer "
        "termination is not model adequacy. Conditional SEs omit variance-parameter "
        "uncertainty; no p-values, confidence intervals or Satterthwaite degrees of "
        "freedom are supplied. REML log likelihood is restricted and cannot support "
        "ordinary fixed-design ML comparisons. Hosts own matching, scale normalization "
        "and independence assumptions; repeated folds do not become independent groups "
        "merely through identifier encoding. Fixed resource caps are part of the contract."
    ),
    "implementation": (
        "Optional statically linked MIT mixeff-rs=1.0.0-rc.5 with defaults disabled, "
        "no NLopt/R/Python/formula-string runtime. Programmatic AST contains only "
        "explicit numeric fixed columns and intercept-only random terms. Stable "
        "caller-buffer C ABI preserves feature-off BACKEND_UNAVAILABLE behavior, "
        "technical-error atomicity and caught Rust panic/C++ exception boundaries. "
        "Real native checks include analytic balanced ML/REML variance/covariance "
        "targets, independent statsmodels0.14.6 crossed/nested fixtures, rank and "
        "variance-identifiability counterexamples, genuine boundary, budget exhaustion, "
        "repeated fits, arbitrary supported strides and output ownership. Actual "
        "optimized native/WASM/package size qualification belongs to aggregate gates, "
        "not to a successful standalone archive build."
    ),
    "provenance": "; ".join((
        _REPO+"cpp/backends/mixeff/src/lib.rs",
        _REPO+"cpp/backends/mixeff/Cargo.lock",
        _REPO+"cpp/include/n4m/mixed_effects.h",
        _REPO+"cpp/src/c_api/c_api_mixed_effects.cpp",
        _REPO+"cpp/tests/test_mixed_effects.cpp",
        _REPO+"parity/fixtures/mixed_effects_reference_v1.json",
        _REPO+"parity/generate_mixed_effects_reference.py",
        _REPO+"docs/MIXED_EFFECTS.md",
    )),
}
# Keep the shared scientific-content schema consistent with the other records.
SCIENTIFIC_CONTENT["model_selection_mixed_effects"]["limitations"] = SCIENTIFIC_CONTENT[
    "model_selection_mixed_effects"
].pop("caveats")
