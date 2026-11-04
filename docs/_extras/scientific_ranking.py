"""Reviewed comparison-block ranking and supplied-variance effect synthesis."""

_REPO = "https://github.com/GBeurier/nirs4all-methods/blob/main/"
_SOURCE = _REPO + "cpp/src/core/ranking/ranking.cpp"
_HEADER = _REPO + "cpp/include/n4m/ranking.h"
_TEST = _REPO + "bindings/python/tests/test_ranking.py"
_FIXTURES = _REPO + "parity/fixtures/ranking_reference_v1.json"

SCIENTIFIC_CONTENT: dict[str, dict[str, str]] = {
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
