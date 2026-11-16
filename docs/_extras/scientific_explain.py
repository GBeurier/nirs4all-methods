"""Model explanations: Shapley-family, surrogate and perturbation attributions (ABI 2.23)."""

_REPO = "https://github.com/GBeurier/nirs4all-methods/blob/main/"
_HEADER = _REPO + "cpp/include/n4m/explain.h"
_C_API = _REPO + "cpp/src/c_api/c_api_explain.cpp"
_COMMON = _REPO + "cpp/src/core/explain/explain_common.cpp"
_CPP_TEST = _REPO + "cpp/tests/test_explain.cpp"
_PY_TEST = _REPO + "bindings/python/tests/test_explain.py"
_JS_TEST = _REPO + "bindings/js/test/run_explain.mjs"
_DOC = _REPO + "docs/EXPLAIN.md"
_SCHEMA = _REPO + "docs/explain/explanation_v1.schema.json"

SCIENTIFIC_CONTENT: dict[str, dict[str, str]] = {
    "diagnostics_explain_model_agnostic": {
        "title": "Ask/tell group Shapley, LIME and occlusion explanations of any model",
        "paper": (
            "Shapley (1953), A Value for n-Person Games, in Contributions to the Theory of "
            "Games II, Annals of Mathematics Studies 28, 307-317, "
            "https://doi.org/10.1515/9781400881970-018. Permutation sampling: Strumbelj and "
            "Kononenko (2014), Explaining prediction models and individual predictions with "
            "feature contributions, Knowledge and Information Systems 41(3), 647-665, "
            "https://doi.org/10.1007/s10115-013-0679-x; antithetic (forward/reverse) orders: "
            "Mitchell, Cooper, Frank and Holmes (2022), Sampling Permutations for Shapley Value "
            "Estimation, Journal of Machine Learning Research 23, "
            "https://www.jmlr.org/papers/v23/21-0439.html. Kernel SHAP: Lundberg and Lee (2017), "
            "A Unified Approach to Interpreting Model Predictions, NeurIPS 30, "
            "https://arxiv.org/abs/1705.07874. LIME: Ribeiro, Singh and Guestrin (2016), "
            "\"Why Should I Trust You?\": Explaining the Predictions of Any Classifier, KDD, "
            "1135-1144, https://doi.org/10.1145/2939672.2939778. Occlusion: Zeiler and Fergus "
            "(2014), Visualizing and Understanding Convolutional Networks, ECCV, LNCS 8689, "
            "818-833, https://doi.org/10.1007/978-3-319-10590-1_53. Interventional value "
            "function: Janzing, Minorics and Bloebaum (2020), Feature relevance quantification "
            "in explainable AI: A causal problem, AISTATS, PMLR 108, 2907-2916, "
            "https://proceedings.mlr.press/v108/janzing20a.html."
        ),
        "principle": (
            "Feature groups partitioning the explained features are the players of a coalition "
            "game. The value v(S) is the host model output when the groups in S keep the "
            "explained row's values and absent groups take baseline values: each background "
            "row averaged over the background (interventional), the background column mean, or "
            "a straight-line bridge across each maximal run of absent contiguous groups. "
            r"Exact SHAP enumerates all $2^G$ coalitions ($G\le 16$) with Shapley weights "
            r"$|S|!(G-|S|-1)!/G!$. Permutation SHAP draws uniform group orders and averages the "
            "marginal contributions of each forward chain and its antithetic reverse chain; each "
            "chain telescopes from v(empty) to v(full), so the estimate is additive by "
            r"construction and its standard error is $\mathrm{sd}/\sqrt{P}$ over P draws. Kernel "
            "SHAP solves the Shapley-kernel weighted least squares with the efficiency constraint "
            "eliminated exactly; it enumerates every proper coalition when the budget allows "
            "(then equal to exact SHAP) and otherwise samples coalition sizes proportionally to "
            "their kernel mass with complement pairing. LIME fits a weighted ridge surrogate of "
            "v(z) on binary group masks with an exponential kernel of the cosine distance to "
            r"the all-present mask. Occlusion reports $f(x)-v(\text{all except } g)$."
        ),
        "use_cases": (
            "Explaining any fitted regressor or classifier output (prediction in original units, "
            "one class probability or a decision score) whose predictions the host can compute in "
            "batches with any backend, including Python, WebAssembly, ONNX or asynchronous "
            "workers. For NIR spectra, grouped explanations over contiguous wavelength windows or "
            "chemically informed bands with the interpolation baseline, exact SHAP for a handful "
            "of bands, and permutation SHAP with standard errors for larger group counts."
        ),
        "limitations": (
            "Status: exact SHAP, permutation SHAP and occlusion are implemented; Kernel SHAP and "
            "LIME are experimental, validated for exactness under full enumeration, efficiency "
            "and linear recovery but not for parity with the shap or lime packages on sampled "
            "budgets. Sampled Kernel SHAP is consistent, not unbiased, and reports no standard "
            "error; LIME and occlusion are not additive and occlusion ignores interactions. "
            "Absent groups take baseline values independently of present groups, so correlated "
            "spectral features can be evaluated off the data manifold; interpolation mitigates "
            "but does not remove this, and no conditional baseline is provided. Attributions "
            "describe the model's response under the declared value function, not causal effects "
            "in the measured system. Cost grows as 2^G for exact SHAP and linearly with the "
            "background size; inputs and predictions must be finite."
        ),
        "implementation": (
            "A C++17 ask/tell session in the ABI 2.23 n4m/explain.h surface never calls a model: "
            "it emits deterministic batches of perturbed rows (PCG64 stream seeded by the caller) "
            "and receives host predictions, with memory bounded by one batch and the exact "
            "evaluation count reported up front. Protocol misuse is rejected without losing the "
            "session; nonfinite predictions poison it. Fixed-width and range grouping helpers and "
            "per-group summaries are native. Python n4m.explain and JavaScript explainModel only "
            "marshal the ABI. Tests check exact, permutation and full-enumeration Kernel SHAP "
            "against brute-force Shapley values and pin parity with shap 0.51 ExactExplainer and "
            "PermutationExplainer."
        ),
        "provenance": "; ".join((
            _HEADER,
            _REPO + "cpp/src/core/explain/explain_session.cpp",
            _REPO + "cpp/src/core/explain/grouping.cpp",
            _COMMON, _C_API, _CPP_TEST, _PY_TEST, _JS_TEST, _DOC, _SCHEMA,
        )),
    },
    "diagnostics_explain_linear_shap": {
        "title": "Exact interventional SHAP of affine models",
        "paper": (
            "Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions, "
            "Advances in Neural Information Processing Systems 30 (Linear SHAP under feature "
            "independence), https://arxiv.org/abs/1705.07874. The interventional value function "
            "follows Janzing, Minorics and Bloebaum (2020), AISTATS, PMLR 108, 2907-2916, "
            "https://proceedings.mlr.press/v108/janzing20a.html."
        ),
        "principle": (
            r"For an affine model $f(z)=z\cdot B+c$ with coefficients B (features by outputs) "
            "expressed in the explained space and a reference vector r of background column means, "
            r"the attribution of group g to output o is $\sum_{j\in g}B_{jo}(x_j-r_j)$ and the "
            r"expected value is $r\cdot B_{:,o}+c_o$. Because the model is additive, this equals "
            "exact interventional SHAP against the background, and group sums are exact group "
            "Shapley values; attributions plus the expected value reproduce the prediction."
        ),
        "use_cases": (
            "Exact, evaluation-free explanations of PLS, PCR, Ridge and any linear pipeline whose "
            "coefficients and intercept are expressed in the space being explained, per wavelength "
            "or over spectral groups, and as an oracle for sampled model-agnostic estimators."
        ),
        "limitations": (
            "Valid only when the supplied coefficients represent the explained function in the "
            "explained space; coefficients of a terminal estimator after preprocessing explain the "
            "transformed features, not raw spectra. Interventional attributions ignore feature "
            "correlation, so strongly collinear wavelengths can receive large opposing values. "
            "The reference is caller supplied; inputs, coefficients, intercept and reference must "
            "be finite."
        ),
        "implementation": (
            "A dependency-free C++17 kernel behind n4m_explain_linear_v1 accepts strided F64 views, "
            "an optional feature-to-group map and caller-owned output buffers, and returns "
            "attributions, expected values and optional predictions. Python n4m.explain.linear_shap "
            "and JavaScript linearShap marshal the ABI. Tests check agreement with permutation "
            "SHAP and the group game, and pin parity with shap 0.51 LinearExplainer for Ridge and "
            "PLSRegression."
        ),
        "provenance": "; ".join((
            _HEADER,
            _REPO + "cpp/src/core/explain/linear_shap.cpp",
            _COMMON, _C_API, _CPP_TEST, _PY_TEST, _DOC,
        )),
    },
    "diagnostics_explain_tree_shap": {
        "title": "Path-dependent TreeSHAP over generic flat tree ensembles",
        "paper": (
            "Lundberg, Erion, Chen, DeGrave, Prutkin, Nair, Katz, Himmelfarb, Bansal and Lee "
            "(2020), From local explanations to global understanding with explainable AI for "
            "trees, Nature Machine Intelligence 2, 56-67, Algorithm 2, "
            "https://doi.org/10.1038/s42256-019-0138-9."
        ),
        "principle": (
            "Path-dependent TreeSHAP computes exact Shapley values of the conditional expectation "
            "estimated by each tree, where an absent feature follows both children weighted by "
            "their training covers. Algorithm 2 tracks, along each root-to-leaf path, the "
            "proportion of coalitions of each size that reach the node (extend and unwind of the "
            "unique-feature path), giving polynomial cost in depth instead of exponential cost in "
            "features. Zero fractions are cover[child]/(cover[left]+cover[right]) and the expected "
            "value is the same recursive cover-weighted mean, so additivity is exact for every "
            "tree. The ensemble output is base_offset plus tree_scale times the sum of leaf values; "
            "grouped values are sums of per-feature values."
        ),
        "use_cases": (
            "Exact explanations of exported decision trees, random forests (tree_scale 1/n_trees) "
            "and gradient-boosted ensembles (learning rate and base offset), with multi-output "
            "leaves, scikit-learn/LightGBM or XGBoost split conventions and default-left handling "
            "of missing values."
        ),
        "limitations": (
            "Path-dependent values condition on the training distribution encoded by the covers; "
            "they are not interventional SHAP against a background and differ from it when "
            "features are dependent. Grouped values are sums of per-feature values, not group-game "
            "values. Covers must be finite and positive on reachable nodes, thresholds and leaf "
            "values finite and nodes must form a tree; NaN inputs require default_left. Model "
            "exporters are host responsibilities."
        ),
        "implementation": (
            "A C++17 kernel behind n4m_explain_tree_path_dependent_v1, written after the reference "
            "shap tree_shap.h with an explicit stack so deep trees cannot exhaust small WebAssembly "
            "stacks, consumes a flat ensemble of concatenated nodes with per-tree offsets and local "
            "child indices. Python n4m.explain.tree_shap and JavaScript treeShap marshal the ABI. "
            "Tests compare a brute-force conditional-expectation oracle and pin parity with shap "
            "0.51 TreeExplainer (tree_path_dependent) for scikit-learn trees, forests and gradient "
            "boosting."
        ),
        "provenance": "; ".join((
            _HEADER,
            _REPO + "cpp/src/core/explain/tree_shap.cpp",
            _COMMON, _C_API, _CPP_TEST, _PY_TEST, _DOC,
        )),
    },
}
