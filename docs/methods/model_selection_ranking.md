# `rank` — n4m.ranking.rank

_Namespace_: **`n4m.ranking`** · _Fully-qualified_: `n4m.ranking.rank` · _Catalog id_: `model_selection.ranking`

## API surface

**C ABI (ABI 2):** [`n4m_ranking_options_init_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ranking.h#L37) · [`n4m_ranking_compute_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ranking.h#L57). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.ranking import rank`

**Signature:** [`rank(scores: ArrayLike, *, method: str = 'average_rank', present: ArrayLike | None = None, higher_is_better: bool = False, tie_tolerance: float = 0.0, **method_options: Any)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ranking.py#L65)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `scores` | `ArrayLike` | `required` |
| `method` | `str` | `'average_rank'` |
| `present` | `ArrayLike \| None` | `None` |
| `higher_is_better` | `bool` | `False` |
| `tie_tolerance` | `float` | `0.0` |
| `method_options` | `Any` | `required` |

## Explanations

### Bibliographic source

Weng and Lin (2011), A Bayesian Approximation Method for Online Ranking, Journal of Machine Learning Research 12, 267-300: https://www.jmlr.org/papers/v12/weng11a.html. The two-player Gaussian moment update follows Microsoft's TrueSkill1 description: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/. Friedman's available complete-block approximation is independently documented at https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html.

### Mathematical principle

Rows are matched task/dataset/split comparison blocks; columns are arbitrary candidates. Only jointly present columns produce a pair outcome, with ties inside an absolute metric tolerance. Elo updates a rating by K times observed outcome minus its base-10 logistic win probability. Batch Bradley-Terry maximizes the pairwise logistic log likelihood minus an L2 penalty, with ties counted as half wins. TrueSkill1 applies the two-player truncated Gaussian mean/variance update; Weng-Lin applies Algorithm 1 to two singleton teams, with gamma_i=sigma_i/c. Available-case mean ranks average within-block tied ranks. For k candidates and N complete blocks, the Friedman statistic is $12\sum_j(R_j-N(k+1)/2)^2/[Nk(k+1)C]$, where R_j is a complete-block rank sum and $C=1-\sum(t^3-t)/[N(k^3-k)]$ is the tie correction. Its reported p-value is the chi-square upper tail with k-1 degrees of freedom.

### Appropriate uses

Descriptive comparison of pipelines or host-defined model, preprocessing, parameter and split candidates. Shared candidates connect incomplete dataset panels: A/B on one dataset and B/C on another can support a transitive connected rating without a direct A/C row. The host retains identities, matching policy and dependence assumptions.

### Limits and validation

This is not a causal component attribution or a TrueSkill2, team or multiplayer factor-graph implementation. Replaying all pairs from a multi-candidate block creates correlated evidence; Bayesian posterior SDs do not account for those correlations or repeated folds. Online ratings depend on supplied row/pair order. Global ranks are unavailable for disconnected graphs or nonconverged Bradley-Terry fits. Mean ranks use every observed block while Friedman uses complete blocks only; its asymptotic p-value requires independent blocks and adequate sample size. Fewer than two complete blocks/candidates or all complete-block ties make it unavailable.

### Implementation

The dependency-free C++17 kernel owns all numerical updates. ABI 2.21 exposes initialized versioned options, strided F64 scores, a required explicit 0/1 presence mask and caller-owned candidate/summary buffers. Python and actual JS/WASM wrappers marshal this ABI; they do not reimplement ranking. Unsupported variants are refused, masked nonfinite values are allowed, and failed calls preserve output buffers. Resource/offset checks bound supported requests. Coverage, connectivity, win/tie/loss counts, convergence and uncertainty kind accompany scores. Pinned trueskill 0.4.5, openskill 6.2.0 and SciPy 1.18.1 fixtures independently qualify the implemented variants; R/MATLAB ranking wrappers are not claimed.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/ranking/ranking.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ranking.h; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_ranking.py; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/ranking_reference_v1.json; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/ranking_contract.v1.json; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/generate_ranking_fixtures.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/js/test/run_ranking.mjs

## Catalog note

Comparable-block rankings: Elo, L2 batch Bradley-Terry, pairwise TrueSkill1, pairwise Weng-Lin Bradley-Terry, mean ranks and tie-corrected complete-block Friedman chi-square approximation. Explicit missingness, connectivity, convergence and posterior approximation warnings.


_See also_: [methods index](index.md).