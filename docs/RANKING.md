# Comparable-block ranking (ABI 2.21)

`n4m.ranking.rank` and the JavaScript `rank` function share dependency-free
C++17 kernels through the versioned public C ABI in `n4m/ranking.h`.
No reference package is required at runtime. Python and JS/WASM are supported;
R and MATLAB wrappers for this new surface are deferred. Gaussian mixed effects
and forest fANOVA have separate optional static backends and thin Python/JS
bindings in `n4m.component_analysis`; see `MIXED_EFFECTS.md` and `FANOVA.md`.

## Comparison identity and transitivity

A matrix row is one comparable task/dataset/split block. A column is an arbitrary
candidate, such as a pipeline, model, preprocessing chain, parameter setting,
feature selection strategy, or split strategy. The host owns candidate identities,
context matching, grouping, and dependence policy. A value from another row
never supplies a missing head-to-head comparison.

The methods can infer a ranking through shared candidates across datasets.
A and B observed together on dataset D1, followed by B and C on D2, connects
A to C even if A and C never appear in the same row:

```python
from n4m.ranking import rank

result = rank(
    [[1.0, 2.0, float("nan")], [float("nan"), 1.0, 2.0]],
    present=[[1, 1, 0], [0, 1, 1]],
    method="trueskill",
)
assert result["coverage"]["complete_blocks"] == 0
assert result["coverage"]["connected_components"] == 1
assert [c["rank"] for c in result["candidates"]] == [1.0, 2.0, 3.0]
```

There is no required full-candidate intersection for ratings. Only the Friedman
statistic uses complete blocks. Disconnected comparison graphs have no global
rank. Candidate scores remain available, but their relative origins/priors do
not establish an ordering between disconnected groups. Entirely absent
candidates are explicitly null and count as isolated graph nodes.

Component rankings are descriptive conditional comparisons, not causal component
attributions: the host must match surrounding configurations when this is the
intended question. The core does not silently compare unmatched parameter or
preprocessing effects.

## Variants and assumptions

| Method | Native variant | Score | Uncertainty |
| --- | --- | --- | --- |
| `average_rank` | Available-case mean of within-block average ranks | Lower is better | None |
| `elo` | Logistic base-10 Elo, sequential pairs | Higher is better | None |
| `bradley_terry` | Batch L2-regularized logistic strengths, ties as half wins | Higher is better | None |
| `trueskill` / `trueskill_1v1` | TrueSkill1 Gaussian two-player moment update with draw margin | Posterior mean, higher is better | Approximate posterior SD |
| `openskill` / `weng_lin_bradley_terry` | Weng-Lin Bradley-Terry two singleton teams, gamma_i=sigma_i/c | Posterior mean, higher is better | Approximate posterior SD |

Online methods process rows in supplied order, then pairs `(0,1), (0,2), ...`.
They make one pass. Expanding a block with several candidates into pairs creates
correlated evidence. This is a documented composite approximation; posterior
SD is not a standard error and does not account for dependence between pairs,
overlapping splits, or repeated folds. No TrueSkill2, team, multiplayer factor
graph, Plackett-Luce, causal attribution, or joint mixed-effects fit is claimed.
Unsupported methods and option names are rejected.

Defaults come from the native initializer: mean 25, Elo K=32 and scale=400,
prior SD=25/3, beta=25/6, dynamics=0, draw margin=0.740466587452148,
BT L2=1, gradient tolerance=1e-8, maximum iterations=10000. Dynamics are added
per replayed pair when explicitly enabled. Weng-Lin variance shrinkage floor
is 1e-12. BT returns convergence status and withholds global ranks when its
iteration budget is exhausted; it does not invent standard errors.

## Missingness, tolerance, and inference

Presence is explicit: native and JS calls require a row-major 0/1 mask; Python
uses an all-present mask by default. Unmasked NaN/Inf is an error. Masked
values may be nonfinite. Inputs are borrowed for a call and never retained.
The C ABI accepts F64 matrix strides in elements, including column-major and
positive-stride slices. Negative strides are rejected by C; Python explicitly
copies a negative-stride array into a supported layout.

`higher_is_better` controls metric direction. `tie_tolerance` is absolute in
metric units. A head-to-head result is a tie when its absolute difference is
within tolerance. Within-block ranks use sorted bounded-range groups anchored
at the best value: `[0, .75, 1.5]` at tolerance 1 has ranks `[1.5, 1.5, 3]`.
Tolerance closeness is nontransitive, so pairwise ties can differ from rank
groups in that example. Global score ranks use exact score equality.

Mean ranks include every observed block; the number of candidates per block
can differ, and coverage must be inspected before comparing these means.
Friedman's tie-corrected statistic uses complete blocks only. Its p-value is
a chi-square approximation with k-1 degrees of freedom, conditional on
independent blocks and adequate sample size; the host must assess both.
It is unavailable with fewer than two complete blocks/candidates or when all
complete blocks are tied. This is no post-hoc pairwise significance test.

`paired_relative_gain` computes `(baseline-candidate)/baseline` only for
finite strictly-positive paired losses. Its sample SD describes dispersion.
Optional DerSimonian-Laird random effects require caller-supplied positive
sampling variances of these relative effects, including paired covariance.
Raw loss variances and fold spread are not substituted. Pooled effect,
standard error, tau-squared, Q and I-squared are null without variances or
with fewer than two independent study effects. No confidence interval is
fabricated from repeated folds.

## Output and limits

The machine-readable [result schema](ranking_contract.v1.json) and
[`ranking_result_v1.json`](../parity/fixtures/ranking_result_v1.json) provide a
reusable transitive-bridge example for host integration.

The host-facing result carries `schema_version: "n4m.ranking/1"`, canonical
variant, effective options, score direction, uncertainty kind, candidate index,
score/SD/mean rank/global rank, observation/match/win/tie/loss counts, component
IDs, graph and complete-block coverage, Friedman approximation metadata,
convergence, and warnings. Nonfinite unavailable fields become JSON null.
Candidate IDs and block IDs are attached by the host, preserving its vocabulary.

The C ABI is caller-buffer based: ten F64 fields per candidate and nine summary
fields documented in the header. Failure leaves both buffers unchanged.
The native limits are 10000 candidates (2000 for dense batch BT), 100 million
input cells and 100 million actual co-observed pairs. Offset arithmetic is
checked before reading a strided view. Nonfinite/overflowing computations
are rejected; numerical extremes are not replaced by invented estimates.

## References and validation

The formulas follow [Weng and Lin (2011)](https://www.jmlr.org/papers/v12/weng11a.html),
Algorithm 1 specialized to two singleton teams, and Microsoft's
[TrueSkill1 description](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/).
[The Cochrane handbook](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10)
describes inverse-variance random-effects assumptions. Fixture provenance and
pinned independent implementations are stored in
`parity/fixtures/ranking_reference_v1.json`; regenerate with
`parity/generate_ranking_fixtures.py`. Reference dependencies are test-only.
Native, Python, and actual WASM tests cover connected incomplete graphs,
unsupported variants, explicit absence, direction, tolerance ties, strided
buffers, output atomicity, and independent reference parity.


The supervised `MethodSpec` training/timing registry and its estimator dashboard
are not applicable to this matrix-valued ranking procedure. No supervised
adapter or benchmark result is fabricated. Numerical parity is covered by the
ranking fixtures and dedicated ABI/binding tests; a dedicated ranking timing
registry is deferred. Linux native and Emscripten execution were validated.
macOS/Windows symbol snapshots are derived from the Linux name set; native
execution on those platforms was not run in this batch.
