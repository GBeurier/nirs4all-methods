# `asha` — asynchronous successive halving (pruner)

**Role:** `optimization` · **kind:** `n4m_pruner_kind_t = N4M_PRUNER_ASHA` · **since:** ABI 2.1 (F2)

Asynchronous Successive Halving. At each rung (`step` of `n4m_optimizer_tell_intermediate`), a trial survives only if its score is in the **top 1/reduction_factor** of the peers that have reached the same rung; otherwise it is pruned. The decision is *asynchronous* — made on the fly against whoever has reported at that rung — so it never blocks on a synchronised cohort (the property that makes it sound for a per-`tell` verdict, per `FINETUNING_ROADMAP.md` §3).

Like all pruners it is **orthogonal to the sampler** and works on whatever intermediate-score axis the caller supplies (PLS `n_components` learning curve, subsample fraction, DL epochs). The reduction factor is `opts.reduction_factor` (**default 3** when left 0); a trial is only evaluated once at least `reduction_factor` peers have reached its rung. **Ties survive:** the rule counts only strictly-better peers, so trials tied at the cutoff are all kept and the surviving set may slightly exceed 1/reduction_factor.

> **Fidelity-axis caveat (roadmap §2c):** ASHA/Hyperband assume rank-preservation across rungs. The intended native fidelity is the PLS `n_components` learning curve / subsample fraction / epochs — **not** a CV-fold fraction. The fidelity *engine* that produces those rung scores (from a single NIPALS/SIMPLS fit) is a separate F2 deliverable; `asha` only consumes the stream.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.pruner = N4M_PRUNER_ASHA;
/* per trial, per rung: */
int32_t prune = 0;
n4m_optimizer_tell_intermediate(opt, trial_id, rung, rung_score, &prune);
```

## Parity

- **Tier B (decision-level):** the promote/prune verdict is an RNG-free function of the rung history + reduction factor; verified against a canned history in the C++ tests. A cross-reference fixture vs Optuna's `SuccessiveHalvingPruner` lands with the Track-Q parity machinery.

## References

- Jamieson & Talwalkar, *Non-stochastic Best Arm Identification and HPO*, AISTATS (2016); Karnin, Koren & Somekh, ICML (2013); Li et al. (ASHA), MLSys (2020). See `_finetuning_bibliography.bib`.
