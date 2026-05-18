# MixupAugmenter

## Algorithm

`c4a_aug_mixup` augments a batch of spectra by replacing every row with a
convex combination of two original rows drawn from the same batch.

Given an input matrix $X \in \mathbb{R}^{n \times p}$, the algorithm:

1. Draws a Fisher-Yates permutation $\pi \in S_n$ of the sample indices.
2. Draws $\lambda_i \sim \mathrm{Beta}(\alpha, \alpha)$ for $i = 0, \dots, n-1$.
3. Writes
$$X^{\mathrm{aug}}_{i,:} = \lambda_i \cdot X_{i,:} + (1-\lambda_i) \cdot X_{\pi(i),:}$$

The output keeps the same shape $(n, p)$ as the input. **Each output row is
a mix of two rows from the SAME batch** — this is the canonical "within-batch
mixup" used in vision and audio augmentation pipelines, applied here to NIR
spectra.

## RNG consumption

For each `_apply` call the PCG64 stream consumes:

- $O(n \log n)$ uint64 draws for the Fisher-Yates permutation (one
  `integers(0, k+1)` per swap step).
- $\approx 2n$ next_double draws on average for the Beta sampler (Cheng's
  BB / Joehnk's BC depending on $\alpha$).

## Determinism

A fixed PCG64 state at call time yields a bit-exact output. Seeding the
RNG with `c4a_rng_pcg64_set_seed(rng, S)` before each `_apply` produces a
reproducible stream regardless of prior consumption.

## Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `alpha`   | $> 0$      | Shape of the symmetric Beta distribution used for $\lambda$. |

## Reference

`nirs4all.operators.augmentation.spectral.MixupAugmenter`. The reference
implementation uses `numpy.random.Generator.permutation` followed by
`Generator.beta(alpha, alpha, size=(n,1))`.

For a recent overview of mixup, see Zhang et al. 2018, "mixup: Beyond
Empirical Risk Minimization".
