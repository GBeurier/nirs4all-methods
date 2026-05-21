# LocalMixupAugmenter

## Algorithm

`n4m_aug_local_mixup` is a neighborhood-aware variant of mixup. For each
sample, it draws a random neighbor among the $k$ nearest non-self rows and
mixes the two with a $\mathrm{Beta}(\alpha, \alpha)$ weight.

For each $i \in \{0, \dots, n-1\}$:

1. Compute the squared-Euclidean distance from $X_{i,:}$ to every row of
   $X$.
2. Partial-sort to obtain the $k+1$ smallest indices (the first one is
   $i$ itself).
3. Draw `pick ∈ [0, k)` uniformly via PCG64 integers and select
   `neighbor = indices[1 + pick]`.
4. Draw $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$.
5. Write $X^{\mathrm{aug}}_{i,:} = \lambda \cdot X_{i,:} + (1-\lambda) \cdot X_{\mathrm{neighbor},:}$.

## RNG consumption

Per row: 1 `integers(0, k)` draw + 1 Beta draw (Cheng's BB / Joehnk's BC).

## Parameters

| Parameter      | Constraint | Meaning |
|----------------|------------|---------|
| `alpha`        | $> 0$      | Shape of the symmetric Beta. |
| `k_neighbors`  | $\geq 1$ and $\leq n-1$ | Number of non-self neighbors to choose among. |

## Reference

`nirs4all.operators.augmentation.spectral.LocalMixupAugmenter`. The reference
implementation uses `sklearn.neighbors.NearestNeighbors` with default
parameters (brute force, Euclidean distance), then
`rng.choice(indices[i, 1:])` and `rng.beta(alpha, alpha)`.
