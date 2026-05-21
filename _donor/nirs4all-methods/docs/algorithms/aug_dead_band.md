# DeadBandAugmenter

## Algorithm

Simulates detector dead bands by replacing random contiguous spectral
regions with noise.

Per-sample mode (`variation_scope = 0`):

```
for each row i in 0..n-1:
    u ~ Uniform(0, 1)
    if u < probability:
        for b in 0..n_bands:
            width ~ UniformInt[width_low, width_high]   # inclusive
            start ~ UniformInt[0, max(1, cols - width))
            end   = min(start + width, cols)
            row[start:end] ~ Normal(0, noise_std)
```

Batch mode (`variation_scope = 1`):

```
u ~ Uniform(0, 1)
if u < probability:
    for b in 0..n_bands:
        width, start, end = ... (as above)
        # all rows get the same band positions, but
        # the noise samples are still drawn independently
        # with size = (rows, end - start)
        X[:, start:end] ~ Normal(0, noise_std)
```

## Determinism

When `probability = 0` the augmenter is the identity. The parity fixture
exercises this case (`dead_band_zero_probability`).

## Reference

`nirs4all.operators.augmentation.synthesis.DeadBandAugmenter`.
