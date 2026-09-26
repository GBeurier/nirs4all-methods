# Current method-science reference index

This generated index covers the scientific records rendered in the method pages. It complements the [lossless historical bibliography](../bibliography.md): the records below retain their source text and DOI/URL provenance, but no new BibTeX is emitted without a separately reviewed structured source.

- Curated records: **220**
- Required fields: bibliographic source, principle, uses, limits, implementation, and provenance.
- Links are checked for offline DOI/URL syntax by this generator; they are not network-fetched.

## Source inputs

- `docs/_extras/methods_bibliography.py` — SHA-256 `140554600e9d2f835e1d11a383d9d389c730d86c5205a5f742ea8465824e6d30`
- `docs/_extras/scientific_aom.py` — SHA-256 `43aa30df6c782d2b4a407c09e734e8d1d5a53dbfc21fddae55b916dc16800f93`
- `docs/_extras/scientific_augmentation_filter_split.py` — SHA-256 `a32c7074bf922c3dcaad3923f9e2bc985c96d5efb4bf14dea04abbff62a7e76d`
- `docs/_extras/scientific_legacy.py` — SHA-256 `b1f2a131fe74b33659a0d29b9480bef356efefa2f1d1bf5e453de1daaaedb87d`
- `docs/_extras/scientific_remaining.py` — SHA-256 `9b9d58596371dcc1fd96d0c1976a3058e640f405a5a6e4e5da49e07ea892b75d`

## References by documentation page

### [`aom_chain_ridge_pls`](aom_chain_ridge_pls.md) — Strict-chain AOM Ridge-PLS selector

No canonical publication defines this strict-chain Ridge-PLS product surface. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_chain_ridge_pls.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

### [`aom_chain_sweep_run`](aom_chain_sweep_run.md) — AOM strict-chain sweep

No single canonical paper defines this strict-chain ABI sweep. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_chain_sweep.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

### [`aom_operator_pls_stack`](aom_operator_pls_stack.md) — AOM operator latent-score Ridge stack

No single paper defines this ABI-2 composition. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_operator_pls_stack.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ensemble.py

### [`aom_pls`](aom_pls.md) — AOM-PLS (global adaptive operator selection)

Beurier, G., Reiter, R., Noûs, C., Rouan, L. & Cornet, D. (2026). *Reframing preprocessing selection as model-internal calibration in near-infrared spectroscopy: a large-scale benchmark of operator-adaptive PLS and Ridge models*. arXiv:2605.13587. https://arxiv.org/abs/2605.13587.

**Provenance:** Current implementation: [cpp/src/core/aom_preprocessing.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_preprocessing.cpp).

### [`aom_pls_superblock`](aom_pls_superblock.md) — AOM PLS superblock

No publication specifies this ABI-2 superblock wrapper; it composes the AOM-PLS family. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

### [`aom_pop_aom_chain_fixed_fit`](aom_pop_aom_chain_fixed_fit.md) — AOM fixed-chain refit

No single canonical paper defines this fixed-candidate ABI wrapper. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_method_result.cpp

### [`aom_pop_aom_chain_screen_refit`](aom_pop_aom_chain_screen_refit.md) — AOM chain screen and refit

No single canonical paper defines this screen/refit orchestration surface. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_campaign.py

### [`aom_pop_calibration`](aom_pop_calibration.md) — Versioned AOM branch calibration

No canonical publication defines this complete ABI-2 calibration protocol. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_calibration.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_aom_calibration.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_calibration.py

### [`aom_pop_linear_ridge_stack`](aom_pop_linear_ridge_stack.md) — Nested-OOF linear Ridge stack

No canonical publication defines this exact AOM candidate bank and affine-export surface; it is a product-specific stacked generalization protocol. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/linear_ridge_stack.py; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/linear_stack.cpp

### [`aom_pop_linear_stack_compress`](aom_pop_linear_stack_compress.md) — Affine linear-stack compression

No separate publication defines this ABI-2 deployment operation; it is the exact affine composition used by the versioned AOM stack contract. Source contract: https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ensemble.h.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/linear_stack.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_linear_stack.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/linear_ridge_stack.py

### [`aom_preprocess`](aom_preprocess.md) — AOM (Adaptive Operator Mixture) preprocessing bank

Beurier, G., Reiter, R., Noûs, C., Rouan, L. & Cornet, D. (2026). *Reframing preprocessing selection as model-internal calibration in near-infrared spectroscopy: a large-scale benchmark of operator-adaptive PLS and Ridge models*. arXiv:2605.13587. https://arxiv.org/abs/2605.13587 — introduces operator-adaptive PLS (AOM-PLS / POP-PLS) and the bench against 50+ NIRS datasets that the git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls` is calibrated against.

**Provenance:** Current implementation: [cpp/src/core/aom_preprocessing.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_preprocessing.cpp).

### [`aom_ridge_active_superblock`](aom_ridge_active_superblock.md) — Active AOM Ridge superblock

No canonical paper defines this active-superblock variant. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

### [`aom_ridge_blender`](aom_ridge_blender.md) — AOM Ridge out-of-fold simplex blender

No canonical paper defines this exact n4m candidate bank; it uses non-negative stacking. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_ridge_blender.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ensemble.py

### [`aom_ridge_global`](aom_ridge_global.md) — AOM Ridge global selector

No canonical paper defines this strict-linear Ridge route. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

### [`aom_ridge_mkl_superblock`](aom_ridge_mkl_superblock.md) — AOM Ridge MKL-light superblock

No canonical publication defines this product-specific MKL-light surface. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

### [`aom_ridge_pls_superblock`](aom_ridge_pls_superblock.md) — AOM Ridge-PLS superblock

No canonical publication defines this combined superblock head. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

### [`aom_ridge_superblock`](aom_ridge_superblock.md) — AOM Ridge superblock

No canonical paper defines this ABI-2 superblock wrapper. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/compose/aom_superblock.py

### [`aom_robust_hpo`](aom_robust_hpo.md) — Strict-linear AOM robust-HPO screen

No single canonical paper defines this ABI-2 compact/wide screen. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_robust_hpo.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_method_result.cpp

### [`aom_staged_chain_campaign`](aom_staged_chain_campaign.md) — Staged strict-chain AOM campaign

No single canonical paper defines this staged orchestration layer. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_campaign.py

### [`aom_sweep_run`](aom_sweep_run.md) — AOM strict-chain sweep

The global AOM-PLS paper motivates operator selection; this ABI-2 sweep has no separate canonical paper. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_sweep.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

### [`approximate_press`](approximate_press.md) — PRESS by leave-one-out refitting

Allen, D. M. (1974). *The relationship between variable selection and data augmentation and a method for prediction*. Technometrics 16(1), 125–127. Verified primary link: [https://doi.org/10.1080/00401706.1974.10489157](https://doi.org/10.1080/00401706.1974.10489157).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`aug_band_mask`](aug_band_mask.md) — Random contiguous-band masking

No single spectroscopy paper defines this operator. It is analogous to structured feature dropout, with exact band sampling defined by the native source.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/band_mask.h

### [`aug_band_perturb`](aug_band_perturb.md) — Local band gain-and-offset perturbation

No canonical paper specifies this bandwise affine perturbation. It is a native spectral corruption heuristic related to offset/gain augmentation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/band_perturb.h

### [`aug_batch_effect`](aug_batch_effect.md) — Offset, slope, and gain batch effects

No canonical publication defines this three-component simulator. It is an internal instrument/session perturbation model related to affine spectral augmentation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/batch_effect.h

### [`aug_channel_dropout`](aug_channel_dropout.md) — Independent spectral-channel dropout

No canonical NIR publication defines this cellwise dropout. The exact mask and replacement rules are implementation-defined.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/channel_dropout.h

### [`aug_dead_band`](aug_dead_band.md) — Random noisy detector dead bands

No canonical paper defines this simulator. The number, width, probability, and replacement-noise rules are implementation-specific.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/dead_band.h

### [`aug_detector_rolloff`](aug_detector_rolloff.md) — Detector sensitivity roll-off artifact

No single paper defines the five detector presets in this implementation. They are literature-inspired internal curves; the enum and native source are the auditable specification.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/detector_rolloff.h

### [`aug_edge_artifacts`](aug_edge_artifacts.md) — Configured pipeline of four edge artifacts

No paper defines this composite. It is an implementation-defined pipeline combining truncated peaks, curvature, stray light, and detector roll-off.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_augmenters_edge_splines_random.cpp#L426

### [`aug_edge_curve`](aug_edge_curve.md) — Smooth detector-edge curvature

No canonical paper defines the smile/frown/asymmetric templates. They are internal edge-response heuristics with formulas fixed by the native source.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/edge_curvature.h

### [`aug_emsc_distort`](aug_emsc_distort.md) — Random EMSC-like affine and polynomial distortion

Martens & Stark (1991), *Extended multiplicative signal correction and spectral interference subtraction*, Journal of Pharmaceutical and Biomedical Analysis 9, 625–635, DOI 10.1016/0731-7085(91)80188-F (https://doi.org/10.1016/0731-7085(91)80188-F). This operator generates EMSC-shaped distortions; it does not perform EMSC correction.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/emsc_distort.h; https://doi.org/10.1016/0731-7085(91)80188-F

### [`aug_gauss_jitter`](aug_gauss_jitter.md) — Random Gaussian smoothing jitter

No unique paper defines randomizing the smoothing width. The operator is a stochastic Gaussian convolution whose source defines boundary behavior.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/gauss_jitter.h

### [`aug_gaussian_noise`](aug_gaussian_noise.md) — Gaussian additive noise scaled to each spectrum

There is no single canonical paper for additive-noise augmentation. Bjerrum, Glahder & Skov (2017), *Data Augmentation of Spectral Data for CNN Based Deep Chemometrics*, arXiv:1710.01927 (https://arxiv.org/abs/1710.01927), is a spectroscopy-specific precedent for perturbing spectra during training.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/gaussian_noise.h; https://arxiv.org/abs/1710.01927

### [`aug_hetero_noise`](aug_hetero_noise.md) — Signal-dependent heteroscedastic noise

No single publication defines this affine noise law. It is an internal measurement-noise heuristic documented by the native implementation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/hetero_noise.h

### [`aug_instrument_broaden`](aug_instrument_broaden.md) — Gaussian instrumental broadening from FWHM

No single paper defines this implementation; Gaussian line-spread convolution is a standard instrumental-resolution model, with exact kernel construction specified in source.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/instrument_broaden.h

### [`aug_linear_drift`](aug_linear_drift.md) — Random affine baseline drift

No unique paper defines this augmenter. Additive offset and slope perturbation is a spectroscopy augmentation pattern used by Bjerrum et al. (2017), arXiv:1710.01927 (https://arxiv.org/abs/1710.01927).

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/drift/linear_drift.h; https://arxiv.org/abs/1710.01927

### [`aug_local_clip`](aug_local_clip.md) — Random local 90th-percentile clipping

No canonical paper specifies this saturation heuristic. Its fixed 90th-percentile rule is defined by the native implementation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/local_clip.h

### [`aug_local_mixup`](aug_local_mixup.md) — Nearest-neighbor constrained mixup

This is an internal local variant of Zhang et al.'s mixup (2018), arXiv:1710.09412 (https://arxiv.org/abs/1710.09412); no separate canonical paper defines the specific k-nearest-neighbor rule used here.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/mixup/local_mixup.h; https://arxiv.org/abs/1710.09412

### [`aug_local_warp`](aug_local_warp.md) — Piecewise-linear local wavelength warp

No paper is canonical for this exact local-warp heuristic. The native source, rather than the historical cubic Python variant, defines the shipped behavior.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/wavelength/local_warp.h

### [`aug_magnitude_warp`](aug_magnitude_warp.md) — Smooth multiplicative magnitude warp

No paper is canonical for this exact spectral warp. The current native linear interpolation path is the normative algorithm.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/magnitude_warp.h

### [`aug_mixup`](aug_mixup.md) — Within-batch convex mixup of spectra

Zhang, Cisse, Dauphin & Lopez-Paz (2018), *mixup: Beyond Empirical Risk Minimization*, ICLR, arXiv:1710.09412 (https://arxiv.org/abs/1710.09412).

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/mixup/mixup.h; https://arxiv.org/abs/1710.09412

### [`aug_moisture`](aug_moisture.md) — Water-activity and moisture-band perturbation

No canonical paper specifies this combined sigmoid/free-bound-water model. It is an internal heuristic with source-defined 1435 and 1930 wavelength bands.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/environmental/moisture.h

### [`aug_multiplicative_noise`](aug_multiplicative_noise.md) — Per-spectrum multiplicative gain noise

No publication uniquely defines this implementation. It belongs to the family of spectral offset/slope/gain augmentation discussed by Bjerrum et al. (2017), arXiv:1710.01927 (https://arxiv.org/abs/1710.01927).

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/multiplicative_noise.h; https://arxiv.org/abs/1710.01927

### [`aug_particle_size`](aug_particle_size.md) — Particle-size and path-length scatter heuristic

No single paper defines this simulator. It is physically motivated by diffuse scattering but uses an explicit empirical power law documented in the source, not a Mie or Kubelka–Munk solver.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/particle_size.h

### [`aug_path_length`](aug_path_length.md) — Multiplicative path-length perturbation

There is no paper canonical to this random simulator. Its physical motivation is the multiplicative path-length/scatter term treated by Martens & Stark (1991), DOI 10.1016/0731-7085(91)80188-F (https://doi.org/10.1016/0731-7085(91)80188-F).

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/drift/path_length.h; https://doi.org/10.1016/0731-7085(91)80188-F

### [`aug_poly_drift`](aug_poly_drift.md) — Random polynomial baseline drift

No canonical publication specifies these random coefficient ranges. The method is a controlled extension of offset/slope spectral augmentation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/drift/poly_drift.h

### [`aug_random_x_op`](aug_random_x_op.md) — Independent random elementwise arithmetic

No canonical paper defines this operator. It is a deliberately simple internal perturbation baseline; the native source is the algorithmic specification.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/random/random_x_op.h

### [`aug_rotate_translate`](aug_rotate_translate.md) — Random hinged rotation-and-translation pattern

No canonical publication defines this piecewise-linear augmenter. Its exact hinge construction and scaling are internal nirs4all behavior.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/random/rotate_translate.h

### [`aug_scatter_sim`](aug_scatter_sim.md) — MSC-style affine scatter simulation

Geladi, MacDougall & Martens (1985), *Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat*, Applied Spectroscopy 39, 491–500, DOI 10.1366/0003702854248656 (https://doi.org/10.1366/0003702854248656). The augmenter simulates the affine effects that MSC is designed to remove.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/scatter_sim_msc.h; https://doi.org/10.1366/0003702854248656

### [`aug_spike_noise`](aug_spike_noise.md) — Sparse impulsive spike injection

No canonical paper defines this exact spike simulator. It is an internal artifact model; the implementation source is the normative specification.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/spike_noise.h

### [`aug_spline_curve_simplify`](aug_spline_curve_simplify.md) — Curve-control cubic-spline simplification

No canonical paper defines this augmenter. It shares the internal interpolating B-spline engine with x simplification and exists for historical parity.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_curve_simplification.h

### [`aug_spline_smooth`](aug_spline_smooth.md) — Deterministic natural-cubic spline smoothing

Reinsch (1967), *Smoothing by spline functions*, Numerische Mathematik 10, 177–183, DOI 10.1007/BF02162161 (https://doi.org/10.1007/BF02162161), provides the classical smoothing-spline basis; the exact fixed smoothing choice here is implementation-specific.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_smoothing.h; https://doi.org/10.1007/BF02162161

### [`aug_spline_x_perturb`](aug_spline_x_perturb.md) — Spline-based x-axis perturbation

No canonical paper defines this random x-grid perturbation. The native implementation is a parity-oriented analogue of SciPy spline resampling.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_x_perturbations.h

### [`aug_spline_x_simplify`](aug_spline_x_simplify.md) — Cubic-spline reconstruction from sparse x controls

No canonical paper defines this augmentation heuristic. It uses a classical interpolating B-spline but the control subset and parity rules are implementation-specific.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_simplify_common.h

### [`aug_spline_y_perturb`](aug_spline_y_perturb.md) — Smooth additive spline perturbation

No publication canonically defines this control-point noise augmenter. Its random-control construction is specified by the native source.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_y_perturbations.h

### [`aug_stray_light`](aug_stray_light.md) — Stray-light distortion in transmittance space

No canonical publication defines this edge-enhanced profile. The physical mixing equation is standard, while its wavelength profile and optional truncation are source-defined.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/stray_light.h

### [`aug_temperature`](aug_temperature.md) — Region-specific temperature perturbation heuristic

No publication canonically defines the six coefficient presets used here. This is an internal phenomenological model; source coefficients and regions are normative.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/environmental/temperature.h

### [`aug_truncated_peak`](aug_truncated_peak.md) — Off-range Gaussian peak tails

No canonical paper defines this artifact generator. It is an internal model of bands whose centers lie just outside the measured interval.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/truncated_peak.h

### [`aug_unsharp_mask`](aug_unsharp_mask.md) — Unsharp spectral masking

Unsharp masking is a classical signal/image sharpening construction, but no single NIR paper defines this stochastic parameterization; source code is normative.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/unsharp_mask.h

### [`aug_wavelength_shift`](aug_wavelength_shift.md) — Rigid wavelength-axis shift

No unique paper defines this interpolation augmenter. It is an internal model of wavelength-registration uncertainty.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/wavelength/wavelength_shift.h

### [`aug_wavelength_stretch`](aug_wavelength_stretch.md) — Wavelength-axis stretch about its center

No canonical publication specifies this augmenter; it is a wavelength-scale calibration heuristic whose exact interpolation is defined by the source.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/wavelength/wavelength_stretch.h

### [`bagging_pls`](bagging_pls.md) — Bagging PLS

Breiman, L. (1996). *Bagging predictors*. Machine Learning 24(2), 123–140. — adapted for PLS by various chemometric authors. Verified primary link: [https://doi.org/10.1023/A:1018054314350](https://doi.org/10.1023/A:1018054314350).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`bipls_select`](bipls_select.md) — biPLS — Backward Interval PLS

Leardi, R. & Nørgaard, L. (2004). *Sequential application of backward interval partial least squares and genetic algorithms for the selection of relevant spectral regions*. Journal of Chemometrics 18(11), 486–497. Verified primary link: [https://doi.org/10.1002/cem.893](https://doi.org/10.1002/cem.893).

**Provenance:** Current implementation: [cpp/src/core/bipls_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/bipls_selection.cpp).

### [`boosting_pls`](boosting_pls.md) — Boosting PLS

Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics 29(5), 1189–1232. — adapted for PLS as a base learner. Verified primary link: [https://doi.org/10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`bve_select`](bve_select.md) — BVE — Backward Variable Elimination

Mehmood, T. et al. (2011). *A Partial Least Squares based algorithm for parsimonious variable selection*. Algorithms for Molecular Biology 6, 27. DOI [10.1186/1748-7188-6-27](https://doi.org/10.1186/1748-7188-6-27).

**Provenance:** Current implementation: [cpp/src/core/bve_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/bve_selection.cpp).

### [`cars_select`](cars_select.md) — CARS — Competitive Adaptive Reweighted Sampling

Li, H., Liang, Y., Xu, Q. & Cao, D. (2009). *Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration*. Analytica Chimica Acta 648(1), 77–84. Verified primary link: [https://doi.org/10.1016/j.aca.2009.06.046](https://doi.org/10.1016/j.aca.2009.06.046).

**Provenance:** Current implementation: [cpp/src/core/cars_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/cars_selection.cpp).

### [`continuum_regression`](continuum_regression.md) — Continuum Regression (Stone & Brooks 1990)

Stone, M. & Brooks, R. J. (1990). *Continuum regression: Cross-validated sequentially constructed prediction embracing ordinary least squares, partial least squares and principal components regression*. Journal of the Royal Statistical Society: Series B 52(2), 237--269. DOI [10.1111/j.2517-6161.1990.tb01786.x](https://doi.org/10.1111/j.2517-6161.1990.tb01786.x).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`cppls`](cppls.md) — Canonical Powered PLS (CPPLS)

Indahl, U. G., Liland, K. H. & Næs, T. (2009). *Canonical partial least squares---a unified PLS approach to classification and regression problems*. Journal of Chemometrics 23(9--10), 495--504. DOI [10.1002/cem.1243](https://doi.org/10.1002/cem.1243).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`di_pls`](di_pls.md) — Domain-Invariant PLS (di-PLS)

Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S. & Moser, B. A. (2018). *Domain-invariant partial-least-squares regression*. Analytical Chemistry 90(11), 6693–6701. Verified primary link: [https://doi.org/10.1021/acs.analchem.8b00498](https://doi.org/10.1021/acs.analchem.8b00498).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`diagnostics_pls_diagnostics`](diagnostics_pls_diagnostics.md) — PLS model diagnostics: Hotelling T², Q residuals, and DModX

Hotelling (1931), *The Generalization of Student's Ratio*, https://doi.org/10.1214/aoms/1177732979; Jackson & Mudholkar (1979), *Control Procedures for Residuals Associated With Principal Component Analysis*, https://doi.org/10.1080/00401706.1979.10489779; Wold et al. (2001), *PLS-regression: a basic tool of chemometrics*, https://doi.org/10.1016/S0169-7439(01)00155-1.

**Provenance:** https://doi.org/10.1214/aoms/1177732979; https://doi.org/10.1080/00401706.1979.10489779; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_diagnostics.cpp

### [`diagnostics_regression_metrics`](diagnostics_regression_metrics.md) — Regression and NIR prediction metrics

No single paper defines this API bundle. RPD usage in NIR calibration is reviewed by Williams & Sobering (1993); RPIQ by Bellon-Maurel et al. (2010), https://doi.org/10.1016/j.trac.2010.07.005. RMSE, MAE, bias, SEP, R², and NRMSE use the explicit formulas in the implementation.

**Provenance:** https://doi.org/10.1016/j.trac.2010.07.005; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/nirs_metrics.c

### [`ds`](ds.md) — Direct Standardisation

Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). *Multivariate instrument standardisation*. Analytical Chemistry 63(23), 2750–2756. Verified primary link: [https://doi.org/10.1021/ac00023a016](https://doi.org/10.1021/ac00023a016).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`ecr`](ecr.md) — ECR — Elastic Component Regression

Li, H.-D., Liang, Y.-Z. & Xu, Q.-S. (2010). *Uncover the path from PCR to PLS via elastic component regression*. Chemometrics and Intelligent Laboratory Systems 104(2), 341--346. DOI [10.1016/j.chemolab.2010.08.003](https://doi.org/10.1016/j.chemolab.2010.08.003).

**Provenance:** Current implementation: [cpp/src/core/ecr.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/ecr.cpp).

### [`emcuve_select`](emcuve_select.md) — EMCUVE — Ensemble MC-UVE

Han, Q.-J., Wu, H.-L., Cai, C.-B., Xu, L. & Yu, R.-Q. (2008). *An ensemble of Monte Carlo uninformative variable elimination for wavelength selection*. Analytica Chimica Acta 612(2), 121–125. https://doi.org/10.1016/j.aca.2008.02.032 — extends the MC-UVE procedure of Cai et al. (2008) (`stability_select`) by aggregating independent MC-UVE rounds through a vote rule.

**Provenance:** Current implementation: [cpp/src/core/emcuve_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/emcuve_selection.cpp).

### [`filter_composite`](filter_composite.md) — Boolean composition of sample keep masks

No paper canonically defines this composition wrapper. Boolean aggregation of precomputed masks is an implementation utility, not a new outlier-detection algorithm.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/composite.h

### [`filter_correlation`](filter_correlation.md) — Absolute Pearson-correlation feature filter

There is no single canonical feature-selection paper for this use of Pearson correlation. The absolute-score threshold and top-k policy are implementation-specific.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L1138

### [`filter_leverage`](filter_leverage.md) — Hat-matrix or PCA-score leverage filter

Rousseeuw & van Zomeren (1990), *Unmasking Multivariate Outliers and Leverage Points*, JASA 85, 633–639, DOI 10.1080/01621459.1990.10474920 (https://doi.org/10.1080/01621459.1990.10474920), provides the leverage/outlier context; the fallback and thresholds here are implementation-specific.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/high_leverage.h; https://doi.org/10.1080/01621459.1990.10474920

### [`filter_quality`](filter_quality.md) — Rule-based row-level spectral quality filter

No canonical paper defines this conjunction of data-quality checks. It is an explicit internal quality-control rule set whose thresholds are user policy.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/spectral_quality.h

### [`filter_variance`](filter_variance.md) — Population-variance feature filter

Variance thresholding is a basic descriptive-statistics filter with no single canonical method paper. The shipped scoring and tie behavior are defined by the ABI implementation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L1138

### [`filter_x_outlier`](filter_x_outlier.md) — Multistrategy X-space outlier filter

The facade combines distinct methods: Liu, Ting & Zhou (2008), Isolation Forest, DOI 10.1109/ICDM.2008.17 (https://doi.org/10.1109/ICDM.2008.17); Breunig et al. (2000), LOF, DOI 10.1145/342009.335388 (https://doi.org/10.1145/342009.335388); and Rousseeuw & Van Driessen (1999), FAST-MCD, DOI 10.1080/00401706.1999.10485670 (https://doi.org/10.1080/00401706.1999.10485670). Mahalanobis and PCA Q/T² options are classical.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/x_outlier.h; https://doi.org/10.1109/ICDM.2008.17; https://doi.org/10.1145/342009.335388; https://doi.org/10.1080/00401706.1999.10485670

### [`filter_y_outlier`](filter_y_outlier.md) — Univariate target outlier filter

No single paper defines this four-rule facade. IQR fences, z scores, percentiles, and scaled MAD are classical robust/descriptive rules; the exact NumPy-compatible percentile interpolation is source-defined.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/y_outlier.h

### [`fused_sparse_pls`](fused_sparse_pls.md) — Fused-sparse PLS

Implementation-specific heuristic inspired by fused sparsity; no canonical fused-lasso solver is implemented.

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`ga_select`](ga_select.md) — GA-PLS — Genetic Algorithm variable selection

Leardi, R. (2000). *Application of genetic algorithm–PLS for feature selection in spectral data sets*. Journal of Chemometrics 14(5–6), 643–655. Verified primary link: [https://doi.org/10.1002/1099-128X(200009/12)14:5/6%3C643::AID-CEM621%3E3.0.CO;2-E](https://doi.org/10.1002/1099-128X(200009/12)14:5/6%3C643::AID-CEM621%3E3.0.CO;2-E).

**Provenance:** Current implementation: [cpp/src/core/ga_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/ga_selection.cpp).

### [`gpr_pls`](gpr_pls.md) — Gaussian Process on PLS scores

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §6.4 (Gaussian Processes). — combined with a preliminary PLS dimensionality reduction for spectroscopy. Verified primary link: [https://link.springer.com/book/9780387310732](https://link.springer.com/book/9780387310732).

**Provenance:** Current implementation: [cpp/src/core/gpr_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/gpr_pls.cpp).

### [`group_sparse_pls`](group_sparse_pls.md) — Group-sparse PLS (Liquet 2016)

Liquet, B., de Micheaux, P. L., Hejblum, B. P. & Thiébaut, R. (2016). *Group and sparse group partial least squares approaches applied in genomics context*. Bioinformatics 32(1), 35–42. Verified primary link: [https://doi.org/10.1093/bioinformatics/btv535](https://doi.org/10.1093/bioinformatics/btv535).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`interval_generator`](interval_generator.md) — Fixed and overlapping interval expansion

No canonical paper: this is a deterministic interval-construction and column-copy operation; `interval_fit` is its normative definition.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L1225-L1303

### [`interval_select`](interval_select.md) — iPLS — Interval PLS (moving-window)

Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., Munck, L. & Engelsen, S. B. (2000). *Interval partial least-squares regression (iPLS): a comparative chemometric study with an example from near-infrared spectroscopy*. Applied Spectroscopy 54(3), 413–419. Verified primary link: [https://doi.org/10.1366/0003702001949500](https://doi.org/10.1366/0003702001949500).

**Provenance:** Current implementation: [cpp/src/core/interval_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/interval_selection.cpp).

### [`ipw_select`](ipw_select.md) — IPW — Iterative Predictor Weighting

Forina, M., Casolino, C. & Pizarro Millán, C. (1999). *Iterative predictor weighting (IPW) PLS: a technique for the elimination of useless predictors in regression problems*. Journal of Chemometrics 13(2), 165–184. https://doi.org/10.1002/(SICI)1099-128X(199903/04)13:2<165::AID-CEM535>3.0.CO;2-Y

**Provenance:** Current implementation: [cpp/src/core/ipw_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/ipw_selection.cpp).

### [`irf_select`](irf_select.md) — Interval Random Frog (IRF) selection

Yun, Y.-H. et al. (2013). *An efficient method of wavelength interval selection based on random frog for multivariate spectral calibration*. Spectrochimica Acta Part A 111, 31--36. DOI [10.1016/j.saa.2013.03.083](https://doi.org/10.1016/j.saa.2013.03.083). The shipped libPLS-compatible path may differ in details; it is not the random-forest IRF of Basu et al.

**Provenance:** Current implementation: [cpp/src/core/irf_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/irf_selection.cpp).

### [`iriv_select`](iriv_select.md) — IRIV — Iteratively Retaining Informative Variables

Yun, Y. H., Wang, W. T., Tan, M. L., Liang, Y. Z., Li, H. D., Cao, D. S., Lu, H. M. & Xu, Q. S. (2014). *A strategy that iteratively retains informative variables for selecting optimal variable subset in multivariate calibration*. Analytica Chimica Acta 807, 36–43. Verified primary link: [https://doi.org/10.1016/j.aca.2013.11.032](https://doi.org/10.1016/j.aca.2013.11.032).

**Provenance:** Current implementation: [cpp/src/core/iriv_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/iriv_selection.cpp).

### [`kernel_pls_rbf`](kernel_pls_rbf.md) — Kernel PLS (Rosipal & Trejo 2001)

Rosipal, R. & Trejo, L. J. (2001). *Kernel partial least squares regression in reproducing kernel Hilbert space*. Journal of Machine Learning Research 2, 97–123. Verified primary link: [https://www.jmlr.org/papers/v2/rosipal01a.html](https://www.jmlr.org/papers/v2/rosipal01a.html).

**Provenance:** Current implementation: [cpp/src/core/kernel_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/kernel_pls.cpp).

### [`lw_pls`](lw_pls.md) — Locally-Weighted PLS (LW-PLS)

Centner, V. & Massart, D. L. (1998). *Optimization in locally weighted regression*. Analytical Chemistry 70(19), 4206--4211. DOI [10.1021/ac980208r](https://doi.org/10.1021/ac980208r).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`mb_pls`](mb_pls.md) — Multi-block PLS (Westerhuis 1998)

Westerhuis, J. A., Kourti, T. & MacGregor, J. F. (1998). *Analysis of multiblock and hierarchical PCA and PLS models*. Journal of Chemometrics 12(5), 301–321. Verified primary link: [https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5%3C301::AID-CEM515%3E3.0.CO;2-S](https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5%3C301::AID-CEM515%3E3.0.CO;2-S).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`mir_pls`](mir_pls.md) — MIR-PLS inverse-regression variant

Implementation-specific inverse-regression construction; no canonical paper is claimed.

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`missing_aware_nipals`](missing_aware_nipals.md) — Missing-aware NIPALS

Walczak, B. & Massart, D. L. (2001). *Dealing with missing data: Part I and Part II*. Chemometrics and Intelligent Laboratory Systems 58(1), 15--27 and 29--42. DOIs [10.1016/S0169-7439(01)00131-9](https://doi.org/10.1016/S0169-7439(01)00131-9) and [10.1016/S0169-7439(01)00132-0](https://doi.org/10.1016/S0169-7439(01)00132-0).

**Provenance:** Current implementation: [cpp/src/core/model.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/model.cpp).

### [`models_ensembles_moment_stack`](models_ensembles_moment_stack.md) — Moment-stack ensemble

No canonical paper defines this n4m moment-stack product route; its implementation is a documented linear stack over moment-compatible candidate predictions.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ensemble.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/moment_facade.py

### [`models_pls_pls_regression`](models_pls_pls_regression.md) — PLS regression with a selectable solver

Wold, Sjöström & Eriksson (2001), *PLS-regression: a basic tool of chemometrics*, Chemometrics and Intelligent Laboratory Systems 58, 109–130, https://doi.org/10.1016/S0169-7439(01)00155-1; de Jong (1993), *SIMPLS: an alternative approach to partial least squares regression*, https://doi.org/10.1016/0169-7439(93)85002-X.

**Provenance:** https://doi.org/10.1016/S0169-7439(01)00155-1; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/model.cpp

### [`models_regularized_ridge`](models_regularized_ridge.md) — Ridge regression

Hoerl, A. E. & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/ridge.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/estimators/regression/regularized.py

### [`n_pls`](n_pls.md) — N-way PLS (Trilinear PLS, Bro 1996)

Bro, R. (1996). *Multiway calibration. Multilinear PLS*. Journal of Chemometrics 10(1), 47–61. Verified primary link: [https://doi.org/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C](https://doi.org/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`o2pls`](o2pls.md) — O2-PLS (two-way orthogonal)

Trygg, J. & Wold, S. (2003). *O2-PLS, a two-block (X–Y) latent variable regression method with an integral OSC filter*. Journal of Chemometrics 17(1), 53–64. Verified primary link: [https://doi.org/10.1002/cem.775](https://doi.org/10.1002/cem.775).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`on_pls`](on_pls.md) — OnPLS (Orthogonal N-block PLS)

Löfstedt, T. & Trygg, J. (2011). *OnPLS — a novel multiblock method for the modelling of predictive and orthogonal variation*. Journal of Chemometrics 25(8), 441–455. Verified primary link: [https://doi.org/10.1002/cem.1388](https://doi.org/10.1002/cem.1388).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`one_se_rule`](one_se_rule.md) — One-SE rule for component selection

Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer, §7.10. Verified primary link: [https://doi.org/10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`opls`](opls.md) — Orthogonal PLS (OPLS)

Trygg, J. & Wold, S. (2002). *Orthogonal projections to latent structures (O-PLS)*. Journal of Chemometrics 16(3), 119--128. DOI [10.1002/cem.695](https://doi.org/10.1002/cem.695).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`pcr`](pcr.md) — Principal Components Regression

Massy, W. F. (1965). *Principal components regression in exploratory statistical research*. Journal of the American Statistical Association 60(309), 234--256. DOI [10.1080/01621459.1965.10480787](https://doi.org/10.1080/01621459.1965.10480787).

**Provenance:** Current implementation: [cpp/src/core/model.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/model.cpp).

### [`pds`](pds.md) — Piecewise Direct Standardisation

Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). *Multivariate instrument standardization*. Analytical Chemistry 63(23), 2750–2756. https://doi.org/10.1021/ac00023a016 — same paper as `ds`; PDS is introduced in §3 (piecewise local regression with a sliding window of width 2w+1).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`pls`](pls.md) — PLS regression (SIMPLS)

de Jong, S. (1993). *SIMPLS: an alternative approach to partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 18(3), 251--263. DOI [10.1016/0169-7439(93)85002-X](https://doi.org/10.1016/0169-7439(93)85002-X).

**Provenance:** Current implementation: [cpp/src/core/model.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/model.cpp).

### [`pls_cox`](pls_cox.md) — PLS survival pseudo-response approximation

Implementation-specific approximation inspired by PLS survival modelling; it is not a Cox partial-likelihood estimator.

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`pls_diagnostic_dmodx`](pls_diagnostic_dmodx.md) — DModX (distance to the model in X)

Eriksson, L., Byrne, T., Johansson, E., Trygg, J. & Vikström, C. (2013). *Multi- and Megavariate Data Analysis. Basic Principles and Applications*, 3rd ed., Umetrics Academy, §4.7. Verified primary catalogue: [https://search.worldcat.org/title/891362497](https://search.worldcat.org/title/891362497).

**Provenance:** Current implementation: [cpp/src/core/pls_diagnostics.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_diagnostics.cpp).

### [`pls_diagnostic_q`](pls_diagnostic_q.md) — Q residual (squared prediction error)

Jackson, J. E. & Mudholkar, G. S. (1979). *Control procedures for residuals associated with principal component analysis*. Technometrics 21(3), 341–349. Verified primary link: [https://doi.org/10.1080/00401706.1979.10489779](https://doi.org/10.1080/00401706.1979.10489779).

**Provenance:** Current implementation: [cpp/src/core/pls_diagnostics.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_diagnostics.cpp).

### [`pls_diagnostic_t2`](pls_diagnostic_t2.md) — Hotelling T² score

Hotelling, H. (1931). *The generalization of Student's ratio*. Annals of Mathematical Statistics 2(3), 360--378. DOI [10.1214/aoms/1177732979](https://doi.org/10.1214/aoms/1177732979).

**Provenance:** Current implementation: [cpp/src/core/pls_diagnostics.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_diagnostics.cpp).

### [`pls_glm`](pls_glm.md) — PLS-GLM compatibility entry point

No canonical paper validates the shipped SIMPLS compatibility path as a Poisson/generalized-linear model; it is not attributed to a GLM solver.

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`pls_lda`](pls_lda.md) — PLS-LDA

Barker, M. & Rayens, W. (2003). *Partial least squares for discrimination*. Journal of Chemometrics 17(3), 166--173. DOI [10.1002/cem.785](https://doi.org/10.1002/cem.785).

**Provenance:** Current implementation: [cpp/src/core/pls_lda.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_lda.cpp).

### [`pls_logistic`](pls_logistic.md) — PLS-logistic regression

Bastien, P., Esposito Vinzi, V. & Tenenhaus, M. (2005). *PLS generalised linear regression*. Computational Statistics & Data Analysis 48(1), 17–46. Verified primary link: [https://doi.org/10.1016/j.csda.2004.02.005](https://doi.org/10.1016/j.csda.2004.02.005).

**Provenance:** Current implementation: [cpp/src/core/pls_logistic.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_logistic.cpp).

### [`pls_monitoring`](pls_monitoring.md) — PLS monitoring (T² + Q with control limits)

Kourti, T. & MacGregor, J. F. (1996). *Multivariate SPC methods for process and product monitoring and control*. Journal of Quality Technology 28(4), 409–428. Verified primary link: [https://doi.org/10.1080/00224065.1996.11979699](https://doi.org/10.1080/00224065.1996.11979699).

**Provenance:** Current implementation: [cpp/src/core/pls_monitoring.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_monitoring.cpp).

### [`pls_qda`](pls_qda.md) — PLS-QDA

Pérez-Enciso, M. & Tenenhaus, M. (2003). *Prediction of clinical outcome with microarray data: a partial least squares discriminant analysis (PLS-DA) approach*. Human Genetics 112, 581--592. DOI [10.1007/s00439-003-0921-9](https://doi.org/10.1007/s00439-003-0921-9).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`pop_pls`](pop_pls.md) — POP-PLS (per-component operator selection)

Beurier, G., Reiter, R., Noûs, C., Rouan, L. & Cornet, D. (2026). *Reframing preprocessing selection as model-internal calibration in near-infrared spectroscopy: a large-scale benchmark of operator-adaptive PLS and Ridge models*. arXiv:2605.13587. https://arxiv.org/abs/2605.13587.

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`pp_airpls`](pp_airpls.md) — Adaptive iteratively reweighted penalized least squares (airPLS)

Zhang, Chen & Liang (2010), *Baseline correction using adaptive iteratively reweighted penalized least squares*, Analyst 135, 1138–1146, https://doi.org/10.1039/B922045C.

**Provenance:** https://doi.org/10.1039/B922045C; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/airpls.c

### [`pp_area`](pp_area.md) — Area normalization

No single canonical paper: total-area normalization is a direct normalization rule. The exact sum, absolute-sum, and trapezoidal definitions are specified by the implementation source.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/area_normalization.c

### [`pp_arpls`](pp_arpls.md) — Asymmetrically reweighted penalized least squares (arPLS)

Baek et al. (2015), *Baseline correction using asymmetrically reweighted penalized least squares smoothing*, Analyst 140, 250–257, https://doi.org/10.1039/C4AN01061B.

**Provenance:** https://doi.org/10.1039/C4AN01061B; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/arpls.c

### [`pp_asls`](pp_asls.md) — Asymmetric least-squares baseline correction (AsLS)

Eilers & Boelens (2005), *Baseline correction with asymmetric least squares smoothing*, Leiden University Medical Centre technical report; algorithmic source also summarized at https://doi.org/10.1039/C4AN01061B.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/asls.c; https://doi.org/10.1039/C4AN01061B

### [`pp_baseline`](pp_baseline.md) — Training-set column mean centering

No single canonical paper: this is the standard centering operation used by PCA, PLS, and linear models. The implementation source is the normative definition.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/baseline.c

### [`pp_beads`](pp_beads.md) — Baseline estimation and denoising with sparsity (BEADS)

Ning, Selesnick & Duval (2014), *Chromatogram baseline estimation and denoising using sparsity (BEADS)*, Chemometrics and Intelligent Laboratory Systems 139, 156–167, https://doi.org/10.1016/j.chemolab.2014.09.014.

**Provenance:** https://doi.org/10.1016/j.chemolab.2014.09.014; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/beads.c

### [`pp_cow_align`](pp_cow_align.md) — Correlation optimized warping (COW)

Nielsen et al. (1998), *Alignment of single and multiple wavelength chromatographic profiles for chemometric data analysis using correlation optimised warping*, Journal of Chemometrics 12, 521–538, https://doi.org/10.1002/(SICI)1099-128X(199811/12)12:6%3C521::AID-CEM515%3E3.0.CO;2-Z.

**Provenance:** https://doi.org/10.1002/(SICI)1099-128X(199811/12)12:6%3C521::AID-CEM515%3E3.0.CO;2-Z; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_crop`](pp_crop.md) — Wavelength-column cropping

No canonical paper: cropping is an indexing operation, defined here by Python's half-open slice convention `[start, end)`.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/crop.c

### [`pp_derivate`](pp_derivate.md) — Order-$d$ finite-difference derivative

No unique canonical paper: this is repeated forward finite differencing. For the spectroscopic rationale see Norris & Williams (1984), *Optimization of mathematical treatments of raw near-infrared signal*.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/derivate.c

### [`pp_detrend`](pp_detrend.md) — Polynomial spectral detrending

Barnes, Dhanoa & Lister (1989), *Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra*, Applied Spectroscopy 43, 772–777, https://doi.org/10.1366/0003702894202201.

**Provenance:** https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/detrend.c

### [`pp_direct_standardization`](pp_direct_standardization.md) — Direct standardization (DS)

Wang et al. (1991), *Improvement of multivariate calibration through instrument standardization*, Analytical Chemistry 63, 2750–2756, https://doi.org/10.1021/ac00023a016.

**Provenance:** https://doi.org/10.1021/ac00023a016; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_dtw_align`](pp_dtw_align.md) — Dynamic time warping spectral alignment

Sakoe & Chiba (1978), *Dynamic programming algorithm optimization for spoken word recognition*, IEEE TASSP 26, 43–49, https://doi.org/10.1109/TASSP.1978.1163055.

**Provenance:** https://doi.org/10.1109/TASSP.1978.1163055; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_emsc`](pp_emsc.md) — Extended multiplicative scatter correction (EMSC)

Martens & Stark (1991), *Extended multiplicative signal correction and spectral interference subtraction*, Journal of Pharmaceutical and Biomedical Analysis 9, 625–635, https://doi.org/10.1016/0731-7085(91)80188-F.

**Provenance:** https://doi.org/10.1016/0731-7085(91)80188-F; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/emsc.c

### [`pp_epo`](pp_epo.md) — External parameter orthogonalization (EPO)

Roger, Chauchard & Bellon-Maurel (2003), *EPO-PLS external parameter orthogonalisation of PLS application to temperature-independent measurement of sugar content of intact fruits*, Chemometrics and Intelligent Laboratory Systems 66, 191–204, https://doi.org/10.1016/S0169-7439(03)00051-0.

**Provenance:** https://doi.org/10.1016/S0169-7439(03)00051-0; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/orthogonalization/epo.c

### [`pp_fck_static`](pp_fck_static.md) — Static fractional convolutional-kernel bank

No canonical paper uniquely defines this n4m operator. It is an implementation-specific bank of fractional convolutional kernels; its generated kernel equation and fixed $\sigma=3$ are defined by `fck_kernel.h` and `fck_static.c`.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/specialized/fck_static.c; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/common/fck_kernel.h

### [`pp_first_derivative`](pp_first_derivative.md) — Shape-preserving first numerical derivative

No unique paper defines `numpy.gradient`; the spectroscopic use of derivatives is discussed by Norris & Williams (1984), *Optimization of mathematical treatments of raw near-infrared signal*.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/first_derivative.c; https://numpy.org/doc/stable/reference/generated/numpy.gradient.html

### [`pp_flex_pca`](pp_flex_pca.md) — Flexible principal component analysis

Pearson (1901), *On Lines and Planes of Closest Fit to Systems of Points in Space*, Philosophical Magazine 2, 559–572, https://doi.org/10.1080/14786440109462720.

**Provenance:** https://doi.org/10.1080/14786440109462720; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/feature_selection/flexible_pca.c

### [`pp_flex_svd`](pp_flex_svd.md) — Flexible truncated singular-value decomposition

Eckart & Young (1936), *The approximation of one matrix by another of lower rank*, Psychometrika 1, 211–218, https://doi.org/10.1007/BF02288367.

**Provenance:** https://doi.org/10.1007/BF02288367; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/feature_selection/flexible_svd.c

### [`pp_frac_to_pct`](pp_frac_to_pct.md) — Fraction-to-percent signal conversion

No canonical paper: percent is exactly the conventional unit conversion $x_{\%}=100x$.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/fraction_to_percent.c

### [`pp_from_absorbance`](pp_from_absorbance.md) — Absorbance-to-reflectance/transmittance conversion

Beer (1852), *Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten*, Annalen der Physik 162, 78–88, https://doi.org/10.1002/andp.18521620505.

**Provenance:** https://doi.org/10.1002/andp.18521620505; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/from_absorbance.c

### [`pp_gaussian`](pp_gaussian.md) — Gaussian smoothing and derivative filtering

No single canonical spectroscopic paper: this is discrete Gaussian convolution. The public reference behaviour is SciPy `gaussian_filter1d`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html.

**Provenance:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/smoothing/gaussian.c

### [`pp_haar`](pp_haar.md) — Single-level Haar discrete wavelet transform

Haar (1910), *Zur Theorie der orthogonalen Funktionensysteme*, Mathematische Annalen 69, 331–371, https://doi.org/10.1007/BF01456326.

**Provenance:** https://doi.org/10.1007/BF01456326; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/haar.c

### [`pp_iasls`](pp_iasls.md) — Improved asymmetric least-squares baseline correction (IAsLS)

He et al. (2014), *Baseline correction for Raman spectra using an improved asymmetric least squares method*, Analytical Methods 6, 4402–4407, https://doi.org/10.1039/C4AY00068D.

**Provenance:** https://doi.org/10.1039/C4AY00068D; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/iasls.c

### [`pp_icoshift_align`](pp_icoshift_align.md) — Interval correlation optimized shifting (icoshift-style)

Savorani, Tomasi & Engelsen (2010), *icoshift: A versatile tool for the rapid alignment of 1D NMR spectra*, Journal of Magnetic Resonance 202, 190–202, https://doi.org/10.1016/j.jmr.2009.11.012.

**Provenance:** https://doi.org/10.1016/j.jmr.2009.11.012; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_imodpoly`](pp_imodpoly.md) — Improved modified-polynomial baseline correction (IModPoly)

Gan, Ruan & Mo (2006), *Baseline correction by improved iterative polynomial fitting with automatic threshold*, Chemometrics and Intelligent Laboratory Systems 82, 59–65, https://doi.org/10.1016/j.chemolab.2005.08.009.

**Provenance:** https://doi.org/10.1016/j.chemolab.2005.08.009; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/imodpoly.c

### [`pp_kbins_disc`](pp_kbins_disc.md) — Per-feature integer k-bin discretization

No single canonical paper: this is uniform or empirical-quantile scalar quantization. Scikit-learn documents the corresponding estimator at https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html.

**Provenance:** https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/kbins_discretizer.c

### [`pp_kubelka_munk`](pp_kubelka_munk.md) — Kubelka–Munk remission transform

Kubelka & Munk (1931), *Ein Beitrag zur Optik der Farbanstriche*, Zeitschrift für Technische Physik 12, 593–601; English translation: https://doi.org/10.1002/col.5080010404.

**Provenance:** https://doi.org/10.1002/col.5080010404; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/kubelka_munk.c

### [`pp_local_centering`](pp_local_centering.md) — Source-to-target local centering

No unique canonical paper: this is a mean-shift domain-adaptation rule, fully specified by the implementation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_localized_msc`](pp_localized_msc.md) — Localized moving-window multiplicative scatter correction

No single paper canonically defines this implementation. It is a moving-window extension of Geladi, MacDougall & Martens (1985), https://doi.org/10.1366/0003702854248656.

**Provenance:** https://doi.org/10.1366/0003702854248656; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_log`](pp_log.md) — Element-wise logarithmic transform

No canonical paper: this is the mathematical logarithm with an implementation-defined offset policy.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/log_transform.c

### [`pp_lsnv`](pp_lsnv.md) — Local standard normal variate (LSNV)

No single canonical paper defines this sliding-window variant. It locally applies the SNV concept of Barnes, Dhanoa & Lister (1989), https://doi.org/10.1366/0003702894202201.

**Provenance:** https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/local_snv.c

### [`pp_modpoly`](pp_modpoly.md) — Modified-polynomial baseline correction (ModPoly)

Lieber & Mahadevan-Jansen (2003), *Automated method for subtraction of fluorescence from biological Raman spectra*, Applied Spectroscopy 57, 1363–1367, https://doi.org/10.1366/000370203322554518.

**Provenance:** https://doi.org/10.1366/000370203322554518; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/modpoly.c

### [`pp_msc`](pp_msc.md) — Multiplicative scatter correction (MSC)

Geladi, MacDougall & Martens (1985), *Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat*, Applied Spectroscopy 39, 491–500, https://doi.org/10.1366/0003702854248656.

**Provenance:** https://doi.org/10.1366/0003702854248656; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/msc.c

### [`pp_normalize`](pp_normalize.md) — Column-wise normalization

No canonical paper: the default is column L2 normalization; a non-default feature range selects column-wise min–max scaling. The source defines this mode switch.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/normalize.c

### [`pp_norris_williams`](pp_norris_williams.md) — Norris–Williams segment-gap derivative

Norris & Williams (1984), *Optimization of mathematical treatments of raw near-infrared signal in the measurement of protein in hard red spring wheat*, Cereal Chemistry 61, 158–165 (bibliographic record: https://www.cerealsgrains.org/publications/cc/backissues/1984/Documents/61_158.pdf).

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/norris_williams.c; https://www.cerealsgrains.org/publications/cc/backissues/1984/Documents/61_158.pdf

### [`pp_osc`](pp_osc.md) — Orthogonal signal correction (OSC)

Wold et al. (1998), *Orthogonal signal correction of near-infrared spectra*, Chemometrics and Intelligent Laboratory Systems 44, 175–185, https://doi.org/10.1016/S0169-7439(98)00109-9.

**Provenance:** https://doi.org/10.1016/S0169-7439(98)00109-9; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/orthogonalization/osc.c

### [`pp_pct_to_frac`](pp_pct_to_frac.md) — Percent-to-fraction signal conversion

No canonical paper: this is exactly the unit conversion $x=x_{\%}/100$.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/percent_to_fraction.c

### [`pp_piecewise_direct_standardization`](pp_piecewise_direct_standardization.md) — Piecewise direct standardization (PDS)

Bouveresse & Massart (1996), *Improvement of the piecewise direct standardization procedure for the transfer of NIR spectra for multivariate calibration*, Chemometrics and Intelligent Laboratory Systems 32, 201–213, https://doi.org/10.1016/0169-7439(95)00074-7.

**Provenance:** https://doi.org/10.1016/0169-7439(95)00074-7; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_piecewise_msc`](pp_piecewise_msc.md) — Piecewise multiplicative scatter correction

No single paper canonically defines this fixed-window variant. It applies the MSC model of Geladi, MacDougall & Martens (1985), https://doi.org/10.1366/0003702854248656, independently by interval.

**Provenance:** https://doi.org/10.1366/0003702854248656; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_piecewise_snv`](pp_piecewise_snv.md) — Piecewise standard normal variate

No single paper canonically defines this interval variant. It applies the SNV normalization of Barnes, Dhanoa & Lister (1989), https://doi.org/10.1366/0003702894202201, independently by interval.

**Provenance:** https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_range_disc`](pp_range_disc.md) — Fixed-edge range discretization

No canonical paper: this is scalar quantization against user-supplied ordered edges, equivalent in concept to `numpy.digitize`: https://numpy.org/doc/stable/reference/generated/numpy.digitize.html.

**Provenance:** https://numpy.org/doc/stable/reference/generated/numpy.digitize.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/range_discretizer.c

### [`pp_resample`](pp_resample.md) — Fixed-length normalized-axis resampling

No unique canonical paper: this is piecewise-linear interpolation on normalized sample positions, matching SciPy `interp1d(kind='linear')`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html.

**Provenance:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/resample_transformer.c

### [`pp_resampler`](pp_resampler.md) — Wavelength-grid resampler

No single canonical paper: the supported linear, nearest, and not-a-knot cubic interpolants follow SciPy interpolation semantics documented at https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html.

**Provenance:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/resampler.c

### [`pp_rnv`](pp_rnv.md) — Robust normal variate (RNV)

Guo, Wu & Massart (1999), *The robust normal variate transform for pattern recognition with near-infrared data*, Analytica Chimica Acta 382, 87–103, https://doi.org/10.1016/S0003-2670(98)00737-5.

**Provenance:** https://doi.org/10.1016/S0003-2670(98)00737-5; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/robust_snv.c

### [`pp_robust_direct_standardization`](pp_robust_direct_standardization.md) — Trimmed robust direct standardization

No single canonical paper defines this implementation. It robustifies direct standardization from Wang et al. (1991), https://doi.org/10.1021/ac00023a016, by iterative residual trimming.

**Provenance:** https://doi.org/10.1021/ac00023a016; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp

### [`pp_rolling_ball`](pp_rolling_ball.md) — Rolling-ball morphological baseline correction

Kneen & Annegarn (1996), *Algorithm for fitting XRF, SEM and PIXE X-ray spectra backgrounds*, Nuclear Instruments and Methods in Physics Research B 109, 209–213, https://doi.org/10.1016/0168-583X(95)00908-6.

**Provenance:** https://doi.org/10.1016/0168-583X(95)00908-6; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/rolling_ball.c

### [`pp_saps`](pp_saps.md) — Score-augmented projection standardization (SAPS)

No canonical publication defines this exact n4m transform. It is an internal score-augmented linear standardization inspired by projection-based calibration transfer; the source is the normative specification.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L356-L496

### [`pp_savgol`](pp_savgol.md) — Savitzky–Golay smoothing and differentiation

Savitzky & Golay (1964), *Smoothing and Differentiation of Data by Simplified Least Squares Procedures*, Analytical Chemistry 36, 1627–1639, https://doi.org/10.1021/ac60214a047.

**Provenance:** https://doi.org/10.1021/ac60214a047; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/savitzky_golay.c

### [`pp_second_derivative`](pp_second_derivative.md) — Shape-preserving second numerical derivative

No unique paper defines two `numpy.gradient` passes; the spectroscopic derivative rationale follows Norris & Williams (1984), while the numerical reference is https://numpy.org/doc/stable/reference/generated/numpy.gradient.html.

**Provenance:** https://numpy.org/doc/stable/reference/generated/numpy.gradient.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/second_derivative.c

### [`pp_simple_scale`](pp_simple_scale.md) — Column-wise min–max scaling to [0, 1]

No canonical paper: this is the affine min–max transform $(x_j-\min x_j)/(\max x_j-\min x_j)$.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/simple_scale.c

### [`pp_slope_bias`](pp_slope_bias.md) — Slope-and-bias prediction correction

No single canonical paper defines this two-parameter post-calibration operation. It is ordinary least-squares bias/slope correction, with source code as the exact specification.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L499-L523

### [`pp_snip`](pp_snip.md) — Statistics-sensitive nonlinear iterative peak clipping (SNIP)

Ryan et al. (1988), *SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications*, Nuclear Instruments and Methods B 34, 396–402, https://doi.org/10.1016/0168-583X(88)90063-8.

**Provenance:** https://doi.org/10.1016/0168-583X(88)90063-8; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/snip.c

### [`pp_snv`](pp_snv.md) — Standard normal variate (SNV)

Barnes, Dhanoa & Lister (1989), *Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra*, Applied Spectroscopy 43, 772–777, https://doi.org/10.1366/0003702894202201.

**Provenance:** https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/snv.c

### [`pp_to_absorbance`](pp_to_absorbance.md) — Reflectance/transmittance-to-absorbance conversion

Beer (1852), *Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten*, Annalen der Physik 162, 78–88, https://doi.org/10.1002/andp.18521620505.

**Provenance:** https://doi.org/10.1002/andp.18521620505; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/to_absorbance.c

### [`pp_vsn`](pp_vsn.md) — Variable-sorting normalization (VSN-style weighted SNV)

Rabatel, Marini, Walczak & Roger (2020), *VSN: Variable sorting for normalization*, Journal of Chemometrics 34, e3164, https://doi.org/10.1002/cem.3164. The shipped correlation-weighted rule is a compact VSN-style implementation, not a claim of full paper parity.

**Provenance:** https://doi.org/10.1002/cem.3164; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L560-L635

### [`pp_wavelet`](pp_wavelet.md) — Single-level discrete wavelet coefficient transform

Mallat (1989), *A theory for multiresolution signal decomposition: the wavelet representation*, IEEE TPAMI 11, 674–693, https://doi.org/10.1109/34.192463.

**Provenance:** https://doi.org/10.1109/34.192463; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet.c; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/common/wavelet_kernels.c

### [`pp_wavelet_denoise`](pp_wavelet_denoise.md) — Multilevel wavelet VisuShrink denoising

Donoho & Johnstone (1994), *Ideal spatial adaptation by wavelet shrinkage*, Biometrika 81, 425–455, https://doi.org/10.1093/biomet/81.3.425.

**Provenance:** https://doi.org/10.1093/biomet/81.3.425; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_denoise.c

### [`pp_wavelet_features`](pp_wavelet_features.md) — Multilevel wavelet summary features

No single canonical paper defines this exact four-statistic descriptor. It uses the multiresolution analysis of Mallat (1989), https://doi.org/10.1109/34.192463, with implementation-specific summaries.

**Provenance:** https://doi.org/10.1109/34.192463; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_features.c

### [`pp_wavelet_pca`](pp_wavelet_pca.md) — PCA of multilevel wavelet coefficients

No unique paper defines this composite. It combines Mallat's DWT (https://doi.org/10.1109/34.192463) with Pearson PCA (https://doi.org/10.1080/14786440109462720).

**Provenance:** https://doi.org/10.1109/34.192463; https://doi.org/10.1080/14786440109462720; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_pca.c

### [`pp_wavelet_svd`](pp_wavelet_svd.md) — Truncated SVD of multilevel wavelet coefficients

No unique paper defines this composite. It combines Mallat's DWT (https://doi.org/10.1109/34.192463) with Eckart–Young low-rank approximation (https://doi.org/10.1007/BF02288367).

**Provenance:** https://doi.org/10.1109/34.192463; https://doi.org/10.1007/BF02288367; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_svd.c

### [`pp_weighted_snv`](pp_weighted_snv.md) — Weighted standard normal variate

No single canonical paper defines this exact weighted estimator. It generalizes the SNV transform of Barnes, Dhanoa & Lister (1989), https://doi.org/10.1366/0003702894202201.

**Provenance:** https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L526-L635

### [`pp_xcorr_align`](pp_xcorr_align.md) — Whole-spectrum cross-correlation alignment

No single canonical paper defines this bounded implementation. It applies the standard discrete cross-correlation lag estimator; the source defines padding and tie behaviour.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L798-L900

### [`pso_select`](pso_select.md) — PSO-PLS — Particle Swarm Optimisation

Kennedy, J. & Eberhart, R. (1995). *Particle swarm optimization*. IEEE ICNN 1995, vol. 4, 1942–1948. — binary PSO variant used for variable selection. Verified primary link: [https://doi.org/10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).

**Provenance:** Current implementation: [cpp/src/core/pso_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pso_selection.cpp).

### [`random_frog_select`](random_frog_select.md) — Random Frog

Li, H., Xu, Q. & Liang, Y. (2012). *Random frog: an efficient reversible jump Markov chain Monte Carlo-like approach for variable selection*. Analytica Chimica Acta 740, 20–26. Verified primary link: [https://doi.org/10.1016/j.aca.2012.06.031](https://doi.org/10.1016/j.aca.2012.06.031).

**Provenance:** Current implementation: [cpp/src/core/random_frog_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/random_frog_selection.cpp).

### [`random_subspace_pls`](random_subspace_pls.md) — Random-subspace PLS

Ho, T. K. (1998). *The random subspace method for constructing decision forests*. IEEE TPAMI 20(8), 832–844. — adapted for PLS regressors. Verified primary link: [https://doi.org/10.1109/34.709601](https://doi.org/10.1109/34.709601).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`randomization_select`](randomization_select.md) — Randomisation test (Y-permutation)

Westad, F. & Martens, H. (2000). *Variable selection in near infrared spectroscopy based on significance testing in partial least squares regression*. JNIRS 8(2), 117–124. Verified primary link: [https://doi.org/10.1255/jnirs.271](https://doi.org/10.1255/jnirs.271).

**Provenance:** Current implementation: [cpp/src/core/randomization_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/randomization_selection.cpp).

### [`recursive_pls`](recursive_pls.md) — Recursive (moving-window) PLS

Helland, K., Berntsen, H. E., Borgen, O. S. & Martens, H. (1992). *Recursive algorithm for partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 14(1--3), 129--137. DOI [10.1016/0169-7439(92)80098-O](https://doi.org/10.1016/0169-7439(92)80098-O).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`rep_select`](rep_select.md) — REP — Recursive Elimination of Predictors

Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). *A review of variable selection methods in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 118, 62–69. https://doi.org/10.1016/j.chemolab.2012.07.010 — same review as `shaving_select`; §3.3 *Recursive elimination* introduces the fixed-count variant implemented here.

**Provenance:** Current implementation: [cpp/src/core/rep_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/rep_selection.cpp).

### [`ridge_pls`](ridge_pls.md) — Ridge-augmented PLS

Hoerl, A. E. & Kennard, R. W. (1970). *Ridge regression: biased estimation for nonorthogonal problems*. Technometrics 12(1), 55–67. — combined with PLS via Tikhonov regularisation of the inner regression. Verified primary link: [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`robust_pls`](robust_pls.md) — Robust PLS (Partial Robust M-regression)

Serneels, S., Croux, C., Filzmoser, P. & Van Espen, P. J. (2005). *Partial robust M-regression*. Chemometrics and Intelligent Laboratory Systems 79(1--2), 55--64. DOI [10.1016/j.chemolab.2005.04.007](https://doi.org/10.1016/j.chemolab.2005.04.007).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`rosa`](rosa.md) — ROSA (Response-Oriented Sequential Alternation)

Liland, K. H., Næs, T. & Indahl, U. G. (2016). *ROSA---a fast extension of partial least squares regression for multiblock data analysis*. Journal of Chemometrics 30(11), 651--662. DOI [10.1002/cem.2824](https://doi.org/10.1002/cem.2824).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`scars_select`](scars_select.md) — SCARS — Stability-CARS

Zheng, K., Li, Q., Wang, J., Geng, J., Cao, P., Sui, T., Wang, X. & Du, Y. (2012). *Stability competitive adaptive reweighted sampling (SCARS) and its applications to multivariate calibration of NIR spectra*. Chemometrics and Intelligent Laboratory Systems 112, 48–54. Verified primary link: [https://doi.org/10.1016/j.chemolab.2012.01.002](https://doi.org/10.1016/j.chemolab.2012.01.002).

**Provenance:** Current implementation: [cpp/src/core/scars_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/scars_selection.cpp).

### [`shaving_select`](shaving_select.md) — Shaving (recursive elimination)

Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). *A review of variable selection methods in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 118, 62–69 (§3.2 Shaving). Verified primary link: [https://doi.org/10.1016/j.chemolab.2012.07.010](https://doi.org/10.1016/j.chemolab.2012.07.010).

**Provenance:** Current implementation: [cpp/src/core/shaving_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/shaving_selection.cpp).

### [`sipls_select`](sipls_select.md) — siPLS — Synergy Interval PLS

Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., Munck, L. & Engelsen, S. B. (2000). *Interval partial least-squares regression (iPLS): a comparative chemometric study with an example from near-infrared spectroscopy*. Applied Spectroscopy 54(3), 413–419 — same paper as `interval_select`; siPLS is the synergy-combinations extension proposed in §3. Verified primary link: [https://doi.org/10.1366/0003702001949500](https://doi.org/10.1366/0003702001949500).

**Provenance:** Current implementation: [cpp/src/core/sipls_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/sipls_selection.cpp).

### [`so_pls`](so_pls.md) — Sequential and Orthogonalised PLS (SO-PLS)

Næs, T., Tomic, O., Mevik, B.-H. & Martens, H. (2011). *Path modelling by sequential PLS regression*. Journal of Chemometrics 25(1), 28–40. Verified primary link: [https://doi.org/10.1002/cem.1357](https://doi.org/10.1002/cem.1357).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`spa_select`](spa_select.md) — SPA — Successive Projections Algorithm

Araújo, M. C. U., Saldanha, T. C. B., Galvão, R. K. H., Yoneyama, T., Chame, H. C. & Visani, V. (2001). *The successive projections algorithm for variable selection in spectroscopic multicomponent analysis*. Chemometrics and Intelligent Laboratory Systems 57(2), 65–73. Verified primary link: [https://doi.org/10.1016/S0169-7439(01)00119-8](https://doi.org/10.1016/S0169-7439(01)00119-8).

**Provenance:** Current implementation: [cpp/src/core/spa_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/spa_selection.cpp).

### [`sparse_pls_da`](sparse_pls_da.md) — Sparse PLS-DA (Lê Cao 2008)

Lê Cao, K.-A., Rossouw, D., Robert-Granié, C. & Besse, P. (2008). *A sparse PLS for variable selection when integrating omics data*. Statistical Applications in Genetics and Molecular Biology 7(1). Verified primary link: [https://doi.org/10.2202/1544-6115.1390](https://doi.org/10.2202/1544-6115.1390).

**Provenance:** Current implementation: [cpp/src/c_api/c_api_method_result.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_method_result.cpp).

### [`sparse_simpls`](sparse_simpls.md) — Sparse SIMPLS (Chun & Keleş 2010)

Chun, H. & Keleş, S. (2010). *Sparse partial least squares regression for simultaneous dimension reduction and variable selection*. Journal of the Royal Statistical Society: Series B 72(1), 3--25. DOI [10.1111/j.1467-9868.2009.00723.x](https://doi.org/10.1111/j.1467-9868.2009.00723.x).

**Provenance:** Current implementation: [cpp/src/core/model.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/model.cpp).

### [`split_binned_strat_group_kfold`](split_binned_strat_group_kfold.md) — Binned, group-preserving round-robin folds

No canonical paper defines this simplified grouped heuristic. It is inspired by binned stratification and `StratifiedGroupKFold`, but the native source explicitly documents a different assignment algorithm.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/binned_strat_group_kfold.h

### [`split_kbins_stratified`](split_kbins_stratified.md) — Continuous-target binned stratified split

No single paper defines this composition of discretization and stratified shuffle split. The parity target is the scikit-learn `KBinsDiscretizer` plus `StratifiedShuffleSplit` behavior documented at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/kbins_stratified.h; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

### [`split_kennard_stone`](split_kennard_stone.md) — Kennard–Stone maximin calibration split

Kennard & Stone (1969), *Computer Aided Design of Experiments*, Technometrics 11, 137–148, DOI 10.1080/00401706.1969.10490666 (https://doi.org/10.1080/00401706.1969.10490666).

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/kennard_stone.h; https://doi.org/10.1080/00401706.1969.10490666

### [`split_kmeans`](split_kmeans.md) — K-means++ representative-sample split

Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*, SODA, 1027–1035 (https://dl.acm.org/doi/10.5555/1283383.1283494). The surrounding representative split is implementation-specific.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/kmeans.h; https://dl.acm.org/doi/10.5555/1283383.1283494

### [`split_split_splitter`](split_split_splitter.md) — SPlit sequential data-twinning split

Joseph & Vakayil (2021/2022), *SPlit: An Optimal Method for Data Splitting*, Technometrics 64, 166–176, DOI 10.1080/00401706.2021.1921037 (https://doi.org/10.1080/00401706.2021.1921037). The shipped routine is the paper-inspired sequential nearest-neighbor algorithm documented in source.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/split_splitter.h; https://doi.org/10.1080/00401706.2021.1921037

### [`split_spxy`](split_spxy.md) — SPXY joint X–Y maximin split

Galvão et al. (2005), *A method for calibration and validation subset partitioning*, Talanta 67, 736–740, DOI 10.1016/j.talanta.2005.03.025 (https://doi.org/10.1016/j.talanta.2005.03.025).

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/spxy.h; https://doi.org/10.1016/j.talanta.2005.03.025

### [`split_spxy_fold`](split_spxy_fold.md) — Alternating maximin SPXY K-fold assignment

This is an internal K-fold extension of Galvão et al.'s SPXY criterion (2005), DOI 10.1016/j.talanta.2005.03.025 (https://doi.org/10.1016/j.talanta.2005.03.025); the alternating fold assignment itself is not defined by that paper.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L146; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/spxy_fold.h; https://doi.org/10.1016/j.talanta.2005.03.025

### [`split_spxy_g_fold`](split_spxy_g_fold.md) — Group-preserving SPXY K-fold assignment

This is an internal grouped extension of SPXY (Galvão et al., 2005, DOI 10.1016/j.talanta.2005.03.025, https://doi.org/10.1016/j.talanta.2005.03.025); no canonical paper defines its representative aggregation.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/spxy_g_fold.h; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L202; https://doi.org/10.1016/j.talanta.2005.03.025

### [`split_systematic_circular`](split_systematic_circular.md) — Systematic circular sampling over sorted targets

No canonical paper defines this exact rotate-and-systematically-sample implementation. It is an internal systematic sampling heuristic whose seeded offset and rounding rules are source-defined.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/systematic_circular.h

### [`st_select`](st_select.md) — ST-PLS — Score Threshold selection

Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). *A review of variable selection methods in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 118, 62–69. https://doi.org/10.1016/j.chemolab.2012.07.010 — same review as `shaving_select`; §3.4 *Score-threshold methods* covers the deterministic-threshold family implemented here.

**Provenance:** Current implementation: [cpp/src/core/st_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/st_selection.cpp).

### [`stability_select`](stability_select.md) — MC-UVE (Monte-Carlo coefficient stability)

Cai, W., Li, Y. & Shao, X. (2008). *A variable selection method based on uninformative variable elimination for multivariate calibration of near-infrared spectra*. Chemometrics and Intelligent Laboratory Systems 90(2), 188–194. Verified primary link: [https://doi.org/10.1016/j.chemolab.2007.10.001](https://doi.org/10.1016/j.chemolab.2007.10.001).

**Provenance:** Current implementation: [cpp/src/core/stability_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/stability_selection.cpp).

### [`t2_select`](t2_select.md) — Hotelling T² loading selection

Mehmood, T. (2016). *Hotelling T² based variable selection in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 154, 23–28. https://doi.org/10.1016/j.chemolab.2016.03.020 — proposes T²-PLS, the loading-weights-level Hotelling T² selector. See also Wold, Sjöström & Eriksson (2001), Chemometrics and Intelligent Laboratory Systems 58(2), 109–130 §6.2 for the original T²-vs-VIP discussion in PLS.

**Provenance:** Current implementation: [cpp/src/core/t2_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/t2_selection.cpp).

### [`utilities_hotelling_t2`](utilities_hotelling_t2.md) — Standalone PCA Hotelling T² statistic

Hotelling (1931), *The Generalization of Student's Ratio*, Annals of Mathematical Statistics 2, 360–378, https://doi.org/10.1214/aoms/1177732979.

**Provenance:** https://doi.org/10.1214/aoms/1177732979; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/hotelling_t2.c

### [`utilities_moments`](utilities_moments.md) — Moment sufficient-statistics utilities

No single canonical paper defines the ABI utility surface; it implements standard first- and second-moment sufficient statistics used by linear-model updates.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/lowlevel/moments.py; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/lowlevel.h

### [`utilities_q_residuals`](utilities_q_residuals.md) — Standalone PCA Q residuals (squared prediction error)

Jackson & Mudholkar (1979), *Control Procedures for Residuals Associated With Principal Component Analysis*, Technometrics 21, 341–349, https://doi.org/10.1080/00401706.1979.10489779.

**Provenance:** https://doi.org/10.1080/00401706.1979.10489779; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/q_residuals.c

### [`utilities_signal_type_detector`](utilities_signal_type_detector.md) — Heuristic spectral signal-type detector

No canonical scientific paper defines these thresholds. This is an n4m compatibility heuristic, and `signal_type_detector.c` is the authoritative specification.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/signal_type_detector.c

### [`utilities_sweep`](utilities_sweep.md) — Native candidate sweep utility

No canonical scientific paper defines this orchestration utility; it is a reproducible enumeration and scoring surface.

**Provenance:** https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/sweep.py; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h

### [`utilities_transfer_metrics`](utilities_transfer_metrics.md) — Source–target representation transfer metrics

This bundle has no single canonical paper. Its components include linear CKA (Kornblith et al., 2019, https://proceedings.mlr.press/v97/kornblith19a.html), Procrustes analysis (Gower, 1975, https://doi.org/10.1007/BF02291478), and trustworthiness (Venna & Kaski, 2001, https://doi.org/10.1016/S0893-6080(01)00150-6).

**Provenance:** https://proceedings.mlr.press/v97/kornblith19a.html; https://doi.org/10.1007/BF02291478; https://doi.org/10.1016/S0893-6080(01)00150-6; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/transfer_metrics.c

### [`uve_select`](uve_select.md) — UVE — Uninformative Variable Elimination

Centner, V., Massart, D. L., de Noord, O. E., de Jong, S., Vandeginste, B. M. & Sterna, C. (1996). *Elimination of uninformative variables for multivariate calibration*. Analytical Chemistry 68(21), 3851–3858. Verified primary link: [https://doi.org/10.1021/ac960321m](https://doi.org/10.1021/ac960321m).

**Provenance:** Current implementation: [cpp/src/core/uve_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/uve_selection.cpp).

### [`variable_select_coef`](variable_select_coef.md) — Coefficient-magnitude selection

Martens, H. & Næs, T. (1989). *Multivariate Calibration*, §5. — the simplest ranking baseline. Verified primary catalogue: [https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282](https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282).

**Provenance:** Current implementation: [cpp/src/core/variable_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/variable_selection.cpp).

### [`variable_select_sr`](variable_select_sr.md) — Selectivity Ratio

Rajalahti, T., Arneberg, R., Berven, F. S., Myhr, K.-M., Ulvik, R. J. & Kvalheim, O. M. (2009). *Biomarker discovery in mass spectral profiles by means of selectivity ratio plot*. Chemometrics and Intelligent Laboratory Systems 95(1), 35–48. Verified primary link: [https://doi.org/10.1016/j.chemolab.2008.08.004](https://doi.org/10.1016/j.chemolab.2008.08.004).

**Provenance:** Current implementation: [cpp/src/core/variable_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/variable_selection.cpp).

### [`variable_select_vip`](variable_select_vip.md) — VIP (Variable Importance in Projection)

Wold, S., Sjöström, M. & Eriksson, L. (2001). *PLS-regression: a basic tool of chemometrics*. Chemometrics and Intelligent Laboratory Systems 58(2), 109–130. Verified primary link: [https://doi.org/10.1016/S0169-7439(01)00155-1](https://doi.org/10.1016/S0169-7439(01)00155-1).

**Provenance:** Current implementation: [cpp/src/core/variable_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/variable_selection.cpp).

### [`vip_spa_select`](vip_spa_select.md) — VIP-seeded SPA

Hybrid heuristic combining VIP ranking and the Successive Projections Algorithm. See registry notes; no single canonical paper.

**Provenance:** Current implementation: [cpp/src/core/vip_spa_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/vip_spa_selection.cpp).

### [`vissa_select`](vissa_select.md) — VISSA — Variable Iterative Space-Shrinkage

Deng, B. C., Yun, Y. H., Liang, Y. Z. & Yi, L. Z. (2015). *A new strategy to prevent over-fitting in partial least squares models based on model population analysis*. Analytica Chimica Acta 880, 32--41. DOI [10.1016/j.aca.2015.04.045](https://doi.org/10.1016/j.aca.2015.04.045).

**Provenance:** Current implementation: [cpp/src/core/vissa_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/vissa_selection.cpp).

### [`weighted_pls`](weighted_pls.md) — Sample-weighted PLS

Martens, H. & Næs, T. (1989). *Multivariate Calibration*. Wiley. §4.5 'Weighted regression for non-i.i.d. errors'. Verified primary catalogue: [https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282](https://search.worldcat.org/title/Multivariate-calibration/oclc/19847282).

**Provenance:** Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).

### [`wvc_select`](wvc_select.md) — WVC — Weighted Variable Contribution

Andries, J. P. M. & Vander Heyden, Y. (2011). *Improved variable reduction in partial least squares modelling based on predictive-property-ranked variables and adaptation of partial least squares complexity*. Analytica Chimica Acta 705(1–2), 292–305. Verified primary link: [https://doi.org/10.1016/j.aca.2011.06.037](https://doi.org/10.1016/j.aca.2011.06.037).

**Provenance:** Current implementation: [cpp/src/core/wvc_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/wvc_selection.cpp).

### [`wvc_threshold_select`](wvc_threshold_select.md) — WVC-threshold selection

Andries, J. P. M. & Vander Heyden, Y. (2011). *Improved variable reduction in partial least squares modelling based on predictive-property-ranked variables and adaptation of partial least squares complexity*. Analytica Chimica Acta 705(1–2), 292–305. https://doi.org/10.1016/j.aca.2011.06.037 — same paper as `wvc_select`; introduces both the top-$k$ ranking and the threshold / factor-of-mean rules used here.

**Provenance:** Current implementation: [cpp/src/core/wvc_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/wvc_selection.cpp).
