# SPDX-License-Identifier: CECILL-2.1

test_that("22 X-only native augmentations preserve shape and seeded result", {
    X <- matrix(rep(1 + 0.02 * (0:31), each = 8L), nrow = 8L)
    original_X <- X + 0
    original_Y <- seq_len(nrow(X))
    cases <- list(
        gaussian_noise = c(.03), multiplicative_noise = c(.03),
        spike_noise = c(1, 2, .1, .2), hetero_noise = c(.01, .02),
        linear_drift = c(.01, .02, .001, .002), path_length = c(.05, .5),
        band_perturb = c(1, 2, 4, .9, 1.1, -.01, .01),
        band_mask = c(1, 1, 2, 4, 0), channel_dropout = c(.1, 0),
        gauss_jitter = c(.5, 1, 5), unsharp_mask = c(.1, .2, 1, 5),
        local_clip = c(1, 2, 4), rotate_translate = c(.05, .05),
        random_x_op = c(0, .9, 1.1),
        scatter_sim_msc = c(.9, 1.1, -.01, .01),
        dead_band = c(1, 2, 4, .01, .5, 0),
        batch_effect = c(.01, .01, .01, 0), spline_smoothing = numeric(),
        spline_x_perturb = c(3, .2, -.1, .1),
        spline_y_perturb = c(4, .1), spline_x_simplify = c(8, 1),
        spline_curve_simplify = c(8, 1))
    for (kind in names(cases)) {
        first <- n4m_augmentation_apply(kind, X, cases[[kind]], seed = 42)
        second <- n4m_augmentation_apply(kind, X, cases[[kind]], seed = 42)
        expect_identical(first, second, info = kind)
        expect_identical(dim(first), dim(X), info = kind)
        expect_true(all(is.finite(first)), info = kind)
    }
    expect_identical(X, original_X)
    expect_identical(seq_len(nrow(X)), original_Y)
    expect_error(n4m_augmentation_apply("mixup", X, .5), "unknown")
    expect_error(n4m_augmentation_apply("gaussian_noise", X), "parameter")
    expect_error(n4m_augmentation_apply("spike_noise", X,
                                         c(1.5, 2, .1, .2)), "invalid argument")
    expect_error(n4m_augmentation_apply("gaussian_noise", X, .03,
                                         seed = 2^53), "exact")
})
