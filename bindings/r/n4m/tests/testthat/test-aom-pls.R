test_that("AOM-PLS and POP-PLS return prediction matrices", {
    set.seed(42)
    X <- matrix(rnorm(36 * 18), nrow = 36)
    Y <- as.numeric(0.7 * X[, 1] - 0.2 * X[, 2] + rnorm(36, sd = 0.05))

    aom <- aom_pls(X, Y, max_components = 2L, n_operators = 4L, cv = 3L)
    expect_named(aom, c("predictions", "operator_kinds", "operator_scores",
                        "rmse_curves", "selected_operator_index",
                        "selected_n_components", "n_operators",
                        "max_components", "best_score"))
    expect_equal(dim(aom$predictions), c(nrow(X), 1L))
    expect_true(aom$selected_operator_index >= 1L)
    expect_true(aom$selected_n_components >= 1L)
    expect_equal(dim(aompls(X, Y, max_components = 2L,
                            n_operators = 4L, cv = 3L)$predictions),
                 c(nrow(X), 1L))

    pop <- pop_pls(X, Y, max_components = 2L, n_operators = 4L, cv = 3L)
    expect_named(pop, c("predictions", "operator_kinds",
                        "selected_operator_indices", "component_scores",
                        "prefix_scores", "selected_n_components",
                        "n_operators", "max_components", "best_score"))
    expect_equal(dim(pop$predictions), c(nrow(X), 1L))
    expect_true(all(pop$selected_operator_indices >= 1L))
    expect_true(pop$selected_n_components >= 1L)
    expect_equal(dim(poppls(X, Y, max_components = 2L,
                            n_operators = 4L, cv = 3L)$predictions),
                 c(nrow(X), 1L))
})
