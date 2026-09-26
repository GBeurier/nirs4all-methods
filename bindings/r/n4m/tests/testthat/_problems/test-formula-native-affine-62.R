# Extracted from test-formula-native-affine.R:62

# test -------------------------------------------------------------------------
set.seed(12)
X <- matrix(stats::rnorm(30L * 6L), 30L, 6L)
data <- data.frame(X, y = 0.3 + X[, 2L] - 0.2 * X[, 4L])
weighted <- weighted_pls(y ~ ., data, ncomp = 2L,
                           weights = rep(1, nrow(data)))
glm <- pls_glm(y ~ ., data, ncomp = 2L)
for (fit in list(weighted, glm)) {
    expect_null(fit$native_model)
    expect_equal(unname(predict(fit, data)), as.numeric(fit$predictions),
                 tolerance = 1e-10)
  }
