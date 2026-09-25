test_that("R MethodResult solver conventions match the native Python binding", {
    testthat::skip_if_not("n4m" %in% names(getLoadedDLLs()),
                          "n4m DLL not loaded")
    X <- outer(seq_len(12L), seq_len(8L),
               function(i, j) sin(i * j / 7) + i * j / 50)
    y <- 2 + X[, 2L] - 0.3 * X[, 5L]
    expected <- list(
        cppls = c(2.4782440961054117, 2.2890294509225764,
                  2.437026067885301),
        ridge_pls = c(2.4892550678094483, 2.3107822329231467,
                      2.4563965646520964),
        continuum_regression = c(2.3604419059764936,
                                 2.250654439880409, 2.4400245161393572))
    for (method in names(expected)) {
        fit <- n4m_method(method, X, y, 2L)
        expect_equal(as.numeric(fit$predictions[1:3]), expected[[method]],
                     tolerance = 1e-12)
    }
})
