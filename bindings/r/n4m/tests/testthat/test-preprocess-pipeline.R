# SPDX-License-Identifier: CECILL-2.1

test_that("generic pipeline advertises only implemented C operators", {
    capabilities <- n4m_preprocess_capabilities()
    expect_equal(nrow(capabilities), 19L)
    expect_equal(sum(capabilities$supported), 15L)
    expect_equal(capabilities$code, 0:18)
    for (kind in capabilities$kind[!capabilities$supported]) {
        expect_error(n4m_preprocess_step(kind), "not implemented")
    }
    expect_error(n4m_preprocess_step("not_an_operator"), "unknown")
    expect_error(n4m_preprocess_step("snv", Inf), "finite")
    expect_error(n4m_preprocess_fit(matrix(1:12, nrow = 3),
                                    list(n4m_preprocess_step("snv", 1))),
                 "SNV does not accept parameters")
    expect_error(n4m_preprocess_fit(matrix(1:12, nrow = 3),
                                    list(n4m_preprocess_step("savgol_smooth", 7))),
                 "smooth expects zero params or window/poly_degree")
    expect_error(n4m_preprocess_fit(matrix(NA_real_, 2, 2),
                                    list(n4m_preprocess_step("identity"))), "finite")
    single_feature <- matrix(c(1, 2, 3), ncol = 1L)
    fitted <- n4m_preprocess_fit(single_feature,
                                 list(n4m_preprocess_step("identity")))
    expect_equal(n4m_preprocess_transform(fitted, single_feature), single_feature)
})

test_that("fitted generic pipeline matches independent Python n4m held-out oracle", {
    features <- 1:8
    signal <- function(samples) {
        outer(samples, features, function(i, j) {
            sin(i * j / 9) + cos(i + j / 7) + i * j / 100
        })
    }
    train <- signal(1:6)
    held_out <- signal(c(2.5, 7.5))
    fitted <- n4m_preprocess_fit(train, list(
        n4m_preprocess_step("snv"), n4m_preprocess_step("msc")))
    # Python n4m.transform.scatter.SNV + MSC, fit on train only.
    expected <- rbind(
        c(-4.001193774003808, -2.2897281335515838, -0.7449819940745884,
          0.5119733321851917, 1.3908885370727357, 1.8394358338967038,
          1.8472190824668655, 1.446387116008484),
        c(-2.2015814295950307, -2.6519026524658353, -1.5440849758961113,
          0.4949255610220601, 2.225790408871716, 2.604408534032623,
          1.461767129717428, -0.3893225756868496))
    expect_equal(n4m_preprocess_transform(fitted, held_out), expected, tolerance = 1e-10)
    expect_equal(n4m_preprocess_transform(fitted, held_out), expected, tolerance = 1e-10)
    expect_error(n4m_preprocess_transform(fitted, held_out[, -1, drop = FALSE]),
                 "feature width")
    expect_error(n4m_preprocess_fit(train, list(n4m_preprocess_step("osc"))),
                 "pipeline fit")
})

test_that("all 15 implemented operator kinds use native fit and transform", {
    features <- 1:24
    signal <- function(samples) {
        outer(samples, features, function(i, j) {
            sin(i * j / 31) + cos(i + j / 11) + i * j / 1000
        })
    }
    train <- signal(1:30)
    held_out <- signal(c(3.5, 17.5))
    response <- matrix(seq_len(nrow(train)) / nrow(train), ncol = 1L)
    for (kind in n4m_preprocess_capabilities()$kind[1:15]) {
        step <- n4m_preprocess_step(kind)
        target <- if (kind %in% c("osc", "epo")) response else NULL
        fitted <- n4m_preprocess_fit(train, list(step), Y = target)
        result <- n4m_preprocess_transform(fitted, held_out)
        expect_equal(dim(result), dim(held_out), info = kind)
        expect_true(all(is.finite(result)), info = kind)
        bytes <- n4m_preprocess_export(fitted)
        expect_equal(typeof(bytes), "raw", info = kind)
        imported <- n4m_preprocess_import(bytes)
        expect_equal(n4m_preprocess_plan(imported), list(step), info = kind)
        expect_equal(n4m_preprocess_transform(imported, held_out), result,
                     tolerance = 1e-12, info = kind)
        expect_equal(n4m_preprocess_export(imported), bytes, info = kind)
    }
})

test_that("portable fitted preprocessing rejects corruption and width mismatch", {
    x <- outer(1:12, 1:12, function(i, j) sin(i * j / 13) + i / 20)
    fitted <- n4m_preprocess_fit(x, list(
        n4m_preprocess_step("center"), n4m_preprocess_step("snv"),
        n4m_preprocess_step("msc")))
    bytes <- n4m_preprocess_export(fitted)
    imported <- n4m_preprocess_import(unserialize(serialize(bytes, NULL)))
    held_out <- x[c(2, 5), , drop = FALSE]
    expect_equal(n4m_preprocess_transform(imported, held_out),
                 n4m_preprocess_transform(fitted, held_out), tolerance = 1e-12)
    expect_error(n4m_preprocess_transform(imported, held_out[, -1, drop = FALSE]),
                 "feature width")
    expect_error(n4m_preprocess_import(bytes[-length(bytes)]), "pipeline import")
    tampered <- bytes
    tampered[1L] <- as.raw(bitwXor(as.integer(tampered[1L]), 255L))
    expect_error(n4m_preprocess_import(tampered), "pipeline import")
    expect_error(n4m_preprocess_import(integer()), "raw vector")
})

test_that("N4MP preserves the ordered original parameter plan", {
    x <- outer(1:20, 1:21, function(i, j) sin(i * j / 31) + i / 20)
    steps <- list(n4m_preprocess_step("snv"),
                  n4m_preprocess_step("savgol_derivative", c(7, 3, 1, 1)),
                  n4m_preprocess_step("center"))
    fitted <- n4m_preprocess_fit(x, steps)
    imported <- n4m_preprocess_import(n4m_preprocess_export(fitted))
    expect_equal(n4m_preprocess_plan(imported), steps)
    expect_equal(imported$steps, steps)
})
