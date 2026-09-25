test_that("EMSC training reference and predictions match Python n4m", {
    train <- outer(0:5, 0:8, function(i, j)
        1.5 + 0.08 * i + 0.04 * j + 0.11 * sin((i + 1) * (j + 2)))
    test <- outer(0:2, 0:8, function(i, j)
        1.3 + 0.13 * i + 0.05 * j + 0.09 * cos((i + 2) * (j + 1)))
    reference <- emsc_fit(train, degree = 2L)
    python_reference <- c(
        1.696000466935961, 1.7333367622922748, 1.7692831840984675,
        1.800564912737596, 1.7783733174539165, 1.9258582713371728,
        1.9459429653863995, 1.9784937844670896, 2.0119117036926744)
    python_output <- matrix(c(
        1.681629569928991, 1.6731780522772992, 1.8912067488584463,
        1.7877957259738821, 1.7383406880834722, 1.9783220991570485,
        1.9363553544353478, 1.8525197813931515, 2.0983834794917318,
        1.6340271831628432, 1.8567433856423579, 1.6687316727171064,
        1.886819179085552, 1.7449511234941038, 1.9430773420962004,
        1.8613624445682018, 2.027121365852293, 2.016113838395883,
        1.6709638695551141, 1.7368003331632405, 1.8569473952137163,
        1.7036097586483256, 1.8736694670649823, 1.9146255371102772,
        1.8211446875932307, 2.052700032324763, 2.0139615128941575),
        nrow = 3L, byrow = TRUE)
    expect_equal(reference, python_reference, tolerance = 1e-12)
    expect_equal(emsc_transform(test, reference, degree = 2L), python_output,
                 tolerance = 1e-11)
    expect_equal(emsc_transform(test, unserialize(serialize(reference, NULL)), 2L),
                 python_output, tolerance = 1e-11)
    expect_error(emsc_transform(test[, -1L, drop = FALSE], reference),
                 "feature-aligned")
    expect_error(emsc_transform(test, c(reference[-1L], Inf)), "finite")
    expect_error(emsc_fit(train, degree = ncol(train)), "at least")
})
