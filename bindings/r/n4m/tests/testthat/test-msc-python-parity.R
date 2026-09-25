test_that("MSC learns only the train reference and matches Python n4m", {
    train <- outer(0:5, 0:8, function(i, j)
        1.5 + 0.08 * i + 0.04 * j + 0.11 * sin((i + 1) * (j + 2)))
    test <- outer(0:2, 0:8, function(i, j)
        1.3 + 0.13 * i + 0.05 * j + 0.09 * cos((i + 2) * (j + 1)))
    reference <- msc_fit(train)
    python_reference <- c(
        1.696000466935961, 1.7333367622922748, 1.7692831840984675,
        1.800564912737596, 1.7783733174539165, 1.9258582713371728,
        1.9459429653863995, 1.9784937844670896, 2.0119117036926744)
    python_output <- matrix(c(
        1.6683867979921181, 1.6905265708226374, 1.8415343449856123,
        1.8032413720289853, 1.7936342118795683, 1.9494527783427453,
        1.9389027463720252, 1.9013945196668081, 2.0526920263110533,
        1.6212962313316013, 1.8026518389451169, 1.7074225671747292,
        1.8746517958288209, 1.7988020116883372, 1.9417865246187211,
        1.8945614261456967, 2.005114206743126, 1.9934787659254003,
        1.6382313207538064, 1.7171656342920165, 1.8318093053946864,
        1.7393500504306849, 1.8819244508735598, 1.9243452782931842,
        1.8626619385326233, 2.0372262213606072, 2.0070511684703831),
        nrow = 3L, byrow = TRUE)
    expect_equal(reference, python_reference, tolerance = 1e-12)
    expect_equal(msc_transform(test, reference), python_output,
                 tolerance = 1e-12)
    expect_equal(msc_transform(test, unserialize(serialize(reference, NULL))),
                 python_output, tolerance = 1e-12)
    expect_error(msc_transform(test, rep(1, 9)), "numerical failure")
    expect_error(msc_transform(test[, -1L, drop = FALSE], reference),
                 "feature-aligned")
    expect_error(msc_transform(test, c(reference[-1L], Inf)), "finite")
})
