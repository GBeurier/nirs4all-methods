# SNV contract: actual C bridge/refusals, not host-side numerical code.
fixture <- Sys.getenv("N4M_SNV_FIXTURE_PATH", "../../../../../parity/fixtures/snv_parameter_contract_v1.json")
if (!file.exists(fixture)) stop("Set N4M_SNV_FIXTURE_PATH to the shared owner fixture in packaged test context")
doc <- jsonlite::fromJSON(fixture, simplifyVector = FALSE)
stopifnot(identical(sort(names(doc)), sort(c("schema", "matrix", "references", "domains", "typed_values", "default_bits", "provenance"))))
X <- matrix(unlist(doc$matrix), ncol = 4L, byrow = TRUE)
y <- c(1, 2, 4, 3)
legacy <- if (nzchar(Sys.getenv("N4M_SNV_LEGACY_SOURCE"))) {
  e <- new.env(parent = asNamespace("n4m"))
  sys.source(Sys.getenv("N4M_SNV_LEGACY_SOURCE"), envir = e)
  e$snv_transform
} else snv_transform
value <- function(row) {
  switch(row$tag, literal = row$value, nan = NaN, infinity = Inf,
         factor = factor(row$value), difftime = as.difftime(row$value, units = "secs"),
         "NA" = NA, stop("unconsumed R fixture tag: ", row$tag))
}
generic <- function(params) n4m_estimator_transform(n4m_estimator_fit(do.call(n4m_snv, params), X), X)
steps <- function(params) list(list(method_id = "preprocessing.scatter.snv", params = params), "models.regularized.ridge")
for (row in doc$references) {
  local({
    r <- row
    test_that(paste("literal reference", r$id), {
      want <- matrix(unlist(r$values), ncol = 4L, byrow = TRUE)
      expect_equal(do.call(legacy, c(list(X = X), r$params)), want, tolerance = 1e-12)
      expect_equal(generic(r$params), want, tolerance = 1e-12)
    })
  })
}
for (row in doc$typed_values) {
  if (is.null(row$admit$r)) next
  local({
    r <- row
    params <- setNames(list(value(r)), r$parameter)
    test_that(paste("legacy typed", r$id), {
      if (isTRUE(r$admit$r)) expect_equal(dim(do.call(legacy, c(list(X = X), params))), c(4L, 4L))
      else expect_error(do.call(legacy, c(list(X = X), params)))
    })
    test_that(paste("generic typed", r$id), {
      # Public generated role NULL is unset; the legacy scalar is not.
      if (isTRUE(r$admit$r) || is.null(value(r))) expect_equal(dim(generic(params)), c(4L, 4L))
      else expect_error(generic(params))
    })
    test_that(paste("pipeline typed", r$id), {
      if (isTRUE(r$admit$r) || is.null(value(r))) {
        object <- n4m_role_pipeline(steps(params))
        expect_s3_class(n4m_estimator_fit(object, X, y), "n4m_role_pipeline")
      } else expect_error(n4m_role_pipeline(steps(params)))
    })
  })
}
test_that("raw role validation and import cannot bypass SNV types", {
  for (params in list(list(ddof = TRUE), list(ddof = "1"), list(ddof = factor("z")),
                      list(ddof = as.difftime(1, units = "secs")), list(with_std = "false"))) {
    expect_error(.Call("r_n4m_role_pipeline_validate",
                      c("preprocessing.scatter.snv", "models.regularized.ridge"),
                      list(params, list()), PACKAGE = "n4m"))
  }
  valid <- n4m_estimator_fit(n4m_role_pipeline(steps(list())), X, y)
  restored <- n4m_role_pipeline_import(valid$steps, valid$state$n4me)
  expect_equal(predict(restored, X), predict(valid, X), tolerance = 0)
  bad <- valid$steps
  bad[[1]]$params$ddof <- TRUE
  expect_error(n4m_role_pipeline_import(bad, valid$state$n4me))
})
test_that("owner denominator gate is independent of host types", {
  for (ddof in c(4L, 5L, .Machine$integer.max)) {
    expect_error(legacy(X, ddof = ddof))
    expect_error(generic(list(ddof = ddof)))
    expect_equal(legacy(X, with_std = FALSE, ddof = ddof), X - rowMeans(X), tolerance = 0)
  }
})

test_that("raw legacy and classed/nonscalar values preserve typed admission", {
  for (params in list(list(ddof = c(0, 1)), list(ddof = I(1)), list(ddof = complex(real = 1)),
                      list(with_std = c(TRUE, FALSE)), list(with_std = I(TRUE)),
                      list(with_std = factor("a")), list(with_std = "T"))) {
    expect_error(do.call(legacy, c(list(X = X), params)))
    expect_error(generic(params))
  }
  for (row in doc$typed_values) {
    if (is.null(row$admit$r)) next
    params <- list(with_mean = TRUE, with_std = TRUE, ddof = 0L)
    params[row$parameter] <- list(value(row))
    call <- function() .Call("r_n4m_snv_transform", X, params$with_mean, params$with_std, params$ddof, PACKAGE = "n4m")
    if (isTRUE(row$admit$r)) expect_equal(dim(call()), c(4L, 4L)) else expect_error(call())
  }
})
test_that("procedure and raw role import hit the shared SNV check", {
  inputs <- getFromNamespace(".n4m_fit_inputs", "n4m")(X, list())
  expect_error(.Call("r_n4m_procedure_run", "preprocessing.scatter.snv", list(ddof = TRUE),
                    X, NULL, inputs, PACKAGE = "n4m"), "invalid value for parameter")
})
test_that("raw role import hits the shared SNV check", {
  valid <- n4m_estimator_fit(n4m_role_pipeline(steps(list())), X, y)
  payloads <- lapply(valid$state$n4me, function(s) if (is.raw(s)) s else s$n4me)
  expect_error(.Call("r_n4m_role_pipeline_import",
                    c("preprocessing.scatter.snv", "models.regularized.ridge"),
                    list(list(ddof = TRUE), list()), payloads, NULL, PACKAGE = "n4m"),
               "invalid value for parameter")
})
