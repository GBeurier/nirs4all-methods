# SPDX-License-Identifier: CECILL-2.1
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("raw diagnostic fixture JSON path required")
library(n4m)
fixture <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
unlist_recipe <- function(recipe) {
  recipe$source_order <- unlist(recipe$source_order)
  for (name in intersect(c("image", "series"), recipe$source_order)) {
    recipe$encoders[[name]]$n_components <- as.numeric(recipe$encoders[[name]]$n_components)
    recipe$encoders[[name]]$random_state <- as.numeric(recipe$encoders[[name]]$random_state)
  }
  if ("metadata" %in% recipe$source_order) {
    recipe$encoders$metadata$numeric_columns <- 0
    recipe$encoders$metadata$categorical_columns <- 1
  }
  recipe
}
recipe <- unlist_recipe(fixture$recipe)
schemas <- fixture$source_schemas
for (name in names(schemas)) schemas[[name]]$input_shape <- as.numeric(unlist(schemas[[name]]$input_shape))
blocks <- function(raw) lapply(names(raw), function(name) {
  value <- raw[[name]]
  if (name == "metadata") return(do.call(rbind, lapply(value, unlist)))
  shape <- as.integer(unlist(value$shape))
  # Pure raw layout conversion: JSON C-order -> R column-major sample-first.
  aperm(array(as.double(unlist(value$data)), rev(shape)), rev(seq_along(shape)))
}) |> setNames(names(fixture$train))
train <- blocks(fixture$train)
heldout <- blocks(fixture$heldout)
expected <- as.double(unlist(fixture$expected))
check <- function(value) stopifnot(length(value) == length(expected), max(abs(value - expected)) <= 1e-8 * (1 + max(abs(expected))))
state <- jsonlite::base64_dec(fixture$state)
replay <- n4m_multimodal_pipeline_from_state(state, recipe, schemas)
check(predict(replay, heldout))
# Current writer ABI may differ from the historical imported fixture.
replay_before <- n4m_export_state(replay)
z <- n4m_estimator_transform(replay, heldout)
width <- 0L
for (name in recipe$source_order) {
  if (name == "metadata") {
    count <- length(unique(train$metadata[, 2]))
    stopifnot(identical(as.double(z[1, width + 1L + seq_len(count)]), rep(0, count)))
    width <- width + 1L + count
  } else if (recipe$encoders[[name]]$kind == "tensor_pca") {
    width <- width + recipe$encoders[[name]]$n_components
  } else width <- width + prod(schemas[[name]]$input_shape)
}
stopifnot(ncol(z) == width)
float32_sources <- names(schemas)[vapply(schemas, function(schema) identical(schema$dtype, "float32"), logical(1))]
if (length(float32_sources)) {
  # JSON preserves exact f32 values in R doubles. Only declared marshalling
  # converts them; the learned scaler/PCA/Ridge remain entirely native.
  for (name in float32_sources) {
    values <- as.double(train[[name]])
    roundtrip <- readBin(writeBin(values, raw(), size = 4L), numeric(), n = length(values), size = 4L)
    stopifnot(identical(values, roundtrip))
  }
  invalid <- heldout
  invalid[[float32_sources[[1]]]][[1]] <- 0.1
  failure <- try(predict(replay, invalid), silent = TRUE)
  stopifnot(inherits(failure, "try-error"), grepl("float32.*lossless", as.character(failure)))
  stopifnot(identical(n4m_export_state(replay), replay_before))
  check(predict(replay, heldout))
}
broken <- state; broken[[61]] <- as.raw(bitwXor(as.integer(broken[[61]]), 1L))
stopifnot(inherits(try(n4m_multimodal_pipeline_from_state(broken, recipe, schemas), silent = TRUE), "try-error"))
wrong <- schemas; selected <- recipe$source_order[[1]]
wrong[[selected]]$identity <- paste0(wrong[[selected]]$identity, ":wrong-axis")
stopifnot(inherits(try(predict(replay, heldout, source_schemas = wrong), silent = TRUE), "try-error"))
fresh <- n4m_multimodal_pipeline(recipe, schemas)
fresh <- n4m_fit(fresh, train, as.double(unlist(fixture$y)))
check(predict(fresh, heldout))
hydrated <- n4m_multimodal_pipeline_from_state(n4m_export_state(fresh), recipe, schemas)
stopifnot(identical(predict(hydrated, heldout), predict(fresh, heldout)))
before <- predict(fresh, heldout)
numeric <- setdiff(recipe$source_order, "metadata")
if (length(numeric)) train[[numeric[[1]]]][[1]] <- NaN else train$metadata[1, 1] <- "NaN"
stopifnot(inherits(try(n4m_fit(fresh, train, as.double(unlist(fixture$y))), silent = TRUE), "try-error"))
stopifnot(identical(predict(fresh, heldout), before))
n4m_close(replay); n4m_close(hydrated); n4m_close(fresh); n4m_close(fresh)
cat("native R raw multimodal fit/import/schema/unknown-category/cleanup PASS; source dtypes:",
    paste(vapply(schemas, function(schema) schema$dtype, character(1)), collapse = ","), "\n")
