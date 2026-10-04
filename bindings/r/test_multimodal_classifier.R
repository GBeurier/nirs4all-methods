# SPDX-License-Identifier: CECILL-2.1
# Fresh Python/oracle fixture, native fit and native replay; no skip fallback.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("actual raw classifier fixture JSON path required")
library(n4m)
fixture <- jsonlite::fromJSON(args[[1L]], simplifyVector = FALSE)
recipe <- fixture$recipe
recipe$source_order <- unlist(recipe$source_order, use.names = FALSE)
recipe$model$params <- lapply(recipe$model$params, as.numeric)
for (name in intersect(c("image", "series"), recipe$source_order)) {
  recipe$encoders[[name]]$n_components <- as.numeric(recipe$encoders[[name]]$n_components)
  recipe$encoders[[name]]$random_state <- as.numeric(recipe$encoders[[name]]$random_state)
}
if ("metadata" %in% recipe$source_order) {
  recipe$encoders$metadata$numeric_columns <- 0
  recipe$encoders$metadata$categorical_columns <- 1
}
schemas <- fixture$source_schemas
for (name in names(schemas)) schemas[[name]]$input_shape <- as.numeric(unlist(schemas[[name]]$input_shape))
blocks <- function(raw) {
  out <- lapply(names(raw), function(name) {
    value <- raw[[name]]
    if (name == "metadata") return(do.call(rbind, lapply(value, unlist)))
    shape <- as.integer(unlist(value$shape))
    aperm(array(as.double(unlist(value$data)), rev(shape)), rev(seq_along(shape)))
  })
  setNames(out, names(raw))
}
train <- blocks(fixture$train)
heldout <- blocks(fixture$heldout)
labels <- unlist(fixture$label_names, use.names = FALSE)
expected_labels <- unlist(fixture$expected_labels, use.names = FALSE)
expected_probabilities <- do.call(rbind, lapply(fixture$expected_probabilities, function(row) as.double(unlist(row))))
check <- function(model) {
  classes <- n4m_classes(model)
  predictions <- predict(model, heldout)
  stopifnot(is.character(classes) == is.character(labels), all(classes == labels),
            length(predictions) == length(expected_labels), all(predictions == expected_labels),
            is.character(predictions) == is.character(expected_labels))
  probabilities <- predict(model, heldout, type = "prob")
  stopifnot(identical(dim(probabilities), dim(expected_probabilities)),
            identical(colnames(probabilities), as.character(labels)),
            all(is.finite(probabilities)), all(probabilities >= 0 & probabilities <= 1),
            max(abs(rowSums(probabilities) - 1)) < 1e-12,
            max(abs(probabilities - expected_probabilities)) <= 1e-8 * (1 + max(abs(expected_probabilities))))
}
state <- jsonlite::base64_dec(fixture$state)
stopifnot(identical(rawToChar(state[1:4]), "N4MC"))
replay <- n4m_multimodal_classifier_from_state(state, recipe, schemas, class_names = labels)
check(replay)
raw_ids <- n4m_multimodal_classifier_from_state(state, recipe, schemas)
stopifnot(identical(n4m_classes(raw_ids), as.double(seq_along(labels) - 1L)),
          identical(predict(raw_ids, heldout), as.double(match(expected_labels, labels) - 1L)))
n4m_close(raw_ids)
broken <- state; broken[[61L]] <- as.raw(bitwXor(as.integer(broken[[61L]]), 1L))
stopifnot(inherits(try(n4m_multimodal_classifier_from_state(broken, recipe, schemas, labels), silent = TRUE), "try-error"))
wrong <- schemas; wrong[[recipe$source_order[[1L]]]]$identity <- paste0(wrong[[recipe$source_order[[1L]]]]$identity, ":foreign-source")
stopifnot(inherits(try(predict(replay, heldout, source_schemas = wrong), silent = TRUE), "try-error"),
          inherits(try(n4m_multimodal_classifier_from_state(state, recipe, wrong, labels), silent = TRUE), "try-error"),
          inherits(try(n4m_multimodal_classifier_from_state(state, recipe, schemas, labels[-1L]), silent = TRUE), "try-error"),
          inherits(try(n4m_multimodal_classifier_from_state(state, recipe, schemas, rep(labels[[1L]], length(labels))), silent = TRUE), "try-error"))
wrong_recipe <- recipe; wrong_recipe$model$params$n_components <- wrong_recipe$model$params$n_components + 1
stopifnot(inherits(try(n4m_multimodal_classifier_from_state(state, wrong_recipe, schemas, labels), silent = TRUE), "try-error"),
          inherits(try(n4m_multimodal_pipeline(recipe, schemas), silent = TRUE), "try-error"))
fresh <- n4m_multimodal_classifier(recipe, schemas)
unfitted <- fresh
fresh <- n4m_fit(fresh, train, unlist(fixture$y, use.names = FALSE))
n4m_close(unfitted)
check(fresh)
hydrated <- n4m_multimodal_classifier_from_state(n4m_export_state(fresh), recipe, schemas,
                                               class_names = n4m_classes(fresh))
check(hydrated)
before_labels <- predict(fresh, heldout)
before_probabilities <- predict(fresh, heldout, type = "prob")
numeric <- setdiff(recipe$source_order, "metadata")
if (length(numeric)) train[[numeric[[1L]]]][[1L]] <- NaN else train$metadata[1L, 1L] <- "NaN"
stopifnot(inherits(try(n4m_fit(fresh, train, unlist(fixture$y, use.names = FALSE)), silent = TRUE), "try-error"),
          identical(predict(fresh, heldout), before_labels),
          identical(predict(fresh, heldout, type = "prob"), before_probabilities),
          inherits(try(n4m_fit(fresh, blocks(fixture$train), rep(labels[[1L]], length(fixture$y))), silent = TRUE), "try-error"),
          identical(predict(fresh, heldout), before_labels))
check(replay)
n4m_close(replay); n4m_close(hydrated); n4m_close(fresh); n4m_close(fresh)
cat("native R classifier fit/import/labels/probability-order/schema/transactional-cleanup PASS\n")
