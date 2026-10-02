# SPDX-License-Identifier: CECILL-2.1
# Raw object translation only; the complete learned recipe belongs to Methods.
.n4m_mm_keys <- function(value, expected) {
  if (!is.list(value) || !setequal(names(value), expected) ||
      length(value) != length(expected) || anyDuplicated(names(value))) {
    stop("invalid multimodal recipe/schema fields", call. = FALSE)
  }
}
.n4m_mm_recipe <- function(recipe, source_schemas) {
  .n4m_mm_keys(recipe, c("schema_version", "fusion", "source_order", "encoders", "source_weights", "model"))
  order <- unlist(recipe$source_order, use.names = FALSE)
  if (!identical(as.numeric(recipe$schema_version), 1) || !identical(recipe$fusion, "early") ||
      !is.character(order) || length(order) < 1L || length(order) > 4L || anyDuplicated(order) ||
      any(!order %in% c("nir", "image", "series", "metadata"))) stop("expected recipe v1 with 1..4 distinct ordered modalities", call. = FALSE)
  .n4m_mm_keys(recipe$encoders, order)
  .n4m_mm_keys(recipe$source_weights, order)
  .n4m_mm_keys(source_schemas, order)
  for (name in order) {
    .n4m_mm_keys(source_schemas[[name]], c("representation_id", "input_shape", "dtype", "identity"))
    encoder <- recipe$encoders[[name]]
    if (identical(encoder$kind, "standard_scaler")) {
      .n4m_mm_keys(encoder, c("kind", "with_mean", "with_std"))
    } else if (identical(encoder$kind, "tensor_pca")) {
      .n4m_mm_keys(encoder, c("kind", "n_components", "random_state", "whiten"))
    } else if (identical(encoder$kind, "column_transformer")) {
      .n4m_mm_keys(encoder, c("kind", "numeric_columns", "categorical_columns", "with_mean", "with_std", "handle_unknown", "sparse_output", "drop"))
      if (!identical(as.numeric(encoder$numeric_columns), 0) || !identical(as.numeric(encoder$categorical_columns), 1) ||
          !identical(encoder$handle_unknown, "ignore") || !identical(encoder$sparse_output, FALSE) || !is.null(encoder$drop))
        stop("unsupported mixed-column recipe", call. = FALSE)
    } else stop("unsupported encoder", call. = FALSE)
  }
  .n4m_mm_keys(recipe$model, c("method_id", "params"))
  .n4m_mm_keys(recipe$model$params, c("alpha", "center_x", "center_y", "scale_x"))
  if (!identical(recipe$model$method_id, "models.regularized.ridge")) stop("expected native Ridge", call. = FALSE)
  invisible(NULL)
}
.n4m_mm_blocks <- function(blocks, order) {
  order <- unlist(order, use.names = FALSE)
  .n4m_mm_keys(blocks, order)
  blocks <- blocks[order]
  for (name in setdiff(order, "metadata")) {
    x <- blocks[[name]]
    if (!is.numeric(x) || is.null(dim(x))) stop("raw numeric sources need sample-first arrays", call. = FALSE)
    storage.mode(x) <- "double"
    blocks[[name]] <- x
  }
  if (!"metadata" %in% order) return(blocks)
  metadata <- blocks$metadata
  if (is.data.frame(metadata)) {
    if (ncol(metadata) != 2L || !is.character(metadata[[2]])) stop("metadata needs declared numeric/string columns", call. = FALSE)
    metadata <- cbind(as.character(metadata[[1]]), metadata[[2]])
  }
  if (!is.matrix(metadata) || !is.character(metadata) || ncol(metadata) != 2L)
    stop("metadata must retain raw two-column string cells", call. = FALSE)
  blocks$metadata <- metadata
  blocks
}

#' Complete native raw multimodal pipeline
#'
#' An explicit ordered subset of the fixed-shape nir, image, series and metadata
#' sources is encoded and fused natively, followed by centered, unscaled-X Ridge. The UTF-8
#' vocabulary is learned only from fit rows; unseen categories encode as zero.
#' N4MF bytes contain the complete fitted state and no training rows.
#' @param recipe Closed version-one early-fusion recipe.
#' @param source_schemas Named source descriptors; identity strings stay unchanged.
#' @param state Raw N4MF bytes.
#' @export
n4m_multimodal_pipeline <- function(recipe, source_schemas) {
  .n4m_mm_recipe(recipe, source_schemas)
  recipe$source_order <- unlist(recipe$source_order, use.names = FALSE)
  source_schemas <- source_schemas[recipe$source_order]
  structure(list(recipe = recipe, source_schemas = source_schemas,
                 pointer = .Call("r_n4m_multimodal_create", recipe, source_schemas, NULL, PACKAGE = "n4m")),
            class = "n4m_multimodal_pipeline")
}
#' @rdname n4m_multimodal_pipeline
#' @export
n4m_multimodal_pipeline_from_state <- function(state, recipe, source_schemas) {
  .n4m_mm_recipe(recipe, source_schemas)
  recipe$source_order <- unlist(recipe$source_order, use.names = FALSE)
  source_schemas <- source_schemas[recipe$source_order]
  structure(list(recipe = recipe, source_schemas = source_schemas,
                 pointer = .Call("r_n4m_multimodal_create", recipe, source_schemas, state, PACKAGE = "n4m")),
            class = "n4m_multimodal_pipeline")
}
#' @export
n4m_estimator_fit.n4m_multimodal_pipeline <- function(object, X, y = NULL, ...) {
  if (length(list(...))) stop("unsupported multimodal fit inputs", call. = FALSE)
  blocks <- .n4m_mm_blocks(X, object$recipe$source_order)
  if (!is.numeric(y) || !is.null(dim(y)) && !(is.matrix(y) && ncol(y) == 1L))
    stop("one numeric target per row is required", call. = FALSE)
  .Call("r_n4m_multimodal_fit", object$pointer, blocks, object$source_schemas, as.double(y), PACKAGE = "n4m")
  object
}
#' @export
predict.n4m_multimodal_pipeline <- function(object, newdata, source_schemas = object$source_schemas, ...) {
  if (length(list(...))) stop("unsupported prediction options", call. = FALSE)
  .n4m_mm_keys(source_schemas, object$recipe$source_order)
  drop(.Call("r_n4m_multimodal_op", object$pointer, .n4m_mm_blocks(newdata, object$recipe$source_order), source_schemas[object$recipe$source_order], FALSE, PACKAGE = "n4m"))
}
#' @export
n4m_estimator_transform.n4m_multimodal_pipeline <- function(object, X, source_schemas = object$source_schemas, ...) {
  if (length(list(...))) stop("unsupported transform options", call. = FALSE)
  .n4m_mm_keys(source_schemas, object$recipe$source_order)
  .Call("r_n4m_multimodal_op", object$pointer, .n4m_mm_blocks(X, object$recipe$source_order), source_schemas[object$recipe$source_order], TRUE, PACKAGE = "n4m")
}
#' @export
n4m_export_state <- function(object) UseMethod("n4m_export_state")
#' @export
n4m_export_state.n4m_multimodal_pipeline <- function(object) .Call("r_n4m_multimodal_export", object$pointer, PACKAGE = "n4m")
#' @export
n4m_close <- function(object) UseMethod("n4m_close")
#' @export
n4m_close.n4m_multimodal_pipeline <- function(object) invisible(.Call("r_n4m_multimodal_close", object$pointer, PACKAGE = "n4m"))
