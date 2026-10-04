# SPDX-License-Identifier: CECILL-2.1
# Raw marshalling and label tables only; every learned operation is native.
.n4m_mm_class_table <- function(values) {
  values <- .n4m_label_table(values)
  if (length(values) < 2L || is.numeric(values) &&
      any(values != floor(values) | abs(values) > 2^53 - 1))
    stop("class labels require at least two unique strings or exact integers", call. = FALSE)
  if (length(values) > 65536L || is.character(values) && any(nchar(enc2utf8(values), type = "bytes") > 1048576))
    stop("class labels exceed native resource bounds", call. = FALSE)
  values
}

#' Native raw multimodal PLS-logistic classifier
#'
#' Fits the same raw source encoders as the regression pipeline, followed by
#' the existing native PLS-logistic classifier. Class columns use the native
#' class order; original labels remain an explicit typed table.
#' @param recipe Closed early-fusion recipe with PLS-logistic model.
#' @param source_schemas Named raw source schemas.
#' @param state Raw N4MC bytes containing no training rows.
#' @param class_names Optional original labels in native class-column order.
#' @export
n4m_multimodal_classifier <- function(recipe, source_schemas) {
  .n4m_mm_recipe(recipe, source_schemas, classifier = TRUE)
  recipe$source_order <- unlist(recipe$source_order, use.names = FALSE)
  source_schemas <- source_schemas[recipe$source_order]
  structure(list(recipe = recipe, source_schemas = source_schemas, label_names = NULL,
                 pointer = .Call("r_n4m_multimodal_classifier_create", recipe,
                                 source_schemas, NULL, PACKAGE = "n4m")),
            class = "n4m_multimodal_classifier")
}

#' @rdname n4m_multimodal_classifier
#' @export
n4m_multimodal_classifier_from_state <- function(state, recipe, source_schemas,
                                               class_names = NULL) {
  .n4m_mm_recipe(recipe, source_schemas, classifier = TRUE)
  levels <- if (is.null(class_names)) NULL else .n4m_mm_class_table(class_names)
  recipe$source_order <- unlist(recipe$source_order, use.names = FALSE)
  source_schemas <- source_schemas[recipe$source_order]
  pointer <- .Call("r_n4m_multimodal_classifier_create", recipe, source_schemas,
                   state, PACKAGE = "n4m")
  ok <- FALSE
  on.exit(if (!ok) .Call("r_n4m_multimodal_classifier_close", pointer, PACKAGE = "n4m"))
  ids <- .Call("r_n4m_multimodal_classifier_classes", pointer, PACKAGE = "n4m")
  if (!is.null(levels) && length(levels) != length(ids))
    stop("class_names length differs from native class columns", call. = FALSE)
  out <- structure(list(recipe = recipe, source_schemas = source_schemas,
                        label_names = levels, pointer = pointer),
                   class = "n4m_multimodal_classifier")
  ok <- TRUE
  out
}

#' @export
n4m_estimator_fit.n4m_multimodal_classifier <- function(object, X, y = NULL, ...) {
  if (length(list(...))) stop("unsupported classifier fit inputs", call. = FALSE)
  blocks <- .n4m_mm_blocks(X, object$recipe$source_order)
  if (is.factor(y)) y <- as.character(y)
  if (!(is.character(y) || is.numeric(y)) || !is.null(dim(y)) ||
      length(y) != dim(blocks[[1L]])[[1L]] || anyNA(y))
    stop("expected one nonmissing string or integer class label per row", call. = FALSE)
  names <- .n4m_mm_class_table(unique(y))
  names <- sort(names, method = "radix")
  if (length(y) * length(names) > 16777216)
    stop("classifier fit matrix exceeds native bounds", call. = FALSE)
  ids <- as.double(match(y, names) - 1L)
  # Fit a separate owner, so failure leaves the supplied object's labels and
  # fitted state usable. Assign the return value, as for other n4m_fit methods.
  candidate <- n4m_multimodal_classifier(object$recipe, object$source_schemas)
  ok <- FALSE
  on.exit(if (!ok) n4m_close(candidate))
  .Call("r_n4m_multimodal_classifier_fit", candidate$pointer, blocks,
        candidate$source_schemas, ids, PACKAGE = "n4m")
  fitted_ids <- .Call("r_n4m_multimodal_classifier_classes", candidate$pointer, PACKAGE = "n4m")
  if (!identical(fitted_ids, as.double(seq_along(names) - 1L)))
    stop("native class order differs from the encoded label table", call. = FALSE)
  candidate$label_names <- names
  ok <- TRUE
  candidate
}

#' @export
n4m_classes.n4m_multimodal_classifier <- function(object) {
  ids <- .Call("r_n4m_multimodal_classifier_classes", object$pointer, PACKAGE = "n4m")
  if (is.null(object$label_names)) ids else object$label_names
}

#' @export
predict.n4m_multimodal_classifier <- function(object, newdata,
                                            type = c("class", "prob", "decision"),
                                            source_schemas = object$source_schemas, ...) {
  if (length(list(...))) stop("unsupported classifier prediction inputs", call. = FALSE)
  type <- match.arg(type)
  .n4m_mm_keys(source_schemas, object$recipe$source_order)
  out <- .Call("r_n4m_multimodal_classifier_op", object$pointer,
               .n4m_mm_blocks(newdata, object$recipe$source_order),
               source_schemas[object$recipe$source_order], type, PACKAGE = "n4m")
  if (type == "class") {
    ids <- .Call("r_n4m_multimodal_classifier_classes", object$pointer, PACKAGE = "n4m")
    positions <- match(out, ids)
    if (anyNA(positions)) stop("native prediction contains an undeclared class ID", call. = FALSE)
    return(if (is.null(object$label_names)) out else object$label_names[positions])
  }
  colnames(out) <- as.character(n4m_classes(object))
  out
}

#' @export
n4m_estimator_transform.n4m_multimodal_classifier <- function(object, X,
                                                           source_schemas = object$source_schemas, ...) {
  if (length(list(...))) stop("unsupported classifier transform inputs", call. = FALSE)
  .n4m_mm_keys(source_schemas, object$recipe$source_order)
  .Call("r_n4m_multimodal_classifier_op", object$pointer,
        .n4m_mm_blocks(X, object$recipe$source_order),
        source_schemas[object$recipe$source_order], "transform", PACKAGE = "n4m")
}

#' @export
n4m_export_state.n4m_multimodal_classifier <- function(object) {
  .Call("r_n4m_multimodal_classifier_export", object$pointer, PACKAGE = "n4m")
}

#' @export
n4m_close.n4m_multimodal_classifier <- function(object) {
  invisible(.Call("r_n4m_multimodal_classifier_close", object$pointer, PACKAGE = "n4m"))
}
