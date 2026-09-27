# SPDX-License-Identifier: CECILL-2.1

# Native role pipelines (ABI 2.14): a trained linear recipe of catalog
# estimators. Recipe validation, fit-input routing, the feature-name check and
# the per-step N4ME states are native (n4m_role_pipeline_*); this file only
# converts R objects, strictly (no recycling), and keeps class label names.

.n4m_role_names <- c(`1` = "transformer", `2` = "regressor", `4` = "classifier",
                     `8` = "selector", `16` = "sample_filter")

# list(method_id, params) from "id", "n4m:id", list(method_id =, params =) or
# list(class = "n4m:id", params =) (the recipe tokens of the v8 envelopes).
.n4m_role_step <- function(step) {
  if (is.character(step) && length(step) == 1L) {
    return(list(method_id = sub("^n4m:", "", step), params = list()))
  }
  if (!is.list(step)) stop("a role pipeline step is a method id or a list", call. = FALSE)
  id <- if (!is.null(step$method_id)) step$method_id else step$class
  if (!is.character(id) || length(id) != 1L) {
    stop("a role pipeline step needs one method id", call. = FALSE)
  }
  params <- if (is.null(step$params)) list() else as.list(step$params)
  list(method_id = sub("^n4m:", "", id), params = params[!vapply(params, is.null, logical(1))])
}

.n4m_role_ids <- function(steps) vapply(steps, `[[`, "", "method_id")
.n4m_role_params <- function(steps) lapply(steps, `[[`, "params")

# Numeric matrix and its column names (NULL: positional).
.n4m_role_x <- function(X) {
  names <- colnames(X)
  if (is.data.frame(X)) {
    if (!all(vapply(X, is.numeric, logical(1)))) stop("X columns must be numeric", call. = FALSE)
    X <- as.matrix(X)
  }
  if (!is.matrix(X) || !is.numeric(X)) stop("X must be a numeric matrix or data frame", call. = FALSE)
  storage.mode(X) <- "double"
  dimnames(X) <- NULL
  list(X = X, names = if (is.null(names)) NULL else as.character(names))
}

# Responses: a numeric vector with one value per row, or a matrix with one row
# per row of X. Anything else is refused (R would otherwise recycle it).
.n4m_role_y <- function(y, n) {
  if (is.matrix(y) && is.numeric(y)) {
    if (nrow(y) != n) stop(sprintf("y has %d rows; X has %d", nrow(y), n), call. = FALSE)
    storage.mode(y) <- "double"
    return(y)
  }
  if (!is.numeric(y) || !is.null(dim(y))) {
    stop("y must be a numeric vector or matrix", call. = FALSE)
  }
  if (length(y) != n) stop(sprintf("y has %d values; X has %d rows", length(y), n), call. = FALSE)
  matrix(as.double(y), ncol = 1L)
}

# Class ids for the core and the label names (NULL for integer labels). Names
# sort in C-locale order, as the other bindings do.
.n4m_role_labels <- function(y, n) {
  if (!is.null(dim(y)) || length(y) != n) {
    stop(sprintf("class labels must be a vector of %d values", n), call. = FALSE)
  }
  if (is.factor(y) || is.character(y)) {
    levels <- sort(unique(as.character(y)), method = "radix")
    return(list(ids = match(as.character(y), levels) - 1, levels = levels))
  }
  if (!is.numeric(y) || any(y != round(y))) {
    stop("class labels must be a factor, characters or integers", call. = FALSE)
  }
  list(ids = as.double(y), levels = NULL)
}

.n4m_role_state <- function(object, pointer, names, levels, y_vector) {
  object$state <- list2env(list(
    pointer = pointer,
    # Kept so saveRDS()/readRDS() rehydrate without refitting (a local
    # checkpoint; n4m_role_pipeline_export() refuses training rows by default).
    n4me = .Call("r_n4m_role_pipeline_export", pointer, TRUE, PACKAGE = "n4m"),
    feature_names = names, levels = levels, y_vector = y_vector,
    info = .Call("r_n4m_role_pipeline_info", pointer, PACKAGE = "n4m")))
  object
}

.n4m_role_pointer <- function(object) {
  state <- object$state
  if (is.null(state)) stop("n4m role pipeline is not fitted", call. = FALSE)
  if (!isTRUE(.Call("r_n4m_estimator_alive", state$pointer, PACKAGE = "n4m"))) {
    state$pointer <- .Call("r_n4m_role_pipeline_import", .n4m_role_ids(object$steps),
                           .n4m_role_params(object$steps), state$n4me, state$feature_names,
                           PACKAGE = "n4m")
  }
  state$pointer
}

.n4m_role_classifier <- function(object) {
  roles <- object$state$info$steps$role
  roles[length(roles)] == 4L
}

.n4m_role_op <- function(object, newdata, kind) {
  x <- .n4m_role_x(newdata)
  .Call("r_n4m_role_pipeline_op", .n4m_role_pointer(object), x$X, x$names, kind,
        PACKAGE = "n4m")
}

#' Native role pipelines
#'
#' \code{n4m_role_pipeline()} builds a trained linear recipe of catalog
#' estimators: sample filters (training rows only), transformers and selectors,
#' then one regressor or classifier. The recipe is validated natively (role
#' order, final step, parameters). \code{n4m_estimator_fit()} fits every step
#' natively: \code{y} holds responses for a final regressor (a vector or a
#' matrix, one row per sample) or class labels for a final classifier; steps
#' that require \code{y} receive every response column (or the class ids),
#' sample filters drop training rows of every row-aligned input, and inputs
#' no step uses are refused. Column names of \code{X} are stored and checked
#' at every later call (a reordered, renamed or missing column is refused);
#' matrices without column names are positional. The fitted state is one N4ME
#' payload per stateful step: \code{n4m_role_pipeline_export()} refuses states
#' that embed training rows unless \code{allow_training_rows = TRUE}, and
#' \code{n4m_role_pipeline_import()} refuses states that contradict the recipe
#' (method, parameters, widths). The same states are read by the Python,
#' JS/WASM and Rust bindings.
#'
#' @param steps A list of steps: method ids (\code{"models.pls.cppls"} or
#'   \code{"n4m:models.pls.cppls"}), \code{list(method_id =, params = list())}
#'   or recipe tokens \code{list(class = "n4m:<id>", params = list())}.
#' @param object A role pipeline.
#' @param X,newdata Numeric matrix or data frame (rows are samples).
#' @param y Responses (regression) or class labels (classification).
#' @param sample_weight,groups,feature_groups,blocks,axis,X_target,fold_ids
#'   Optional fit inputs, routed natively to the steps that declare them.
#' @param type For a final classifier: \code{"class"}, \code{"prob"} or
#'   \code{"decision"}.
#' @param allow_training_rows Export states that embed training rows.
#' @param states List of N4ME raw vectors, one per stateful step (or the list
#'   returned by \code{n4m_role_pipeline_export()}).
#' @param feature_names Optional input column names of the imported pipeline.
#' @param class_names Optional label names of an imported classifier.
#' @param x A role pipeline to print.
#' @param ... Unused.
#' @return \code{n4m_role_pipeline()} an unfitted pipeline;
#'   \code{n4m_estimator_fit()} and \code{n4m_role_pipeline_import()} a fitted
#'   one; \code{predict()} and \code{n4m_estimator_transform()} numeric results
#'   (class labels for \code{type = "class"}); \code{n4m_role_pipeline_export()}
#'   a list of \code{list(method_id, n4me, contains_training_rows)};
#'   \code{n4m_role_pipeline_steps()} a data frame describing each step.
#' @name n4m_role_pipeline
NULL

#' @rdname n4m_role_pipeline
#' @export
n4m_role_pipeline <- function(steps) {
  if (!is.list(steps) && !is.character(steps)) stop("steps must be a list", call. = FALSE)
  steps <- lapply(steps, .n4m_role_step)
  .Call("r_n4m_role_pipeline_validate", .n4m_role_ids(steps), .n4m_role_params(steps),
        PACKAGE = "n4m")
  structure(list(steps = steps, state = NULL), class = "n4m_role_pipeline")
}

#' @rdname n4m_role_pipeline
#' @export
n4m_estimator_fit.n4m_role_pipeline <- function(object, X, y = NULL, sample_weight = NULL,
                                                groups = NULL, feature_groups = NULL,
                                                blocks = NULL, axis = NULL, X_target = NULL,
                                                fold_ids = NULL, ...) {
  x <- .n4m_role_x(X)
  n <- nrow(x$X)
  labels <- NULL
  levels <- NULL
  y_matrix <- NULL
  if (!is.null(y) && .n4m_role_final_classifier(object$steps)) {
    encoded <- .n4m_role_labels(y, n)
    labels <- encoded$ids
    levels <- encoded$levels
  } else if (!is.null(y)) {
    y_matrix <- .n4m_role_y(y, n)
  }
  inputs <- list(labels = labels, sample_weight = sample_weight, groups = groups,
                 feature_groups = feature_groups, blocks = blocks, axis = axis,
                 X_target = if (is.null(X_target)) NULL else .n4m_role_x(X_target)$X,
                 fold_ids = fold_ids)
  inputs <- inputs[!vapply(inputs, is.null, logical(1))]
  pointer <- .Call("r_n4m_role_pipeline_fit", .n4m_role_ids(object$steps),
                   .n4m_role_params(object$steps), x$X, y_matrix, inputs, x$names,
                   PACKAGE = "n4m")
  .n4m_role_state(object, pointer, x$names, levels, !is.null(y) && is.null(dim(y)))
}

# TRUE when the final step plays the classifier role (as the core decides it).
.n4m_role_final_classifier <- function(steps) {
  roles <- .n4m_method_roles[[steps[[length(steps)]]$method_id]]
  "classifier" %in% roles && !("regressor" %in% roles)
}

#' @rdname n4m_role_pipeline
#' @export
predict.n4m_role_pipeline <- function(object, newdata, type = c("class", "prob", "decision"),
                                      ...) {
  .n4m_role_pointer(object)
  if (!.n4m_role_classifier(object)) {
    out <- .n4m_role_op(object, newdata, 0L)
    return(if (isTRUE(object$state$y_vector) && ncol(out) == 1L) out[, 1L] else out)
  }
  type <- match.arg(type)
  if (type == "class") {
    ids <- .n4m_role_op(object, newdata, 4L)
    levels <- object$state$levels
    return(if (is.null(levels)) ids else factor(levels[ids + 1], levels = levels))
  }
  out <- .n4m_role_op(object, newdata, if (type == "prob") 3L else 2L)
  classes <- object$state$info$classes
  colnames(out) <- if (is.null(object$state$levels)) classes else object$state$levels[classes + 1]
  out
}

#' @rdname n4m_role_pipeline
#' @export
n4m_estimator_transform.n4m_role_pipeline <- function(object, X, ...) {
  .n4m_role_op(object, X, 1L)
}

#' @rdname n4m_role_pipeline
#' @export
n4m_role_pipeline_export <- function(object, allow_training_rows = FALSE) {
  if (!is.logical(allow_training_rows) || length(allow_training_rows) != 1L ||
      is.na(allow_training_rows)) {
    stop("allow_training_rows must be TRUE or FALSE", call. = FALSE)
  }
  states <- .Call("r_n4m_role_pipeline_export", .n4m_role_pointer(object),
                  allow_training_rows, PACKAGE = "n4m")
  steps <- object$state$info$steps
  stateful <- which(steps$state_index >= 0L)
  Map(function(bytes, k) list(method_id = steps$method_id[k], n4me = bytes,
                              contains_training_rows = steps$contains_training_rows[k]),
      states, stateful)
}

#' @rdname n4m_role_pipeline
#' @export
n4m_role_pipeline_import <- function(steps, states, feature_names = NULL, class_names = NULL) {
  object <- n4m_role_pipeline(steps)
  payloads <- lapply(states, function(s) if (is.raw(s)) s else s$n4me)
  names <- if (is.null(feature_names)) NULL else as.character(feature_names)
  pointer <- .Call("r_n4m_role_pipeline_import", .n4m_role_ids(object$steps),
                   .n4m_role_params(object$steps), payloads, names, PACKAGE = "n4m")
  levels <- if (is.null(class_names)) NULL else as.character(class_names)
  object <- .n4m_role_state(object, pointer, names, levels, FALSE)
  # N4ME does not record the caller's target shape: one output is a vector.
  widths <- object$state$info$steps$n_features_out
  object$state$y_vector <- widths[length(widths)] == 1
  object
}

#' @rdname n4m_role_pipeline
#' @export
n4m_role_pipeline_steps <- function(object) {
  .n4m_role_pointer(object)
  steps <- object$state$info$steps
  data.frame(method_id = steps$method_id,
             role = unname(.n4m_role_names[as.character(steps$role)]),
             state_index = steps$state_index,
             contains_training_rows = steps$contains_training_rows,
             n_features_in = steps$n_features_in, n_features_out = steps$n_features_out,
             stringsAsFactors = FALSE)
}

#' @rdname n4m_role_pipeline
#' @export
print.n4m_role_pipeline <- function(x, ...) {
  state <- if (is.null(x$state)) "unfitted" else "fitted"
  cat("<n4m role pipeline (", state, "): ", paste(.n4m_role_ids(x$steps), collapse = " -> "),
      ">\n", sep = "")
  invisible(x)
}
