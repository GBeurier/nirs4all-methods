# SPDX-License-Identifier: CECILL-2.1

# Generic native estimator roles (ABI 2.13). Each constructor generated in
# estimator_roles_generated.R returns an unfitted specification whose S3
# classes are its role interfaces (n4m_regressor, n4m_transformer, ...).
# Parameters, defaults, required inputs, fitting and the N4ME fitted state
# are native; this file only marshals R objects.

.n4m_role_classes <- c(regressor = "n4m_regressor", classifier = "n4m_classifier",
                       transformer = "n4m_transformer", selector = "n4m_selector",
                       sample_filter = "n4m_sample_filter", splitter = "n4m_splitter",
                       augmenter = "n4m_augmenter", generic = "n4m_procedure")
.n4m_procedure_roles <- c("splitter", "augmenter", "generic")

.n4m_cap_serializable <- 128L  # N4M_CAP_SERIALIZABLE

.n4m_estimator <- function(method_id, roles, params) {
  structure(
    list(method_id = method_id, params = params, state = NULL),
    class = c(unname(.n4m_role_classes[roles]),
              if (!any(roles %in% .n4m_procedure_roles)) "n4m_estimator", "n4m_method")
  )
}

.n4m_as_matrix <- function(X, name = "X") {
  if (!is.null(dim(X)) && length(dim(X)) > 2L) {
    stop(name, " must be a matrix; got an array with ", length(dim(X)), " dimensions",
         call. = FALSE)
  }
  X <- as.matrix(X)
  if (!is.numeric(X)) stop(name, " must be numeric", call. = FALSE)
  storage.mode(X) <- "double"
  X
}

# Inputs are matched to the rows (or columns) of X exactly: R recycling and
# reshaping never apply, so a target of the wrong length is an error, not a
# different problem.

# A vector (a one-column matrix is accepted) of exactly `n` entries.
.n4m_vector <- function(v, name, n, what) {
  if (!is.null(dim(v))) {
    if (length(dim(v)) != 2L || ncol(v) != 1L) {
      stop(name, " must be a vector (", what, "); got dimensions ",
           paste(dim(v), collapse = " x "), call. = FALSE)
    }
    v <- v[, 1L]
  }
  if (length(v) != n) {
    stop(sprintf("%s must have length %d (%s); got %d", name, n, what, length(v)),
         call. = FALSE)
  }
  v
}

# Integer-valued numbers (class ids, groups, fold ids, block sizes), within
# int64 (the native type): nothing is truncated or wrapped.
.n4m_integers <- function(v, name) {
  if (is.factor(v)) v <- as.integer(v)
  if (!is.numeric(v) || anyNA(v) || any(!is.finite(v)) || any(v != round(v))) {
    stop(name, " must contain finite integers", call. = FALSE)
  }
  if (any(v < -2^63 | v >= 2^63)) stop(name, " must fit int64", call. = FALSE)
  v
}

# Class ids for the core and the label table they index (the shared label
# contract). Integer labels are the ids (no table) and must fit int64; other
# numbers, and strings (tabled by `string_levels`), become a table whose
# positions are the ids. Missing and non-finite labels are refused.
.n4m_encode_labels <- function(y, string_levels) {
  if (is.factor(y) || is.character(y)) {
    if (anyNA(y)) stop("class labels must not be missing (NA)", call. = FALSE)
    levels <- string_levels(y)
    return(list(ids = match(as.character(y), levels) - 1, levels = levels))
  }
  if (!is.numeric(y)) stop("class labels must be a factor, characters or numbers", call. = FALSE)
  if (any(is.na(y) & !is.nan(y))) stop("class labels must not be missing (NA)", call. = FALSE)
  if (!all(is.finite(y))) stop("class labels must be finite (no NaN or infinity)", call. = FALSE)
  if (all(y == round(y))) {
    if (any(y < -2^63 | y >= 2^63)) stop("class labels must fit int64", call. = FALSE)
    return(list(ids = as.double(y), levels = NULL))
  }
  levels <- sort(unique(as.double(y)))
  list(ids = match(y, levels) - 1, levels = levels)
}

# Predicted labels: a factor over string labels. Numeric labels stay numbers:
# a factor prints them with 15 digits, which merges close fractions.
.n4m_label_values <- function(ids, levels) {
  if (is.null(levels)) ids
  else if (is.character(levels)) factor(levels[ids + 1], levels = levels)
  else levels[ids + 1]
}

# Class labels as column names, with enough digits to keep numbers distinct.
.n4m_label_text <- function(labels) {
  text <- as.character(labels)
  if (is.numeric(labels) && anyDuplicated(text)) text <- formatC(labels, digits = 17, format = "g")
  text
}

# Targets as an n x q double matrix: a vector of length n is one column, a
# matrix must have n rows.
.n4m_targets <- function(y, n) {
  if (is.data.frame(y)) y <- as.matrix(y)
  if (!is.null(dim(y))) {
    if (length(dim(y)) != 2L) {
      stop("y must be a vector or a matrix; got an array with ", length(dim(y)),
           " dimensions", call. = FALSE)
    }
    if (nrow(y) != n || ncol(y) < 1L) {
      stop(sprintf("y must have %d rows (one per row of X); got %d x %d", n, nrow(y),
                   ncol(y)), call. = FALSE)
    }
  } else if (length(y) != n) {
    stop(sprintf("y must have length %d (one per row of X); got %d", n, length(y)),
         call. = FALSE)
  }
  if (!is.numeric(y) && !is.logical(y)) stop("y must be numeric", call. = FALSE)
  matrix(as.double(y), nrow = n)
}

# Optional fit inputs checked against X; NULL entries are dropped.
.n4m_fit_inputs <- function(X, inputs) {
  inputs <- inputs[!vapply(inputs, is.null, logical(1))]
  n <- nrow(X)
  p <- ncol(X)
  rows <- "one per row of X"
  for (name in intersect(c("labels", "groups", "fold_ids"), names(inputs))) {
    inputs[[name]] <- .n4m_integers(.n4m_vector(inputs[[name]], name, n, rows), name)
  }
  if (!is.null(inputs$sample_weight)) {
    inputs$sample_weight <- as.double(.n4m_vector(inputs$sample_weight, "sample_weight", n, rows))
  }
  if (!is.null(inputs$feature_groups)) {
    inputs$feature_groups <- .n4m_integers(
      .n4m_vector(inputs$feature_groups, "feature_groups", p, "one per column of X"),
      "feature_groups")
  }
  if (!is.null(inputs$axis)) {
    inputs$axis <- as.double(.n4m_vector(inputs$axis, "axis", p, "one per column of X"))
  }
  if (!is.null(inputs$blocks)) {
    blocks <- inputs$blocks
    inputs$blocks <- .n4m_integers(.n4m_vector(blocks, "blocks", length(blocks), "block sizes"),
                                   "blocks")
  }
  if (!is.null(inputs$X_target)) inputs$X_target <- .n4m_as_matrix(inputs$X_target, "X_target")
  inputs
}

# The external pointer does not survive saveRDS(); the N4ME bytes do.
.n4m_pointer <- function(object) {
  state <- object$state
  if (is.null(state)) stop("n4m estimator is not fitted", call. = FALSE)
  if (!isTRUE(.Call("r_n4m_estimator_alive", state$pointer, PACKAGE = "n4m"))) {
    if (is.null(state$n4me)) {
      stop("this estimator's state is not serializable; refit it after readRDS()",
           call. = FALSE)
    }
    state$pointer <- .Call("r_n4m_estimator_import", state$n4me, PACKAGE = "n4m")
  }
  state$pointer
}

#' Generic native estimator roles
#'
#' Constructors such as \code{n4m_pls_regression()} or \code{n4m_cppls()}
#' return an unfitted estimator whose classes are its role interfaces:
#' \code{n4m_regressor} (\code{predict}), \code{n4m_classifier}
#' (\code{predict} with \code{type = "class"}, \code{"prob"} or
#' \code{"decision"}, \code{n4m_classes}), \code{n4m_transformer}
#' (\code{n4m_estimator_transform}) and \code{n4m_selector}
#' (\code{n4m_estimator_transform}, \code{n4m_selected_indices}) and
#' \code{n4m_sample_filter} (\code{n4m_sample_mask}, train-only). Fit with
#' \code{n4m_estimator_fit()}. The fitted state is kept
#' as portable N4ME bytes, so \code{saveRDS()}/\code{readRDS()} and the Python
#' and JS/WASM bindings reuse it without refitting.
#'
#' @param object An estimator from one of the generated constructors.
#' @param X Numeric matrix (rows are samples).
#' @param y Numeric response vector (one value per row of \code{X}) or matrix
#'   (one row per row of \code{X}) for regressors; class labels (factor,
#'   character or numeric vector, one per row) for classifiers: integer labels
#'   are the class ids (within int64), strings and other finite numbers become
#'   a label table; missing (NA) and non-finite labels are refused. Nothing is
#'   recycled: another length is an error.
#' @param sample_weight,groups,feature_groups,blocks,axis,X_target,fold_ids
#'   Optional fit inputs; each method declares which ones it requires and the
#'   native core refuses the others.
#' @param newdata Numeric matrix of new samples.
#' @param type For classifiers: \code{"class"} (labels), \code{"prob"}
#'   (class probabilities, for methods that define them) or \code{"decision"}
#'   (method-defined class scores).
#' @param allow_training_rows \code{n4m_estimator_export()} refuses a state
#'   that embeds training rows (\code{n4m_contains_training_rows()}, e.g.
#'   kernel PLS, GPR-PLS, LW-PLS) unless this is \code{TRUE}: sharing the
#'   export shares those rows. \code{saveRDS()} keeps them, as a checkpoint of
#'   the in-process object.
#' @param bytes Raw vector produced by \code{n4m_estimator_export()}.
#' @param ... Unused.
#' @return \code{n4m_estimator_fit()} returns the fitted estimator (a failed fit
#'   leaves the object passed in unchanged); \code{predict()} and
#'   \code{n4m_estimator_transform()} return numeric results; \code{n4m_estimator_export()} a raw
#'   vector; \code{n4m_estimator_import()} a fitted estimator.
#' @name n4m_estimator_roles
NULL

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_fit <- function(object, X, y = NULL, ...) UseMethod("n4m_estimator_fit")

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_fit.n4m_estimator <- function(object, X, y = NULL, sample_weight = NULL, groups = NULL,
                                  feature_groups = NULL, blocks = NULL, axis = NULL,
                                  X_target = NULL, fold_ids = NULL, ...) {
  X <- .n4m_as_matrix(X)
  levels <- NULL
  labels <- NULL
  if (inherits(object, "n4m_classifier") && !is.null(y)) {
    # The core works on integer class ids; other labels are encoded here.
    y <- .n4m_vector(y, "class labels", nrow(X), "one per row of X")
    encoded <- .n4m_encode_labels(
      y, function(y) if (is.factor(y)) levels(droplevels(y)) else sort(unique(y)))
    labels <- encoded$ids
    levels <- encoded$levels
    y <- NULL
  }
  y_matrix <- if (is.null(y)) NULL else .n4m_targets(y, nrow(X))
  inputs <- .n4m_fit_inputs(X, list(labels = labels, sample_weight = sample_weight,
                                    groups = groups, feature_groups = feature_groups,
                                    blocks = blocks, axis = axis, X_target = X_target,
                                    fold_ids = fold_ids))
  params <- object$params[!vapply(object$params, is.null, logical(1))]
  pointer <- .Call("r_n4m_estimator_fit", object$method_id, params, X, y_matrix,
                   inputs, PACKAGE = "n4m")
  caps <- .Call("r_n4m_estimator_info", pointer, PACKAGE = "n4m")$capabilities
  serializable <- bitwAnd(as.integer(caps), .n4m_cap_serializable) != 0
  # The object is replaced only here, after a successful fit: a failed refit
  # leaves the caller's fitted object (state, levels) untouched. The N4ME
  # bytes kept in the object are the saveRDS() checkpoint of this in-process
  # object, so they keep training rows like the object does;
  # n4m_estimator_export() is the explicit, shareable export.
  object$state <- list2env(list(
    pointer = pointer,
    n4me = if (serializable) .Call("r_n4m_estimator_export", pointer, TRUE, PACKAGE = "n4m"),
    # One prediction column comes back as a vector unless y was a one-column
    # matrix (a survival (time, event) response gives one risk column).
    y_vector = !is.null(y) && !identical(ncol(y), 1L),
    levels = levels
  ))
  object
}

#' @rdname n4m_estimator_roles
#' @export
predict.n4m_regressor <- function(object, newdata, ...) {
  out <- .Call("r_n4m_estimator_predict", .n4m_pointer(object), .n4m_as_matrix(newdata),
               PACKAGE = "n4m")
  if (isTRUE(object$state$y_vector) && ncol(out) == 1L) out[, 1L] else out
}

#' @rdname n4m_estimator_roles
#' @export
predict.n4m_classifier <- function(object, newdata, type = c("class", "prob", "decision"),
                                   ...) {
  type <- match.arg(type)
  pointer <- .n4m_pointer(object)
  X <- .n4m_as_matrix(newdata)
  if (type == "class") {
    ids <- .Call("r_n4m_estimator_predict_labels", pointer, X, PACKAGE = "n4m")
    return(.n4m_label_values(ids, object$state$levels))
  }
  entry <- if (type == "prob") "r_n4m_estimator_predict_proba" else
    "r_n4m_estimator_decision_function"
  out <- .Call(entry, pointer, X, PACKAGE = "n4m")
  colnames(out) <- .n4m_label_text(n4m_classes(object))
  out
}

#' @rdname n4m_estimator_roles
#' @export
n4m_sample_mask <- function(object, X, y = NULL) UseMethod("n4m_sample_mask")

#' @rdname n4m_estimator_roles
#' @export
n4m_sample_mask.n4m_sample_filter <- function(object, X, y = NULL) {
  X <- .n4m_as_matrix(X)
  y_matrix <- if (is.null(y)) NULL else .n4m_targets(y, nrow(X))
  .Call("r_n4m_estimator_apply_mask", .n4m_pointer(object), X, y_matrix, PACKAGE = "n4m")
}

#' @rdname n4m_estimator_roles
#' @export
n4m_classes <- function(object) UseMethod("n4m_classes")

#' @rdname n4m_estimator_roles
#' @export
n4m_classes.n4m_classifier <- function(object) {
  ids <- .Call("r_n4m_estimator_classes", .n4m_pointer(object), PACKAGE = "n4m")
  levels <- object$state$levels
  if (is.null(levels)) ids else levels[ids + 1]
}

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_transform <- function(object, X, ...) UseMethod("n4m_estimator_transform")

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_transform.n4m_transformer <- function(object, X, ...) {
  .Call("r_n4m_estimator_transform", .n4m_pointer(object), .n4m_as_matrix(X), PACKAGE = "n4m")
}

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_transform.n4m_selector <- function(object, X, ...) {
  .Call("r_n4m_estimator_transform", .n4m_pointer(object), .n4m_as_matrix(X), PACKAGE = "n4m")
}

#' @rdname n4m_estimator_roles
#' @export
n4m_selected_indices <- function(object) UseMethod("n4m_selected_indices")

#' @rdname n4m_estimator_roles
#' @export
n4m_selected_indices.n4m_selector <- function(object) {
  .Call("r_n4m_estimator_selected_indices", .n4m_pointer(object), PACKAGE = "n4m") + 1
}

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_export <- function(object, allow_training_rows = FALSE) {
  if (is.null(object$state)) stop("n4m estimator is not fitted", call. = FALSE)
  if (is.null(object$state$n4me)) {
    stop("this estimator's state is not serializable", call. = FALSE)
  }
  .Call("r_n4m_estimator_export", .n4m_pointer(object), isTRUE(allow_training_rows),
        PACKAGE = "n4m")
}

#' @rdname n4m_estimator_roles
#' @export
n4m_contains_training_rows <- function(object) {
  .Call("r_n4m_estimator_contains_training_rows", .n4m_pointer(object), PACKAGE = "n4m")
}

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_import <- function(bytes) {
  pointer <- .Call("r_n4m_estimator_import", bytes, PACKAGE = "n4m")
  info <- .Call("r_n4m_estimator_info", pointer, PACKAGE = "n4m")
  roles <- .n4m_method_roles[[info$method_id]]
  if (is.null(roles)) stop("no R role mapping for ", info$method_id, call. = FALSE)
  object <- .n4m_estimator(info$method_id, roles, info$params)
  object$state <- list2env(list(pointer = pointer, n4me = bytes,
                                y_vector = info$n_outputs == 1))
  object
}

#' @export
print.n4m_method <- function(x, ...) {
  roles <- sub("^n4m_", "", setdiff(class(x), c("n4m_estimator", "n4m_method")))
  state <- if (!inherits(x, "n4m_estimator")) "" else if (is.null(x$state)) ", unfitted" else ", fitted"
  cat("<n4m ", x$method_id, " (", paste(roles, collapse = ", "), ")", state, ">\n", sep = "")
  invisible(x)
}

.n4m_procedure_run <- function(object, X, y = NULL, inputs = list()) {
  X <- .n4m_as_matrix(X)
  y_matrix <- if (is.null(y)) NULL else .n4m_targets(y, nrow(X))
  inputs <- .n4m_fit_inputs(X, inputs)
  params <- object$params[!vapply(object$params, is.null, logical(1))]
  .Call("r_n4m_procedure_run", object$method_id, params, X, y_matrix, inputs, PACKAGE = "n4m")
}

#' Generic native procedures
#'
#' Constructors such as \code{n4m_kennard_stone()}, \code{n4m_gaussian_noise()}
#' or \code{n4m_regression_metrics()} return a procedure: a catalog method run
#' once, without fitted state. Its class is its role:
#' \code{n4m_splitter} (\code{n4m_split}), \code{n4m_augmenter}
#' (\code{n4m_augment}, train-only) or \code{n4m_procedure}
#' (\code{n4m_run}). Parameters, seeds included, are native and shared with
#' the Python and JS/WASM bindings.
#'
#' @param object A procedure from one of the generated constructors.
#' @param X Numeric matrix (rows are samples).
#' @param y Optional target vector or matrix, when the method uses it.
#' @param groups Optional sample groups for group-aware splitters.
#' @param axis Optional spectral axis for axis-dependent augmenters
#'   (wavelengths in nm for the augmenters with nanometre constants).
#' @param ... Further named inputs (\code{X_target}, \code{fold_ids}, ...).
#' @return \code{n4m_split()} a list of folds, each \code{list(train, test)}
#'   of 1-based row indices; \code{n4m_augment()} the augmented matrix, or
#'   \code{list(X, Y)} for the augmenters that mix rows (mixup; \code{y}
#'   required, mixed with the same draw and returned with its shape);
#'   \code{n4m_run()} a named list of the native outputs.
#' @name n4m_procedures
NULL

#' @rdname n4m_procedures
#' @export
n4m_split <- function(object, X, y = NULL, groups = NULL) UseMethod("n4m_split")

#' @rdname n4m_procedures
#' @export
n4m_split.n4m_splitter <- function(object, X, y = NULL, groups = NULL) {
  out <- .n4m_procedure_run(object, X, y, list(groups = groups))
  lapply(out[[".folds"]], function(f) list(train = f[[1L]] + 1, test = f[[2L]] + 1))
}

#' @rdname n4m_procedures
#' @export
n4m_augment <- function(object, X, y = NULL, axis = NULL) UseMethod("n4m_augment")

#' @rdname n4m_procedures
#' @export
n4m_augment.n4m_augmenter <- function(object, X, y = NULL, axis = NULL) {
  out <- .n4m_procedure_run(object, X, y, list(axis = axis))
  if (is.null(out[["Y"]])) return(out[["X"]])
  Y <- out[["Y"]]
  if (is.null(dim(y))) Y <- as.vector(Y)
  list(X = out[["X"]], Y = Y)
}

#' @rdname n4m_procedures
#' @export
n4m_run <- function(object, X, y = NULL, ...) UseMethod("n4m_run")

#' @rdname n4m_procedures
#' @export
n4m_run.n4m_procedure <- function(object, X, y = NULL, ...) {
  .n4m_procedure_run(object, X, y, list(...))
}

#' Native manifest and method lookup
#'
#' \code{n4m_manifest_json()} returns the native manifest (every catalog
#' method's roles, DAG-ML node kinds, fit inputs and typed parameters with
#' defaults) as a JSON string, the same document every n4m binding reads.
#' \code{n4m_constructor()} returns the generated constructor of a catalog
#' method id, so pipelines can be built from method ids.
#'
#' @param method_id Catalog method id, for example \code{"models.pls.cppls"}.
#' @return A JSON string, or a constructor function.
#' @name n4m_manifest
NULL

#' @rdname n4m_manifest
#' @export
n4m_manifest_json <- function() .Call("r_n4m_manifest_json", PACKAGE = "n4m")

#' @rdname n4m_manifest
#' @export
n4m_constructor <- function(method_id) {
  name <- .n4m_method_constructors[method_id]
  if (is.na(name)) stop("no n4m role class for '", method_id, "'", call. = FALSE)
  get(name, envir = asNamespace("n4m"))
}

