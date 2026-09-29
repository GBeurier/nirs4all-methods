# SPDX-License-Identifier: CECILL-2.1

# The declarations below are plain R values. Only the native engine samples,
# tracks, prunes and serializes trials; the R layer never implements an optimizer.

#' Integer search axis for the native optimizer
#' @param low,high Inclusive bounds, exactly representable as binary64 integers.
#' @param step Positive integer grid step.
#' @param log Whether to sample the axis on a logarithmic scale.
#' @return A search-axis declaration for [n4m_optimizer()].
#' @export
n4m_param_int <- function(low, high, step = 1, log = FALSE) {
  .n4m_exact_integer(low, "low")
  .n4m_exact_integer(high, "high")
  .n4m_exact_integer(step, "step")
  .n4m_flag(log, "log")
  list(kind = if (log) 2L else 0L, low = as.double(low),
       high = as.double(high), step = as.double(step))
}

#' Floating-point search axis for the native optimizer
#' @inheritParams n4m_param_int
#' @param step Zero for continuous sampling, otherwise a positive grid step.
#' @export
n4m_param_float <- function(low, high, step = 0, log = FALSE) {
  for (item in list(low = low, high = high, step = step)) {
    if (!is.numeric(item) || length(item) != 1L || !is.finite(item))
      stop("float bounds and step must be finite scalar numbers", call. = FALSE)
  }
  .n4m_flag(log, "log")
  list(kind = if (log) 3L else 1L, low = as.double(low),
       high = as.double(high), step = as.double(step))
}

#' Homogeneous categorical search axis
#' @param choices Nonempty character, double, integer or logical vector without missing values.
#' @param type Optional native category type. Use `"integer"` for exactly
#'   representable integer choices larger than R's 32-bit integer range.
#' @export
n4m_param_categorical <- function(choices, type = NULL) {
  if (!is.atomic(choices) || !is.null(dim(choices)) || length(choices) < 1L ||
      !typeof(choices) %in% c("character", "double", "integer", "logical") ||
      anyNA(choices) || (is.numeric(choices) && any(!is.finite(choices))))
    stop("choices must be a nonempty homogeneous atomic vector without NA", call. = FALSE)
  inferred <- switch(typeof(choices), character = "string", double = "float",
                     integer = "integer", logical = "boolean")
  if (is.null(type)) type <- inferred
  type <- match.arg(type, c("string", "integer", "float", "boolean"))
  if (type == "integer") {
    if (!is.numeric(choices) || any(choices != floor(choices)) || any(abs(choices) > 2^53))
      stop("integer choices must be binary64-exact integers", call. = FALSE)
  } else if (type != inferred) stop("category type disagrees with choices", call. = FALSE)
  list(kind = 4L, choices = choices,
       category_type = match(type, c("string", "integer", "float", "boolean")) - 1L)
}

#' Ordered numeric search axis
#' @param choices Ordered finite numeric values.
#' @export
n4m_param_ordinal <- function(choices) {
  if (!is.numeric(choices) || length(choices) < 1L || any(!is.finite(choices)))
    stop("ordinal choices must be a nonempty finite numeric vector", call. = FALSE)
  list(kind = 5L, choices = as.double(choices))
}

#' Sorted tuple search axis
#' @param length Number of elements.
#' @inheritParams n4m_param_int
#' @param integer Whether elements are integers.
#' @export
n4m_param_sorted_tuple <- function(length, low, high, integer = FALSE) {
  .n4m_int32(length, "length")
  .n4m_flag(integer, "integer")
  if (!is.numeric(low) || !is.numeric(high) || length(low) != 1L ||
      length(high) != 1L || !is.finite(low) || !is.finite(high))
    stop("tuple bounds must be finite scalar numbers", call. = FALSE)
  list(kind = 6L, length = as.integer(length), low = as.double(low),
       high = as.double(high), integer = integer)
}

#' Search-space constraint passed to the native optimizer
#' @param kind One of mutex, requires, exclude, condition_in, condition_not_in.
#' @param refs Parameter names, ordered as required by the C ABI.
#' @param labels Matching categorical/ordinal labels; empty strings mean bare references.
#' @export
n4m_constraint <- function(kind, refs, labels = rep("", length(refs))) {
  kind <- match.arg(kind, c("mutex_group", "mutex", "requires", "exclude",
                            "condition_in", "condition_not_in"))
  if (!is.character(refs) || !is.character(labels) || length(refs) < 1L ||
      length(refs) != length(labels) || anyNA(refs) || anyNA(labels))
    stop("refs and labels must be equal nonempty character vectors", call. = FALSE)
  if (kind == "mutex") kind <- "mutex_group"
  list(kind = match(kind, c("mutex_group", "requires", "exclude",
                            "condition_in", "condition_not_in")) - 1L,
       refs = refs, labels = labels)
}

.n4m_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value))
    stop(name, " must be TRUE or FALSE", call. = FALSE)
}

.n4m_exact_integer <- function(value, name) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value != floor(value) || abs(value) > 2^53)
    stop(name, " must be an exactly representable integer", call. = FALSE)
}

.n4m_int32 <- function(value, name) {
  .n4m_exact_integer(value, name)
  if (value < -2147483648 || value > 2147483647)
    stop(name, " must fit int32", call. = FALSE)
}

.n4m_choice <- function(value, choices, name) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !value %in% choices)
    stop(name, " must be one of: ", paste(choices, collapse = ", "), call. = FALSE)
  match(value, choices) - 1L
}

#' Create a native ask/tell optimizer
#' @param space Named list of axes built with `n4m_param_*` constructors.
#' @param constraints List of [n4m_constraint()] declarations.
#' @param sampler Native sampler name.
#' @param pruner Native pruner name.
#' @param direction Score direction; `auto` follows the metric.
#' @param eval_mode Native fold aggregation policy.
#' @param metric Native metric name.
#' @param liar Constant-liar policy for parallel asks.
#' @param n_startup_trials Number of startup trials for adaptive samplers.
#' @param seed Nonnegative exact integer seed (up to 2^53).
#' @param timeout_seconds Native wall-clock timeout, zero for none.
#' @param max_resource Resource limit for Hyperband; zero otherwise.
#' @param reduction_factor ASHA/Hyperband reduction factor; zero for default.
#' @return An `n4m_optimizer` handle. Run [n4m_optimizer_ask()] and report scores.
#' @export
n4m_optimizer <- function(space, constraints = list(), sampler = "random",
                          pruner = "none", direction = "auto", eval_mode = "mean",
                          metric = "rmse", liar = "none", n_startup_trials = 10L,
                          seed = 0, timeout_seconds = 0, max_resource = 0L,
                          reduction_factor = 0L) {
  if (!is.list(space) || length(space) < 1L || is.null(names(space)) ||
      anyNA(names(space)) || any(!nzchar(names(space))) || anyDuplicated(names(space)))
    stop("space must be a nonempty named list with unique names", call. = FALSE)
  if (!is.list(constraints)) stop("constraints must be a list", call. = FALSE)
  .n4m_exact_integer(seed, "seed")
  if (seed < 0) stop("seed must be nonnegative", call. = FALSE)
  for (item in list(n_startup_trials = n_startup_trials, max_resource = max_resource,
                    reduction_factor = reduction_factor)) .n4m_int32(item, "option")
  if (!is.numeric(timeout_seconds) || length(timeout_seconds) != 1L ||
      !is.finite(timeout_seconds) || timeout_seconds < 0)
    stop("timeout_seconds must be finite and nonnegative", call. = FALSE)
  metric_values <- c(rmse = 0L, mse = 1L, mae = 2L, r2 = 3L, accuracy = 16L,
                     balanced_accuracy = 17L, f1 = 18L, logloss = 19L)
  if (!is.character(metric) || length(metric) != 1L || is.na(metric) ||
      !metric %in% names(metric_values))
    stop("unknown metric", call. = FALSE)
  options <- list(
    sampler = .n4m_choice(sampler, c("random", "sobol", "lhs", "ternary", "ga",
                                     "pso", "cmaes", "tpe", "gp_ei"), "sampler"),
    pruner = .n4m_choice(pruner, c("none", "median", "asha", "hyperband", "racing"), "pruner"),
    direction = .n4m_choice(direction, c("auto", "minimize", "maximize"), "direction"),
    eval_mode = .n4m_choice(eval_mode, c("best", "mean", "robust_best"), "eval_mode"),
    metric = unname(metric_values[[metric]]),
    liar = .n4m_choice(liar, c("none", "min", "mean", "max"), "liar"),
    n_startup_trials = as.integer(n_startup_trials), seed = as.double(seed),
    timeout_seconds = as.double(timeout_seconds), max_resource = as.integer(max_resource),
    reduction_factor = as.integer(reduction_factor)
  )
  ptr <- .Call("r_n4m_optimizer_create", space, constraints, options, PACKAGE = "n4m")
  structure(list(ptr = ptr, space = space, constraints = constraints), class = "n4m_optimizer")
}

.n4m_optimizer_handle <- function(optimizer) {
  if (!inherits(optimizer, "n4m_optimizer") || !is.list(optimizer) ||
      typeof(optimizer$ptr) != "externalptr")
    stop("expected an n4m_optimizer", call. = FALSE)
  optimizer$ptr
}

#' Release a native optimizer handle immediately
#' @param optimizer An `n4m_optimizer`.
#' @return Invisibly `TRUE`. A second close is harmless.
#' @export
n4m_optimizer_close <- function(optimizer) {
  invisible(.Call("r_n4m_optimizer_close", .n4m_optimizer_handle(optimizer), PACKAGE = "n4m"))
}

#' Request a trial from the native optimizer
#' @param optimizer An `n4m_optimizer`.
#' @return Trial ID, typed parameter values, native status and resource rung.
#' @export
n4m_optimizer_ask <- function(optimizer) {
  .Call("r_n4m_optimizer_ask", .n4m_optimizer_handle(optimizer), optimizer$space,
        PACKAGE = "n4m")
}

#' Request a native batch of trials
#' @param optimizer An `n4m_optimizer`.
#' @param n Number of requested trials, between 0 and 10000.
#' @return List of committed trials in ask order. On a partial native error,
#'   the returned list retains committed trials and has `native_status` attribute;
#'   the caller must terminalize or recover those trials.
#' @export
n4m_optimizer_ask_batch <- function(optimizer, n) {
  .n4m_int32(n, "n")
  if (n < 0 || n > 10000) stop("n must be between 0 and 10000", call. = FALSE)
  .Call("r_n4m_optimizer_ask_batch", .n4m_optimizer_handle(optimizer),
        optimizer$space, as.integer(n), PACKAGE = "n4m")
}

#' Queue a native warm-start candidate
#' @param optimizer An `n4m_optimizer`.
#' @param parameters Named numeric vector of axis values. Categorical values
#'   are zero-based choice indices, following the C ABI.
#' @export
n4m_optimizer_enqueue <- function(optimizer, parameters) {
  if (!is.numeric(parameters) || length(parameters) < 1L ||
      is.null(names(parameters)) || anyNA(names(parameters)) ||
      any(!nzchar(names(parameters))) || anyDuplicated(names(parameters)) ||
      any(!is.finite(parameters)))
    stop("parameters must be a nonempty named finite numeric vector", call. = FALSE)
  values <- as.double(parameters)
  names(values) <- names(parameters)
  invisible(.Call("r_n4m_optimizer_enqueue", .n4m_optimizer_handle(optimizer),
                  values, PACKAGE = "n4m"))
}

#' Report a terminal trial result
#' @param optimizer An `n4m_optimizer`.
#' @param trial Trial returned by [n4m_optimizer_ask()], or its exact numeric ID.
#' @param score Finite score for completed trials; NULL otherwise.
#' @param status `completed`, `pruned`, `failed` or `cancelled`.
#' @param error Optional plain or versioned native error text for failed/cancelled trials.
#' @export
n4m_optimizer_tell <- function(optimizer, trial, score = NULL, status = "completed", error = NULL) {
  id <- if (is.list(trial)) trial$id else trial
  .n4m_exact_integer(id, "trial id")
  .n4m_choice(status, c("completed", "pruned", "failed", "cancelled"), "status")
  if (identical(status, "completed")) {
    if (!is.numeric(score) || length(score) != 1L || !is.finite(score))
      stop("completed trials need one finite score", call. = FALSE)
  } else if (!is.null(score)) stop("only completed trials accept a score", call. = FALSE)
  if (!is.null(error) && (!is.character(error) || length(error) != 1L || is.na(error)))
    stop("error must be NULL or one string", call. = FALSE)
  invisible(.Call("r_n4m_optimizer_tell", .n4m_optimizer_handle(optimizer),
                  as.double(id), match(status, c("running", "completed", "pruned",
                                                "failed", "cancelled")) - 1L,
                  score, error, PACKAGE = "n4m"))
}

#' Report an intermediate score and ask the native pruner for a decision
#' @param optimizer An `n4m_optimizer`.
#' @param trial Trial or exact numeric ID.
#' @param step Nonnegative resource step.
#' @param score Finite intermediate score.
#' @return `TRUE` when the native optimizer pruned the trial.
#' @export
n4m_optimizer_intermediate <- function(optimizer, trial, step, score) {
  id <- if (is.list(trial)) trial$id else trial
  .n4m_exact_integer(id, "trial id")
  .n4m_int32(step, "step")
  if (!is.numeric(score) || length(score) != 1L || !is.finite(score))
    stop("score must be one finite number", call. = FALSE)
  .Call("r_n4m_optimizer_intermediate", .n4m_optimizer_handle(optimizer),
        as.double(id), as.integer(step), as.double(score), PACKAGE = "n4m")
}

#' Return the native optimizer's best completed trial
#' @param optimizer An `n4m_optimizer`.
#' @return List with `trial` and `score`. Errors if no completed trial exists.
#' @export
n4m_optimizer_best <- function(optimizer) {
  .Call("r_n4m_optimizer_best", .n4m_optimizer_handle(optimizer),
        optimizer$space, PACKAGE = "n4m")
}

#' Save a portable N4MOPT checkpoint with its R search-space declaration
#' @param optimizer An `n4m_optimizer`.
#' @return List containing native raw N4MOPT bytes and R axis declarations.
#' @export
n4m_optimizer_save <- function(optimizer) {
  list(blob = .Call("r_n4m_optimizer_save", .n4m_optimizer_handle(optimizer), PACKAGE = "n4m"),
       space = optimizer$space, constraints = optimizer$constraints)
}

#' Restore a native optimizer from a checkpoint
#' @param checkpoint Result of [n4m_optimizer_save()].
#' @return Resumed `n4m_optimizer` handle.
#' @export
n4m_optimizer_load <- function(checkpoint) {
  if (!is.list(checkpoint) || typeof(checkpoint$blob) != "raw" ||
      !is.list(checkpoint$space) || !is.list(checkpoint$constraints))
    stop("expected an n4m optimizer checkpoint", call. = FALSE)
  ptr <- .Call("r_n4m_optimizer_load", checkpoint$blob, PACKAGE = "n4m")
  structure(list(ptr = ptr, space = checkpoint$space,
                 constraints = checkpoint$constraints), class = "n4m_optimizer")
}

#' Read an owning snapshot of the complete native trial trace
#' @param optimizer An `n4m_optimizer`.
#' @param since_id Inclusive nonnegative trial ID filter.
#' @return Named list of native trace v1 fields; matrices use R column-major
#'   layout and int64 vectors become character if a value exceeds the exact
#'   binary64 integer range. See `docs/methods/optimization.md` for the key schema.
#' @export
n4m_optimizer_trials <- function(optimizer, since_id = 0) {
  .n4m_exact_integer(since_id, "since_id")
  if (since_id < 0) stop("since_id must be nonnegative", call. = FALSE)
  .Call("r_n4m_optimizer_trials", .n4m_optimizer_handle(optimizer),
        as.double(since_id), PACKAGE = "n4m")
}
