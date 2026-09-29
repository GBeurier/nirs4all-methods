# SPDX-License-Identifier: CECILL-2.1
# Base-R integration gate for the native ask/tell binding. Run after R CMD INSTALL.
suppressPackageStartupMessages(library(n4m))

expect_error <- function(expr, pattern) {
  actual <- tryCatch({ force(expr); NULL }, error = function(e) conditionMessage(e))
  if (is.null(actual) || !grepl(pattern, actual, fixed = TRUE))
    stop("expected error containing '", pattern, "'; got: ", actual)
}

space <- list(
  mode = n4m_param_categorical(c("plain", "sparse")),
  components = n4m_param_int(1, 4),
  penalty = n4m_param_float(0.05, 0.5),
  weight = n4m_param_ordinal(c(0.2, 0.8)),
  knots = n4m_param_sorted_tuple(2, 0, 1)
)
conditions <- list(n4m_constraint("condition_in", c("penalty", "mode"), c("", "sparse")))
new_optimizer <- function() n4m_optimizer(space, constraints = conditions, seed = 42)
objective <- function(trial) {
  p <- trial$parameters
  penalty <- if (is.null(p$penalty)) 0.2 else p$penalty
  (p$components - 2)^2 + penalty + abs(p$weight - 0.2) + sum(p$knots)
}

# Same ordered event stream must produce the same native decisions and trace.
reference <- new_optimizer()
resumable <- new_optimizer()
for (i in seq_len(4)) {
  a <- n4m_optimizer_ask(reference)
  b <- n4m_optimizer_ask(resumable)
  stopifnot(identical(a, b), a$status == 0L, a$id == i - 1)
  stopifnot(identical(n4m_optimizer_intermediate(reference, a, 0L, objective(a)), FALSE))
  stopifnot(identical(n4m_optimizer_intermediate(resumable, b, 0L, objective(b)), FALSE))
  n4m_optimizer_tell(reference, a, objective(a))
  n4m_optimizer_tell(resumable, b, objective(b))
}
checkpoint <- n4m_optimizer_save(resumable)
stopifnot(is.raw(checkpoint$blob), length(checkpoint$blob) > 100L)
resumable <- n4m_optimizer_load(unserialize(serialize(checkpoint, NULL)))
for (i in seq_len(4)) {
  a <- n4m_optimizer_ask(reference)
  b <- n4m_optimizer_ask(resumable)
  stopifnot(identical(a, b))
  n4m_optimizer_tell(reference, a, objective(a))
  n4m_optimizer_tell(resumable, b, objective(b))
}
stopifnot(identical(n4m_optimizer_best(reference), n4m_optimizer_best(resumable)))
trace <- n4m_optimizer_trials(resumable)
stopifnot(trace$trace_format_version == 1, trace$n_trials == 8,
          identical(as.double(trace$trial_ids_i64), as.double(0:7)),
          all(trace$trial_status == 1), nrow(trace$trial_param_values) == 8,
          ncol(trace$trial_param_values) == 6,
          length(trace$trial_param_active) == 48,
          length(trace$trial_intermediate_steps) == 4)
stopifnot(n4m_optimizer_trials(resumable, since_id = 7)$n_trials == 1)
snapshot <- trace
rm(resumable)
invisible(gc())
stopifnot(snapshot$n_trials == 8)

# Batch asks and warm starts use the same C ABI and retain committed trials.
queued <- n4m_optimizer(list(components = n4m_param_int(1, 4)), seed = 7)
n4m_optimizer_enqueue(queued, c(components = 3))
batch <- n4m_optimizer_ask_batch(queued, 3L)
stopifnot(length(batch) == 3L, batch[[1]]$parameters$components == 3,
          identical(n4m_optimizer_ask_batch(queued, 0L), list()))
for (trial in batch) n4m_optimizer_tell(queued, trial, trial$parameters$components)
stopifnot(n4m_optimizer_best(queued)$score >= 1)
n4m_optimizer_close(queued)
n4m_optimizer_close(queued)
expect_error(n4m_optimizer_ask(queued), "closed")
large_category <- n4m_optimizer(list(code = n4m_param_categorical(
  c(2147483648, 2147483649), type = "integer")), seed = 3)
stopifnot(n4m_optimizer_ask(large_category)$parameters$code %in%
            c(2147483648, 2147483649))

# Malformed checkpoint/input must fail before exposing a native handle.
broken <- checkpoint
broken$blob[1] <- as.raw(0)
expect_error(n4m_optimizer_load(broken), "optimizer_load")
expect_error(n4m_optimizer(list(x = n4m_param_int(1, 2)), eval_mode = "best"),
             "optimizer_create")
expect_error(n4m_optimizer_tell(queued, batch[[1]], Inf), "finite score")
expect_error(n4m_optimizer_trials(queued, since_id = -1), "nonnegative")
expect_error(n4m_param_categorical(c(1.5, 2.5), type = "integer"), "exact integers")
cat("R native optimizer ask/tell, trace, batch and checkpoint OK\n")
