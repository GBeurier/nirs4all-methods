# SPDX-License-Identifier: CECILL-2.1
# Exercise the authoritative HPO registry through the R binding and compare
# native proposals, pruning decisions and the owning rich trace to Python goldens.
suppressPackageStartupMessages(library(n4m))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L)
  stop("usage: test_optimizer_goldens.R SPECS.json GOLDEN_DIR", call. = FALSE)
if (!requireNamespace("jsonlite", quietly = TRUE))
  stop("jsonlite is required for the cross-binding golden gate", call. = FALSE)
specs <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
stopifnot(length(specs) %in% c(14L, 45L))

same <- function(actual, expected, label) {
  result <- all.equal(actual, expected, tolerance = 0, check.attributes = FALSE)
  if (!isTRUE(result))
    stop(label, ": ", paste(result, collapse = "; "), call. = FALSE)
}

axis <- function(declaration) {
  kind <- declaration$kind
  values <- unlist(declaration$args, use.names = FALSE)
  if (kind == "int")
    return(n4m_param_int(values[[1]], values[[2]],
                         if (length(values) > 2L) values[[3]] else 1))
  if (kind == "log_int")
    return(n4m_param_int(values[[1]], values[[2]], log = TRUE))
  if (kind == "float")
    return(n4m_param_float(values[[1]], values[[2]],
                           if (length(values) > 2L) values[[3]] else 0))
  if (kind == "log_float")
    return(n4m_param_float(values[[1]], values[[2]], log = TRUE))
  if (kind == "categorical")
    return(n4m_param_categorical(values))
  if (kind == "ordinal")
    return(n4m_param_ordinal(as.double(values)))
  stop("unsupported HPO axis: ", kind, call. = FALSE)
}

pool_string <- function(bytes, offsets, index) {
  first <- as.integer(offsets[[index]]) + 1L
  last <- as.integer(offsets[[index + 1L]])
  if (last < first) return("")
  rawToChar(as.raw(bytes[first:last]))
}

check_trace <- function(trace, spec, golden) {
  same(trace$trace_format_version, 1, paste(spec$id, "trace version"))
  same(trace$n_trials, length(golden), paste(spec$id, "trial count"))
  same(trace$n_params, length(spec$space), paste(spec$id, "parameter count"))
  same(trace$n_intermediates,
       sum(vapply(golden, function(item) length(item$intermediates), integer(1))),
       paste(spec$id, "intermediate count"))
  same(trace$n_events, 1 + max(vapply(golden, function(item) as.double(item$terminal_at),
                                       numeric(1))), paste(spec$id, "event count"))
  for (i in seq_along(golden)) {
    expected <- golden[[i]]
    label <- paste(spec$id, "trial", i - 1L)
    same(trace$trial_ids_i64[[i]], expected$id, paste(label, "id"))
    same(trace$trial_ask_sequence[[i]], expected$asked_at, paste(label, "ask event"))
    same(as.integer(trace$trial_status[1L, i]), expected$status, paste(label, "status"))
    same(trace$trial_terminal_sequence[[i]], expected$terminal_at,
         paste(label, "terminal event"))
    if (expected$status == 1L)
      same(trace$trial_scores[1L, i], expected$score, paste(label, "score"))
    else if (!is.nan(trace$trial_scores[1L, i]))
      stop(label, " has an unexpected score", call. = FALSE)

    for (j in seq_along(spec$space)) {
      declaration <- spec$space[[j]]
      name <- declaration$name
      cell <- (i - 1L) * length(spec$space) + j
      same(trace$trial_param_active[[cell]], 1L, paste(label, name, "active"))
      value <- if (declaration$kind == "categorical")
        pool_string(trace$trial_param_label_utf8, trace$trial_param_label_offsets, cell)
      else trace$trial_param_values[i, j]
      same(value, expected$params[[name]], paste(label, name, "trace parameter"))
    }

    start <- as.integer(trace$trial_intermediate_offsets[[i]]) + 1L
    end <- as.integer(trace$trial_intermediate_offsets[[i + 1L]])
    reports <- if (end < start) integer() else seq.int(start, end)
    if (!is.null(spec$intermediate) && nzchar(spec$intermediate)) {
      same(length(reports), length(expected$intermediates),
           paste(label, "intermediate count"))
      for (k in seq_along(reports)) {
        index <- reports[[k]]
        event <- expected$intermediates[[k]]
        same(trace$trial_intermediate_sequence[[index]], event$sequence,
             paste(label, "intermediate event"))
        same(trace$trial_intermediate_steps[[index]], event$step,
             paste(label, "intermediate step"))
        same(trace$trial_intermediate_scores[1L, index], event$score,
             paste(label, "intermediate score"))
        same(as.logical(trace$trial_intermediate_should_prune[[index]]), event$should_prune,
             paste(label, "pruning decision"))
      }
      pruned <- Filter(function(event) isTRUE(event$should_prune), expected$intermediates)
      same(if (length(pruned)) pruned[[1]]$step else -1L, expected$pruned_at,
           paste(label, "prune step"))
    } else same(length(reports), 0L, paste(label, "unexpected intermediates"))

    code <- pool_string(trace$trial_error_code_utf8, trace$trial_error_code_offsets, i)
    message <- pool_string(trace$trial_error_message_utf8,
                           trace$trial_error_message_offsets, i)
    if (is.null(expected$error)) {
      same(code, "", paste(label, "unexpected error code"))
      same(message, "", paste(label, "unexpected error message"))
      same(trace$trial_error_retryable[[i]], 0L, paste(label, "unexpected retry policy"))
    } else {
      same(code, expected$error$code, paste(label, "error code"))
      same(message, expected$error$message, paste(label, "error message"))
      same(as.logical(trace$trial_error_retryable[[i]]), expected$error$retryable,
           paste(label, "retryable"))
    }
  }
}

for (spec in specs) {
  golden <- jsonlite::fromJSON(file.path(args[[2]], paste0(spec$id, ".json")),
                               simplifyVector = FALSE)
  space <- lapply(spec$space, axis)
  names(space) <- vapply(spec$space, function(item) item$name, character(1))
  optimizer <- n4m_optimizer(space, sampler = spec$sampler, pruner = spec$pruner,
                             direction = spec$direction,
                             n_startup_trials = spec$n_startup_trials, seed = spec$seed,
                             max_resource = spec$max_resource,
                             reduction_factor = spec$reduction_factor)
  tryCatch({
    batch_size <- spec$max_in_flight
    checkpoint_after <- if (is.null(spec$checkpoint_after_trials))
      0L else as.integer(spec$checkpoint_after_trials)
    resumed <- FALSE
    for (start in seq.int(1L, spec$n_trials, by = batch_size)) {
      size <- min(batch_size, spec$n_trials - start + 1L)
      batch <- lapply(seq_len(size), function(unused) n4m_optimizer_ask(optimizer))
      for (trial in batch) {
        expected <- golden[[as.integer(trial$id) + 1L]]
        same(sort(names(trial$parameters)), sort(names(expected$params)),
             paste(spec$id, "trial", trial$id, "parameter names"))
        same(trial$parameters[sort(names(trial$parameters))],
             expected$params[sort(names(expected$params))],
             paste(spec$id, "trial", trial$id, "proposal"))
      }
      order <- if (length(spec$tell_order))
        as.integer(unlist(spec$tell_order, use.names = FALSE)) + 1L else seq_len(size)
      order <- order[order <= size]
      pruned <- rep(FALSE, size)
      if (!is.null(spec$intermediate) && nzchar(spec$intermediate)) {
        for (step in seq.int(0L, spec$intermediate_steps - 1L)) {
          for (position in order) {
            if (pruned[[position]]) next
            trial <- batch[[position]]
            event <- golden[[as.integer(trial$id) + 1L]]$intermediates[[step + 1L]]
            same(event$step, step, paste(spec$id, "intermediate score tape"))
            decision <- n4m_optimizer_intermediate(optimizer, trial, step, event$score)
            same(decision, event$should_prune,
                 paste(spec$id, "trial", trial$id, "pruner"))
            if (decision) pruned[[position]] <- TRUE
          }
        }
      }
      for (position in order) {
        if (pruned[[position]]) next
        trial <- batch[[position]]
        expected <- golden[[as.integer(trial$id) + 1L]]
        if (trial$id %in% unlist(spec$failed_trial_ids, use.names = FALSE)) {
          n4m_optimizer_tell(optimizer, trial, status = "failed",
                             error = paste0("n4m.error.v1|", expected$error$code,
                                            "|0|", expected$error$message))
        } else n4m_optimizer_tell(optimizer, trial, score = expected$score)
      }
      if (checkpoint_after > 0L && start + size - 1L == checkpoint_after) {
        prefix <- n4m_optimizer_trials(optimizer)
        same(prefix$n_trials, checkpoint_after, paste(spec$id, "checkpoint prefix"))
        if (any(prefix$trial_status == 0L))
          stop(spec$id, ": checkpoint contains a running trial", call. = FALSE)
        checkpoint <- n4m_optimizer_save(optimizer)
        restored <- n4m_optimizer_load(checkpoint)
        n4m_optimizer_close(optimizer)
        optimizer <- restored
        resumed <- TRUE
      }
    }
    same(resumed, checkpoint_after > 0L, paste(spec$id, "checkpoint exercised"))
    check_trace(n4m_optimizer_trials(optimizer), spec, golden)
    cat("HPO R golden:", spec$id, "(", length(golden), "trials )\n")
  }, finally = n4m_optimizer_close(optimizer))
}
cat("All", length(specs), "HPO R golden traces match Python/native\n")
