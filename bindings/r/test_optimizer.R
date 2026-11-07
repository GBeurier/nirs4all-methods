# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
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

# Every log-domain proposal must survive running and completed checkpoints.
log_domains <- list(c(1e-5, 1e-2), c(5, 5000), c(7, 7000), c(50, 50000),
                    c(1000, 1e6), c(1e15, 1e16), c(1e18, 9e18),
                    c(2e19, 9e20), c(1e21, 9e21), c(0.5, 2.5), c(1, 1000))
float_bits <- function(value) writeBin(as.double(value), raw(), size = 8, endian = "big")
for (domain in log_domains) {
  low <- domain[1]
  high <- domain[2]
  sp <- list(scale = n4m_param_float(low, high, log = TRUE), aux = n4m_param_float(0, 1))
  opt <- n4m_optimizer(sp, sampler = 'sobol', seed = 19)
  a <- n4m_optimizer_ask(opt)
  stopifnot(is.finite(a$parameters$scale), a$parameters$scale >= low, a$parameters$scale <= high)
  if (low %in% c(0.5, 1)) stopifnot(identical(float_bits(a$parameters$scale), float_bits(low)))
  running <- n4m_optimizer_load(n4m_optimizer_save(opt))
  stopifnot(identical(as.integer(n4m_optimizer_trials(running)$trial_status), 0L))
  n4m_optimizer_tell(opt, a, 0.5)
  n4m_optimizer_tell(running, a, 0.5)
  stopifnot(identical(as.integer(n4m_optimizer_trials(opt)$trial_status), 1L))
  completed <- n4m_optimizer_load(n4m_optimizer_save(opt))
  stopifnot(identical(as.integer(n4m_optimizer_trials(completed)$trial_status), 1L))
  expected <- n4m_optimizer_ask(opt)
  for (restored in list(running, completed)) {
    actual <- n4m_optimizer_ask(restored)
    stopifnot(identical(actual$id, expected$id))
    for (name in c("scale", "aux"))
      stopifnot(identical(float_bits(actual$parameters[[name]]), float_bits(expected$parameters[[name]])))
  }
  n4m_optimizer_close(running)
  n4m_optimizer_close(completed)
  n4m_optimizer_close(opt)
}
check_staged_continuation <- function(opt, trial, score, names) {
  running <- n4m_optimizer_load(n4m_optimizer_save(opt))
  on.exit(n4m_optimizer_close(running), add = TRUE)
  statuses <- as.integer(n4m_optimizer_trials(running)$trial_status)
  stopifnot(tail(statuses, 1) == 0L, all(head(statuses, -1) == 1L))
  n4m_optimizer_tell(opt, trial, score)
  n4m_optimizer_tell(running, trial, score)
  completed <- n4m_optimizer_load(n4m_optimizer_save(opt))
  on.exit(n4m_optimizer_close(completed), add = TRUE)
  stopifnot(all(as.integer(n4m_optimizer_trials(completed)$trial_status) == 1L))
  expected <- n4m_optimizer_ask(opt)
  for (restored in list(running, completed)) {
    actual <- n4m_optimizer_ask(restored)
    stopifnot(identical(actual$id, expected$id))
    for (name in names)
      stopifnot(identical(float_bits(actual$parameters[[name]]), float_bits(expected$parameters[[name]])))
  }
}
for (domain in list(c(1e15, 1e16), c(2e19, 9e20))) {
  for (sampler in c('cmaes', 'tpe')) {
    sp <- list(scale = n4m_param_float(domain[1], domain[2], log = TRUE), aux = n4m_param_float(0, 1))
    opt <- n4m_optimizer(sp, sampler = sampler, direction = 'minimize', seed = 19)
    for (i in seq_len(60)) {
      trial <- n4m_optimizer_ask(opt)
      value <- trial$parameters$scale
      stopifnot(is.finite(value), value >= domain[1], value <= domain[2])
      if (i == 60) check_staged_continuation(opt, trial, -log(value), c("scale", "aux"))
      else n4m_optimizer_tell(opt, trial, -log(value))
    }
    n4m_optimizer_close(opt)
  }
}
log_int_space <- list(scale = n4m_param_int(1e15, 2^53 - 1, log = TRUE))
log_int_opt <- n4m_optimizer(log_int_space, sampler = 'sobol', seed = 19)
log_int_trial <- n4m_optimizer_ask(log_int_opt)
stopifnot(log_int_trial$parameters$scale >= 1e15, log_int_trial$parameters$scale <= 2^53 - 1)
check_staged_continuation(log_int_opt, log_int_trial, 0.5, "scale")
n4m_optimizer_close(log_int_opt)

# Genuine pre-fix EMPTY checkpoints; no payload bytes are reconstructed.
hex_raw <- function(x) as.raw(strtoi(substring(x, seq(1, nchar(x), 2), seq(2, nchar(x), 2)), 16L))
domain_checkpoint <- list(blob = hex_raw("4e344d4f50540d0a0100000020000000e800000000000000c0000000000000002a4561ca5cd613b13c0000000000000001000000010000007801000000763b7730d142eeff763b7730d142ee7f000000000000000000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000eb9af1ffa822dc3e00000000000000000000000000000000000000000000000000000000e41aeb13812ec382"), space = list(x = n4m_param_float(-1.7e308, 1.7e308)), constraints = list())
expect_error(n4m_optimizer_load(domain_checkpoint), "optimizer checkpoint option/search-space contract mismatch")
domain_checkpoint <- list(blob = hex_raw("4e344d4f50540d0a0100000020000000e800000000000000c00000000000000061667f6d1e6e72713c0000000000000001000000010000007806000000763b7730d142eeff763b7730d142ee7f000000000000000000000000000000000000000000000200000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000fb3f8cb2ac6bdc3e000000000000000000000000000000000000000000000000000000008f9e37096cfc04b1"), space = list(x = n4m_param_sorted_tuple(2, -1.7e308, 1.7e308)), constraints = list())
expect_error(n4m_optimizer_load(domain_checkpoint), "optimizer checkpoint option/search-space contract mismatch")
expect_error(n4m_optimizer(list(x = n4m_param_float(-1.7e308, 1.7e308))), "optimizer_create")
expect_error(n4m_optimizer(list(x = n4m_param_sorted_tuple(2, -1.7e308, 1.7e308))), "optimizer_create")

# Domain stages use live references and compare semantic trace fields, not durations.
domain_trace <- function(opt) {
  trace <- n4m_optimizer_trials(opt)
  trace$trial_duration <- NULL
  trace
}
domain_stages <- function(space, sampler, seed, validate) {
  stages <- list(c(0, 0), c(1, 1), c(1, 0), c(12, 1), c(12, 0), c(13, 1))
  for (stage in stages) {
    original <- n4m_optimizer(space, sampler = sampler, direction = "minimize", seed = seed)
    pending <- NULL
    if (stage[1] > 0) for (i in seq_len(stage[1])) {
      pending <- n4m_optimizer_ask(original)
      validate(pending)
      if (!(stage[2] == 1 && i == stage[1])) n4m_optimizer_tell(original, pending, ((i-1) %% 7)/7)
    }
    captured <- domain_trace(original)
    restored <- n4m_optimizer_load(n4m_optimizer_save(original))
    stopifnot(identical(restored$space, original$space),
              identical(restored$constraints, original$constraints),
              identical(domain_trace(restored), captured))
    if (stage[2] == 1) {
      n4m_optimizer_tell(original, pending, 0.5)
      n4m_optimizer_tell(restored, pending, 0.5)
    }
    for (i in seq_len(6)) {
      a <- n4m_optimizer_ask(original)
      b <- n4m_optimizer_ask(restored)
      validate(a); validate(b)
      stopifnot(identical(a$id, b$id), identical(names(a$parameters), names(b$parameters)))
      for (name in names(a$parameters))
        stopifnot(identical(float_bits(a$parameters[[name]]), float_bits(b$parameters[[name]])))
      n4m_optimizer_tell(original, a, ((i-1) %% 7)/7)
      n4m_optimizer_tell(restored, b, ((i-1) %% 7)/7)
    }
    n4m_optimizer_close(restored)
    n4m_optimizer_close(original)
  }
}
domain_logs <- list(
  list(FALSE, 0.1, 0.1 + 2^-56), list(FALSE, 1e300, 1e300 * (1 + 1e-14)),
  list(FALSE, 1e-5, 1e-5 + 2^-68), list(TRUE, 1e15, 1e15+1),
  list(TRUE, 1e15, 1e15+2), list(TRUE, 2^53-2, 2^53),
  list(TRUE, 1e15, 1e15+3), list(FALSE, 1, 1 + 2^-52),
  list(FALSE, 5, 5), list(FALSE, 0.5, 2.5), list(FALSE, 1, 1000))
for (domain in domain_logs) for (sampler in c("gp_ei", "tpe")) for (seed in c(19, 3)) {
  low <- domain[[2]]; high <- domain[[3]]
  axis <- if (domain[[1]]) n4m_param_int(low, high, log = TRUE) else n4m_param_float(low, high, log = TRUE)
  space <- list(x = axis, aux = n4m_param_float(0, 1))
  domain_stages(space, sampler, seed, function(t) {
    stopifnot(is.finite(t$parameters$x), t$parameters$x >= low, t$parameters$x <= high)
  })
}
p <- 2^52; q <- 2^53
integer_tuple_domains <- list(c(0,10),c(-5,5),c(1000,1002),c(-1000,-998),c(2^40,2^40+2),
  c(2^50,2^50+2),c(2^51,2^51+2),c(-2^51-2,-2^51),c(p,p),c(p,p+1),c(p,p+2),
  c(p+1000,p+1003),c(q-3,q-1),c(q-2,q),c(-q,-q+2),c(-q,-q),c(-p-2,-p),c(-q,q),c(-q+1,q-1))
for (domain in integer_tuple_domains) for (sampler in c("random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei")) for (len in c(1L,3L)) {
  low <- domain[1]; high <- domain[2]
  space <- list(x = n4m_param_sorted_tuple(len, low, high, integer = TRUE), aux = n4m_param_float(0, 1))
  domain_stages(space, sampler, 19, function(t) {
    values <- t$parameters$x
    stopifnot(length(values) == len, all(is.finite(values)), all(values == trunc(values)),
              all(values >= low), all(values <= high), !is.unsorted(values))
  })
}
span_max <- .Machine$double.xmax; span_power <- 2^1023
bad_spans <- list(c(-1.7e308,1.7e308),c(-span_max,span_max),c(-span_power,span_max/2),c(-span_power,span_power))
good_spans <- list(c(-span_max/2,span_max/2),c(0,span_max),c(-8e307,8e307),c(-span_power,span_max/2-2^970))
for (domain in bad_spans) for (tuple_axis in c(FALSE, TRUE)) for (aux in c(FALSE, TRUE)) {
  axis <- if (tuple_axis) n4m_param_sorted_tuple(2,domain[1],domain[2]) else n4m_param_float(domain[1],domain[2])
  space <- list(x = axis)
  if (aux) space$aux <- n4m_param_float(0, 1)
  expect_error(n4m_optimizer(space, seed = 19), "optimizer_create")
}
for (domain in good_spans) for (tuple_axis in c(FALSE, TRUE)) for (sampler in c("random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei")) {
  low <- domain[1]; high <- domain[2]
  axis <- if (tuple_axis) n4m_param_sorted_tuple(2,low,high) else n4m_param_float(low,high)
  space <- list(x = axis, aux = n4m_param_float(0,1))
  domain_stages(space, sampler, 19, function(t) {
    values <- t$parameters$x
    stopifnot(all(is.finite(values)), all(values >= low), all(values <= high), !is.unsorted(values))
  })
}
for (sampler in c("random", "tpe", "gp_ei")) {
  low <- 1e15; high <- low+2
  opt <- n4m_optimizer(list(x = n4m_param_int(low, high, log = TRUE), aux = n4m_param_float(0,1)), sampler = sampler, seed = 19)
  if (sampler == "gp_ei") {
    expect_error(n4m_optimizer_enqueue(opt, c(x = low, aux = 0.25)), "optimizer_enqueue")
  } else {
    for (value in c(low, high, low+1)) n4m_optimizer_enqueue(opt, c(x = value, aux = 0.25))
    for (value in c(low, high, low+1)) {
      t <- n4m_optimizer_ask(opt); stopifnot(t$parameters$x == value); n4m_optimizer_tell(opt, t, 0.5)
    }
  }
  n4m_optimizer_close(opt)
}
for (sampler in c("random", "tpe", "gp_ei")) {
  low <- 1e300; high <- 1e300 * (1 + 1e-14); middle <- low + (high-low)/2
  stopifnot(middle > low, middle < high)
  opt <- n4m_optimizer(list(x = n4m_param_float(low, high, log = TRUE), aux = n4m_param_float(0,1)), sampler = sampler, seed = 19)
  if (sampler == "gp_ei") {
    expect_error(n4m_optimizer_enqueue(opt, c(x = low, aux = 0.25)), "optimizer_enqueue")
  } else {
    for (value in c(low, high, middle)) n4m_optimizer_enqueue(opt, c(x = value, aux = 0.25))
    for (value in c(low, high, middle)) {
      trial <- n4m_optimizer_ask(opt)
      stopifnot(identical(float_bits(trial$parameters$x), float_bits(value)))
      n4m_optimizer_tell(opt, trial, 0.5)
    }
  }
  n4m_optimizer_close(opt)
}
# RANDOM shares the scalar draw stream only within the non-phantom-cell range.
oracle_low <- 2^52; oracle_high <- oracle_low + 1; oracle_n <- 6000L
scalar <- n4m_optimizer(list(x = n4m_param_int(oracle_low, oracle_high)), sampler = "random", seed = 19)
scalar_values <- numeric(oracle_n)
for (i in seq_len(oracle_n)) {
  trial <- n4m_optimizer_ask(scalar); scalar_values[i] <- trial$parameters$x
  n4m_optimizer_tell(scalar, trial, 0)
}
n4m_optimizer_close(scalar)
for (len in c(1L, 3L)) {
  tuple <- n4m_optimizer(list(x = n4m_param_sorted_tuple(len, oracle_low, oracle_high, integer = TRUE)), sampler = "random", seed = 19)
  upper <- 0L
  for (i in seq_len(oracle_n / len)) {
    trial <- n4m_optimizer_ask(tuple)
    expected <- sort(scalar_values[((i-1)*len+1):(i*len)])
    stopifnot(identical(float_bits(trial$parameters$x), float_bits(expected)))
    if (len == 1 && trial$parameters$x == oracle_high) upper <- upper + 1L
    n4m_optimizer_tell(tuple, trial, 0)
  }
  if (len == 1) stopifnot(abs(upper/oracle_n - 0.5) <= 0.03)
  n4m_optimizer_close(tuple)
}
# Rechecksummed GP proposal mutations remain corrupt after the inverse fix.
fnv_refresh <- function(blob) {
  hi <- 3421674724; lo <- 2216829733; base <- 2^32
  for (byte in as.integer(blob[seq_len(length(blob)-8L)])) {
    lo <- lo - (lo %% 256) + bitwXor(as.integer(lo %% 256), byte)
    product <- lo * 435
    carry <- floor(product / base)
    next_lo <- product %% base
    hi <- (hi * 435 + carry + (lo * 256) %% base) %% base
    lo <- next_lo
  }
  encode <- function(value) as.raw(sapply(0:3, function(i) floor(value / 256^i) %% 256))
  blob[(length(blob)-7L):length(blob)] <- c(encode(lo), encode(hi))
  blob
}
raw_hits <- function(blob, pattern) {
  which(vapply(seq_len(length(blob)-length(pattern)-7L), function(i)
    identical(blob[i:(i+length(pattern)-1L)], pattern), logical(1)))
}
gp_space <- list(x = n4m_param_float(0,1))
gp <- n4m_optimizer(gp_space, sampler = "gp_ei", seed = 19)
gp_trial <- n4m_optimizer_ask(gp)
gp_checkpoint <- n4m_optimizer_save(gp)
gp_pattern <- writeBin(as.double(gp_trial$parameters$x), raw(), size = 8, endian = "little")
gp_hits <- raw_hits(gp_checkpoint$blob, gp_pattern)
stopifnot(length(gp_hits) == 2)
for (bad in c(NaN, -0.1, 1.5)) {
  damaged <- gp_checkpoint
  damaged$blob[gp_hits[2]:(gp_hits[2]+7L)] <- writeBin(as.double(bad), raw(), size = 8, endian = "little")
  damaged$blob <- fnv_refresh(damaged$blob)
  expect_error(n4m_optimizer_load(damaged), "optimizer_load")
}
n4m_optimizer_close(gp)
gp_space <- list(x = n4m_param_float(0.1, 0.1+2^-56, log = TRUE))
gp <- n4m_optimizer(gp_space, sampler = "gp_ei", seed = 19)
marker <- as.raw(c(255,255,255,255,255,255,239,63))
for (count in seq_len(3)) {
  trial <- n4m_optimizer_ask(gp); n4m_optimizer_tell(gp, trial, 0.5)
  checkpoint <- n4m_optimizer_save(gp)
  stopifnot(length(raw_hits(checkpoint$blob, marker)) == count)
  restored <- n4m_optimizer_load(checkpoint); n4m_optimizer_close(restored)
}
n4m_optimizer_close(gp)

# Python bigint oracle applied to seed-19 RANDOM raw-unit values.
h <- 2^53
expected <- c(h,h-2,h-1,h-2,h,h,h-1,h-1,h-2,h-1,h,h,h,h,h,h)
for (kind in c("int", "float")) {
  axis <- if (kind == "int") n4m_param_int(h-2,h,1) else n4m_param_float(h-2,h,1)
  opt <- n4m_optimizer(list(x=axis), sampler="random", seed=19)
  counts <- integer(3)
  for (i in seq_len(6000)) {
    trial <- n4m_optimizer_ask(opt); x <- trial$parameters$x
    stopifnot(x >= h-2, x <= h)
    if (i <= length(expected)) stopifnot(identical(float_bits(x), float_bits(expected[i])))
    counts[x-(h-2)+1] <- counts[x-(h-2)+1] + 1L
    n4m_optimizer_tell(opt, trial, 0)
  }
  stopifnot(identical(counts, c(2015L,2005L,1980L)))
  n4m_optimizer_close(opt)
}
near_ulp <- list(c(1-2^-52,1,2^-53,3), c(0.5-2^-53,0.5,2^-54,3),
                 c(2^60-256,2^60,128,3), c(1-3*2^-53,1,2^-53,4),
                 c(2^52,2^52+4,1.5,3))
for (domain in near_ulp) {
  opt <- n4m_optimizer(list(x=n4m_param_float(domain[1],domain[2],domain[3])), sampler="random",seed=19)
  values <- numeric(6000)
  for (i in seq_along(values)) {
    trial <- n4m_optimizer_ask(opt); values[i] <- trial$parameters$x
    stopifnot(values[i]>=domain[1], values[i]<=domain[2])
    n4m_optimizer_tell(opt,trial,0)
  }
  keys <- vapply(values, function(x) paste(as.character(float_bits(x)), collapse=""), character(1))
  counts <- table(keys)
  stopifnot(length(counts)==domain[4], all(abs(as.numeric(counts)/6000-1/domain[4])<=0.03))
  n4m_optimizer_close(opt)
}
for (domain in list(c(h-3,h,2),c(-h,h,3))) for (kind in c("int","float"))
  for (sampler in c("random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei")) {
    axis <- if(kind=="int") n4m_param_int(domain[1],domain[2],domain[3]) else n4m_param_float(domain[1],domain[2],domain[3])
    space <- list(x=axis, aux=n4m_param_float(0,1))
    domain_stages(space,sampler,19,function(trial) {
      x <- trial$parameters$x
      stopifnot(x>=domain[1], x<=domain[2], x%%domain[3]==domain[1]%%domain[3])
    })
  }

# Python bigint oracle for a wide step-3 integer grid.
wide_expected <- c(4239843483439303,-6906221722996298,-2457079771300949,-7517534437616789,5116916499969259,8257295123674171,-417271150248371,1254295467694255,-4530896056998680,452284686773314,7284296237484454,7676065770718294,5637129696437785,6717941993400823,5031873289515835,4454328880214974)
for (kind in c("int","float")) {
  axis <- if(kind=="int") n4m_param_int(-h,h,3) else n4m_param_float(-h,h,3)
  opt <- n4m_optimizer(list(x=axis),sampler="random",seed=19)
  for (x in wide_expected) { trial <- n4m_optimizer_ask(opt); stopifnot(identical(float_bits(trial$parameters$x),float_bits(x))); n4m_optimizer_tell(opt,trial,0) }
  n4m_optimizer_close(opt)
}
for (d in list(list(-h,h,3,c(1,7,13,h-1),numeric()),
               list(2^52,2^52+16,8,c(2^52+8),c(2^52+1,2^52+7,2^52+9)),
               list(h-3,h,2,numeric(),c(h)))) {
  opt <- n4m_optimizer(list(x=n4m_param_float(d[[1]],d[[2]],d[[3]])),sampler="random",seed=19)
  for (x in d[[5]]) expect_error(n4m_optimizer_enqueue(opt,c(x=x)),"optimizer_enqueue")
  for (x in d[[4]]) {n4m_optimizer_enqueue(opt,c(x=x)); trial <- n4m_optimizer_ask(opt)
    stopifnot(identical(float_bits(trial$parameters$x),float_bits(x))); n4m_optimizer_tell(opt,trial,0)}
  n4m_optimizer_close(opt)
}
opt <- n4m_optimizer(list(x=n4m_param_int(h-3,h,2)),sampler="random",seed=19)
expect_error(n4m_optimizer_enqueue(opt,c(x=h)),"optimizer_enqueue")
n4m_optimizer_close(opt)

# All required integer-grid domains; vectors use exact Python integer arithmetic.
for (d in list(
  list(9007199254740990,9007199254740992,1,c(9007199254740992,9007199254740990,9007199254740991,9007199254740990,9007199254740992,9007199254740992,9007199254740991,9007199254740991,9007199254740990,9007199254740991,9007199254740992,9007199254740992,9007199254740992,9007199254740992,9007199254740992,9007199254740992)),
  list(9007199254740989,9007199254740992,2,c(9007199254740991,9007199254740989,9007199254740989,9007199254740989,9007199254740991,9007199254740991,9007199254740989,9007199254740991,9007199254740989,9007199254740991,9007199254740991,9007199254740991,9007199254740991,9007199254740991,9007199254740991,9007199254740991)),
  list(9007199254740987,9007199254740992,3,c(9007199254740990,9007199254740987,9007199254740987,9007199254740987,9007199254740990,9007199254740990,9007199254740987,9007199254740990,9007199254740987,9007199254740990,9007199254740990,9007199254740990,9007199254740990,9007199254740990,9007199254740990,9007199254740990)),
  list(4503599627370496,4503599627370503,8,c(4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496,4503599627370496)),
  list(562949953421312,562949953421315,4,c(562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312,562949953421312)),
  list(281474976710656,281474976710659,4,c(281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656,281474976710656)),
  list(-9007199254740992,9007199254740992,1,c(4239843483439302,-6906221722996298,-2457079771300950,-7517534437616788,5116916499969258,8257295123674170,-417271150248372,1254295467694256,-4530896056998678,452284686773314,7284296237484452,7676065770718292,5637129696437782,6717941993400820,5031873289515832,4454328880214974)),
  list(-9007199254740992,9007199254740992,3,c(4239843483439303,-6906221722996298,-2457079771300949,-7517534437616789,5116916499969259,8257295123674171,-417271150248371,1254295467694255,-4530896056998680,452284686773314,7284296237484454,7676065770718294,5637129696437785,6717941993400823,5031873289515835,4454328880214974)),
  list(-9007199254740992,9007199254740992,7,c(4239843483439301,-6906221722996299,-2457079771300949,-7517534437616788,5116916499969263,8257295123674173,-417271150248369,1254295467694258,-4530896056998679,452284686773314,7284296237484458,7676065770718299,5637129696437781,6717941993400823,5031873289515831,4454328880214979)),
  list(-1,9007199254740992,1,c(6623521369090147,1050488765872346,3275059741720020,744832408562101,7062057877355125,8632247189207581,4294964052246309,5130747361217624,2238151598871156,4729741970757153,8145747746112722,8341632512729642,7322164475589387,7862570624070906,7019536272128412,6730764067477983)),
  list(-3,9007199254740992,1,c(6623521369090146,1050488765872344,3275059741720019,744832408562099,7062057877355125,8632247189207581,4294964052246308,5130747361217623,2238151598871154,4729741970757152,8145747746112722,8341632512729642,7322164475589387,7862570624070906,7019536272128412,6730764067477982)),
  list(0,9007199254740992,1,c(6623521369090147,1050488765872347,3275059741720021,744832408562102,7062057877355125,8632247189207581,4294964052246310,5130747361217624,2238151598871157,4729741970757153,8145747746112722,8341632512729642,7322164475589387,7862570624070906,7019536272128412,6730764067477983)),
  list(0,9007199254740990,1,c(6623521369090146,1050488765872346,3275059741720020,744832408562101,7062057877355124,8632247189207580,4294964052246309,5130747361217623,2238151598871156,4729741970757152,8145747746112721,8341632512729641,7322164475589386,7862570624070905,7019536272128411,6730764067477982)),
  list(-9007199254740992,9007199254740992,5,c(4239843483439303,-6906221722996302,-2457079771300952,-7517534437616792,5116916499969258,8257295123674168,-417271150248372,1254295467694253,-4530896056998682,452284686773313,7284296237484453,7676065770718293,5637129696437783,6717941993400818,5031873289515833,4454328880214973)),
  list(-9007199254740992,9007199254740992,9,c(4239843483439309,-6906221722996304,-2457079771300952,-7517534437616792,5116916499969262,8257295123674177,-417271150248368,1254295467694255,-4530896056998680,452284686773311,7284296237484460,7676065770718297,5637129696437782,6717941993400826,5031873289515838,4454328880214977)),
  list(-9007199254740992,9007199254740992,11,c(4239843483439303,-6906221722996307,-2457079771300954,-7517534437616794,5116916499969255,8257295123674171,-417271150248378,1254295467694250,-4530896056998682,452284686773307,7284296237484456,7676065770718293,5637129696437781,6717941993400817,5031873289515834,4454328880214975))
)) for (kind in c("int","float")) {
  axis <- if(kind=="int") n4m_param_int(d[[1]],d[[2]],d[[3]]) else n4m_param_float(d[[1]],d[[2]],d[[3]])
  opt <- n4m_optimizer(list(x=axis),sampler="random",seed=19)
  for (x in d[[4]]) {trial <- n4m_optimizer_ask(opt);stopifnot(identical(float_bits(trial$parameters$x),float_bits(x)));n4m_optimizer_tell(opt,trial,0)}
  n4m_optimizer_close(opt)
}

# T4f: adaptive bounds, own enqueue and independently generated rational vectors.
t4f_r1 <- list(c(-1,1-2^-52,3*2^-53,400),c(-0.5,0.5-2^-53,1.5*2^-53,400))
t4f_r2 <- list(c(1.3094115832905704,9.007642868376131e21,3.020767460720702,240),c(1.3552258541085935,3.1816648682723785e21,9.932851196532727,240))
t4f_r4 <- list(c(-1,1,3*2^-53),c(-2^43,2^43,3/1024),c(-2^54,2^54,6))
t4f_r3 <- c(t4f_r4,list(c(-1,1,1.25*2^-53),c(-1,1,1.5*2^-53)))
for(d in c(t4f_r1,t4f_r2)) for(sampler in c("cmaes","pso","ga","tpe")) {
 space <- list(x=n4m_param_float(d[1],d[2],d[3]),aux=n4m_param_float(0,1));o<-n4m_optimizer(space,sampler=sampler,seed=19,direction="minimize");outside<-0L
 for(i in seq_len(d[4])) {t<-n4m_optimizer_ask(o);x<-t$parameters$x;if(x<d[1]||x>d[2])outside<-outside+1L;n4m_optimizer_tell(o,t,-x)}
 stopifnot(outside==0L);copy<-n4m_optimizer_load(n4m_optimizer_save(o));n4m_optimizer_close(copy);n4m_optimizer_close(o)
}
for(d in t4f_r3) {space<-list(x=n4m_param_float(d[1],d[2],d[3]));a<-n4m_optimizer(space,sampler="random",seed=19);b<-n4m_optimizer(space,sampler="random",seed=3)
 for(i in seq_len(300)){t<-n4m_optimizer_ask(a);x<-t$parameters$x;n4m_optimizer_enqueue(b,c(x=x));u<-n4m_optimizer_ask(b);stopifnot(identical(float_bits(x),float_bits(u$parameters$x)));n4m_optimizer_tell(a,t,0);n4m_optimizer_tell(b,u,0)}
 n4m_optimizer_close(a);n4m_optimizer_close(b)
}
t4f_vectors <- list(
 c("3fde203af977a18e","bfe8892c0c8a0e4a","bfd17566ede808aa","bfeab52861bdac95","3fe22dcef65374eb","3fed55f756a6f43b","bfa7b8182af81b30","3fc1d319af2e62bc","bfe018d371fddb18","3fa9b59bc17ac420","3fe9e10761ac89a6","3feb455755894456","3fe406f0891be619","3fe7ddeec22369f7","3fe1e0764bf9f33b","3fdfa660853c677c"),
 c("428e203af977a18e","c298892c0c8a0e4a","c2817566ede808aa","c29ab52861bdac95","42922dcef65374eb","429d55f756a6f43b","c257b8182af81b30","4271d319af2e62bc","c29018d371fddb18","4259b59bc17ac420","4299e10761ac89a6","429b455755894456","429406f0891be619","4297ddeec22369f7","4291e0764bf9f33b","428fa660853c677c"),
 c("433e203af977a18e","c348892c0c8a0e4a","c3317566ede808aa","c34ab52861bdac95","43422dcef65374eb","434d55f756a6f43b","c307b8182af81b30","4321d319af2e62bc","c34018d371fddb18","4309b59bc17ac420","4349e10761ac89a6","434b455755894456","434406f0891be619","4347ddeec22369f7","4341e0764bf9f33b","433fa660853c677c")
)
for(j in seq_along(t4f_r4)){d<-t4f_r4[[j]];o<-n4m_optimizer(list(x=n4m_param_float(d[1],d[2],d[3])),sampler="random",seed=19)
 for(expected in t4f_vectors[[j]]){t<-n4m_optimizer_ask(o);actual<-paste(as.character(float_bits(t$parameters$x)),collapse="");stopifnot(identical(actual,expected));n4m_optimizer_tell(o,t,0)};n4m_optimizer_close(o)}
for(d in c(t4f_r1,t4f_r3)) for(sampler in c("random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei")) {
 space<-list(x=n4m_param_float(d[1],d[2],d[3]),aux=n4m_param_float(0,1));domain_stages(space,sampler,19,function(t)stopifnot(t$parameters$x>=d[1],t$parameters$x<=d[2]))
}
for(d in c(t4f_r1,t4f_r3)) for(sampler in c("random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei")) {
 space<-list(x=n4m_param_float(d[1],d[2],d[3]),aux=n4m_param_float(0,1));o<-n4m_optimizer(space,sampler=sampler,seed=19)
 if(sampler %in% c("ga","pso","cmaes","gp_ei")){expect_error(n4m_optimizer_enqueue(o,c(x=d[1],aux=.25)),"optimizer_enqueue");n4m_optimizer_close(o);next}
 n4m_optimizer_enqueue(o,c(x=d[1],aux=.25));copy<-n4m_optimizer_load(n4m_optimizer_save(o));stopifnot(identical(copy$space,o$space),identical(copy$constraints,o$constraints))
 for(i in seq_len(6)){a<-n4m_optimizer_ask(o);b<-n4m_optimizer_ask(copy);stopifnot(identical(a$id,b$id),identical(float_bits(a$parameters$x),float_bits(b$parameters$x)),identical(float_bits(a$parameters$aux),float_bits(b$parameters$aux)));if(i==1)stopifnot(identical(float_bits(a$parameters$x),float_bits(d[1])));n4m_optimizer_tell(o,a,.5);n4m_optimizer_tell(copy,b,.5)}
 n4m_optimizer_close(copy);n4m_optimizer_close(o)
}

# Genuine name-and-rank targeted checkpoint mutations, prepared from public checkpoints.
loader_grid_bad <- c(
  "4e344d4f50540d0a010000002000000050010000000000002201000000000000fefd47e44137aed63c000000000000000100000001000000780000000000000000000000000000000000002440000000000000084001000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000db5912a55f50163f010000000000000002000000000000000100000000000000000000000100000001000000780000000000001c40ffffffff000000000101000000000000000000d03f01f44d02eab9c4f23e00000000000000000000000000000100000000000000f87c41c9ec45f73e00000000000000000000000000000000000000000000000000000079f206fca54cbe86",
  "4e344d4f50540d0a010000002000000050010000000000002201000000000000fefd47e44137aed63c000000000000000100000001000000780000000000000000000000000000000000002440000000000000084001000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000049d81abe4359f13e010000000000000001000000000000000100000000000000000000000100000001000000780000000000001c40ffffffff00000000010000000000000000000000000000000000000000000000000000000000000000000000ffffffffffffffff2dd5cd116e3cdc3e00000000000000000000000000000000000000000000000000000055db2ba5873406ea",
  "4e344d4f50540d0a0100000020000000500100000000000022010000000000007279c6e12b96c67b3c00000000000000010000000100000078010000000000000000000000000000000000f03f9a9999999999b93f00000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000086c613419c87f33e01000000000000000200000000000000010000000000000000000000010000000100000078333333333333d33fffffffff000000000101000000000000000000d03f0155a485752ab3dd3e00000000000000000000000000000100000000000000f008a9065ab4e23e0000000000000000000000000000000000000000000000000000008453efa0f3c528c4",
  "4e344d4f50540d0a01000000200000005001000000000000220100000000000086412335e4f789433c0000000000000001000000010000007801000000000000000000f0bf000000000000f03f000000000000b83c00000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000060062128d08aec3e01000000000000000200000000000000010000000000000000000000010000000100000078ffffffffffffefbfffffffff000000000101000000000000000000d03f01aff75f0ab6e1d53e0000000000000000000000000000010000000000000068c24a93bf6cdc3e000000000000000000000000000000000000000000000000000000b80d6507cf98d4c0",
  "4e344d4f50540d0a0100000020000000e001000000000000b1010000000000009e900a20946c3bd2af0000000000000002000000040000006d6f64650400000000000000000000000000000000000000000000000000000000000000000002000000030000006f6666020000006f6e020000000000000000000000000000000000f03f0000000000010000007801000000000000000000f03f0000000000001440000000000000004000000000000000000000000000000000000000010000000300000002000000010000007800000000040000006d6f6465020000006f6e0d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000ddfd5cfab8b2fb3e0100000000000000020000000000000001000000000000000000000002000000040000006d6f6465000000000000000000000000030000006f66660101000000780000000000001040ffffffff000000000001000000000000000000d03f0101153dd995e2df3e000000000000000000000000000001000000000000003f384ef356a9e53e000000000000000000000000000000000000000000000000000000006c2ea3ac43e1c8c2",
  "4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000bd0d7ae11a381dfd",
  "4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90"
)
loader_grid_spaces <- list(
 list(x=n4m_param_int(0,10,3)), list(x=n4m_param_int(0,10,3)),
 list(x=n4m_param_float(0,1,0.1)), list(x=n4m_param_float(-1,1,3*2^-53)),
 list(mode=n4m_param_categorical(c("off","on")),x=n4m_param_float(1,5,2)),
 list(x=n4m_param_float(0,4,2)), list(x=n4m_param_float(0,4,2)))
for(i in seq_along(loader_grid_bad)) {
  conditions <- if(i==5) list(n4m_constraint("condition_in",c("x","mode"),c("","on"))) else list()
  expect_error(n4m_optimizer_load(list(blob=hex_raw(loader_grid_bad[i]),space=loader_grid_spaces[[i]],constraints=conditions)),"optimizer_load")
}
loader_grid_good <- list(blob=hex_raw("4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000040ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90"),space=loader_grid_spaces[[6]],constraints=list())
a <- n4m_optimizer_load(loader_grid_good); b <- n4m_optimizer_load(loader_grid_good)
stopifnot(identical(a$space,b$space),identical(a$constraints,b$constraints))
for(i in seq_len(6)){x<-n4m_optimizer_ask(a);y<-n4m_optimizer_ask(b);stopifnot(identical(x$id,y$id),identical(float_bits(x$parameters$x),float_bits(y$parameters$x)));n4m_optimizer_tell(a,x,.5);n4m_optimizer_tell(b,y,.5)}
n4m_optimizer_close(a);n4m_optimizer_close(b)
