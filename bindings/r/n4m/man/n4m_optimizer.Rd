\name{n4m_optimizer}
\alias{n4m_optimizer}
\alias{n4m_optimizer_close}
\alias{n4m_optimizer_ask}
\alias{n4m_optimizer_ask_batch}
\alias{n4m_optimizer_enqueue}
\alias{n4m_optimizer_tell}
\alias{n4m_optimizer_intermediate}
\alias{n4m_optimizer_best}
\alias{n4m_optimizer_save}
\alias{n4m_optimizer_load}
\alias{n4m_optimizer_trials}
\alias{n4m_param_int}
\alias{n4m_param_float}
\alias{n4m_param_categorical}
\alias{n4m_param_ordinal}
\alias{n4m_param_sorted_tuple}
\alias{n4m_constraint}
\title{Native hyperparameter optimization from R}
\description{
Build a typed search space and run the libn4m optimizer through its public C
ABI. Sampling, pruning, ordered trial events, and N4MOPT checkpoints are owned
by the native engine. The R binding marshals values and owns the native handle.
}
\usage{
n4m_param_int(low, high, step = 1, log = FALSE)
n4m_param_float(low, high, step = 0, log = FALSE)
n4m_param_categorical(choices, type = NULL)
n4m_param_ordinal(choices)
n4m_param_sorted_tuple(length, low, high, integer = FALSE)
n4m_constraint(kind, refs, labels = rep("", length(refs)))
n4m_optimizer(space, constraints = list(), sampler = "random", pruner = "none",
  direction = "auto", eval_mode = "mean", metric = "rmse", liar = "none",
  n_startup_trials = 10L, seed = 0, timeout_seconds = 0,
  max_resource = 0L, reduction_factor = 0L)
n4m_optimizer_close(optimizer)
n4m_optimizer_ask(optimizer)
n4m_optimizer_ask_batch(optimizer, n)
n4m_optimizer_enqueue(optimizer, parameters)
n4m_optimizer_tell(optimizer, trial, score = NULL, status = "completed", error = NULL)
n4m_optimizer_intermediate(optimizer, trial, step, score)
n4m_optimizer_best(optimizer)
n4m_optimizer_trials(optimizer, since_id = 0)
n4m_optimizer_save(optimizer)
n4m_optimizer_load(checkpoint)
}
\arguments{
\item{low, high}{Finite inclusive axis bounds. Integer axes use binary64-exact integers.}
\item{step}{Grid step; zero means continuous for floating axes.}
\item{log}{Whether native sampling uses a logarithmic axis.}
\item{choices}{Nonempty homogeneous categorical vector, or ordered finite numeric values.}
\item{type}{Optional native category type: \code{string}, \code{integer}, \code{float}, or \code{boolean}.}
\item{length}{Number of elements in a sorted tuple.}
\item{integer}{Whether tuple elements are integers.}
\item{kind}{Constraint kind: \code{mutex_group} (also \code{mutex}), \code{requires}, \code{exclude},
\code{condition_in}, or \code{condition_not_in}.}
\item{refs, labels}{Ordered parameter references and matching labels for a constraint.}
\item{space}{Nonempty named list of axis declarations. Declaration order is significant.}
\item{constraints}{List of constraint declarations.}
\item{sampler}{One of \code{random}, \code{sobol}, \code{lhs}, \code{ternary},
\code{ga}, \code{pso}, \code{cmaes}, \code{tpe}, \code{gp_ei}.}
\item{pruner}{One of \code{none}, \code{median}, \code{asha}, \code{hyperband}, \code{racing}.}
\item{direction}{\code{auto}, \code{minimize}, or \code{maximize}.}
\item{eval_mode}{Native evaluation mode. Only \code{mean} is currently implemented.}
\item{metric}{Native metric name: \code{rmse}, \code{mse}, \code{mae}, \code{r2},
\code{accuracy}, \code{balanced_accuracy}, \code{f1}, or \code{logloss}.}
\item{liar}{Native constant-liar policy. Only \code{none} is currently implemented.}
\item{n_startup_trials}{Startup trial count for adaptive sampling and median pruning.}
\item{seed}{Nonnegative binary64-exact native seed.}
\item{timeout_seconds}{Finite nonnegative wall-clock timeout, zero for none.}
\item{max_resource}{Positive resource limit for Hyperband, zero otherwise.}
\item{reduction_factor}{ASHA/Hyperband factor, zero for native default.}
\item{optimizer}{An \code{n4m_optimizer} handle.}
\item{n}{Requested batch size, zero through 10000.}
\item{parameters}{Named numeric warm-start values; categorical values are zero-based indices.}
\item{trial}{A returned trial record or its exact numeric ID.}
\item{score}{Finite score for a completed trial; \code{NULL} otherwise.}
\item{status}{Terminal trial status: \code{completed}, \code{pruned}, \code{failed}, or \code{cancelled}.}
\item{error}{Optional plain or versioned native error text for failed or cancelled trials.}
\item{since_id}{Inclusive nonnegative trial-ID filter.}
\item{checkpoint}{List returned by \code{n4m_optimizer_save()}.}
}
\value{
Axis and constraint constructors return plain R declarations. \code{n4m_optimizer()}
and \code{n4m_optimizer_load()} return native handles. \code{n4m_optimizer_close()}
releases one immediately; garbage collection also releases it. \code{n4m_optimizer_ask()}
returns a list containing an exact numeric \code{id}, typed \code{parameters},
native \code{status}, and \code{rung}. Inactive conditional axes have \code{NULL}
values. Batch asks return a list of committed trials; if a non-benign native
error occurs after commits, the list has a \code{native_status} attribute and
the caller must terminalize or recover those trials. \code{n4m_optimizer_best()}
returns the best completed trial and score. \code{n4m_optimizer_trials()}
returns the complete owning native trace v1 as a named list; int64 vectors
become character when a value exceeds binary64's exact integer range.
\code{n4m_optimizer_save()} returns a portable raw N4MOPT blob with the R axis
declarations needed to decode trials after \code{n4m_optimizer_load()}.
}
\details{
The native checkpoint contains the authoritative ordered search space and
optimizer state. Keep the accompanying R axis declarations unchanged: they
are used only to decode trial values because the C trial accessor does not
enumerate its original declarations. Host evaluation remains the caller's
responsibility. A binding does not make a Python-only estimator runnable in R.
For the trace key/shape contract, see \code{docs/methods/optimization.md} in
the source repository. Unsupported native options fail at creation.
}
\examples{
space <- list(components = n4m_param_int(1, 4))
opt <- n4m_optimizer(space, seed = 42)
trial <- n4m_optimizer_ask(opt)
n4m_optimizer_tell(opt, trial, (trial$parameters$components - 2)^2)
n4m_optimizer_best(opt)$score
restored <- n4m_optimizer_load(n4m_optimizer_save(opt))
n4m_optimizer_trials(restored)$n_trials
}
