# SPDX-License-Identifier: CECILL-2.1
# Continue a Python N4MOPT checkpoint in R, then produce the reciprocal one.
suppressPackageStartupMessages(library(n4m))
directory <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(length(directory) == 1L, dir.exists(directory))
space <- list(x = n4m_param_int(1, 5), a = n4m_param_float(0.1, 0.9))
expected <- scan(file.path(directory, "expected.txt"), quiet = TRUE)

python_path <- file.path(directory, "python.n4mopt")
python_blob <- readBin(python_path, "raw", n = file.info(python_path)$size)
from_python <- n4m_optimizer_load(list(blob = python_blob, space = space,
                                       constraints = list()))
next_trial <- n4m_optimizer_ask(from_python)
actual <- c(next_trial$id, next_trial$parameters$x, next_trial$parameters$a)
stopifnot(identical(actual, expected))
n4m_optimizer_close(from_python)

from_r <- n4m_optimizer(space, seed = 42)
for (i in seq_len(3)) {
  trial <- n4m_optimizer_ask(from_r)
  n4m_optimizer_tell(from_r, trial, trial$parameters$x + trial$parameters$a)
}
writeBin(n4m_optimizer_save(from_r)$blob, file.path(directory, "r.n4mopt"))
next_trial <- n4m_optimizer_ask(from_r)
stopifnot(identical(c(next_trial$id, next_trial$parameters$x,
                      next_trial$parameters$a), expected))
cat("R resumed Python N4MOPT checkpoint exactly\n")
