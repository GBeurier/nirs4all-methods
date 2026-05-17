# R tier-1: .Call r_p4a_sparse_simpls_fit @ lambda=0 (== pure SIMPLS).
suppressMessages(library(pls4all))
suppressMessages(library(jsonlite))

argv <- commandArgs(trailingOnly = TRUE)
csv_path <- argv[[1]]
nc <- as.integer(argv[[2]])
runs <- as.integer(argv[[3]])

mat <- read.csv(csv_path)
X <- as.matrix(mat[, -ncol(mat)])
y <- as.matrix(mat[, ncol(mat)])

times <- numeric(runs)
for (i in seq_len(runs)) {
  t0 <- proc.time()[["elapsed"]]
  res <- sparse_simpls_fit(X, y, n_components = nc, sparsity_lambda = 0.0)
  invisible(res$predictions)
  times[i] <- (proc.time()[["elapsed"]] - t0) * 1000.0
}
if (runs >= 3L) times <- times[-1L]
out <- list(ok = TRUE,
            n_runs = length(times),
            median_ms = median(times),
            min_ms = min(times),
            max_ms = max(times))
cat(toJSON(out, auto_unbox = TRUE), "\n", sep = "")
