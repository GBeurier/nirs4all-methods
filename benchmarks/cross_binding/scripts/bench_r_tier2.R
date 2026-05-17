# R tier-2: formula+S3 `pls(y ~ ., data)` via pls4all.
suppressMessages(library(pls4all))
suppressMessages(library(jsonlite))

argv <- commandArgs(trailingOnly = TRUE)
csv_path <- argv[[1]]
nc <- as.integer(argv[[2]])
runs <- as.integer(argv[[3]])

df <- read.csv(csv_path)

times <- numeric(runs)
for (i in seq_len(runs)) {
  t0 <- proc.time()[["elapsed"]]
  fit <- pls(y ~ ., data = df, ncomp = nc, algo = "pls_simpls")
  invisible(predict(fit, newdata = df))
  times[i] <- (proc.time()[["elapsed"]] - t0) * 1000.0
}
if (runs >= 3L) times <- times[-1L]
out <- list(ok = TRUE,
            n_runs = length(times),
            median_ms = median(times),
            min_ms = min(times),
            max_ms = max(times))
cat(toJSON(out, auto_unbox = TRUE), "\n", sep = "")
