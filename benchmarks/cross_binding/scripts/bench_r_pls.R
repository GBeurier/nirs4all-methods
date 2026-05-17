# External R: pls::plsr SIMPLS.
suppressMessages(library(pls))
suppressMessages(library(jsonlite))

argv <- commandArgs(trailingOnly = TRUE)
csv_path <- argv[[1]]
nc <- as.integer(argv[[2]])
runs <- as.integer(argv[[3]])

df <- read.csv(csv_path)

times <- numeric(runs)
for (i in seq_len(runs)) {
  t0 <- proc.time()[["elapsed"]]
  fit <- pls::plsr(y ~ ., data = df, ncomp = nc, method = "simpls",
                    validation = "none", scale = FALSE)
  invisible(predict(fit, newdata = df, ncomp = nc))
  times[i] <- (proc.time()[["elapsed"]] - t0) * 1000.0
}
if (runs >= 3L) times <- times[-1L]
out <- list(ok = TRUE,
            n_runs = length(times),
            median_ms = median(times),
            min_ms = min(times),
            max_ms = max(times))
cat(toJSON(out, auto_unbox = TRUE), "\n", sep = "")
