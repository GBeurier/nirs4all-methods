function [stats, last_preds] = pls4all_bench_run(fit_predict_seeded, runs, seed_base)
% Helper: run one unmeasured warmup, then `fit_predict_seeded(seed)` for
% seed in [base..base+runs-1]. Return timed stats + last predictions vector.
if runs < 1
    error('runs must be >= 1');
end
fit_predict_seeded(seed_base);
samples = zeros(runs, 1);
last_preds = [];
for i = 1:runs
    t0 = tic();
    last_preds = fit_predict_seeded(seed_base + (i - 1));
    samples(i) = toc(t0) * 1000.0;
end
stats = struct();
stats.ok = true;
stats.n_runs = length(samples);
stats.median_ms = median(samples);
stats.min_ms = min(samples);
stats.max_ms = max(samples);
end
