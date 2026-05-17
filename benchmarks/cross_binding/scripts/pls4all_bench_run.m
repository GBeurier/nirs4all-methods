function [stats, last_preds] = pls4all_bench_run(fit_predict_seeded, runs, seed_base)
% Helper: run `fit_predict_seeded(seed)` for seed in [base..base+runs-1],
% return stats struct + last predictions vector (warmup discarded if runs>=3).
samples = zeros(runs, 1);
last_preds = [];
for i = 1:runs
    t0 = tic();
    last_preds = fit_predict_seeded(seed_base + (i - 1));
    samples(i) = toc(t0) * 1000.0;
end
if runs >= 3
    timed = samples(2:end);
else
    timed = samples;
end
stats = struct();
stats.ok = true;
stats.n_runs = length(timed);
stats.median_ms = median(timed);
stats.min_ms = min(timed);
stats.max_ms = max(timed);
end
