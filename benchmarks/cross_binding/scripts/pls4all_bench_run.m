function [stats, last_preds] = pls4all_bench_run(fit_predict_seeded, runs, seed_base)
% Adaptive benchmark helper. One timed warm-up is used to classify very
% slow cells; otherwise a timed probe chooses the final sample count.
max_runs = max(1, floor(runs));
% Number of warm-up executions discarded before the first timed probe.
% A single warm-up does not amortise Octave classdef + MEX JIT and BLAS
% thread-pool spin-up, which decay over the first few calls.
warmup_runs = str2double(getenv("BENCH_WARMUP_RUNS"));
if isnan(warmup_runs) || warmup_runs < 1
    warmup_runs = 3;
end

    function [target, statistic, decision] = adaptive_target(probe_ms)
        if probe_ms > 60000.0
            target = 2; statistic = 'single'; decision = 'probe_gt_60s';
        elseif probe_ms > 30000.0
            target = min(max_runs, 2); statistic = 'single'; decision = 'probe_gt_30s';
        elseif probe_ms > 5000.0
            target = min(max_runs, 3); statistic = 'mean'; decision = 'probe_gt_5s';
        elseif probe_ms > 1000.0
            target = min(max_runs, 10); statistic = 'median'; decision = 'probe_gt_1s';
        elseif probe_ms > 100.0
            target = min(max_runs, 20); statistic = 'median'; decision = 'probe_gt_100ms';
        else
            target = min(max_runs, 40); statistic = 'median'; decision = 'probe_le_100ms';
        end
    end

    function m = bench_median(v)
        v = sort(v(:));
        n = length(v);
        mid = floor((n + 1) / 2);
        if mod(n, 2) == 1
            m = v(mid);
        else
            m = (v(mid) + v(mid + 1)) / 2.0;
        end
    end

    function m = bench_mean(v)
        v = v(:);
        m = sum(v) / length(v);
    end

    function s = make_stats(samples_, statistic_, warmup_ms_, decision_, total_runs_, warmup_included_)
        if length(samples_) == 1
            statistic_ = 'single';
        end
        if strcmp(statistic_, 'mean')
            reported = bench_mean(samples_);
        else
            reported = bench_median(samples_);
        end
        s = struct();
        s.ok = true;
        s.n_runs = length(samples_);
        s.median_ms = reported;
        s.reported_ms = reported;
        s.sample_median_ms = bench_median(samples_);
        s.mean_ms = bench_mean(samples_);
        s.min_ms = min(samples_);
        s.max_ms = max(samples_);
        s.warmup_ms = warmup_ms_;
        s.warmup_included = warmup_included_;
        s.timing_statistic = statistic_;
        s.timing_decision = decision_;
        s.max_runs = max_runs;
        s.total_runs = total_runs_;
    end

% Discard `warmup_runs` executions on seed_base so the timed probe is warm.
warmup_preds = [];
warmup_ms = 0.0;
for w = 1:warmup_runs
    t0 = tic();
    warmup_preds = fit_predict_seeded(seed_base);
    warmup_ms = toc(t0) * 1000.0;
end
if warmup_ms > 300000.0
    samples = warmup_ms;
    stats = make_stats(samples, 'single', warmup_ms, 'warmup_gt_5min', 1, true);
    last_preds = warmup_preds;
    return;
end
if max_runs < 2
    samples = warmup_ms;
    stats = make_stats(samples, 'single', warmup_ms, 'max_runs_1_warmup_only', 1, true);
    last_preds = warmup_preds;
    return;
end

t0 = tic();
last_preds = fit_predict_seeded(seed_base);
probe_ms = toc(t0) * 1000.0;
samples = probe_ms;

[target, statistic, decision] = adaptive_target(probe_ms);
target_samples = max(1, target - 1);
for i = 2:target_samples
    t0 = tic();
    last_preds = fit_predict_seeded(seed_base + (i - 1));
    samples(end + 1, 1) = toc(t0) * 1000.0;
end
stats = make_stats(samples, statistic, warmup_ms, decision, warmup_runs + length(samples), false);
end
