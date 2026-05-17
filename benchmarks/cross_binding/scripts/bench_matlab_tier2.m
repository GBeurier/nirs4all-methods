% MATLAB / Octave tier-2: pls4all.fit("pls", ...) — idiomatic
% classdef interface around the legacy pls_fit MEX.
csv_path = getenv("BENCH_CSV");
nc = str2double(getenv("BENCH_NCOMP"));
runs = str2double(getenv("BENCH_NRUNS"));

mat = csvread(csv_path, 1, 0);
X = mat(:, 1:end-1);
y = mat(:, end);

times = zeros(runs, 1);
for i = 1:runs
    t0 = tic();
    mdl = pls4all.fit("pls", X, y, "NumComponents", nc);
    p = predict(mdl, X);
    times(i) = toc(t0) * 1000.0;
end
if runs >= 3
    times = times(2:end);
end
out = sprintf('{"ok":true,"n_runs":%d,"median_ms":%g,"min_ms":%g,"max_ms":%g}', ...
              length(times), median(times), min(times), max(times));
disp(out)
