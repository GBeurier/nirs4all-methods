% MATLAB / Octave tier-1: pls4all.sparse_simpls @ lambda=0 (pure SIMPLS
% via the dispatcher MEX).
csv_path = getenv("BENCH_CSV");
nc = str2double(getenv("BENCH_NCOMP"));
runs = str2double(getenv("BENCH_NRUNS"));

mat = csvread(csv_path, 1, 0);  % skip header row
X = mat(:, 1:end-1);
y = mat(:, end);

times = zeros(runs, 1);
for i = 1:runs
    t0 = tic();
    res = pls4all.sparse_simpls(X, y, nc, 0.0);
    p = res.predictions;
    times(i) = toc(t0) * 1000.0;
end
if runs >= 3
    times = times(2:end);
end
out = sprintf('{"ok":true,"n_runs":%d,"median_ms":%g,"min_ms":%g,"max_ms":%g}', ...
              length(times), median(times), min(times), max(times));
disp(out)
