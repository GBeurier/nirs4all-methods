% External MATLAB / Octave: statistics::plsregress (SIMPLS).
pkg load statistics
csv_path = getenv("BENCH_CSV");
nc = str2double(getenv("BENCH_NCOMP"));
runs = str2double(getenv("BENCH_NRUNS"));

mat = csvread(csv_path, 1, 0);
X = mat(:, 1:end-1);
y = mat(:, end);
n = size(X, 1);

times = zeros(runs, 1);
for i = 1:runs
    t0 = tic();
    [~, ~, ~, ~, BETA, ~, ~] = plsregress(X, y, nc);
    p = [ones(n, 1) X] * BETA;
    times(i) = toc(t0) * 1000.0;
end
if runs >= 3
    times = times(2:end);
end
out = sprintf('{"ok":true,"n_runs":%d,"median_ms":%g,"min_ms":%g,"max_ms":%g}', ...
              length(times), median(times), min(times), max(times));
disp(out)
