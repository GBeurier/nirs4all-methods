function [X, y] = pls4all_bench_load_xy(csv_dir, n, p, seed)
% Load the orchestrator-cached CSV for (n, p, seed). All backends share
% the same file → cross-language parity comparison is meaningful.
csv = fullfile(csv_dir, sprintf("data_%dx%d_seed%d.csv", n, p, seed));
% csvread skips a header row by passing 1 for the starting row index.
arr = csvread(csv, 1, 0);
X = arr(:, 1:end-1);
y = arr(:, end);
end
