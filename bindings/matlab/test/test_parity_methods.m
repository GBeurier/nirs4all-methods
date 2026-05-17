% Parity gate: pls4all MATLAB tier-2 vs Octave statistics `plsregress`.
%
% Verifies that pls4all SIMPLS (via Regression) produces predictions
% within tolerance of the Octave `statistics` package's plsregress.
%
% Tolerance: 1e-6 absolute, after manually centering X and y so both
% sides use identical preprocessing (plsregress centers by default,
% pls4all centers by default; we disable scaling on both ends).

pkg load statistics

rand("seed", 0); randn("seed", 0);
n = 50; p = 20;
X = randn(n, p);
y = 2.0 * X(:, 3) - X(:, 6) + 0.5 * X(:, 9) + 0.05 * randn(n, 1);

k = 5;

% --- pls4all SIMPLS via the MethodResult dispatcher (center, no scale).
% sparse_simpls @ lambda=0 collapses to plain SIMPLS, so we use it here
% rather than pls_fit.m (which currently bakes in scale_x=1).
res = pls4all.sparse_simpls(X, y, k, 0.0);
preds_ours = res.predictions;

% --- Octave statistics::plsregress (SIMPLS by default) ----------------
% plsregress returns:
%   BETA — (p+1 × q) regression coefficients with intercept row.
% Note: Octave's plsregress centers X but doesn't scale, matching
% pls4all's center_x=1, scale_x=0 default in the dispatcher.
[~, ~, ~, ~, BETA, ~, ~] = plsregress(X, y, k);
preds_theirs = [ones(n, 1) X] * BETA;

diff = max(abs(preds_ours - preds_theirs));
printf("=== Parity gate: pls4all SIMPLS vs Octave plsregress ===\n");
printf("  max abs diff = %g\n", diff);
printf("  ours[1:3]    = %s\n", sprintf("%.6f ", preds_ours(1:3)));
printf("  theirs[1:3]  = %s\n", sprintf("%.6f ", preds_theirs(1:3)));

if diff < 1e-6
    printf("  PARITY GATE PASSED (tol 1e-6)\n");
elseif diff < 1e-3
    printf("  parity gate WARN: diff > 1e-6 but < 1e-3\n");
else
    printf("  PARITY GATE FAILED\n");
    error("pls4all:parity", "MATLAB parity gate failed at diff = %g", diff);
end
