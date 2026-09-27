function model = GCU(rank, max_iter, tol)
% GCU Global nonnegative factors with a training-fitted fixed basis.
if nargin < 1, rank = 16; end
if nargin < 2, max_iter = 60; end
if nargin < 3, tol = 0.001; end
model = n4m.SpectralEncoder([1, 64, rank, 0, 1, 0, max_iter, tol]);
end
