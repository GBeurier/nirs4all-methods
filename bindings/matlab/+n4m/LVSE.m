function model = LVSE(width, rank, overlap, standardize, snv)
% LVSE Local variance subspaces, fitted only on training spectra.
if nargin < 1, width = 64; end
if nargin < 2, rank = 4; end
if nargin < 3, overlap = 0; end
if nargin < 4, standardize = true; end
if nargin < 5, snv = false; end
model = n4m.SpectralEncoder([0, width, rank, overlap, standardize, snv, 60, 0.001]);
end
