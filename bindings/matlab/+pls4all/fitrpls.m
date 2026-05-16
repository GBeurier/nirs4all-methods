function mdl = fitrpls(X, y, varargin)
% fitrpls — Train a PLS regression model.
%
%   MDL = pls4all.fitrpls(X, y) — fits a 2-component PLS regression.
%   MDL = pls4all.fitrpls(X, y, "NumComponents", K) — K components.
%
% Returns an instance of pls4all.Regression. Mirrors the calling
% convention of MATLAB Statistics Toolbox's fitrtree / fitrnet
% factories: positional (X, y) followed by Name-Value option pairs.
%
% Recognized Name-Value pairs:
%   "NumComponents"   positive integer (default 2)
%
% Example:
%
%     mdl  = pls4all.fitrpls(X, y, "NumComponents", 5);
%     yhat = predict(mdl, Xnew);
%
% See also: pls4all.Regression, predict, loss, score.

    p = inputParser;
    p.addRequired("X", @isnumeric);
    p.addRequired("y", @isnumeric);
    p.addParameter("NumComponents", 2, ...
        @(x) isnumeric(x) && isscalar(x) && x >= 1);
    p.parse(X, y, varargin{:});

    mdl = pls4all.Regression(X, y, p.Results.NumComponents);
end
