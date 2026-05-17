function mdl = fit(algo, X, y, varargin)
% pls4all.fit  Unified factory for tier-2 PLS regressors.
%
%   mdl = pls4all.fit("sparse_simpls", X, y, "NumComponents", 5, ...
%                       "Lambda", 0.05)
%   mdl = pls4all.fit("cppls",         X, y, "NumComponents", 5, ...
%                       "Gamma", 0.7)
%   mdl = pls4all.fit("weighted_pls",  X, y, "NumComponents", 5, ...
%                       "Weights", w)
%   mdl = pls4all.fit("mb_pls",        X, y, "NumComponents", 3, ...
%                       "BlockSizes", [10 10])
%   mdl = pls4all.fit("pls_glm",       X, y, "NumComponents", 5, ...
%                       "Family", "gaussian")
%   mdl = pls4all.fit("mir_pls",       X, y, "NumComponents", 5)
%   mdl = pls4all.fit("pls",           X, y, "NumComponents", 5)
%
% NumComponents is required (no sensible default for this surface).
% Returns one of pls4all.SparsePlsRegression / CpplsRegression /
% WeightedPlsRegression / MbPlsRegression / GlmRegression /
% MirRegression / Regression, all of which implement
% `predict / loss / score`.
if isstring(algo), algo = char(algo); end

p = inputParser();
p.addParameter("NumComponents", 2, @(x) isscalar(x) && x > 0);
p.addParameter("Lambda",        0.05, @isscalar);
p.addParameter("Gamma",         0.5,  @isscalar);
p.addParameter("Weights",       [],   @(x) isnumeric(x) && isvector(x));
p.addParameter("BlockSizes",    [],   @(x) isnumeric(x) && isvector(x));
p.addParameter("Family",        "gaussian", ...
    @(x) any(strcmpi(string(x), ["gaussian", "poisson"])));
p.KeepUnmatched = false;
p.parse(varargin{:});
k = p.Results.NumComponents;

switch lower(algo)
    case {"pls", "pls_simpls", "simpls"}
        mdl = pls4all.Regression(X, y, k);
    case "sparse_simpls"
        mdl = pls4all.SparsePlsRegression(X, y, k, p.Results.Lambda);
    case "cppls"
        mdl = pls4all.CpplsRegression(X, y, k, p.Results.Gamma);
    case "weighted_pls"
        if isempty(p.Results.Weights)
            error("pls4all:nargin", "weighted_pls requires Weights");
        end
        mdl = pls4all.WeightedPlsRegression(X, y, k, p.Results.Weights);
    case "mb_pls"
        if isempty(p.Results.BlockSizes)
            error("pls4all:nargin", "mb_pls requires BlockSizes");
        end
        mdl = pls4all.MbPlsRegression(X, y, k, p.Results.BlockSizes);
    case "pls_glm"
        mdl = pls4all.GlmRegression(X, y, k, p.Results.Family);
    case "mir_pls"
        mdl = pls4all.MirRegression(X, y, k);
    otherwise
        error("pls4all:bad_arg", "unknown algorithm: %s", algo);
end
end
