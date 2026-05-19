function mdl = fit(algo, X, y, varargin)
% pls4all.fit  Unified factory for tier-2 PLS regressors.
%
% Supported algorithms (alphabetical):
%   "bagging_pls"    → BaggingPlsRegression
%   "boosting_pls"   → BoostingPlsRegression
%   "continuum_regression" → ContinuumRegression
%   "cppls"          → CpplsRegression
%   "ecr"            → EcrRegression
%   "kernel_pls"     → KernelPlsRegression (in-sample only)
%   "mb_pls"         → MbPlsRegression           (needs BlockSizes)
%   "mir_pls"        → MirRegression
%   "missing_aware_nipals" → MissingAwareNipalsRegression
%   "n_pls"          → NPlsRegression            (needs ModeJ, ModeK)
%   "o2pls"          → O2plsRegression
%   "opls"           → OplsRegression
%   "pcr"            → PcrRegression
%   "pls" | "pls_simpls" → Regression (SIMPLS)
%   "pls_glm"        → GlmRegression             (Family: gaussian/poisson)
%   "recursive_pls"  → RecursivePlsRegression (in-sample only)
%   "ridge_pls"      → RidgePlsRegression
%   "robust_pls"     → RobustPlsRegression
%   "sparse_simpls"  → SparsePlsRegression
%   "weighted_pls"   → WeightedPlsRegression     (needs Weights)
%
% Common Name-Value parameters:
%   NumComponents      (default 2)
%   Lambda             (sparsity_lambda / ridge_lambda — context-dependent)
%   Gamma              (CPPLS gamma; kernel_pls gamma)
%   Alpha              (ECR alpha)
%   Tau                (continuum regression tau)
%   HuberK / MaxIrlsIter (robust_pls)
%   WindowSize         (recursive_pls)
%   ModeJ / ModeK      (n_pls)
%   NEstimators / LearningRate / Seed / FeaturesPerSubspace
%                       (bagging / boosting / random_subspace)
%   KernelType / Coef0 / Degree (kernel_pls)
%   NPredictive / NXOrthogonal / NYOrthogonal (o2pls)
%   Family             (pls_glm: "gaussian" | "poisson")
%   Weights            (weighted_pls; numeric vector)
%   BlockSizes         (mb_pls; integer vector summing to size(X, 2))
if isstring(algo), algo = char(algo); end

p = inputParser();
p.addParameter("NumComponents",         2,  @(x) isscalar(x) && x > 0);
p.addParameter("Lambda",                0.05, @isscalar);
p.addParameter("Gamma",                 0.5,  @isscalar);
p.addParameter("Alpha",                 0.5,  @isscalar);
p.addParameter("Tau",                   0.5,  @isscalar);
p.addParameter("HuberK",                1.345, @isscalar);
p.addParameter("MaxIrlsIter",           20,   @isscalar);
p.addParameter("WindowSize",            50,   @isscalar);
p.addParameter("ModeJ",                 0,    @isscalar);
p.addParameter("ModeK",                 0,    @isscalar);
p.addParameter("NEstimators",           50,   @isscalar);
p.addParameter("LearningRate",          0.1,  @isscalar);
p.addParameter("FeaturesPerSubspace",   10,   @isscalar);
p.addParameter("Seed",                  0,    @isscalar);
p.addParameter("KernelType",            1,    @isscalar);
p.addParameter("Coef0",                 1.0,  @isscalar);
p.addParameter("Degree",                3,    @isscalar);
p.addParameter("NPredictive",           2,    @isscalar);
p.addParameter("NXOrthogonal",          1,    @isscalar);
p.addParameter("NYOrthogonal",          1,    @isscalar);
p.addParameter("Weights",               [],   @(x) isnumeric(x) && isvector(x));
p.addParameter("BlockSizes",            [],   @(x) isnumeric(x) && isvector(x));
p.addParameter("Family",                "gaussian", ...
    @(x) any(strcmpi(string(x), ["gaussian", "poisson"])));
p.KeepUnmatched = false;
p.parse(varargin{:});
k = p.Results.NumComponents;

switch lower(algo)
    case {"pls", "pls_simpls", "simpls"}
        mdl = pls4all.Regression(X, y, k);
    case "pcr"
        mdl = pls4all.PcrRegression(X, y, k);
    case {"opls", "opls_nipals"}
        mdl = pls4all.OplsRegression(X, y, k);
    case "sparse_simpls"
        mdl = pls4all.SparsePlsRegression(X, y, k, p.Results.Lambda);
    case "cppls"
        mdl = pls4all.CpplsRegression(X, y, k, p.Results.Gamma);
    case "ecr"
        mdl = pls4all.EcrRegression(X, y, k, p.Results.Alpha);
    case "weighted_pls"
        if isempty(p.Results.Weights)
            error("pls4all:nargin", "weighted_pls requires Weights");
        end
        mdl = pls4all.WeightedPlsRegression(X, y, k, p.Results.Weights);
    case "robust_pls"
        mdl = pls4all.RobustPlsRegression(X, y, k, ...
            p.Results.HuberK, p.Results.MaxIrlsIter);
    case "ridge_pls"
        mdl = pls4all.RidgePlsRegression(X, y, k, p.Results.Lambda);
    case "continuum_regression"
        mdl = pls4all.ContinuumRegression(X, y, k, p.Results.Tau);
    case "recursive_pls"
        mdl = pls4all.RecursivePlsRegression(X, y, k, p.Results.WindowSize);
    case "n_pls"
        if p.Results.ModeJ <= 0 || p.Results.ModeK <= 0
            error("pls4all:nargin", "n_pls requires ModeJ and ModeK > 0");
        end
        mdl = pls4all.NPlsRegression(X, y, k, p.Results.ModeJ, p.Results.ModeK);
    case "kernel_pls"
        mdl = pls4all.KernelPlsRegression(X, y, k, ...
            p.Results.KernelType, p.Results.Gamma, p.Results.Coef0, p.Results.Degree);
    case "o2pls"
        mdl = pls4all.O2plsRegression(X, y, p.Results.NPredictive, ...
            p.Results.NXOrthogonal, p.Results.NYOrthogonal);
    case "mb_pls"
        if isempty(p.Results.BlockSizes)
            error("pls4all:nargin", "mb_pls requires BlockSizes");
        end
        mdl = pls4all.MbPlsRegression(X, y, k, p.Results.BlockSizes);
    case "pls_glm"
        mdl = pls4all.GlmRegression(X, y, k, p.Results.Family);
    case "mir_pls"
        mdl = pls4all.MirRegression(X, y, k);
    case "missing_aware_nipals"
        mdl = pls4all.MissingAwareNipalsRegression(X, y, k);
    case "bagging_pls"
        mdl = pls4all.BaggingPlsRegression(X, y, k, ...
            p.Results.NEstimators, p.Results.Seed);
    case "boosting_pls"
        mdl = pls4all.BoostingPlsRegression(X, y, k, ...
            p.Results.NEstimators, p.Results.LearningRate);
    otherwise
        error("pls4all:bad_arg", "unknown algorithm: %s", algo);
end
end
