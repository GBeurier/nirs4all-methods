function mdl = fit(algo, X, y, varargin)
% pls4all.fit  Unified factory for tier-2 PLS regressors.
%
% Supported algorithms (alphabetical):
%   "bagging_pls"    → BaggingPlsRegression
%   "boosting_pls"   → BoostingPlsRegression
%   "continuum_regression" → ContinuumRegression
%   "cppls"          → CpplsRegression
%   "di_pls"         → DiPlsRegression            (needs XTarget)
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
%   DiLambda           (DI-PLS penalty)
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
%   XTarget            (di_pls; numeric matrix)
%   BlockSizes         (mb_pls; integer vector summing to size(X, 2))
%   Params             struct passed to GenericMethodResult fallback
%   PredictionKey      MethodResult field returned by generic predict()
if isstring(algo), algo = char(algo); end

opts = struct( ...
    "NumComponents",       2, ...
    "Lambda",              0.05, ...
    "Gamma",               0.5, ...
    "Alpha",               0.5, ...
    "DiLambda",            1.0, ...
    "Tau",                 0.5, ...
    "HuberK",              1.345, ...
    "MaxIrlsIter",         20, ...
    "WindowSize",          50, ...
    "ModeJ",               0, ...
    "ModeK",               0, ...
    "NEstimators",         50, ...
    "LearningRate",        0.1, ...
    "FeaturesPerSubspace", 10, ...
    "Seed",                0, ...
    "KernelType",          1, ...
    "Coef0",               1.0, ...
    "Degree",              3, ...
    "NPredictive",         2, ...
    "NXOrthogonal",        1, ...
    "NYOrthogonal",        1, ...
    "Weights",             [], ...
    "XTarget",             [], ...
    "BlockSizes",          [], ...
    "Params",              struct(), ...
    "PredictionKey",       "predictions", ...
    "Family",              "gaussian");

if mod(numel(varargin), 2) ~= 0
    error("pls4all:nargin", "Name-value arguments must come in pairs");
end
for i = 1:2:numel(varargin)
    name = varargin{i};
    if isstring(name) && isscalar(name), name = char(name); end
    if ~ischar(name)
        error("pls4all:nargin", "Parameter names must be char or string scalars");
    end
    value = varargin{i + 1};
    switch lower(name)
    case "numcomponents"
        opts.NumComponents = value;
    case "lambda"
        opts.Lambda = value;
    case "gamma"
        opts.Gamma = value;
    case "alpha"
        opts.Alpha = value;
    case "dilambda"
        opts.DiLambda = value;
    case "tau"
        opts.Tau = value;
    case "huberk"
        opts.HuberK = value;
    case "maxirlsiter"
        opts.MaxIrlsIter = value;
    case "windowsize"
        opts.WindowSize = value;
    case "modej"
        opts.ModeJ = value;
    case "modek"
        opts.ModeK = value;
    case "nestimators"
        opts.NEstimators = value;
    case "learningrate"
        opts.LearningRate = value;
    case "featurespersubspace"
        opts.FeaturesPerSubspace = value;
    case "seed"
        opts.Seed = value;
    case "kerneltype"
        opts.KernelType = value;
    case "coef0"
        opts.Coef0 = value;
    case "degree"
        opts.Degree = value;
    case "npredictive"
        opts.NPredictive = value;
    case "nxorthogonal"
        opts.NXOrthogonal = value;
    case "nyorthogonal"
        opts.NYOrthogonal = value;
    case "weights"
        opts.Weights = value;
    case "xtarget"
        opts.XTarget = value;
    case "blocksizes"
        opts.BlockSizes = value;
    case "params"
        opts.Params = value;
    case "predictionkey"
        opts.PredictionKey = value;
    case "family"
        opts.Family = value;
    otherwise
        error("pls4all:unknownParameter", "Unknown parameter: %s", name);
    end
end

if isstring(opts.PredictionKey), opts.PredictionKey = char(opts.PredictionKey); end
if isstring(opts.Family), opts.Family = char(opts.Family); end
if ~any(strcmpi(char(opts.Family), {"gaussian", "poisson"}))
    error("pls4all:nargin", "Family must be 'gaussian' or 'poisson'");
end

k = opts.NumComponents;
algo_l = lower(algo);
generic_algo = algo_l;
generic_params = opts.Params;
prediction_key = char(opts.PredictionKey);

if strcmp(algo_l, "kernel_pls_rbf")
    algo_l = "kernel_pls";
    generic_algo = "kernel_pls";
    if ~isfield(generic_params, "kernel_type")
        generic_params.kernel_type = int32(1);
    end
elseif strcmp(algo_l, "approximate_press")
    generic_algo = "approximate_press_compute";
    if isfield(generic_params, "max_components")
        k = double(generic_params.max_components);
    end
elseif strcmp(algo_l, "one_se_rule")
    generic_algo = "one_se_rule_compute";
    if isfield(generic_params, "max_components")
        k = double(generic_params.max_components);
    end
elseif any(strcmp(algo_l, {"pls_diagnostic_t2", "pls_diagnostic_q", "pls_diagnostic_dmodx"}))
    generic_algo = "pls_diagnostics_compute";
elseif strcmp(algo_l, "pls_monitoring")
    generic_algo = "pls_monitoring_run";
elseif strcmp(algo_l, "variable_select_vip")
    generic_algo = "variable_select_rank";
    generic_params.method = int32(0);
elseif strcmp(algo_l, "variable_select_coef")
    generic_algo = "variable_select_rank";
    generic_params.method = int32(1);
elseif strcmp(algo_l, "variable_select_sr")
    generic_algo = "variable_select_rank";
    generic_params.method = int32(2);
end

switch algo_l
case {"pls", "pls_simpls", "simpls"}
    mdl = pls4all.Regression(X, y, k);
case "pcr"
    mdl = pls4all.PcrRegression(X, y, k);
case {"opls", "opls_nipals"}
    mdl = pls4all.OplsRegression(X, y, k);
case "sparse_simpls"
    mdl = pls4all.SparsePlsRegression(X, y, k, opts.Lambda);
case "cppls"
    mdl = pls4all.CpplsRegression(X, y, k, opts.Gamma);
case "ecr"
    mdl = pls4all.EcrRegression(X, y, k, opts.Alpha);
case "di_pls"
    if isempty(opts.XTarget)
        error("pls4all:nargin", "di_pls requires XTarget");
    end
    mdl = pls4all.DiPlsRegression(X, y, k, ...
        opts.XTarget, opts.DiLambda);
case "weighted_pls"
    if isempty(opts.Weights)
        error("pls4all:nargin", "weighted_pls requires Weights");
    end
    mdl = pls4all.WeightedPlsRegression(X, y, k, opts.Weights);
case "robust_pls"
    mdl = pls4all.RobustPlsRegression(X, y, k, ...
        opts.HuberK, opts.MaxIrlsIter);
case "ridge_pls"
    mdl = pls4all.RidgePlsRegression(X, y, k, opts.Lambda);
case "continuum_regression"
    mdl = pls4all.ContinuumRegression(X, y, k, opts.Tau);
case "recursive_pls"
    mdl = pls4all.RecursivePlsRegression(X, y, k, opts.WindowSize);
case "n_pls"
    if opts.ModeJ <= 0 || opts.ModeK <= 0
        error("pls4all:nargin", "n_pls requires ModeJ and ModeK > 0");
    end
    mdl = pls4all.NPlsRegression(X, y, k, opts.ModeJ, opts.ModeK);
case "kernel_pls"
    mdl = pls4all.KernelPlsRegression(X, y, k, ...
        opts.KernelType, opts.Gamma, opts.Coef0, opts.Degree);
case "o2pls"
    mdl = pls4all.O2plsRegression(X, y, opts.NPredictive, ...
        opts.NXOrthogonal, opts.NYOrthogonal);
case "mb_pls"
    if isempty(opts.BlockSizes)
        error("pls4all:nargin", "mb_pls requires BlockSizes");
    end
    mdl = pls4all.MbPlsRegression(X, y, k, opts.BlockSizes);
case "pls_glm"
    mdl = pls4all.GlmRegression(X, y, k, opts.Family);
case "mir_pls"
    mdl = pls4all.MirRegression(X, y, k);
case "missing_aware_nipals"
    mdl = pls4all.MissingAwareNipalsRegression(X, y, k);
case "bagging_pls"
    mdl = pls4all.BaggingPlsRegression(X, y, k, ...
        opts.NEstimators, opts.Seed);
case "boosting_pls"
    mdl = pls4all.BoostingPlsRegression(X, y, k, ...
        opts.NEstimators, opts.LearningRate);
otherwise
    mdl = pls4all.GenericMethodResult(generic_algo, X, y, k, ...
                                      generic_params, prediction_key);
end
end
