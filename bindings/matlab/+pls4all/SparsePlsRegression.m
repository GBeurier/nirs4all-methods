classdef SparsePlsRegression < pls4all.MethodResultRegression
% pls4all.SparsePlsRegression — Sparse SIMPLS (Chun & Keles 2010)
% as a tier-2 classdef. Construct via the factory:
%
%   mdl = pls4all.fitrsparsepls(X, y, "NumComponents", 5, "Lambda", 0.05)
%
% or directly:
%
%   mdl = pls4all.SparsePlsRegression(X, y, n_components, sparsity_lambda)

    properties (SetAccess = private)
        SparsityLambda = 0.05
    end

    methods
        function obj = SparsePlsRegression(X, y, n_components, sparsity_lambda)
            if nargin < 4 || isempty(sparsity_lambda)
                sparsity_lambda = 0.05;
            end
            if isvector(y), y = y(:); end
            if size(X, 1) ~= size(y, 1)
                error("pls4all:shape", "X and y must have the same number of rows");
            end
            res = pls4all.sparse_simpls(X, y, n_components, sparsity_lambda);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.SparsityLambda = double(sparsity_lambda);
            obj.Method = "sparse_simpls";
            obj.Extra = struct("sparsity_lambda", obj.SparsityLambda);
        end
    end
end
