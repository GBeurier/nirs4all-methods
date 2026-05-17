classdef ContinuumRegression < pls4all.MethodResultRegression
% pls4all.ContinuumRegression  Continuum regression (tau ∈ [0, 1]).
    properties (SetAccess = private)
        Tau = 0.5
    end
    methods
        function obj = ContinuumRegression(X, y, n_components, tau)
            if nargin < 4 || isempty(tau), tau = 0.5; end
            if isvector(y), y = y(:); end
            res = pls4all.continuum_regression(X, y, n_components, tau);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.Tau = double(tau);
            obj.Method = "continuum_regression";
            obj.Extra = struct("tau", obj.Tau);
        end
    end
end
