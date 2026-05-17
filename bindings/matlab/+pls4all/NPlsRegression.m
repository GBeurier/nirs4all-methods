classdef NPlsRegression < pls4all.MethodResultRegression
% pls4all.NPlsRegression  N-PLS (3-way tensor) regression (Bro 1996).
    properties (SetAccess = private)
        ModeJ
        ModeK
    end
    methods
        function obj = NPlsRegression(X_flat, y, n_components, mode_j, mode_k)
            if isvector(y), y = y(:); end
            res = pls4all.n_pls(X_flat, y, n_components, mode_j, mode_k);
            obj = obj.absorb_result(res, n_components, size(X_flat, 2));
            obj.ModeJ = int32(mode_j); obj.ModeK = int32(mode_k);
            obj.Method = "n_pls";
            obj.Extra = struct("mode_j", obj.ModeJ, "mode_k", obj.ModeK);
        end
    end
end
