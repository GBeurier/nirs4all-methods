classdef MbPlsRegression < pls4all.MethodResultRegression
% pls4all.MbPlsRegression — Multi-block PLS.
% predict uses the stored intercept directly (coefficients are already
% in original X scale + intercept folds in y_mean - x_mean @ coef).
    properties (SetAccess = private)
        BlockSizes
        BlockWeights
        XScale
    end
    methods
        function obj = MbPlsRegression(X, y, n_components, block_sizes)
            if nargin < 4
                error("pls4all:nargin", "block_sizes required");
            end
            if isvector(y), y = y(:); end
            res = pls4all.mb_pls(X, y, n_components, block_sizes);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.BlockSizes   = int32(block_sizes(:));
            obj.BlockWeights = res.block_weights;
            obj.XScale       = res.x_scale;
            obj.Method       = "mb_pls";
        end
    end
end
