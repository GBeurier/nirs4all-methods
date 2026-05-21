classdef GenericMethodResult
% pls4all.GenericMethodResult  Classdef fallback for MethodResult kernels.
%
% This keeps the tier-2 factory usable for selectors, diagnostics,
% transformers and paper-only smoke methods whose natural output is not a
% reusable regression coefficient model. predict() returns the stored
% in-sample output selected by PredictionKey.

    properties (SetAccess = protected)
        Algo
        Result
        PredictionKey
        NumPredictors
        NumRows
        Output
    end

    methods
        function obj = GenericMethodResult(algo, X, Y, n_components, params, prediction_key)
            if nargin < 5 || isempty(params)
                params = struct();
            end
            if nargin < 6 || isempty(prediction_key)
                prediction_key = "predictions";
            end
            obj.Algo = char(algo);
            obj.PredictionKey = char(prediction_key);
            obj.NumPredictors = size(X, 2);
            obj.NumRows = size(X, 1);
            obj.Result = pls4all.n4m_method_fit_mex( ...
                obj.Algo, double(X), double(Y), int32(n_components), params);
            obj.Output = pls4all.GenericMethodResult.materialize_result( ...
                obj.Result, obj.PredictionKey, obj.NumPredictors);
        end

        function yhat = predict(obj, X)
            if size(X, 2) ~= obj.NumPredictors
                error("pls4all:shape", ...
                      "X has %d features, but method was fitted with %d", ...
                      size(X, 2), obj.NumPredictors);
            end
            yhat = obj.Output;
        end

        function Xt = transform(obj, X)
            Xt = predict(obj, X);
        end

        function res = raw_result(obj)
            res = obj.Result;
        end
    end

    methods (Static)
        function out = materialize_result(res, pkey, p)
            pkey = char(pkey);
            if any(strcmp(pkey, {"selected_indices", "mask", "support"}))
                if isfield(res, "mask")
                    out = double(reshape(res.mask, 1, []));
                    return;
                end
                if isfield(res, "support")
                    out = double(reshape(res.support, 1, []));
                    return;
                end
                if isfield(res, "selected_indices")
                    sel = double(res.selected_indices(:));
                    mask = zeros(1, p);
                    valid = sel(sel > 0 & sel <= p);
                    if ~isempty(valid)
                        mask(valid) = 1.0;
                    end
                    out = mask;
                    return;
                end
                out = zeros(1, p);
                return;
            end
            if strcmp(pkey, "decision_scores") && isfield(res, "decision_scores")
                out = double(res.decision_scores);
                return;
            end
            if isfield(res, pkey)
                out = double(res.(pkey));
                return;
            end
            if isfield(res, "predictions")
                out = double(res.predictions);
                return;
            end
            if isfield(res, "selected_indices")
                sel = double(res.selected_indices(:));
                mask = zeros(1, p);
                valid = sel(sel > 0 & sel <= p);
                if ~isempty(valid)
                    mask(valid) = 1.0;
                end
                out = mask;
                return;
            end
            error("pls4all:missingField", ...
                  "method result has no field '%s'", pkey);
        end
    end
end
