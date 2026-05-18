function res = selectivity_ratio_select(X, Y, n_components, top_k)
% pls4all.selectivity_ratio_select  Selectivity-ratio feature ranking.
%
%   res = pls4all.selectivity_ratio_select(X, Y, n_components, top_k)
%
% Fits an internal SIMPLS model (store_scores=1) and ranks features by
% the Selectivity Ratio (SR) statistic.
%
% Output struct fields:
%   scores            : 1 x n_features double row of SR scores.
%   selected_indices  : 1 x top_k row vector of 1-based feature indices.
%   n_features, top_k, method (== 2).
if nargin < 4 || isempty(top_k)
    top_k = min(10, size(X, 2));
end
params = struct("method", int32(2), ...
                "top_k", int32(top_k));
res = pls4all.p4a_method_fit_mex("variable_select_rank", ...
                                  double(X), double(Y), ...
                                  int32(n_components), params);
end
