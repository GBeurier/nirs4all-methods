function res = boosting_pls(X, Y, n_components, n_estimators, learning_rate)
% pls4all.boosting_pls  Gradient-boosting stage-wise refit of PLS regressors.
if nargin < 4 || isempty(n_estimators),  n_estimators = 50;   end
if nargin < 5 || isempty(learning_rate), learning_rate = 0.1; end
params = struct("n_estimators",  int32(n_estimators), ...
                "learning_rate", double(learning_rate));
res = pls4all.p4a_method_fit_mex("boosting_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
