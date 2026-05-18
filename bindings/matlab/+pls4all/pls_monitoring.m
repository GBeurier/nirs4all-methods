function res = pls_monitoring(X_reference, Y_reference, X_monitor, ...
                                n_components, alpha)
% pls4all.pls_monitoring  PLS process monitoring (Hotelling T^2 + Q).
%
%   res = pls4all.pls_monitoring(X_ref, Y_ref, X_mon, n_components)
%   res = pls4all.pls_monitoring(X_ref, Y_ref, X_mon, n_components, alpha)
%
% Fits an internal SIMPLS model (store_scores=1) on the reference data,
% derives T^2 / Q control thresholds at confidence level (1 - alpha)
% from the reference rows, then evaluates the monitor rows.
%
% Output struct fields:
%   t2, q                       : 1 x n_monitor row vectors.
%   t2_reference, q_reference   : 1 x n_reference row vectors.
%   t2_alarms, q_alarms,
%   any_alarms                  : 1 x n_monitor int32 flag vectors.
%   t2_threshold, q_threshold,
%   alpha                       : scalars.
if nargin < 5 || isempty(alpha)
    alpha = 0.05;
end
params = struct("X_monitor", double(X_monitor), ...
                "alpha", double(alpha));
res = pls4all.p4a_method_fit_mex("pls_monitoring_run", ...
                                  double(X_reference), ...
                                  double(Y_reference), ...
                                  int32(n_components), params);
end
