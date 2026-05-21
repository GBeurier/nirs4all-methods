function res = pls_cox(X, n_components, survival_times, event_indicators)
% pls4all.pls_cox  PLS-Cox proportional hazards (Breslow baseline hazard).
% survival_times: numeric vector of length size(X, 1).
% event_indicators: 0/1 integer vector of length size(X, 1).
params = struct("survival_times",   double(survival_times(:)), ...
                "event_indicators", int32(event_indicators(:)));
res = pls4all.n4m_method_fit_mex("pls_cox", double(X), zeros(size(X,1), 1), ...
                                  int32(n_components), params);
end
