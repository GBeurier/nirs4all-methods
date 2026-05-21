function res = ds(X_source, X_target)
% pls4all.ds  Direct Standardization (calibration transfer).
params = struct("X_target", double(X_target));
res = pls4all.n4m_method_fit_mex("ds", double(X_source), zeros(size(X_source,1), 1), ...
                                  int32(1), params);
end
