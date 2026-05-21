function res = pls_qda(X, y_labels, n_components)
% pls4all.pls_qda  Quadratic Discriminant Analysis on PLS scores.
% y_labels: integer class IDs in {0, …, n_classes-1}.
params = struct("y_labels", int32(y_labels(:)));
res = pls4all.n4m_method_fit_mex("pls_qda", double(X), zeros(size(X, 1), 1), ...
                                  int32(n_components), params);
end
