function res = pls_lda(X, y_labels, n_components, n_classes)
% pls4all.pls_lda  Linear Discriminant Analysis on PLS scores.
if nargin < 4 || isempty(n_classes), n_classes = max(int32(y_labels(:))) + 1; end
params = struct("y_labels",  int32(y_labels(:)), ...
                "n_classes", int32(n_classes));
res = pls4all.p4a_method_fit_mex("pls_lda", double(X), zeros(size(X,1), 1), ...
                                  int32(n_components), params);
end
