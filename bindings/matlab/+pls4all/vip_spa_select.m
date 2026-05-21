function res = vip_spa_select(X, Y, n_components, vip_threshold, top_k)
% pls4all.vip_spa_select  VIP-then-SPA hybrid (Phase 53).
if nargin < 4 || isempty(vip_threshold), vip_threshold = 0.3; end
if nargin < 5 || isempty(top_k),         top_k = 10;          end
params = struct("vip_threshold", double(vip_threshold), "top_k", int32(top_k));
res = pls4all.n4m_method_fit_mex("vip_spa_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
