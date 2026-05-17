function res = approximate_press(X, Y, max_components)
% pls4all.approximate_press  PRESS-curve component selection.
%
%   res = pls4all.approximate_press(X, Y, max_components)
%
% For each k ∈ {1, …, max_components}, fits SIMPLS and approximates
% PRESS via leverage-inflated in-sample residuals. Returns:
%   press_per_component (1 × max_components)
%   rmse_per_component  (1 × max_components)
%   selected_n_components — argmin PRESS (1-based)
res = pls4all.p4a_method_fit_mex("approximate_press", ...
                                  double(X), double(Y), ...
                                  int32(max_components), struct());
end
