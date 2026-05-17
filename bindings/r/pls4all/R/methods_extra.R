# SPDX-License-Identifier: CeCILL-2.1
#
# Tier-1 R wrappers for the remaining MethodResult fits, selectors and
# diagnostics exposed by the unified dispatcher (r_dispatch.c). The
# 6 fits in methods.R remain as convenience aliases — these add the
# rest.

# ---- MethodResult fits ----------------------------------------------

#' Elastic Component Regression (Liu 2009/2010).
#' @export
ecr_fit <- function(X, Y, n_components, alpha = 0.5) {
    pls4all_method("ecr", X, Y, n_components, params = list(alpha = alpha))
}

#' Domain-Invariant PLS (Nikzad-Langerodi 2018).
#' @export
di_pls_fit <- function(X_source, Y_source, n_components, X_target, di_lambda = 1.0) {
    pls4all_method("di_pls", X_source, Y_source, n_components,
                   params = list(X_target = as.matrix(X_target),
                                  di_lambda = di_lambda))
}

#' Robust PLS via Huber IRLS.
#' @export
robust_pls_fit <- function(X, Y, n_components,
                            huber_k = 1.345, max_irls_iter = 20L) {
    pls4all_method("robust_pls", X, Y, n_components,
                   params = list(huber_k = huber_k,
                                  max_irls_iter = as.integer(max_irls_iter)))
}

#' L2-augmented PLS regression.
#' @export
ridge_pls_fit <- function(X, Y, n_components, ridge_lambda = 1.0) {
    pls4all_method("ridge_pls", X, Y, n_components,
                   params = list(ridge_lambda = ridge_lambda))
}

#' Continuum regression (tau ∈ [0, 1]).
#' @export
continuum_regression_fit <- function(X, Y, n_components, tau = 0.5) {
    pls4all_method("continuum_regression", X, Y, n_components,
                   params = list(tau = tau))
}

#' Moving-window recursive PLS.
#' @export
recursive_pls_fit <- function(X, Y, n_components, window_size = 50L) {
    pls4all_method("recursive_pls", X, Y, n_components,
                   params = list(window_size = as.integer(window_size)))
}

#' N-PLS (3-way tensor) regression. `X` is the flattened (n, mode_j*mode_k) matrix.
#' @export
n_pls_fit <- function(X_flat, Y, n_components, mode_j, mode_k) {
    pls4all_method("n_pls", X_flat, Y, n_components,
                   params = list(mode_j = as.integer(mode_j),
                                  mode_k = as.integer(mode_k)))
}

#' Non-linear kernel PLS (Rosipal & Trejo 2001).
#' @export
kernel_pls_fit <- function(X, Y, n_components,
                            kernel_type = 1L, gamma = 0.0,
                            coef0 = 1.0, degree = 3L) {
    pls4all_method("kernel_pls", X, Y, n_components,
                   params = list(kernel_type = as.integer(kernel_type),
                                  gamma = gamma, coef0 = coef0,
                                  degree = as.integer(degree)))
}

#' O2-PLS (bi-directional OPLS).
#' @export
o2pls_fit <- function(X, Y, n_predictive = 2L,
                       n_x_orthogonal = 1L, n_y_orthogonal = 1L) {
    pls4all_method("o2pls", X, Y, n_predictive,
                   params = list(n_predictive = as.integer(n_predictive),
                                  n_x_orthogonal = as.integer(n_x_orthogonal),
                                  n_y_orthogonal = as.integer(n_y_orthogonal)))
}

#' Sparse PLS-DA classifier (`y_labels` is an integer vector of class IDs).
#' @export
sparse_pls_da_fit <- function(X, y_labels, n_components, sparsity_lambda = 0.05) {
    pls4all_method("sparse_pls_da", X, rep(0, nrow(X)), n_components,
                   params = list(y_labels = as.integer(y_labels),
                                  sparsity_lambda = sparsity_lambda))
}

#' Group-sparse PLS (group L1 across feature groups).
#' @export
group_sparse_pls_fit <- function(X, Y, n_components, group_assignment,
                                  group_lambda = 0.05) {
    pls4all_method("group_sparse_pls", X, Y, n_components,
                   params = list(group_assignment = as.integer(group_assignment),
                                  group_lambda = group_lambda))
}

#' Fused-sparse PLS (L1 + adjacent-coef smoothing).
#' @export
fused_sparse_pls_fit <- function(X, Y, n_components,
                                  l1_lambda = 0.05, fusion_lambda = 0.05) {
    pls4all_method("fused_sparse_pls", X, Y, n_components,
                   params = list(l1_lambda = l1_lambda,
                                  fusion_lambda = fusion_lambda))
}

#' Sequential & Orthogonalised multi-block PLS (Næs et al. 2011).
#' `block_sizes` integer vector summing to ncol(X);
#' `n_components_per_block` integer vector of same length.
#' @export
so_pls_fit <- function(X, Y, block_sizes, n_components_per_block) {
    n_total <- sum(n_components_per_block)
    pls4all_method("so_pls", X, Y, as.integer(n_total),
                   params = list(block_sizes = as.integer(block_sizes),
                                  n_components_per_block =
                                      as.integer(n_components_per_block)))
}

#' OnPLS — Orthogonal multi-block PLS (joint + unique loadings).
#' @export
on_pls_fit <- function(X, Y, n_joint, n_unique_per_block, block_sizes) {
    pls4all_method("on_pls", X, Y, as.integer(n_joint),
                   params = list(block_sizes = as.integer(block_sizes),
                                  n_joint = as.integer(n_joint),
                                  n_unique_per_block =
                                      as.integer(n_unique_per_block)))
}

#' ROSA — Response-Oriented Sequential Alternation.
#' @export
rosa_fit <- function(X, Y, n_components, block_sizes) {
    pls4all_method("rosa", X, Y, n_components,
                   params = list(block_sizes = as.integer(block_sizes)))
}

#' Bagging PLS (bootstrap aggregation of PLS regressors).
#' @export
bagging_pls_fit <- function(X, Y, n_components,
                             n_estimators = 50L, seed = 0L) {
    pls4all_method("bagging_pls", X, Y, n_components,
                   params = list(n_estimators = as.integer(n_estimators),
                                  seed = as.integer(seed)))
}

#' Boosting PLS (stage-wise refit with learning_rate).
#' @export
boosting_pls_fit <- function(X, Y, n_components,
                              n_estimators = 50L, learning_rate = 0.1) {
    pls4all_method("boosting_pls", X, Y, n_components,
                   params = list(n_estimators = as.integer(n_estimators),
                                  learning_rate = learning_rate))
}

#' Random-subspace PLS (Ho 1998).
#' @export
random_subspace_pls_fit <- function(X, Y, n_components,
                                     n_estimators = 50L,
                                     features_per_subspace = 10L,
                                     seed = 0L) {
    pls4all_method("random_subspace_pls", X, Y, n_components,
                   params = list(n_estimators = as.integer(n_estimators),
                                  features_per_subspace =
                                      as.integer(features_per_subspace),
                                  seed = as.integer(seed)))
}

#' Gaussian Process Regression on PLS scores (single-target Y).
#' @export
gpr_pls_fit <- function(X, Y, n_components,
                         length_scale = 1.0, noise_level = 1e-4, seed = 0L) {
    pls4all_method("gpr_pls", X, Y, n_components,
                   params = list(length_scale = length_scale,
                                  noise_level = noise_level,
                                  seed = as.integer(seed)))
}

#' PLS-QDA (Quadratic Discriminant Analysis on PLS scores).
#' @export
pls_qda_fit <- function(X, y_labels, n_components) {
    pls4all_method("pls_qda", X, rep(0, nrow(X)), n_components,
                   params = list(y_labels = as.integer(y_labels)))
}

#' PLS-Cox proportional hazards.
#' @export
pls_cox_fit <- function(X, n_components, survival_times, event_indicators) {
    pls4all_method("pls_cox", X, rep(0, nrow(X)), n_components,
                   params = list(survival_times = as.numeric(survival_times),
                                  event_indicators =
                                      as.integer(event_indicators)))
}

#' Piecewise Direct Standardization (calibration transfer).
#' @export
pds_fit <- function(X_source, X_target, window_half_width = 2L) {
    pls4all_method("pds", X_source, rep(0, nrow(X_source)), 1L,
                   params = list(X_target = as.matrix(X_target),
                                  window_half_width =
                                      as.integer(window_half_width)))
}

#' Direct Standardization (calibration transfer).
#' @export
ds_fit <- function(X_source, X_target) {
    pls4all_method("ds", X_source, rep(0, nrow(X_source)), 1L,
                   params = list(X_target = as.matrix(X_target)))
}

#' Missing-aware NIPALS PLS (Nelson 1996).
#' @export
missing_aware_nipals_fit <- function(X, Y, n_components) {
    pls4all_method("missing_aware_nipals", X, Y, n_components)
}

#' Locally-weighted PLS (Næs & Centner 1998).
#' @export
lw_pls_fit <- function(X, Y, n_components, n_neighbors = 30L) {
    pls4all_method("lw_pls", X, Y, n_components,
                   params = list(n_neighbors = as.integer(n_neighbors)))
}

#' PLS-LDA — Linear Discriminant Analysis on PLS scores.
#' @export
pls_lda_fit <- function(X, y_labels, n_components, n_classes = NULL) {
    if (is.null(n_classes)) n_classes <- max(as.integer(y_labels)) + 1L
    pls4all_method("pls_lda", X, rep(0, nrow(X)), n_components,
                   params = list(y_labels = as.integer(y_labels),
                                  n_classes = as.integer(n_classes)))
}

#' Multinomial logistic regression on PLS scores.
#' @export
pls_logistic_fit <- function(X, y_labels, n_components, n_classes = NULL) {
    if (is.null(n_classes)) n_classes <- max(as.integer(y_labels)) + 1L
    pls4all_method("pls_logistic", X, rep(0, nrow(X)), n_components,
                   params = list(y_labels = as.integer(y_labels),
                                  n_classes = as.integer(n_classes)))
}

# ---- Selectors (no ValidationPlan or with default plan inside C) ----

#' Stability selector (coefficient stability, MCUVE-style).
#' @export
stability_select <- function(X, Y, n_components, top_k = 10L) {
    pls4all_method("stability_select", X, Y, n_components,
                   params = list(top_k = as.integer(top_k)))
}

#' UVE — Uninformative Variable Elimination.
#' @export
uve_select <- function(X, Y, n_components, noise_features = NULL, noise_seed = 0L) {
    if (is.null(noise_features)) noise_features <- ncol(as.matrix(X))
    pls4all_method("uve_select", X, Y, n_components,
                   params = list(noise_features = as.integer(noise_features),
                                  noise_seed = as.integer(noise_seed)))
}

#' Interval selector (iPLS).
#' @export
interval_select <- function(X, Y, n_components, interval_width = 10L, step = 1L) {
    pls4all_method("interval_select", X, Y, n_components,
                   params = list(interval_width = as.integer(interval_width),
                                  step = as.integer(step)))
}

#' Random Frog (Phase 5g).
#' @export
random_frog_select <- function(X, Y, n_components,
                                n_iterations = 100L,
                                initial_size = 30L, min_size = NULL,
                                max_size = NULL, top_k = 10L, seed = 0L) {
    if (is.null(min_size)) min_size <- n_components
    if (is.null(max_size)) max_size <- ncol(as.matrix(X))
    pls4all_method("random_frog_select", X, Y, n_components,
                   params = list(n_iterations = as.integer(n_iterations),
                                  initial_size = as.integer(initial_size),
                                  min_size = as.integer(min_size),
                                  max_size = as.integer(max_size),
                                  top_k = as.integer(top_k),
                                  seed = as.integer(seed)))
}

#' SCARS — Stability + CARS.
#' @export
scars_select <- function(X, Y, n_components,
                          n_iterations = 50L, min_features = 5L,
                          sample_fraction = 0.8, seed = 0L) {
    pls4all_method("scars_select", X, Y, n_components,
                   params = list(n_iterations = as.integer(n_iterations),
                                  min_features = as.integer(min_features),
                                  sample_fraction = sample_fraction,
                                  seed = as.integer(seed)))
}

#' GA-PLS — genetic algorithm variable selection.
#' @export
ga_select <- function(X, Y, n_components,
                       n_generations = 50L, population_size = 50L,
                       min_features = NULL, max_features = NULL,
                       mutation_rate = 0.01, seed = 0L) {
    if (is.null(min_features)) min_features <- n_components
    if (is.null(max_features)) max_features <- ncol(as.matrix(X))
    pls4all_method("ga_select", X, Y, n_components,
                   params = list(n_generations = as.integer(n_generations),
                                  population_size = as.integer(population_size),
                                  min_features = as.integer(min_features),
                                  max_features = as.integer(max_features),
                                  mutation_rate = mutation_rate,
                                  seed = as.integer(seed)))
}

#' PSO-PLS (Binary Particle Swarm Optimization).
#' @export
pso_select <- function(X, Y, n_components,
                        n_swarm = 30L, n_iterations = 50L,
                        w = 0.729, c1 = 1.494, c2 = 1.494, v_max = 4.0,
                        seed = 0L) {
    pls4all_method("pso_select", X, Y, n_components,
                   params = list(n_swarm = as.integer(n_swarm),
                                  n_iterations = as.integer(n_iterations),
                                  w = w, c1 = c1, c2 = c2, v_max = v_max,
                                  seed = as.integer(seed)))
}

#' VISSA — Variable Iterative Space Shrinkage Approach.
#' @export
vissa_select <- function(X, Y, n_components,
                          n_iterations = 20L, n_submodels = 100L,
                          ratio_kept = 0.1, threshold = 0.5,
                          floor_probability = 0.01, seed = 0L) {
    pls4all_method("vissa_select", X, Y, n_components,
                   params = list(n_iterations = as.integer(n_iterations),
                                  n_submodels = as.integer(n_submodels),
                                  ratio_kept = ratio_kept,
                                  threshold = threshold,
                                  floor_probability = floor_probability,
                                  seed = as.integer(seed)))
}

#' Shaving selector.
#' @export
shaving_select <- function(X, Y, n_components,
                            n_steps = 10L, min_features = 5L,
                            shave_fraction = 0.1) {
    pls4all_method("shaving_select", X, Y, n_components,
                   params = list(n_steps = as.integer(n_steps),
                                  min_features = as.integer(min_features),
                                  shave_fraction = shave_fraction))
}

#' BVE-PLS.
#' @export
bve_select <- function(X, Y, n_components, n_steps = 10L, min_features = 5L) {
    pls4all_method("bve_select", X, Y, n_components,
                   params = list(n_steps = as.integer(n_steps),
                                  min_features = as.integer(min_features)))
}

#' T²-PLS — sweep over α-thresholds.
#' @export
t2_select <- function(X, Y, n_components, alpha_thresholds,
                       min_selected = NULL) {
    if (is.null(min_selected)) min_selected <- n_components
    pls4all_method("t2_select", X, Y, n_components,
                   params = list(alpha_thresholds = as.numeric(alpha_thresholds),
                                  min_selected = as.integer(min_selected)))
}

#' WVC-PLS — weighted vector correlation top-k selector.
#' @export
wvc_select <- function(X, Y, n_components, top_k = 10L, normalize = TRUE) {
    pls4all_method("wvc_select", X, Y, n_components,
                   params = list(top_k = as.integer(top_k),
                                  normalize = as.integer(normalize)))
}

#' WVC-threshold selector.
#' @export
wvc_threshold_select <- function(X, Y, n_components,
                                  normalize = TRUE, threshold = 0.0,
                                  threshold_factor = 1.0, min_selected = 1L) {
    pls4all_method("wvc_threshold_select", X, Y, n_components,
                   params = list(normalize = as.integer(normalize),
                                  threshold = threshold,
                                  threshold_factor = threshold_factor,
                                  min_selected = as.integer(min_selected)))
}

#' EMCUVE — ensemble Monte Carlo UVE.
#' @export
emcuve_select <- function(X, Y, n_components,
                           noise_features = NULL, noise_seed = 0L,
                           n_ensembles = 5L, vote_threshold = 0.5) {
    if (is.null(noise_features)) noise_features <- ncol(as.matrix(X))
    pls4all_method("emcuve_select", X, Y, n_components,
                   params = list(noise_features = as.integer(noise_features),
                                  noise_seed = as.integer(noise_seed),
                                  n_ensembles = as.integer(n_ensembles),
                                  vote_threshold = vote_threshold))
}

#' Randomization test selector.
#' @export
randomization_select <- function(X, Y, n_components,
                                  n_permutations = 100L,
                                  randomization_seed = 0L, alpha = 0.05) {
    pls4all_method("randomization_select", X, Y, n_components,
                   params = list(n_permutations = as.integer(n_permutations),
                                  randomization_seed =
                                      as.integer(randomization_seed),
                                  alpha = alpha))
}

#' biPLS — backward interval PLS.
#' @export
bipls_select <- function(X, Y, n_components,
                          interval_width = 10L, min_intervals = 1L) {
    pls4all_method("bipls_select", X, Y, n_components,
                   params = list(interval_width = as.integer(interval_width),
                                  min_intervals = as.integer(min_intervals)))
}

#' siPLS — synergistic interval PLS.
#' @export
sipls_select <- function(X, Y, n_components,
                          interval_width = 10L, combination_size = 2L) {
    pls4all_method("sipls_select", X, Y, n_components,
                   params = list(interval_width = as.integer(interval_width),
                                  combination_size = as.integer(combination_size)))
}

#' REP-PLS.
#' @export
rep_select <- function(X, Y, n_components,
                        n_steps = 10L, min_features = 5L, remove_count = 1L) {
    pls4all_method("rep_select", X, Y, n_components,
                   params = list(n_steps = as.integer(n_steps),
                                  min_features = as.integer(min_features),
                                  remove_count = as.integer(remove_count)))
}

#' IPW-PLS.
#' @export
ipw_select <- function(X, Y, n_components,
                        n_iterations = 10L, top_k = 10L,
                        damping = 0.5, weight_floor = 1e-6) {
    pls4all_method("ipw_select", X, Y, n_components,
                   params = list(n_iterations = as.integer(n_iterations),
                                  top_k = as.integer(top_k),
                                  damping = damping,
                                  weight_floor = weight_floor))
}

#' ST-PLS — score-threshold selector.
#' @export
st_select <- function(X, Y, n_components, thresholds, min_selected = NULL) {
    if (is.null(min_selected)) min_selected <- n_components
    pls4all_method("st_select", X, Y, n_components,
                   params = list(thresholds = as.numeric(thresholds),
                                  min_selected = as.integer(min_selected)))
}

#' IRIV — Iteratively Retains Informative Variables.
#' @export
iriv_select <- function(X, Y, n_components, max_rounds = 20L, seed = 0L) {
    pls4all_method("iriv_select", X, Y, n_components,
                   params = list(max_rounds = as.integer(max_rounds),
                                  seed = as.integer(seed)))
}

#' IRF — Interval Random Frog.
#' @export
irf_select <- function(X, Y, n_components,
                        n_iterations = 100L, window_size = 10L,
                        initial_intervals = 10L, top_k = 5L, seed = 0L) {
    pls4all_method("irf_select", X, Y, n_components,
                   params = list(n_iterations = as.integer(n_iterations),
                                  window_size = as.integer(window_size),
                                  initial_intervals =
                                      as.integer(initial_intervals),
                                  top_k = as.integer(top_k),
                                  seed = as.integer(seed)))
}

#' VIP-SPA hybrid selector.
#' @export
vip_spa_select <- function(X, Y, n_components,
                            vip_threshold = 0.3, top_k = 10L) {
    pls4all_method("vip_spa_select", X, Y, n_components,
                   params = list(vip_threshold = vip_threshold,
                                  top_k = as.integer(top_k)))
}

# ---- Diagnostics ----------------------------------------------------

#' One-SE rule from a (max_components × n_folds) fold RMSE matrix.
#' @export
one_se_rule <- function(fold_rmse_matrix) {
    fr <- as.matrix(fold_rmse_matrix)
    storage.mode(fr) <- "double"
    pls4all_method("one_se_rule_compute", matrix(0, 2, 1), matrix(0, 2, 1),
                   nrow(fr), params = list(fold_rmse_matrix = fr))
}
