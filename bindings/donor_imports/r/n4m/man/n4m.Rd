\name{n4m}
\alias{n4m}
\alias{n4m_abi_version}
\alias{n4m_version_string}
\alias{n4m_library_path}
\alias{n4m_pp_snv_transform}
\alias{n4m_pp_lsnv_transform}
\alias{n4m_pp_rnv_transform}
\alias{n4m_pp_area_transform}
\alias{n4m_pp_normalize_transform}
\alias{n4m_pp_simple_scale_transform}
\alias{n4m_pp_msc_fit_transform}
\alias{n4m_pp_msc_fit_inverse_transform}
\alias{n4m_pp_emsc_fit_transform}
\alias{n4m_pp_baseline_fit_transform}
\alias{n4m_pp_baseline_fit_inverse_transform}
\alias{n4m_pp_savgol_transform}
\alias{n4m_pp_first_derivative_transform}
\alias{n4m_pp_second_derivative_transform}
\alias{n4m_pp_norris_williams_transform}
\alias{n4m_pp_gaussian_transform}
\alias{n4m_pp_to_absorbance_transform}
\alias{n4m_pp_from_absorbance_transform}
\alias{n4m_pp_pct_to_frac_transform}
\alias{n4m_pp_frac_to_pct_transform}
\alias{n4m_pp_kubelka_munk_transform}
\alias{n4m_pp_detrend_transform}
\alias{n4m_pp_asls_transform}
\alias{n4m_pp_airpls_transform}
\alias{n4m_pp_arpls_transform}
\alias{n4m_pp_modpoly_transform}
\alias{n4m_pp_imodpoly_transform}
\alias{n4m_pp_snip_transform}
\alias{n4m_pp_rolling_ball_transform}
\alias{n4m_pp_iasls_transform}
\alias{n4m_pp_beads_transform}
\alias{n4m_pp_wavelet_denoise_transform}
\alias{n4m_pp_wavelet_transform}
\alias{n4m_pp_haar_transform}
\alias{n4m_pp_wavelet_features_transform}
\alias{n4m_pp_osc_fit_transform}
\alias{n4m_pp_epo_fit_transform}
\alias{n4m_pp_epo_fit_inverse_transform}
\alias{n4m_pp_flex_pca_fit_transform}
\alias{n4m_pp_flex_svd_fit_transform}
\alias{n4m_pp_fck_static_transform}
\alias{n4m_pp_crop_transform}
\alias{n4m_pp_resample_transform}
\alias{n4m_pp_resampler_fit_transform}
\alias{n4m_pp_kbins_disc_fit_transform}
\alias{n4m_pp_range_disc_transform}
\alias{n4m_split_kennard_stone_split}
\alias{n4m_split_spxy_split}
\alias{n4m_split_kbins_stratified_split}
\alias{n4m_filter_y_outlier_fit_apply}
\alias{n4m_filter_x_outlier_fit_apply}
\alias{snv}
\alias{lsnv}
\alias{rnv}
\alias{area_normalization}
\alias{normalize}
\alias{simple_scale}
\alias{msc}
\alias{msc_inverse_transform}
\alias{emsc}
\alias{baseline_center}
\alias{baseline_center_inverse_transform}
\alias{savitzky_golay}
\alias{first_derivative}
\alias{second_derivative}
\alias{norris_williams}
\alias{gaussian}
\alias{to_absorbance}
\alias{from_absorbance}
\alias{percent_to_fraction}
\alias{fraction_to_percent}
\alias{kubelka_munk}
\alias{detrend}
\alias{asls}
\alias{airpls}
\alias{arpls}
\alias{modpoly}
\alias{imodpoly}
\alias{snip}
\alias{rolling_ball}
\alias{iasls}
\alias{beads}
\alias{kennard_stone}
\alias{spxy}
\alias{kbins_stratified}
\alias{y_outlier_filter}
\alias{x_outlier_filter}
\alias{wavelet}
\alias{haar}
\alias{wavelet_denoise}
\alias{wavelet_features}
\alias{osc}
\alias{epo}
\alias{epo_inverse_transform}
\alias{flexible_pca}
\alias{flexible_svd}
\alias{fck_static}
\alias{crop_transform}
\alias{resample_transform}
\alias{resampler}
\alias{range_discretizer}
\alias{kbins_discretizer}
\title{Bindings to the N4M C ABI}
\description{
Thin R bindings for the \code{libn4m} C ABI. The package exposes low-level
\code{.Call} wrappers and a small idiomatic R surface for preprocessing,
baseline correction, splitters, filters, and wavelet denoising.
}
\usage{
n4m_abi_version()
n4m_version_string()
n4m_library_path()

snv(X, with_mean = TRUE, with_std = TRUE, ddof = 0L)
lsnv(X, window = 11L, pad_mode = "reflect", constant_value = 0)
rnv(X, with_center = TRUE, with_scale = TRUE, k = 1.4826)
area_normalization(X, method = "sum")
normalize(X, feature_min = -1, feature_max = 1)
simple_scale(X)
msc(X, X_fit = X)
msc_inverse_transform(X, X_fit = X)
emsc(X, X_fit = X, degree = 1L)
baseline_center(X, X_fit = X)
baseline_center_inverse_transform(X, X_fit = X)
savitzky_golay(X, window_length, polyorder, deriv = 0L, delta = 1,
  mode = "mirror", cval = 0)
first_derivative(X, delta = 1, edge_order = 2L)
second_derivative(X, delta = 1, edge_order = 2L)
norris_williams(X, segment = 5L, gap = 5L, derivative_order = 1L,
  delta = 1)
gaussian(X, sigma = 1, order = 0L, mode = "reflect", cval = 0,
  truncate = 4)
to_absorbance(X, is_percent = FALSE, epsilon = 1e-10,
  clip_negative = TRUE)
from_absorbance(X, is_percent = FALSE)
percent_to_fraction(X)
fraction_to_percent(X)
kubelka_munk(X, is_percent = FALSE, epsilon = 1e-10)
detrend(X, polyorder = 1L)
asls(X, lam = 1e6, p = 0.01, max_iter = 50L, tol = 1e-3)
airpls(X, lam = 1e6, max_iter = 50L, tol = 1e-3)
arpls(X, lam = 1e5, max_iter = 50L, tol = 1e-3)
modpoly(X, polyorder = 2L, max_iter = 250L, tol = 1e-3)
imodpoly(X, polyorder = 2L, max_iter = 250L, tol = 1e-3)
snip(X, max_half_window = 20L)
rolling_ball(X, half_window = 20L, smooth_half_window = 0L)
iasls(X, lam = 1e6, p = 0.01, lam_1 = 1e-4, polyorder = 2L,
  diff_order = 2L, max_iter = 50L, tol = 1e-3)
beads(X, lam_0 = 100, lam_1 = 0.5, lam_2 = 0.5, max_iter = 50L,
  tol = 1e-3)
kennard_stone(X, test_size = 0.25)
spxy(X, Y, test_size = 0.25)
kbins_stratified(Y, test_size = 0.25, seed = 0, n_bins = 5L,
  strategy = "uniform")
y_outlier_filter(y, method = "iqr", threshold = NULL, lower_pct = 1,
  upper_pct = 99)
x_outlier_filter(X, method = "mahalanobis", threshold = NULL,
  n_components = min(5L, ncol(X)), contamination = 0.1, seed = 0,
  n_estimators = 100L, max_samples = 256)
wavelet(X, family = "haar", boundary = "periodization")
haar(X)
wavelet_denoise(X, family = "db4", boundary = "periodization",
  level = 3L, threshold_mode = "soft", noise_estimator = "median")
wavelet_features(X, family = "haar", boundary = "periodization",
  max_level = 2L, entropy = "energy")
osc(X, y, n_components = 1L, scale = TRUE)
epo(X, d, scale = TRUE)
epo_inverse_transform(X, d, scale = TRUE)
flexible_pca(X, n_components = 5.0)
flexible_svd(X, n_components = 5.0)
fck_static(X, kernel_size = 5L, filter_orders = c(0.5, 1.0),
  filter_scales = c(1.0, 2.0))
crop_transform(X, start, end)
resample_transform(X, num_samples)
resampler(X, source_wavelengths, target_wavelengths,
  method = "linear", crop_min = 0, crop_max = 0, use_crop = FALSE,
  fill_value = 0, bounds_error = FALSE, extrapolate = FALSE)
range_discretizer(X, edges)
kbins_discretizer(X, n_bins = 5L, strategy = "uniform")
}
\arguments{
\item{X, X_fit}{Numeric matrices.}
\item{y, d}{Numeric response or disturbance vectors.}
\item{Y}{Numeric response vector or matrix.}
\item{with_mean, with_std, clip_negative, is_percent}{Logical options.}
\item{with_center, with_scale, scale, use_crop, bounds_error, extrapolate}{Logical options.}
\item{ddof, degree, window_length, polyorder, deriv, mode, edge_order,
level, max_iter, window, segment, gap, derivative_order, order,
max_half_window, half_window, smooth_half_window, n_components, kernel_size,
start, end, num_samples, n_bins, n_estimators, max_level, diff_order}{Integer or numeric
options passed to \code{libn4m}.}
\item{delta, cval, epsilon, lam, p, tol, test_size, threshold, lower_pct,
upper_pct, constant_value, k, feature_min, feature_max, sigma, truncate,
lam_0, lam_1, lam_2, contamination, seed, max_samples, crop_min, crop_max,
fill_value}{Numeric options passed to \code{libn4m}.}
\item{source_wavelengths, target_wavelengths, edges, filter_orders,
filter_scales}{Numeric vectors passed to \code{libn4m}.}
\item{method, pad_mode, family, boundary, threshold_mode, noise_estimator,
strategy, entropy}{Character options mapped to the corresponding \code{libn4m} enum
values.}
}
\details{
The package vendors the \code{libn4m} source under \code{src/libn4m} and builds
it into the R extension during package installation. \code{N4M_LIB_PATH}
is only used by developer tests that intentionally target an in-tree shared
library.
}
\value{
Metadata helpers return character or integer version information. Operators
return numeric matrices, integer split indices, or logical/integer masks,
depending on the function.
}
\seealso{
\url{https://github.com/GBeurier/nirs4all-methods}
}
