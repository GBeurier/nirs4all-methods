# SPDX-License-Identifier: CECILL-2.1
#
# Gate 1 (binding_parity) tests for the chemometrics4all R binding.
#
# For each of the 15 tier-2 operators, we run every case packaged in the
# corresponding `parity/fixtures/<op>_v1.json` and check that the
# R-binding output matches the libc4a-reference output bit-for-bit within
# the binding-parity tolerance (`1e-6` by default).
#
# Tests can be skipped at session level by setting CHEMOMETRICS4ALL_SKIP_PARITY=1.

skip_if_no_lib <- function() {
  if (identical(Sys.getenv("CHEMOMETRICS4ALL_SKIP_PARITY"), "1")) {
    testthat::skip("CHEMOMETRICS4ALL_SKIP_PARITY=1 set")
  }
  abi <- tryCatch(c4a_abi_version(), error = function(e) NULL)
  if (is.null(abi)) {
    testthat::skip("libc4a is not loaded")
  }
}

# ---- ABI smoke ----

test_that("c4a_abi_version returns 1.x.y", {
  skip_if_no_lib()
  abi <- c4a_abi_version()
  expect_length(abi, 3L)
  expect_gte(abi[1], 1L)
})

# ---- Phase 2/3 preprocessing ----

test_that("SNV: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("snv")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- snv(X,
                with_mean = case$params$with_mean,
                with_std  = case$params$with_std,
                ddof      = case$params$ddof)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref, label = paste0("snv/", case$name))
  }
})

test_that("MSC: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("msc")
  X_fit <- fixture_fit_matrix(fx)
  X     <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    variant <- case$params$variant
    if (is.null(variant) || identical(variant, "transform")) {
      pred <- msc(X, X_fit = X_fit)
      ref  <- case_output_matrix(case, fx)
      expect_binding_parity(pred, ref, label = paste0("msc/", case$name))
    } else {
      # Inverse-transform case: skip — not exposed by the tier-2 R API.
      testthat::skip(sprintf("msc/%s requires inverse_transform (not exposed)",
                             case$name))
    }
  }
})

test_that("EMSC: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("emsc")
  X_fit <- fixture_fit_matrix(fx)
  X     <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- emsc(X, X_fit = X_fit, degree = case$params$degree)
    ref  <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref, label = paste0("emsc/", case$name))
  }
})

test_that("SavitzkyGolay: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("savgol")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- savitzky_golay(X,
                           window_length = case$params$window_length,
                           polyorder     = case$params$polyorder,
                           deriv         = case$params$deriv,
                           delta         = case$params$delta,
                           mode          = case$params$mode,
                           cval          = case$params$cval)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref, label = paste0("savgol/", case$name))
  }
})

test_that("FirstDerivative: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("first_derivative")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- first_derivative(X,
                             delta      = case$params$delta,
                             edge_order = case$params$edge_order)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref,
                          label = paste0("first_derivative/", case$name))
  }
})

test_that("SecondDerivative: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("second_derivative")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- second_derivative(X,
                              delta      = case$params$delta,
                              edge_order = case$params$edge_order)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref,
                          label = paste0("second_derivative/", case$name))
  }
})

test_that("ToAbsorbance: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("to_absorbance")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    # Generator convention: `is_percent=TRUE` cases run on `R * 100.0`. The
    # 100.0 multiplier is exactly representable in IEEE 754, so the scale
    # is bit-exact.
    Xc <- if (isTRUE(case$params$is_percent)) X * 100.0 else X
    pred <- to_absorbance(Xc,
                          is_percent    = case$params$is_percent,
                          epsilon       = case$params$epsilon,
                          clip_negative = case$params$clip_negative)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref,
                          label = paste0("to_absorbance/", case$name))
  }
})

test_that("KubelkaMunk: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("kubelka_munk")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    Xc <- if (isTRUE(case$params$is_percent)) X * 100.0 else X
    pred <- kubelka_munk(Xc,
                         is_percent = case$params$is_percent,
                         epsilon    = case$params$epsilon)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref,
                          label = paste0("kubelka_munk/", case$name))
  }
})

# ---- Phase 5 baseline ----

test_that("Detrend: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("detrend")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- detrend(X, polyorder = case$params$polyorder)
    ref  <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref,
                          label = paste0("detrend/", case$name))
  }
})

test_that("AsLS: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("asls")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- asls(X,
                 lam      = case$params$lam,
                 p        = case$params$p,
                 max_iter = case$params$max_iter,
                 tol      = case$params$tol)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref, label = paste0("asls/", case$name))
  }
})

test_that("AirPLS: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("airpls")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- airpls(X,
                   lam      = case$params$lam,
                   max_iter = case$params$max_iter,
                   tol      = case$params$tol)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref, label = paste0("airpls/", case$name))
  }
})

# ---- Phase 11 splitters ----

test_that("KennardStone: every case matches libc4a indices", {
  skip_if_no_lib()
  fx <- load_fixture("split_kennard_stone")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    out <- kennard_stone(X, test_size = case$params$test_size)
    # Fixture indices are 0-based; our binding returns 1-based.
    expect_equal(out$train_idx - 1L, unlist(case$params$train_idx),
                 info = paste0("kennard_stone/", case$name, " train"))
    expect_equal(out$test_idx  - 1L, unlist(case$params$test_idx),
                 info = paste0("kennard_stone/", case$name, " test"))
  }
})

test_that("SPXY: every case matches libc4a indices", {
  skip_if_no_lib()
  fx <- load_fixture("split_spxy")
  X  <- fixture_input_matrix(fx)
  Y  <- fixture_fit_matrix(fx)
  for (case in fx$cases) {
    out <- spxy(X, Y, test_size = case$params$test_size)
    expect_equal(out$train_idx - 1L, unlist(case$params$train_idx),
                 info = paste0("spxy/", case$name, " train"))
    expect_equal(out$test_idx  - 1L, unlist(case$params$test_idx),
                 info = paste0("spxy/", case$name, " test"))
  }
})

# ---- Phase 12 filters ----

test_that("YOutlierFilter (IQR): every case matches libc4a mask", {
  skip_if_no_lib()
  fx <- load_fixture("filter_y_outlier_iqr")
  y  <- hex_to_double(unlist(fx$y_hex))
  for (case in fx$cases) {
    out <- y_outlier_filter(y,
                            method    = case$params$method,
                            threshold = case$params$threshold,
                            lower_pct = case$params$lower_percentile,
                            upper_pct = case$params$upper_percentile)
    expected <- as.logical(unlist(case$expected_mask) != 0)
    expect_equal(out$mask, expected,
                 info = paste0("y_outlier/", case$name, " mask"))
    es <- case$expected_stats
    expect_equal(out$n_samples,  as.numeric(es$n_samples),
                 info = paste0("y_outlier/", case$name, " n_samples"))
    expect_equal(out$n_kept,     as.numeric(es$n_kept),
                 info = paste0("y_outlier/", case$name, " n_kept"))
    expect_equal(out$n_excluded, as.numeric(es$n_excluded),
                 info = paste0("y_outlier/", case$name, " n_excluded"))
    expected_rate <- hex_to_double(es$exclusion_rate)
    expect_equal(out$exclusion_rate, expected_rate, tolerance = 1e-12,
                 info = paste0("y_outlier/", case$name, " rate"))
  }
})

# ---- Phase 6 wavelets ----

test_that("WaveletDenoise: every case matches libc4a", {
  skip_if_no_lib()
  fx <- load_fixture("wavelet_denoise")
  X <- fixture_input_matrix(fx)
  for (case in fx$cases) {
    pred <- wavelet_denoise(X,
                            family          = case$params$family,
                            boundary        = case$params$mode,
                            level           = case$params$level,
                            threshold_mode  = case$params$threshold_mode,
                            noise_estimator = case$params$noise_estimator)
    ref <- case_output_matrix(case, fx)
    expect_binding_parity(pred, ref,
                          label = paste0("wavelet_denoise/", case$name))
  }
})
