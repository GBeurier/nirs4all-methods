# SPDX-License-Identifier: CeCILL-2.1
#
# Idiomatic Ruby tier-2 wrapper around Pls4all.pls_fit. Provides a
# Rumale-style class with .fit / .predict / .score methods.
#
# Example:
#
#   require "pls4all/sklearn"
#   m = Pls4all::PLSRegression.new(n_components: 5)
#   m.fit(x_flat, y_flat, n, p)
#   preds = m.predict(x_new_flat, n_new, p)
#   r2 = m.score(x_test_flat, y_test_flat, n_test, p)

require "pls4all"

module Pls4all
  # PLS regression estimator wrapping Pls4all.pls_fit. Inputs follow
  # the Rumale convention of row-major float arrays since Ruby has no
  # canonical numeric-matrix type in stdlib.
  class PLSRegression
    attr_reader :n_components, :coefficients, :x_mean, :y_mean,
                :n_features_in, :n_targets

    def initialize(n_components: 2)
      @n_components = n_components
      @coefficients = nil
      @x_mean = nil
      @y_mean = nil
      @n_features_in = 0
      @n_targets = 0
    end

    # Train the model on (X, y).
    #   x    — flat Array<Float> of length n*p (row-major).
    #   y    — flat Array<Float> of length n*q.
    #   n,p  — dimensions of X.
    def fit(x, y, n, p)
      raise ArgumentError, "n and p must be positive" if n <= 0 || p <= 0
      raise ArgumentError, "x.size (#{x.size}) must equal n*p (#{n * p})" if x.size != n * p
      raise ArgumentError, "y.size (#{y.size}) must be a positive multiple of n (#{n})" if y.size == 0 || y.size % n != 0
      q = y.size / n
      res = Pls4all.pls_fit(x, y, n, p, q, @n_components)
      @coefficients = res[:coefficients].dup
      @x_mean = res[:x_mean].dup
      @y_mean = res[:y_mean].dup
      @n_features_in = p
      @n_targets = q
      self
    end

    # Predict on new X. Follows the C ABI convention
    # (cpp/src/core/model.cpp::fill_prediction):
    #   Y_pred = (X - x_mean) @ coef + y_mean
    def predict(x, n_pred, p)
      raise "Model is not fitted" if @coefficients.nil?
      raise ArgumentError, "feature count mismatch" if p != @n_features_in
      q = @n_targets
      preds = Array.new(n_pred * q, 0.0)
      n_pred.times do |i|
        q.times do |t|
          sum = @y_mean[t]
          p.times do |j|
            centered = x[i * p + j] - @x_mean[j]
            sum += centered * @coefficients[j * q + t]
          end
          preds[i * q + t] = sum
        end
      end
      preds
    end

    # R² coefficient of determination, uniform-averaged across targets.
    def score(x, y, n, p)
      preds = predict(x, n, p)
      q = @n_targets
      r2_sum = 0.0
      q.times do |t|
        mean = (0...n).sum { |i| y[i * q + t] } / n.to_f
        ss_res = 0.0
        ss_tot = 0.0
        n.times do |i|
          dr = preds[i * q + t] - y[i * q + t]
          dt = y[i * q + t] - mean
          ss_res += dr * dr
          ss_tot += dt * dt
        end
        next if ss_tot == 0.0 || ss_tot.nan?
        r2_sum += 1 - ss_res / ss_tot
      end
      r2_sum / q.to_f
    end
  end
end
