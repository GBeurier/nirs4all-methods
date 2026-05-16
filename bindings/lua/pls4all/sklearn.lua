-- SPDX-License-Identifier: CeCILL-2.1
--
-- Idiomatic LuaJIT tier-2 wrapper around pls4all.fit. Provides a
-- torch.nn-style class with :fit, :predict, :score methods.
--
-- Usage:
--   local pls4all = require("pls4all")
--   require("pls4all.sklearn")  -- attaches PLSRegression to the module
--   local m = pls4all.PLSRegression(5)
--   m:fit(X, y, n, p)
--   local preds = m:predict(Xnew, n_new, p)
--   local r2 = m:score(Xtest, ytest, n_test, p)
--
-- X / y / preds are flat numeric tables (row-major). For tensor
-- integration with torch, the user can wrap fit/predict in tensor
-- conversions externally.

local pls4all = require("pls4all")

local M = {}
M.__index = M

function pls4all.PLSRegression(n_components)
    local obj = setmetatable({}, M)
    obj.n_components = n_components or 2
    obj.coefficients = nil
    obj.x_mean = nil
    obj.y_mean = nil
    obj.n_features_in = 0
    obj.n_targets = 0
    return obj
end

function M:fit(x, y, n, p)
    if n <= 0 or p <= 0 then
        error("n and p must be positive")
    end
    if #x ~= n * p then
        error(string.format("len(x)=%d must equal n*p=%d", #x, n * p))
    end
    if #y == 0 or #y % n ~= 0 then
        error(string.format("len(y)=%d must be a positive multiple of n=%d", #y, n))
    end
    local q = math.floor(#y / n)
    local res = pls4all.pls_fit(x, y, n, p, q, self.n_components)
    self.coefficients = res.coefficients
    self.x_mean = res.x_mean
    self.y_mean = res.y_mean
    self.n_features_in = p
    self.n_targets = q
    return self
end

function M:predict(x, n_pred, p)
    if not self.coefficients then
        error("Model is not fitted")
    end
    if p ~= self.n_features_in then
        error(string.format(
            "X has %d features but model was fitted with %d",
            p, self.n_features_in))
    end
    local q = self.n_targets
    local preds = {}
    for i = 0, n_pred - 1 do
        for t = 0, q - 1 do
            local sum = self.y_mean[t + 1]
            for j = 0, p - 1 do
                local centered = x[i * p + j + 1] - self.x_mean[j + 1]
                sum = sum + centered * self.coefficients[j * q + t + 1]
            end
            preds[i * q + t + 1] = sum
        end
    end
    return preds
end

function M:score(x, y, n, p)
    local preds = self:predict(x, n, p)
    local q = self.n_targets
    local r2_sum = 0.0
    for t = 0, q - 1 do
        local mean = 0
        for i = 0, n - 1 do
            mean = mean + y[i * q + t + 1]
        end
        mean = mean / n
        local ss_res, ss_tot = 0, 0
        for i = 0, n - 1 do
            local dr = preds[i * q + t + 1] - y[i * q + t + 1]
            local dt = y[i * q + t + 1] - mean
            ss_res = ss_res + dr * dr
            ss_tot = ss_tot + dt * dt
        end
        if ss_tot ~= 0 then
            r2_sum = r2_sum + 1 - ss_res / ss_tot
        end
    end
    return r2_sum / q
end

return pls4all.PLSRegression
