# SPDX-License-Identifier: CeCILL-2.1
#
# Ruby binding around libp4a's `p4a_pls_fit_simple` C ABI helper.
# Uses Fiddle from the Ruby standard library — no extra gems required.
# Parity-gated against the shared cross-binding fixture
# `bindings/js/test/parity_fixture.json` at machine epsilon.

require "fiddle"
require "fiddle/import"

module Pls4all
  # Internal: Fiddle::Importer wired to libp4a. Public callers should
  # use the module-level `version`, `abi_version` and `pls_fit` methods.
  module Lib
    extend Fiddle::Importer

    candidates = []
    candidates << ENV["PLS4ALL_LIB"] if ENV["PLS4ALL_LIB"] && !ENV["PLS4ALL_LIB"].empty?
    candidates += [
      "libp4a.so",
      "libp4a.dylib",
      "p4a.dll",
    ]
    # Walk up from this file to find a CMake dev-release build.
    here = File.expand_path(File.dirname(__FILE__))
    until here == "/" || here.empty?
      %w[libp4a.so libp4a.dylib].each do |name|
        path = File.join(here, "build/dev-release/cpp/src", name)
        candidates << path if File.exist?(path)
      end
      parent = File.dirname(here)
      break if parent == here
      here = parent
    end

    loaded = false
    candidates.each do |path|
      next if path.nil? || path.empty?
      begin
        dlload path
        loaded = true
        break
      rescue Fiddle::DLError
        next
      end
    end
    raise LoadError, "could not load libp4a (tried: #{candidates.inspect})" unless loaded

    extern "const char* p4a_get_version_string(void)"
    extern "unsigned int p4a_get_abi_version_major(void)"
    extern "unsigned int p4a_get_abi_version_minor(void)"
    extern "unsigned int p4a_get_abi_version_patch(void)"
    extern "int p4a_pls_fit_simple(const void*, const void*, int, int, int, int, " \
           "void*, void*, void*, void*)"
  end
  private_constant :Lib

  # Returns the libp4a runtime version, e.g. "0.91.0+abi.1.13.0".
  def self.version
    ptr = Lib.p4a_get_version_string
    return "" if ptr.null?
    ptr.to_s
  end

  # Returns [major, minor, patch] from the libp4a ABI.
  def self.abi_version
    [Lib.p4a_get_abi_version_major,
     Lib.p4a_get_abi_version_minor,
     Lib.p4a_get_abi_version_patch]
  end

  # Bundle of arrays returned by {Pls4all.pls_fit}.
  FitResult = Struct.new(
    :n, :p, :q, :n_components,
    :coefficients,  # length p*q (row-major)
    :x_mean,        # length p
    :y_mean,        # length q
    :predictions,   # length n*q (row-major)
  )

  # Fit a SIMPLS PLS regression on row-major matrices x and y.
  #
  # x must be an Array of Float of length n*p; y must have length n*q.
  # Returns a FitResult. Raises ArgumentError on shape mismatch and
  # RuntimeError on a non-zero status from libp4a.
  def self.pls_fit(x, y, n, p, q, n_components)
    raise ArgumentError, "x length #{x.length} != n*p (#{n * p})" if x.length != n * p
    raise ArgumentError, "y length #{y.length} != n*q (#{n * q})" if y.length != n * q
    raise ArgumentError, "n_components must be >= 1" if n_components < 1

    x_buf = x.pack("E*")                            # IEEE 754 little-endian doubles
    y_buf = y.pack("E*")
    coefs = ([0.0] * (p * q)).pack("E*")
    xm = ([0.0] * p).pack("E*")
    ym = ([0.0] * q).pack("E*")
    preds = ([0.0] * (n * q)).pack("E*")

    status = Lib.p4a_pls_fit_simple(
      x_buf, y_buf, n, p, q, n_components,
      coefs, xm, ym, preds
    )
    raise "p4a_pls_fit_simple failed with status #{status}" if status != 0

    FitResult.new(
      n, p, q, n_components,
      coefs.unpack("E*"),
      xm.unpack("E*"),
      ym.unpack("E*"),
      preds.unpack("E*")
    )
  end
end
