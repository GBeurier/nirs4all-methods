# Installed static C consumer

This standalone C11 consumer qualifies the installed static dependency closure.
It uses only `find_package(N4m)` and `n4m::n4m_c_static`, without repository source
includes or direct archive paths. The C++ language is enabled for the static
implementation's link driver; the consumer itself is compiled as C.

Both analysis engines must be enabled. The test checks a balanced random-
intercept REML fit against analytic coefficients, variances and covariance,
then an exact categorical XOR fANOVA and constant-response refusal.

From the repository root, after building the `analysis-all` artifacts:

```bash
cmake --install build/analysis-all --prefix /tmp/n4m-analysis-static-install
cd cpp/tests/installed_component_analysis
N4M_INSTALL_PREFIX=/tmp/n4m-analysis-static-install cmake --preset installed-static
cmake --build --preset installed-static
ctest --preset installed-static
```

The consumer deliberately requires the engines to work; backend availability
alone cannot satisfy the test.
