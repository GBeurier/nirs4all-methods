# chemometrics4all — Python binding

`ctypes`-based binding for [`libc4a`](../../cpp/), the C ABI of
chemometrics4all.

## Tiers

* **Tier 1 (`chemometrics4all._ffi`)** — raw `ctypes.CDLL` access to the 402
  `c4a_*` symbols. Loads the shared library from
  `CHEMOMETRICS4ALL_LIB_PATH`, the wheel's bundled `lib/` directory, or the
  development build tree (`build/dev-debug/cpp/src/`).

* **Tier 2 (`chemometrics4all.sklearn`)** — hand-written scikit-learn–style
  wrappers (`fit`, `transform`, `fit_transform`) over the 20 most-used
  operators:

  | Family           | Classes                                                              |
  | ---------------- | -------------------------------------------------------------------- |
  | Preprocessings   | `SNV`, `LSNV`, `RNV`, `MSC`, `EMSC`, `SavitzkyGolay`,                |
  |                  | `FirstDerivative`, `SecondDerivative`, `ToAbsorbance`, `KubelkaMunk` |
  | Baseline         | `Detrend`, `AsLS`, `AirPLS`                                          |
  | Splitters        | `KennardStoneSplitter`, `SPXYSplitter`, `KBinsStratifiedSplitter`    |
  | Filters          | `YOutlierFilter`, `XOutlierFilter`                                   |
  | Misc             | `WaveletDenoise`, `GaussianAdditiveNoise`                            |

## binding_parity gate

`tests/test_binding_parity.py` is the Gate 1 contract: for every covered
operator, the binding output must match the libc4a reference fixture in
`parity/fixtures/<op>_v1.json` to within `tolerance=1e-6`.

## Install + test

```bash
cd bindings/python
pip install -e .
CHEMOMETRICS4ALL_LIB_PATH=/path/to/libc4a.so.1.9.0 \
    python -m pytest tests/test_binding_parity.py -v
```
