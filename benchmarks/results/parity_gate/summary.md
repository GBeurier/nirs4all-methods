# Parity gate report

Host: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`
pls4all: `0.1.0+abi.1.9.0`
Python: `3.10.12`
NumPy: `2.2.6`

## Pass quality breakdown

| Quality | Count | Definition |
|---------|------:|------------|
| tight    | 2 | `rmse_rel < 30% of tolerance` — high-confidence parity. |
| moderate | 0 | 30-60% of tolerance — real agreement. |
| loose    | 0 | 60-90% of tolerance — algorithmic divergence likely; passes with margin. |
| **weak** | **0** | **>= 90% of tolerance** — passes barely; tolerance widened to accept stochastic-RNG or algorithmic-convention divergence. **Treat as smoke check, not bit parity.** |

## All rows

Each method is compared against a Python reference and an R reference. Methods without a widely installable external reference are flagged `paper-only` or `(none)` in the lib column.

| Method | Description | Reference (lang / lib / version) | Parity | Quality | RMSE rel | Tolerance | Status |
|--------|-------------|----------------------------------|--------|---------|----------|-----------|--------|
| `rosa` | ROSA — Response-Oriented Sequential Alternation (§19) | python / `numpy` 2.2.6 | ✓ | tight | 5.266e-16 | 1e-06 | ok |
| `rosa` | ROSA — Response-Oriented Sequential Alternation (§19) | R / `multiblock` 0.8.10 | ✓ | tight | 1.362e-15 | 1e-06 | ok |
