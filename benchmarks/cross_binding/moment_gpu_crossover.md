# Moment GPU Crossover

Synthetic score-only `n4m.sweep_run` timing for CPU vs one-GPU CUDA moment screens. `recommended_backend` is source-free and depends only on shape/head timing, not dataset identity.

| head | shape | CPU ms | CUDA default ms | CUDA many-batched ms | best CUDA profile | best CUDA vs CPU | many-batched vs default | recommended |
|---|---:|---:|---:|---:|---|---:|---:|---|
| pls | 256x1024 | 227.389 | 115.589 | 124.100 | default | 1.97x | 0.93x | cuda:default |
| pls | 260x48 | 0.463 | 8.671 | 8.815 | default | 0.05x | 0.98x | cpu |
| pls | 260x256 | 11.604 | 13.204 | 15.304 | default | 0.88x | 0.86x | cpu |
| pls | 512x512 | 84.181 | 39.134 | 41.258 | default | 2.15x | 0.95x | cuda:default |
| ridge | 256x1024 | 168.339 | 67.007 |  | default | 2.51x |  | cuda:default |
| ridge | 260x48 | 1.396 | 5.415 |  | default | 0.26x |  | cpu |
| ridge | 260x256 | 75.546 | 58.862 |  | default | 1.28x |  | cuda:default |
| ridge | 512x512 | 684.388 | 518.294 |  | default | 1.32x |  | cuda:default |
