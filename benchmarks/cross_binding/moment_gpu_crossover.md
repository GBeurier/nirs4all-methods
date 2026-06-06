# Moment GPU Crossover

Synthetic score-only `n4m.sweep_run` timing for CPU vs one-GPU CUDA moment screens. `recommended_backend` is source-free and depends only on shape/head timing, not dataset identity.

| head | shape | CPU ms | CUDA default ms | CUDA many-batched ms | best CUDA profile | best CUDA vs CPU | many-batched vs default | recommended |
|---|---:|---:|---:|---:|---|---:|---:|---|
| pls | 256x1024 | 542.701 | 312.121 | 740.471 | default | 1.74x | 0.42x | cuda:default |
| pls | 260x48 | 1.128 | 57.397 | 32.487 | many_batched | 0.03x | 1.77x | cpu |
| pls | 260x256 | 34.607 | 67.060 | 55.000 | many_batched | 0.63x | 1.22x | cpu |
| pls | 512x512 | 112.756 | 224.345 | 183.722 | many_batched | 0.61x | 1.22x | cpu |
| ridge | 256x1024 | 263.826 | 375.202 |  | default | 0.70x |  | cpu |
| ridge | 260x48 | 4.912 | 117.491 |  | default | 0.04x |  | cpu |
| ridge | 260x256 | 378.230 | 334.641 |  | default | 1.13x |  | cuda:default |
| ridge | 512x512 | 1168.275 | 1558.649 |  | default | 0.75x |  | cpu |
