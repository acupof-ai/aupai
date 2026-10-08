# mlx-lm attribution

This local v42 inference backend integrates the installed `mlx-lm` package.

- Project: MLX LM
- Repository: https://github.com/ml-explore/mlx-lm
- Installed version: 0.32.0
- License: MIT

Reused public interfaces and design patterns:

- `mlx_lm.generate.generate_step` for asynchronous token generation.
- `mlx_lm.models.cache.KVCache` for incremental key/value storage.
- The `model(inputs, cache=...)` compatibility contract.
- The quantized SwitchGLU design based on `mx.gather_qmm`.
- The n-gram token-history cache pattern.

The v42-specific implementation remains in this repository. It includes
HyperConnection, compressed sparse attention, routed MoE rules, SSD-backed
Engram and checkpoint conversion.
