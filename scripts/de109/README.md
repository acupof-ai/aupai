# de-109: materialized entries + flash window, split softmax

CSA2's single softmax runs over `[selected entries ; SWA window keys]`. de-108 flashed
both branches but needed a ragged gather and one flash call per indexer-head group, which
was 2.3-3.2x slower than materialized. de-109 keeps the **entries branch dense and eager**
(it is only `B,H,T,NB`, `NB = T/m = 512`) and moves **only the window branch** to flash's
native causal sliding window — the call `#236` already runs on the PureSWA layers.

## Shape of the cost

The materialized tensor that pressures B8 is the window score matrix `q@k^T` of shape
`B,H,T,T` (the `full` tensor in `_forward_csa2`). The selected-entry score tensor is
`B,H,T,NB` with `NB = T/m = 512`, 1/8 the width; it is not the cost. Flashing the window
alone therefore captures the memory/speed win with no gather and one flash call per layer
over all heads.

## Split softmax

Both branches return an output and a log-sum-exp at fp32:

- entries (eager): `le = logsumexp(se)` over selected entries, `oe = (pe*ste) @ vc`,
  with `pe = softmax(se)` and `ste` the straight-through selector;
- window (flash): `ow, lw = _WindowSWAFlash.apply(...)`, causal,
  `window_size=(n_win-1, 0)`, over the document `cu`.

Combine with `m = max(le, lw)`, `a_e = exp(le-m)`, `a_w = exp(lw-m)`,
`c_e = a_e/(a_e+a_w)`, `c_w = a_w/(a_e+a_w)`, `y = c_e*oe + c_w*ow`.

Backward: `_WindowSWAFlash.backward` receives autograd's total `dout` and `dlse` for the
window output and LSE, and passes `dlse` into `flash_attn.cute._flash_attn_bwd`. The
kernel subtracts the within-branch `g.ow` term itself, and autograd's
`dlse = c_w*(g.ow - g.y)` is exactly the cross-branch term (the de-108 proof; isolated
two-branch kernel test dkw 1.9e-6). `dlse=None` is exact only for standalone attention.

The entries branch stays fully differentiable through `oe = (pe*ste) @ vc`, so the STE
indexer gradient needs no custom bridge: autograd reaches `soft_sel` exactly as on the
materialized path. The entries output MUST be this matmul; materializing `pe*vc` as a
`B,H,T,NB,D` tensor OOMs at B4.

## Scripts

- `check_cpu_parity.py` — float64 materialized vs split, fwd + every q/k/v/x and parameter
  gradient, fixed multi-doc `cu=[0,7,12,24]`; max diff <= 1e-15.
- `check_default_identical.py` — with `csa2_win_flash` unset the layer is byte-identical
  to origin/main; pins the fixed-seed SHA256 of y + all gradients.
- `check_gpu_parity.py` — H20 bf16 fwd/bwd parity at T=512 B=2 and B4/B8 peak GiB + tok/s
  of 10 stacked CSA2 layers at T=4096. Measured 2026-09-11:
  fact `facts/v41.json#v41.de109_win_flash_parity_speed_0911`.

## Gate

`cfg.csa2_win_flash` (default False), CLI `--csa2_win_flash`, in harness `_FROZEN_KEYS`
and `data/mix_scale_run_config.json`. Default path is the materialized launch line.
