# 0830v1 gates

The controller session (aupai-fb) drives these in order. A gate opens only when its
evidence exists as an artifact — not when someone reports it done. Update the status
column in place; do not add a second copy of this table anywhere.

Architecture is fixed for this round: KDA (9 layers) + full causal gated MLA (3 layers,
latent=d/4) + AttnRes, NoPE throughout. Commit b3cad87. Changing it reopens G3.

| gate | opens when | evidence | owner | status |
|---|---|---|---|---|
| G0 harness green | `scripts/harness.py check` exits 0 AND `--selftest` exits 0 | both exit codes | aupai-de | GREEN 2026-08-30 (73cef7b, 16/16 selftest) |
| G1 metric panel | the 200M resolution panel is committed and frozen | `docs/lessons/base_eval_at_200m.md` + 12 `facts/base_eval.json` entries | lessons-b0 | GREEN 2026-08-30, frozen; reporting the panel is now mandatory per run |
| G1b zh minimal pairs | the eval set built to `be.minimal_pair_rules`, n>=277 | the set + its build script | aupai-3b | open; needed before the panel's minimal-pair row can be read, not before G3 |
| G2 profile | step time split by source, percentages summing to ~100 | `facts/efficiency.json` | lessons-e1 | open work; also selects the kernel target |
| G3 first run | 0.2b budget point trained on the new arch | `ckpt_*`, one `experiments.jsonl` row, one `score_matrix.jsonl` row | aupai-fb, aupai-de | blocked: GPU 1-7 held by `0830v1_repeat4` (old sliding-window arch) |
| G4 scaling curve | all six `mix_scale_*` points scored | six score-matrix rows, fitted E + B/D^beta with residuals | aupai-fb | waits on G3 |

Kernel work (tilerl-bench-harness-plan) runs alongside and gates on G2 for target
selection, not on G0. It lands only through the five correctness gates in its brief:
fp64 reference no worse than the kernel it replaces, bit-exact reruns, 200-step
step-by-step loss comparison, `test_arch_compat` coverage, and a working fallback.

Two rules this round does not bend:

- No GPU pretrain while `harness check` is red. A permanent red is the same as no signal.
- No number enters the repo without its measurement config.
