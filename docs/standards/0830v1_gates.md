# 0830v1 gates

The controller session (aupai-fb) drives these in order. A gate opens only when its
evidence exists as an artifact — not when someone reports it done. Update the status
column in place; do not add a second copy of this table anywhere.

Architecture is fixed for this round: KDA (9 layers) + full causal gated MLA (3 layers,
latent=d/4) + AttnRes, NoPE throughout. Commit b3cad87. Changing it reopens G3.

| gate | opens when | evidence | owner | status |
|---|---|---|---|---|
| G0 harness green | `scripts/harness.py check` exits 0 AND `--selftest` exits 0 | both exit codes | aupai-de, lessons-b0 | blocked: facts_well_formed, lessons_have_frontmatter |
| G1 metric panel | the 200M resolution panel is committed and frozen | `docs/lessons/*.md` + `facts/base_eval.json` entries with config | lessons-b0 | open work |
| G2 profile | step time split by source, percentages summing to ~100 | `facts/efficiency.json` | lessons-e1 | open work |
| G3 first run | 0.2b budget point trained on the new arch | `ckpt_*`, one `experiments.jsonl` row, one `score_matrix.jsonl` row | aupai-fb, aupai-de | waits on G0, G1 |
| G4 scaling curve | all six `mix_scale_*` points scored | six score-matrix rows, fitted E + B/D^beta with residuals | aupai-fb | waits on G3 |

Kernel work (tilerl-bench-harness-plan) runs alongside and gates on G2 for target
selection, not on G0. It lands only through the five correctness gates in its brief:
fp64 reference no worse than the kernel it replaces, bit-exact reruns, 200-step
step-by-step loss comparison, `test_arch_compat` coverage, and a working fallback.

Two rules this round does not bend:

- No GPU pretrain while `harness check` is red. A permanent red is the same as no signal.
- No number enters the repo without its measurement config.
