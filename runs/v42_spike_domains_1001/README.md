# arm d grad-spike steps vs their data (de, 2026-10-01)

`arm_d_steps.txt` is the verbatim output of

    python3 scripts/spike_step_domains.py --ckpt ckpt_v42_arch_d_0930.pt.step2000 --steps <s>:<s+1>

run on the pod, CPU-only, beside the live `v42_gate_1001` job, for s in 1600, 1700, 1800, 1900.
Each run prints `VALIDATED rebuilt prefix at step 2000 == row_cursor`, which is the tool's
token-free identity check that the rebuilt plan column is the run's own.

The gnorm at those four steps, read from the pod-only `runs/v42_arch_d_0930.log`:
1600 = 0.07, 1700 = 10064.31, 1800 = 2948.89, 1900 = 0.07. So two spike steps and two calm
steps, both modes of the bimodal distribution.

The tool resolves a step to DOMAIN shares and pool-row ranges. It does not map pool rows back
to source documents, so this artifact can rule out a domain-composition cause and cannot rule
out a single pathological document.

The whole point of keeping the output here: `facts/v41.json` cites it, and
`facts_well_formed` refuses a source path the repo does not hold.
