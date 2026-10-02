# aupai

aupai is the V4.1 coding/math model: a 24-layer v42 stack at 614M active params, trained from
scratch on a 30B-token UltraData gate mix, targeting **HumanEval pass@1 ≥ 30%**. Full rules
are in `AGENTS.md`.

## Model

The gate stack is **v42** (`--arch v42`): the v41f V4.1 implementation, `v41f/lm.V42LM` at
preset `v41f.config.v42_s24`, trained by `v41f/optim.py` at one base lr. The earlier 12-layer
CED line (`v41_ced_0923`) is retired; its code still loads old checkpoints.

| | |
|---|---|
| size | **3,256,557,328 total / 614,145,808 active** (meta-device count at the gate shape) |
| shape | 24 layers, d=1024, 16 heads, head_dim 256, rope_head_dim 64, vocab 32,768 |
| attention | per-layer compression ratios `(0,0,2×10,1×12)`; indexer top-k 512; KV source layers (2,8,12) |
| MoE | 64 routed experts, top-8 + 1 shared, `moe_inter_dim` 640, `torch._grouped_mm` |
| numerics | bf16, `torch.compile`, Liger mHC + RMSNorm, attention logit softcap 50, `attn_impl=fused`, `rope_impl=real` |
| vocabulary | 32,768 slots, `<eos>=1`, `[NUM]=32767`; not tracked, build with `python scripts/build_gate_tokenizer.py` |

Config: `v41f/config.py`; the entry point and flag resolution are in `train.py`; facts in
`facts/v41.json`.

## The gate run

**v42_gate_1001r** — 30B tokens, 38,146 steps, seq 4096, **world 16** across two nodes
(pod 192.168.29.37 ranks 0-7, h20b 192.168.29.36 ranks 8-15). Micro-batch 4 × accum 3 ×
16 ranks × 4096 = **786,432 tokens/step**, identical to the world-8 MB4/accum6 it resumed
from, so the step schedule, warmdown and anneal are unchanged.

```bash
# h20b first (torchrun rendezvous waits on the master), then the pod
bash runs/v42_w16_node1.sh <interrupt-ckpt-basename>                # copied to h20b's /root
bash runs/v42_w16_node0.sh ckpt_v42_gate_1001r.pt.interrupt.stepN   # on the pod
```

Cross-node NCCL needs the vendor topology file at `/var/run/nvidia-topologyd/` — without it
every collective picks a GID-less NIC or finds no local path and hangs. Both launchers carry
the `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` exports and the reason.

The divergence watchdog runs beside the run, read-only, and reports rather than restarts:

```bash
python3 scripts/ced_diverge_watch.py --log runs/v42_gate_1001r.log --name v42_gate_1001r \
  --mem_thresh_gib 88 --g_thresh 1000000000
```

v42's gradient norm is bimodally distributed with a healthy upper mode reaching 2e4
(`facts/v41.json#v41.v42_gnorm_bimodality_not_data_1001`), so the v41-tuned `gnorm > 10` rule
false-kills this stack; the thresholds above are the v42 values.

## Quick start

```bash
uv sync
python scripts/harness.py install-hooks   # ruff E9/F, blob guard, harness check
python scripts/test_arch_compat.py        # CPU fwd/bwd, checkpoint round-trip, doc-mask
python scripts/harness.py check           # repo invariants; CI runs the same
```

A 2,000-document sample (`data/corpus/sample/`, `data/mix_sample.json`) exercises the pipeline.

## Data

The mix file is the **only data path**: `data/mix_v41_gate.json` gives each domain its target
weight, epoch cap, and anneal weight; a missing domain errors, no fallback. Build with
`python datagen/build_corpus.py --domain <d> --source <s>`, decontaminate HumanEval/MBPP via
`python scripts/filter_gate_domains.py --domains <list>` (engine `filters/decontam_ngram.py`,
13-gram), pretokenize with `python scripts/pretokenize_domains.py <domain>`.

## Evaluate

The gate number is HumanEval pass@1 over the **156 decontaminated problems** of 164
(`eval/humaneval_sample.py`, trailing newline stripped before generation; the excluded 8 are
in `runs/contam_r3_he_union.json`). `scripts/eval_watch.py` scores every checkpoint as it is
written. Broader metrics: `python eval/score_matrix.py --ckpt <ckpt> --json
runs/score_matrix.jsonl`. Record every run with `scripts/exp.py start` / `done`. Numbers in
`runs/score_matrix.jsonl` and `facts/*.json` carry their measurement config.

Every GPU or corpus job goes through `python scripts/harness.py launch <name> -- <cmd>`
(experiment row first, card allocation, process monitor). SFT is
`scripts/run_sft.sh <name> <resume_ckpt> <sft.pt>`.

## Commit workflow

1. Own worktree and branch; stage by path, never `git add -A`; one concern per commit,
   English message, subject ending in `(session)`.
2. Code via GitHub PR (`gh pr create`, `--merge` only, never `--squash`), after a second
   reader's `artifact:`/`case:` comment, a `runs/review.jsonl` row, and a green
   `python3 scripts/pr_merge_gate.py <pr>`. Ledger-only commits (`runs/*.jsonl`,
   `EXPERIMENTS.md`) merge without a PR via `scripts/merge_main.sh <branch>`.
3. CI gates every push: ruff, py_compile, `test_arch_compat`, `eqcheck`, `holdout`,
   `harness check` and its `--selftest`.
