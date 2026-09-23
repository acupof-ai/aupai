# aupai

aupai is the V4.1 coding/math model: a small CED transformer targeting
**HumanEval pass@1 ≥ 30% at ~355M active params**, from scratch on a 30B-token UltraData
gate mix. Full rules are in `AGENTS.md`.

## Model
CED — causal encoder-decoder, 12 layers d=1024 / H=8. Layers 0-5 encode causally; each
decoder layer 6-11 projects its own global KV from the encoder boundary state via an unshared
W_KV/W_Z pair. The mask is causal everywhere.

| | |
|---|---|
| size | **3,221,975,040 total / 355,430,400 active** (exact meta-device count at the gate shape) |
| CED | `--ced --ced_enc_layers 6` — bottom 6 encoder, top 6 decoder, per-decoder-layer W_KV/W_Z |
| attention | CSA2: 8-token learned KV entries, indexer top-k, one softmax over global entries + local SWA keys; first two layers SWA-only, window 128; partial RoPE on the last 64 dims (`--rope_dims 64`); no recurrent state |
| MoE | 48 experts, top-3 routed + 1 shared, expert_ffn 1728, every block (`--moe_layers 0-11`) |
| numerics | fp8 Float8Linear fwd+bwd, `torch.compile`, `--csa2_win_flash`, no attention residuals; Muon on 2D weights, AdamW on embeddings and 1D |
| vocabulary | 32,768 slots, `<eos>=1`, `[NUM]=32767`; not tracked, build with `python scripts/build_gate_tokenizer.py` |

Code: `model.py`; facts: `facts/v41.json`. Gate run **v41_ced_0923** — 30B tokens, world 8,
B4/accum6, seq 4096 = 786,432 tokens/step, 38,146 steps; flags, stop rules, read-point
protocol are at `runs/prereg.jsonl#v41_ced_0923`.

## Quick start
```bash
uv sync
python scripts/harness.py install-hooks   # ruff E9/F, blob guard, harness check
python scripts/test_arch_compat.py        # CPU fwd/bwd, checkpoint round-trip, doc-mask
python scripts/harness.py check           # repo invariants; CI runs the same
```
A 2,000-document sample (`data/corpus/sample/`, `data/mix_sample.json`) exercises the pipeline.
## Data
The mix file is the **only data path**: `data/mix_v41_gate.json` gives each domain its
target weight, epoch cap, and anneal weight; a missing domain errors, no fallback. Build with
`python datagen/build_corpus.py --domain <d> --source <s>`, decontaminate HumanEval/MBPP via
`python scripts/filter_gate_domains.py --domains <list>` (engine `filters/decontam_ngram.py`,
13-gram), pretokenize with `python scripts/pretokenize_domains.py <domain>`.

## Train and evaluate
Pretraining is `run_ddp.sh` wrapping `torchrun train.py`; gate line and stop rules are in
`runs/prereg.jsonl#v41_ced_0923`. Every GPU or corpus job goes through
`python scripts/harness.py launch <name> -- <cmd>` (experiment row first, process monitor);
SFT is `scripts/run_sft.sh <name> <resume_ckpt> <sft.pt>`.

The gate number is HumanEval pass@1 (n=164) from `eval/humaneval_sample.py`, with the
prompt's trailing newline stripped before generation (v41_ced_0923 amendment; unstripped arm
beside it). Broader metrics: `python eval/score_matrix.py --ckpt <ckpt> --json runs/score_matrix.jsonl`;
record runs with `scripts/exp.py start` / `done`. Numbers in `runs/score_matrix.jsonl` and
`facts/*.json` carry their measurement config.

## Commit workflow
1. Own worktree and branch; stage by path, never `git add -A`; one concern per commit,
   English message, subject ending in `(session)`.
2. Code via GitHub PR (`gh pr create`, `--merge` only, never `--squash`), after a second
   reader's `artifact:`/`case:` comment, a `runs/review.jsonl` row, and a green
   `python3 scripts/pr_merge_gate.py <pr>`. Ledger-only commits (`runs/*.jsonl`,
   `EXPERIMENTS.md`) merge without a PR via `scripts/merge_main.sh <branch>`.
3. CI gates every push: ruff, py_compile, `test_arch_compat`, `eqcheck`, `holdout`,
   `harness check` and its `--selftest`.
