# aupai

A coding/math LLM model: targeting **HumanEval pass@1 ≥ 30% at ~350M-active**, on the
DeepSeek-V4.1-Flash flat architecture (CSA2 + sliding-window attention, MoE). Pivoted
2026-09-10 by user order (`docs/standards/v41_pivot.md`); the earlier KDA/MLA line and its
0830v1 gates are retired. Working rules, layout, and the run book are in `AGENTS.md`; this
file is the short version.

## Model

| | |
|---|---|
| layers | 12 flat blocks, d=1024 / H=8; 3,209.5M total, ~342.9M active |
| attention | CSA2 (learned 8-token entries + indexer top-k, one softmax over global entries + local SWA keys); first two layers SWA-only, window 128, flash-attn varlen |
| position | partial RoPE on the last 64 dims (`--rope_dims 64`); no recurrent state |
| MoE | 48 experts, top-3 routed + 1 shared, expert_ffn 1728 (`--moe_layers 0-11`) |
| extras | FP8 (e4m3 fwd+bwd) Float8Linear; `torch.compile`; no attention residuals |
| vocabulary | 32,768 slots rebuilt 2026-09-10, `<eos>=1`, `[NUM]=32767`; `data/tokenizer.json` is not in git — copy it from the pod; the pre-rebuild vocab is `data/tokenizer_frozen_0829.json` on the pod |
| optimizer | Muon for 2D weights, AdamW for embeddings and 1D |

Measured smoke ceiling (compiled + flash, facts/v41.json): B8 OOMs at 94.6 GiB pre-step;
B4/accum4 ran 381 steps at 72.64 GiB/rank with no NaN. The gate recipe is B4/accum8 on
world 6 (block 0-5) = 786,432 tokens/step, 38.1K steps over the 30B gate mix; per-rank
peak is the measured 72.6 GiB.

Correctness never depends on which attention package is installed: without flash-attn the
SDPA fallback builds the document mask from `cu_seqlens`.

Interactive parameter/memory calculator: <https://acupof-ai.github.io/aupai/>.

## Quick start

```bash
uv sync
python scripts/harness.py install-hooks        # pre-commit: ruff E9/F, blob guard, harness check
python scripts/test_arch_compat.py             # CPU: fwd/bwd, checkpoint round-trip, doc-mask known answers
python scripts/harness.py check                # repo invariants; CI runs the same
```

The checkout ships a 2,000-document sample corpus (`data/corpus/sample/`, `data/mix_sample.json`)
that exercises the pipeline end to end. The real gate corpus is built by
`python datagen/build_corpus.py --domain <d> --source <s>` into `data/corpus/<domain>/`,
decontaminated with `python filters/decontam_ngram.py <dir>` against HumanEval/MBPP, and
pretokenized CPU-side with `python scripts/pretokenize_domains.py <domain>` into `/data00`.
The mix file (`data/mix_v41_gate.json`) is the only data path: per-domain weight, epoch cap,
anneal weight. A missing mix is an error, not a fallback.

## Run

Every GPU or corpus job starts through one launcher — it writes the experiment row first,
takes its cards from the controller's allocation (world-6 block 0-5 for gate training with
no lane at launch; card 5 is the temporary pre-launch lane; cards 6-7 stay tileRL through
the run), detaches with `setsid`, verifies the startup
gate in the worker log before the job counts as started, and arms a monitor:

```bash
python scripts/harness.py launch <name> --training --hypothesis "..." -- ./run_ddp.sh --mix data/mix_v41_gate.json --name <name>
python scripts/harness.py launch <name> -- python3 eval/score_matrix.py --ckpt <ckpt> --json runs/score_matrix.jsonl
```

The committed gate line is `runs/v41_gate_0911.sh` (draft; launches only on the
controller's explicit go) and its stop rules are `runs/prereg.jsonl#v41_gate_0911`.
SFT: `scripts/run_sft.sh <name> <resume_ckpt> <sft.pt>`. Numbers land in
`runs/score_matrix.jsonl` and `facts/*.json`, each with its measurement config; the gate
number is HumanEval pass@1 via `eval/humaneval_sample.py`.

## Commit workflow — one path

1. Work in your own worktree on your own branch; stage by path (`git add <file>`), never
   `-A`/`-a`. One concern per commit, message in English, subject ending in `(session)`.
2. Code goes through a GitHub PR (`gh pr create`, `--merge` only), second reader approves
   with an `artifact:`/`case:` comment and a `runs/review.jsonl` row; ledger-only commits
   (`runs/*.jsonl`, `EXPERIMENTS.md`) keep `scripts/merge_main.sh <branch>`. The pre-commit
   hook runs ruff E9/F, the blob guard, and `harness check`. A red hook is a red commit.
3. For code, the PR merger pushes the pod in the merge step (`scripts/pod_push.sh`
   `<files>`) and stamps main's sha; for ledger commits the committer pushes. `train.py`
   refuses to start on a drifted pod.
4. CI on push: ruff, `py_compile`, `test_arch_compat`, `eqcheck`, `holdout`, `harness check`
   and `--selftest`.
5. Record every run: `scripts/exp.py start` before, `done` after; tasks live in
   `runs/tasks.jsonl` (`harness task add|done|list`). Status is read from artifacts, never
   from a message.

## Numbers — `--fone`

BPE splits numbers by frequency (1640 → `16|40`). `--fone` gives each number one `[NUM]`
token with a Fourier-encoded value and decodes digits ten-way. The flag changes the data
format everywhere: pack with `datagen/prepare_sft_math.py --fone`; a checkpoint whose flag disagrees
with the pack refuses. `probes/fone_digit_acc.py --ckpt X` scores the digit head.
