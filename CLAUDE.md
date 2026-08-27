# aupai — 200M Chinese LLM (KDA + gated MLA hybrid, optional Attention Residuals)

## Layout
- Root: `train.py`, `sft.py`, `sft_math.py`, `prepare_sft_math.py`, `serve.py`, `chat.py`, `infer.py` (entry points).
- `mathbank/` synthetic math generators (`run_math_short.py`, `vet_programs.py`, `math_programs_*`),
  `datagen/` corpus generation/augmentation, `eval/` benchmarks, `algorithms/` RL, `filters/` data cleaning,
  `scripts/` ops (exp log, eval shards, SFT packing, local inference), `workflows/` corpus JS.

## Entry points
- Pretrain: `run_ddp.sh [train.py flags]` → `torchrun ... train.py --fp8 [--attn_res] [--name X]`.
  Any `--flag` matching `Cfg.<flag>` overrides it (`python train.py --help`).
- SFT: `scripts/run_sft.sh <name> <resume_ckpt> <sft_pt> [sft_math.py args]` (logs + eval + EXPERIMENTS.md).
- Eval: `scripts/eval_hard.sh <ckpt> [ngpu]` on `data/synthetic/math_hard_eval_1k.jsonl` — the metric of
  record. math-500 is saturated (32.2 / 26.8 / 32.0 / 31.0); never conclude from it.
- AttnRes A/B: `NGPU=6 STEPS=500 scripts/run_ablation.sh` (base vs `--attn_res`, same seed).
- Corpus: `python datagen/build_corpus.py --domain web --source fineweb2 --target_tokens 6e9` (clean/dedup/
  cap into `data/corpus/<domain>/`; `--dry --limit N` prints the rejects histogram). Sources are
  interchangeable; the filters are the product. Run every domain (incl. mathbank/synthetic via
  `--source jsonl:<glob>`) through it so the eval holdout filter covers all of them.
- Mix: `data/mix.json` = per-domain weight / epoch cap / anneal weight; when present train.py builds the
  schedule (main phase, then the last `Cfg.anneal_frac` tokens with anneal weights) and consumes it in
  order, so `Cfg.epochs` is forced to 1. Delete or `--mix ""` to fall back to the flat corpus.
- pass@k gate for RL: `python eval/math_hard.py --ckpt X --k 8 --temperature 0.8` (needs pass@8-pass@1 >= 15pt).
- FP8 NaN probe: `COMPILE=1 GC=0 BS=8 MUON=1 STEPS=60 python scripts/nan_probe.py` (pod, GPU).

## Before committing model/optimizer changes
- CI (.github/workflows/ci.yml) runs ruff E9/F, py_compile, test_arch_compat, eqcheck, holdout on every push.
- `python scripts/test_arch_compat.py` (CPU, no GPU deps): AttnRes fwd/bwd, legacy-ckpt round-trip,
  optimizer grouping/schedule/snapshot, KDA decay init. Extend it when you touch those paths.
- `ruff format && ruff check` on touched files (line length 110).
- Old checkpoints keep loading: `HybridLM.load_state_dict` remaps fused keys and auto-disables AttnRes.
  Consumers build the model from `ck["cfg"]`, never from the live `Cfg` class.

## Experiment records
- Every GPU run: `scripts/exp.py start/done` → `runs/experiments.jsonl` → `EXPERIMENTS.md` (hypothesis,
  finding, decision — not just numbers). Checkpoints: `ckpt_{arch}_{tokens}_{date}.pt`, gitignored.
- Best base to date is `ckpt_k4_11b_lr05.pt` (fp8 + attn_res blocks4 + warmup 150 + lr_scale 0.5).
  A fresh clean-corpus pretrain is in progress; `ckpt_k3-mla_2b_step2000.pt` is the older K3 fallback.

## Pod
- 8×H20, all 8 usable (the 6/7 reservation was lifted 2026-08-26). `/work/aupai` on the pod is not a git repo — push files.
- `uv sync` after dependency changes (torch, fla, liger-kernel, torchao are linux-only markers).

## Coordination
- Several Claude sessions share this working tree. Announce before editing `train.py`/`sft*.py`, commit
  promptly, and hand the file back; never edit a file another session has claimed.
- Commit messages in English, one concern per commit; the repo's user-facing text is Chinese.
