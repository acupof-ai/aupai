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
  record. It resolves to about ±1.1pt at a 2-3% pass rate, so test significance before explaining a gap.
  math-500 is saturated (32.2 / 26.8 / 32.0 / 31.0) AND 10.2% of it is memorizable: 51 of its 500
  questions have a near-duplicate in the Belle/mxode training data at Jaccard >= 0.8 carrying the same
  answer (measured 2026-08-28). Its absolute score is inflated; a comparison between checkpoints with
  equal exposure still holds. math-hard is clean by the same scan (top-1 median 0.156, max 0.538).
- AttnRes A/B: `NGPU=6 STEPS=500 scripts/run_ablation.sh` (base vs `--attn_res`, same seed).
- Corpus: `python datagen/build_corpus.py --domain web --source fineweb2 --target_tokens 6e9` (clean/dedup/
  cap into `data/corpus/<domain>/`; `--dry --limit N` prints the rejects histogram). Sources are
  interchangeable; the filters are the product. Run every domain (incl. mathbank/synthetic via
  `--source jsonl:<glob>`) through it so the eval holdout filter covers all of them.
- Mix: `data/mix.json` = per-domain weight / epoch cap / anneal weight; when present train.py builds the
  schedule (main phase, then the last `Cfg.anneal_frac` tokens with anneal weights) and consumes it in
  order, so `Cfg.epochs` is forced to 1. Delete or `--mix ""` to fall back to the flat corpus.
- Numbers (`--fone`): **off by default, and measurement says leave it off.** BPE splits numbers by
  frequency rather than place value (1640 → `16|40`), and `--fone` fixes that -- one `[NUM]` per
  number carrying a Fourier value, ten-way scored per digit. The mechanism works: k6's digit head
  reached 66.5% whole-number exact on held-out math against a 16.4% copy-previous baseline. It still
  bought nothing end to end. Same data, same 6 epochs: plain k5 reached math-hard 3.6%, FoNE k6
  reached 3.2% (z=-0.49, p=0.627), while each beat its own base significantly. So number
  representation is not the constraint here, and `--fone` costs 14% throughput (73K vs 85K
  tok/s/gpu). Worth re-testing on a base that can actually solve these problems.
  `--fone` changes the data format everywhere: pack with `prepare_sft_math.py --fone`, and a
  checkpoint whose flag disagrees with the pack raises. `scripts/fone_digit_acc.py --ckpt X` scores
  the digit head against its two baselines.
- **Every checkpoint is scored with the vocabulary it was trained on.** `data/tokenizer.json` is
  rebuilt in place and ids do not survive a rebuild; size does not identify a vocabulary either.
  Checkpoints carry `vocab_id` (a hash of the id→token map), packs carry the same, and `sft_math.py`
  refuses a mismatch. For an older checkpoint pass `--tokenizer` / `TOKENIZER=`. Three separate bugs
  in one day came from skipping this, the loudest being a k5 SFT that trained at loss 4.77 instead
  of 1.28 with nothing raising.
- Stage end: `scripts/eval_all.sh <ckpt> [tokenizer]` -- math-hard, math-500, the MC suite, and the
  digit head for a FoNE checkpoint, each labelled with what it can and cannot say.
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
- Bases: `ckpt_k4_11b_lr05.pt` and `ckpt_k5_clean_0827.pt` are indistinguishable on math
  (math-500 51.6 vs 51.2, p=0.899; math-hard 2.9 vs 1.9, p=0.152) with k5 holding the better val
  (2.020 vs 2.086). k6_fone adds `--fone`. Recipe: `--fp8 --attn_res --attn_res_blocks 4
  --warmup 150 --lr_scale 0.5`. `ckpt_k3-mla_2b_step2000.pt` is the older K3 fallback.

## Tokenizer / vocabulary
- vocab 32,773 = 32,768 BPE merges (incl `<unk>`/`<eos>`) + 4 chat specials + `[NUM]`.
  `padded_vocab` is 32,832 either way, so adding `[NUM]` resized nothing.
- **Keep 32K.** A fitted vocabulary scaling law (arXiv 2407.13623, N_v ∝ N_nv^0.83) puts the optimum
  for this 166M non-embedding model near 12-20K once overtraining is accounted for; a measured sweep
  on this corpus shows 64K buys +2.8% compression for +33.6M params and **+14% compute per character**
  (the d×V output matmul runs every forward pass — tying halves the params, not the FLOPs; at 32K it
  is already ~17% of FLOPs). Big multilingual vocabs (Qwen3 151,936 / GLM-4 151,552 / DeepSeek-V3
  129,280) buy English + code + 50 languages, not Chinese. Measured head to head on this corpus, ours
  emits FEWER tokens than Qwen3-0.6B: 1.61 vs 1.45 chars/token.
- Train from a STRATIFIED sample, which `scripts/build_tokenizer.py` does (equal per-domain byte
  budget), not from the raw corpus. Feeding all 9.4M docs let web drown everything else: the old
  vocab had no whole-token for common traditional characters and split them into byte pieces, so web
  scored 1.04 chars/token — worse than one token per character. Rebuilding from a 112K-doc stratified
  sample took 5 minutes instead of 45+ and emits **12.5% fewer tokens** on held-out corpus text
  (1.484 vs 1.299 chars/token), better on every domain. Slot spend shifted where you would want it:
  single-character tokens 3,175 -> 4,706 and 5+-character tokens 4,312 -> 7,817. Every pretrain
  before 2026-08-28, k5 included, used the weaker vocab.
- Sample SIZE barely matters; sample BALANCE does. 25K docs vs 395K (16x data, 6x time) moved
  chars/token 2.6093 -> 2.6296 and hanzi occurrence-coverage 99.51% -> 99.62%, and the 5K-10K
  frequency tier stayed at 0% either way — the binding constraint is the 32K vocab budget, not the
  corpus. Of 7,825 distinct hanzi only the top ~1K get whole tokens, which still covers 99.6% of
  occurrences.
- Always train with `initial_alphabet=ByteLevel.alphabet()`. Without it only the byte-alphabet chars
  present in the corpus survive (measured 193/256), which silently drops NUL bytes on the round trip
  and breaks every fast tokenizer library.

## Pod
- 8×H20, all 8 usable (the 6/7 reservation was lifted 2026-08-26). `/work/aupai` on the pod is not a git repo — push files.
- `uv sync` after dependency changes (torch, fla, liger-kernel, torchao are linux-only markers).
- Faster tokenizer libraries do NOT drop in (evaluated 2026-08-28, tokenizing 9.49B tokens at
  2.3M tok/s, ~66 min). `gigatoken` refuses this vocab outright ("no single-byte vocab entry for
  byte 0x00" -- it needs a complete 256-byte base that a Chinese-corpus BPE does not have);
  `fastokens` loads it and matches ids on short strings but raises "character not in vocabulary"
  on real corpus text instead of falling back to `<unk>`; `tiktoken` uses a different regex
  pretokenizer and cannot reproduce ByteLevel BPE ids at all. Adopting any of them means retraining
  the vocab with full byte coverage, which changes every token id. The profile also caps the prize:
  75% of the time is already in the Rust tokenizer running on 90-143 cores, 25% in python. The real
  win is incremental tokenization (only new shards), not a faster library.
- Long jobs need `setsid`, not `nohup`: `pod` runs through `crictl exec`, and when that session ends
  (a dropped tunnel, a tool timeout) the kernel kills the whole process group — `nohup` only blocks
  SIGHUP and does not save it. Launch as
  `pod "cd /work/aupai && setsid nohup bash -c '<cmd> > runs/x.log 2>&1' </dev/null >/dev/null 2>&1 &"`,
  then poll the log in a separate call. A run_sft.sh launched with plain nohup died between the train
  and eval stages this way (2026-08-28): the checkpoint was saved, the eval never ran, and the
  experiments row stayed status="running".

## Coordination
- Several Claude sessions share this working tree. Announce before editing `train.py`/`sft*.py`, commit
  promptly, and hand the file back; never edit a file another session has claimed.
- Commit messages in English, one concern per commit; the repo's user-facing text is Chinese.
