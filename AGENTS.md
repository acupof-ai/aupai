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
- Mix: `data/mix_v3.json` = per-domain weight / epoch cap / anneal weight. train.py builds the schedule
  (main phase, then the last `Cfg.anneal_frac` tokens with anneal weights) and consumes it in order, so
  `Cfg.epochs` is forced to 1. **It is the ONLY data path.** The flat-corpus fallback was deleted
  2026-08-29 along with `data/mix.json`, `load_texts()` and `data/corpus/primary`: it was the branch a
  named-but-missing mix fell through to, training on 244KB in silence. There is nothing to fall back to
  now, so that failure cannot recur. Git ships a 2,000-document sample (4,992 with the
  untracked shards, which `.gitignore`'s `data/corpus/*/` hides) as `data/mix_sample.json` over
  `data/corpus/sample/`, a mix like any other -- getting-started and pod use the same code.
- Numbers (`--fone`): **that "leave it off" verdict was wrong; see the correction below.** BPE splits numbers by
  frequency rather than place value (1640 → `16|40`), and `--fone` fixes that -- one `[NUM]` per
  number carrying a Fourier value, ten-way scored per digit. The mechanism works: k6's digit head
  reached 66.5% whole-number exact on held-out math against a 16.4% copy-previous baseline. It still
  bought nothing end to end. Same data, same 6 epochs: plain k5 reached math-hard 3.6%, FoNE k6
  reached 3.2% (z=-0.49, p=0.627), while each beat its own base significantly. It costs 14%
  throughput (73K vs 85K tok/s/gpu).
  **CORRECTED 2026-08-29.** That comparison held "same data" fixed, and the data is the confound:
  arithmetic appears in this corpus only as `a+b=c`, which Lee et al. (arXiv 2307.03381) show is
  the one format a small transformer never learns. The experiment compared FoNE-plus-unlearnable
  against BPE-plus-unlearnable, so a tie was the only possible outcome and the design could not
  have detected the effect it was testing for. Two signals that DID appear were read backwards:
  the digit head at 66.5% whole-number exact against a 16.4% baseline, and wrong-equation rate
  43.3% -> 32.7% at p~1e-12, the largest effect ever measured on this project. "Arithmetic improved
  and the score did not, therefore representation is not the constraint" inverts the inference --
  representation WAS a constraint and was fixed; the flat score means a SECOND constraint sits
  downstream of it. With a learnable format, FoNE takes bare two-digit arithmetic from **0% to
  16%** against a BPE control trained on the SAME data for the SAME 20 epochs (0/180 first
  number, 3/180 anywhere; Fisher one-sided p=1.2e-7). Termination is a SEPARATE failure the
  representation does not touch (17% vs 21% emit `<eos>`) -- three formats on identical prompts
  leave the answer underdetermined. An earlier "20-32%" here was memorisation: the probe drew
  from an 8,100-pair space the 200K-row training set had exhausted, and 77% of its cases were in
  training. **Synthetic data needs a split on the PROBLEM, not on the row** --
  `arith_curriculum.held_out` hashes it (see EXPERIMENTS.md, k6_arith + its CORRECTION).
  `--fone` changes the data format everywhere: pack with `prepare_sft_math.py --fone`, and a
  checkpoint whose flag disagrees with the pack raises. `scripts/fone_digit_acc.py --ckpt X` scores
  the digit head against its two baselines.
  **It does improve arithmetic, sharply.** Wrong-equation rate in the generated steps:
  k6 32.7% against k5 43.3% at base (p~1e-12), 30.2% against 37.7% after the same SFT
  (p~2e-8), while emitting MORE equations (77% of generations against 66%). That is the
  largest effect measured on this project and it moves the score not at all. Score and
  arithmetic are different questions; `eval_all.sh` reports both.
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

## The harness — `python scripts/harness.py`
- **The single place progress is checked, recorded and advanced.** `check` (invariants,
  exit 1 on failure, in CI), `ledger` (every checkpoint with its provenance AND its
  math-hard on one line), `gaps` (what is NOT measured, stated out loud), `stages`.
- **A stage is done when the measurement that would falsify it exists — not when it
  produced a file.** One night produced three write-ups and zero runs of the metric of
  record, so "which checkpoint is best" was unanswerable while the conclusions read as
  settled. `gaps` exists so that is visible instead of inferred from an absence.
- **A check without a failing case is not a check.** Every entry in `CHECKS` carries a
  `broken()` building a world where its condition is violated, and `--selftest` asserts
  the check reports FAIL there. Four separately written guards for this repo all shipped
  the same defect on one afternoon — satisfied by an empty list, by a missing file, or by
  a deleted call site — and every one of their selftests passed. Add a check only with its
  broken world; the selftest fails otherwise.
- **A probe that asks "did training install X" must measure TEACHER-FORCED and FREE-RUNNING
  in the same run, or its null cannot be read.** `probe_procedure` scored free-running only:
  BOTH went 0.0 -> 0.0 after procedure SFT, which fits the pre-registered cell "coverage was
  not the constraint" and would have retired a correct path. Teacher-forced on the same gold
  text, the digit head went **21.3% -> 57.2%** (McNemar p=5.7e-62). The procedure was
  learned and does not survive the model's own rollout — exposure bias, not missing data.
  The gap between the two numbers IS the diagnosis; either number alone is unreadable.
  ARM B makes the same point from the other side: 7.4% replay moved teacher-forced
  significantly (57.2 -> 61.1, p=6.6e-05) and moved BOTH not at all. One intervention, two
  measurements, opposite outcomes — coverage would have moved both.
- **A null landing in a pre-registered cell does not certify that cell.** Pre-registration
  (`docs/exp_procedure_sft.md`) is what made the missing branch visible instead of absorbing
  the result into an existing one; the amendment there is labelled as written afterwards.
- **Build the broken world by MUTATING A REAL ARTIFACT, never by hand-writing one from the
  check's own source.** All six checks passed `--selftest` while three were dead, because
  each broken world was synthesised from the check author's memory of the input format and
  agreed with the check's bug instead of with production. `no_stale_running` read `date`;
  `exp.py` has only ever written `started`, so every row raised into a bare `except:
  continue` and the check returned PASS having examined ZERO rows, with five runs up to
  three days stale — verbatim the incident it cites. Its broken world hand-wrote a `date`
  key, so both halves believed the same fiction. Both mix worlds wrote `data/mix_test.json`
  while the checks read `cfg_default("mix")`, so they FAILed on "file does not exist" and
  the real logic never once ran. The fixes are structural: `_broken_stale_run` now shells
  out to `scripts/exp.py start` (via `AUPAI_ROOT`) and ages the row it really wrote, and
  `_tmp_repo` writes the mix at the path the checks actually read.
- **`cfg_default` raises instead of returning None.** Annotating `mix = "..."` as
  `mix: str = "..."` makes it an `ast.AnnAssign`; both corpus invariants then reported SKIP
  with the text "chosen on purpose" — an intent nobody expressed — and `check` exited 0. A
  one-token edit no reviewer would flag silently retired two checks, in the file whose own
  thesis is that "could not check" must never read as "checked".
- **The ledger must take names from the SCORES too, not only from disk and command lines.**
  `--name X` attributes a score to `ckpt_X` without `ckpt_X.pt` ever appearing in a command,
  so `ckpt_rl_k4` 4.1% and `ckpt_sft_v5_hard` 3.1% were dropped — and 4.1% is higher than
  the 3.6% the ledger was calling the best on record. The single place progress is read
  from was hiding the top of its own table.
- **CI was RED on a clean checkout at step 4 for the whole checkout's existence**, so no
  step after it — `test_sft_pack`, `eqcheck`, `holdout`, `harness check` — had ever run.
  `loader.py selftest` asserted on gitignored `data/tokenizer.json` where every other
  tokenizer-dependent step prints SKIP. "CI is green" was never evidence of anything.
  `mix_shards_present` was the second: a checkout ships only `data/corpus/sample`, so it is
  now SKIP when NONE of the mix's domains resolve and FAIL only when some do and some
  don't — a permanent red is the same as no signal.
- **`E2E_GPU=7 python scripts/test_e2e.py` tests the JOINS, which is what no other test does.**
  Every stage has a unit test and the chain had none, so the defects that survive are the ones
  between stages: a pack whose fingerprint could never equal any checkpoint's `vocab_id` because
  `prepare_sft.py` hashed `str(id)` before the token, and an SFT that ran at loss 4.77 instead of
  1.28 on weights it had silently reinitialised. It carries ONE artifact through
  mix -> tokenize -> pretrain -> checkpoint -> load -> pack -> SFT -> generate on the sample corpus
  sample and asserts the seams: checkpoint `vocab_id` == tokenizer fingerprint == pack fingerprint,
  cfg comes from `ck["cfg"]`, and cos(pretrained embedding, post-SFT embedding) > 0.9.
- **It also asserts the pretrain actually STEPPED**: cos(fresh init under `Cfg.seed`, checkpoint)
  < 0.9, measured -0.028 after six real steps. Without it every stage passed on a 206M random init
  with the training loop stubbed to zero iterations — the chain test could not see that no training
  had happened, which is the whole class of failure it exists for.
- **There is no CPU half and `E2E_GPU` is required.** DeltaRecurrence is fla/Triton only, so a
  cardless run could only re-check the mix and the vocabulary that `harness.py check` already
  covers, then exit 0 — which reads as "the chain works". It was wired into CI in exactly that
  shape, so the skip path was DELETED rather than documented. **CI does not run it; only a GPU run
  covers the joins.** It will not pick a card on its own: the pod's GPUs are shared, and a test that
  grabs whatever looks free eventually grabs a pretrain's.

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

## Corpus v3 (2026-08-29) — the rebuild
- Recipe and every measurement behind it: `docs/data_recipe_v3.md`; mix: `data/mix_v3.json`.
- **Hand-reading 180 random web documents found 18% worth training on.** The other 82% is
  gambling/adult SEO (brand names injected mid-sentence), product sheets, hospital ads, web novels,
  machine translation, spliced forum fragments, and synonym-substituted plagiarism (`曩昔五年`).
  v2 gave that corpus 88% of an 11.5B-token pretrain while chat got 1%.
- **Traditional -> Simplified was never applied.** 59.4% of the fineweb2 Chinese slice was
  traditional; converting it moved web from 1.04 to **1.45 chars/token**. The opencc table is
  single-codepoint 1:1 only (3,553 entries) -- vocabulary-level differences (軟體/软件) are not
  covered and never were.
- New domains: `textbook` (opencsg/chinese-cosmopedia, 1.74B tok, 100% pass through our own
  filters against web's 18% by hand) and `wiki` (zh, 0.23B). Both scanned by
  `scripts/scan_contamination.py`: zero eval questions in 60,000 documents each. **Run that scan on
  every new source** -- skipping it is finding #1 of docs/review_2026-08-26.md happening again.
- **textbook is synthetic and is capped below web on purpose** (31% against 40%). It could supply
  the whole corpus; SmolLM2 uses Cosmopedia at ~11% against real web, and no benchmark we own could
  detect an overdose -- every MC sits at the 25% chance line.
- Quality filter, two stages (FineWeb-Edu's architecture): the 27B annotates a stratified sample,
  a logistic head on the frozen 200M's mean hidden state learns it and scores all 1.97M documents.
  Student AUC **0.823** against the hand labels, above the 27B teacher's own 0.739 -- the teacher's
  hard yes/no ties cap its AUC, the student's continuous score recovers the ordering.
  Everything cheaper was measured first and failed: spam regex 0.50, char n-grams 0.60, structural
  features 0.62, Qwen3-0.6B 0.539. **Character n-grams rank by topic; the labels split on register.**
- **Cross-source quality comparisons need ONE judge on ONE rubric.** The distilled head is
  trained on the 27B's judgements of WEB PAGES and cannot rank textbook prose: it scores
  cosmopedia below raw web (median -1.67 against -1.33), which is not credible. Judged by the
  27B itself on the same binary rubric: unfiltered web **21.8%** educational, cosmopedia
  **59.3%** -- and an independent 120-document hand audit of a sibling opencsg corpus landed
  on 59%. Two methods, one number.
- **A published quality score is a claim, not a measurement.** cosmopedia's own `score` column
  correlates with ours at Spearman **+0.198** and is non-monotonic across its own bands; the
  same shape appeared in opencsg/Fineweb-Edu-Chinese (bands 52/66/59% usable, top band
  dirtiest). Run `datagen/audit_source_score.py` before using any source's score as a cut.
- **The mix is the source of truth about what the corpus is; the filesystem is not.**
  `data/corpus/web` (unfiltered, 2.99M docs) is kept -- a different quality threshold has
  to be re-cuttable from it -- but anything that enumerates `data/corpus/*` picked it up:
  `build_tokenizer.py` would have drowned its stratified sample in the very documents
  the filter removed, and a mix naming `web` instead of `web_hq` trains on them
  outright. Both fail silently. Take domains from `data/mix_v3.json`.
  Fixed 2026-08-29: `data/corpus/web` is gone from the pod, and `train.py` gained its own `web`-not-in-the-mix guard, on the path `main()` takes, so
  `run_ddp.sh` and a bare `python train.py` are covered and not just the one wrapper.
  `scripts/run_pretrain_v3.sh` still carries its copy: the two can drift, and train.py's is
  the one that matters. A NAMED-BUT-MISSING mix is now an assertion too -- it used to fall
  through to the flat corpus (`data/corpus/primary`, 244KB) in silence, and repointing the
  default at `mix_v3.json` made that likely on a pod that had not received the file.
- **`Cfg.mix` defaulted to `data/mix.json` -- the V2 mix, 88% weight on unfiltered `web`,
  11.5B tokens -- until 2026-08-29.** Any run launched through `run_ddp.sh` without an
  explicit `--mix` before that date trained on the unfiltered corpus. `run_pretrain_v3.sh`
  passed `--mix data/mix_v3.json`; nothing else did. Cfg defaults also disagreed with the
  recorded recipe (warmup 20 against `--warmup 150`, `attn_res` False against `--attn_res
  --attn_res_blocks 4`), so a bare `run_ddp.sh` reproduced neither the data nor the arch.
- **`ckpt_k7_v3` and `ckpt_k6_fone` have never been combined.** k7_v3 is corpus v3
  (3.29B, filtered) with NO FoNE; k6_fone is corpus v2 (11.33B, unfiltered) WITH FoNE.
  Every FoNE conclusion on this project therefore rests on a v2 base, measured against a
  v2 control -- the arithmetic effect is real, but FoNE on filtered data is unmeasured.
- Traps that cost hours, all silent: `--host_cap` is a web-crawl filter and discarded 83.4% of
  Wikipedia (one host) -- pass `--host_cap 0` for any single-source corpus. Sampling a corpus by
  reading shards in sorted order until the quota is met read 8.5% positive where a shard-stratified
  draw read 18.9%. `cuda:1` + fla/Triton raises an illegal memory access (kernels launch on the
  current device) -- use `CUDA_VISIBLE_DEVICES`. Non-contiguous parameters fail cublasGemmEx.

## Chat format
- **ChatML**, owned by `scripts/loader.format_prompt / format_example / format_history`. The old
  homemade `问：/答：` is gone from producers but the eval-contamination regexes in
  `datagen/build_corpus.py` still match it, because the corpus holds documents written that way.
- `datagen/build_corpus.py` renders the pretraining chat domain in ChatML too, so SFT does not have
  to teach the format from nothing in a few hundred steps.
- `scripts/test_sft_pack.py` (CI) checks the loss mask directly: every masked span ends at
  `assistant\n`, no supervised span contains a role marker, the turn terminator IS supervised.
  It exists because the first ChatML commit wrote the prompt into every row twice with the second
  copy supervised -- 40 examples packed into 8 rows instead of 5, and nothing would have reported it.

## Tokenizer / vocabulary
- **`data/tokenizer.json` is rebuilt IN PLACE and the old file is gone.** Every checkpoint
  needs the vocabulary it was trained on, and `ckpt_k6_fone.pt` nearly became unusable this way:
  the corpus-v3 rebuild overwrote the 32,773-token file k6 was trained on with a DIFFERENT
  32,773-token file (`d191af789cdbe597` -> `0bce3584bc24f255`). Same size, every id different,
  so a size check passes and the scores are noise. Recovered only because the local checkout
  still had the old file; it is now `data/tokenizer_k6.json` on the pod.
  **Before rebuilding, copy the current file to `data/tokenizer_<name>.json` for every live
  checkpoint.** `scripts/loader.py` compares fingerprints, which is what makes this survivable.
- `scripts/tokenizer_report.py` measures a vocabulary on four groups -- compression,
  distribution (Zipf deviation, utilisation, undertrained tokens), structure (digit place-value,
  UTF-8 integrity, round-trip, whitespace), and English (fertility, morphology, parity).
  chars/token alone is a weak proxy: arXiv 2506.03101 measures its correlation with downstream
  performance from rho=-0.77 to rho=-0.09 depending on task. It caught that k5's vocabulary was
  round-trip LOSSY (NUL and tab did not survive) and used only 72.3% of its slots.
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
