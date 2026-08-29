# aupai — 200M Chinese LLM (KDA + gated MLA hybrid, optional Attention Residuals)

## Writing rules (all docs, commit messages, and replies)

- No metaphors. They distort.
- No big words, no verdict-first tone, no invented compressed terms, no filler explanation, no
  spoken/speech register.
- Delete anything a competent reader already knows.
- Every rewrite must raise information density. Rephrasing without new information is a no-op.
- 3+ consecutive prose paragraphs: check whether a table, list, or grouping works instead.
- Target: simple, clear, coherent, specific, accurate, complete. Written for a short attention
  span, so short units and strict logical order matter more than completeness of phrasing.

## Layout
- Root: `train.py`, `sft.py`, `sft_math.py`, `prepare_sft_math.py`, `serve.py`, `chat.py`, `infer.py` (entry points).
- `mathbank/` synthetic math generators (`run_math_short.py`, `vet_programs.py`, `math_programs_*`),
  `datagen/` corpus generation/augmentation, `eval/` benchmarks, `algorithms/` RL, `filters/` data cleaning,
  `scripts/` ops (exp log, eval shards, SFT packing, local inference), `workflows/` corpus JS.

## Entry points

| task | command |
|---|---|
| Pretrain | `run_ddp.sh [train.py flags]` → `torchrun ... train.py --fp8 [--attn_res] [--name X]`. Any `--flag` matching `Cfg.<flag>` overrides it |
| SFT | `scripts/run_sft.sh <name> <resume_ckpt> <sft_pt> [sft_math.py args]` — logs, eval, EXPERIMENTS.md |
| Eval, one metric | `scripts/eval_hard.sh <ckpt> [ngpu]` |
| Eval, full matrix | `scripts/eval_all.sh <ckpt> [tokenizer]` — math-hard, math-500, MC suite, digit head |
| Measure everything unscored | `python scripts/harness.py measure` |
| pass@k gate for RL | `python eval/math_hard.py --ckpt X --k 8 --temperature 0.8` — needs pass@8 − pass@1 ≥ 15pt |
| Corpus | `python datagen/build_corpus.py --domain X --source Y --target_tokens 6e9`; `--dry --limit N` prints the rejects histogram |
| AttnRes A/B | `NGPU=6 STEPS=500 scripts/run_ablation.sh` |
| FP8 NaN probe | `COMPILE=1 GC=0 BS=8 MUON=1 STEPS=60 python scripts/nan_probe.py` (pod) |

### What each eval can and cannot say

| set | resolution | caveat |
|---|---|---|
| math-hard, 1032 problems | ±1.1pt at a 2–3% pass rate | The metric of record. Test significance before explaining any gap |
| math-500 | saturated | **10.2% memorizable**: 51 of 500 questions have a near-duplicate in Belle/mxode at Jaccard ≥0.8 with the same answer. Absolute value inflated; comparisons at equal exposure hold |
| MC suite | ARC-E z=9.1, PIQA z=3.6 above chance | The other three sit at the 25% line |

### Mix — `data/mix_v3.json`

Per-domain weight, epoch cap, anneal weight. train.py builds the schedule (main phase, then the
last `Cfg.anneal_frac` of tokens at anneal weights) and consumes it in order, so `Cfg.epochs` is
forced to 1. **It is the only data path.** The flat-corpus fallback was deleted 2026-08-29 with
`data/mix.json`, `load_texts()` and `data/corpus/primary`: a named-but-missing mix fell through to
it and trained on 244KB in silence. Git ships a 2,000-document sample (4,992 with the untracked
shards `.gitignore`'s `data/corpus/*/` hides) as `data/mix_sample.json`.

### Numbers — `--fone`

BPE splits numbers by frequency, not place value (1640 → `16|40`). FoNE gives each number one
`[NUM]` token carrying a Fourier value, scored ten-way per digit.

| measurement | FoNE | control | test |
|---|---|---|---|
| bare two-digit arithmetic, learnable format | **16%** | 0% | Fisher one-sided p=1.2e-7 |
| wrong-equation rate at base | 32.7% | 43.3% | p~1e-12 |
| wrong-equation rate after the same SFT | 30.2% | 37.7% | p~2e-8 |
| digit head, whole-number exact | 66.5% | 16.4% (copy-previous) | — |
| math-hard | 3.2% | 3.6% | z=−0.49, p=0.627 |
| throughput | 73K tok/s/gpu | 85K | −14% |

**It improves arithmetic sharply and moves the score not at all.** Score and arithmetic are
different questions; `eval_all.sh` reports both.

An earlier reading of this table — "arithmetic improved and the score did not, therefore
representation is not the constraint" — inverts the inference. Representation was a constraint
and was fixed; the flat score means a second constraint sits downstream. The original comparison
also held "same data" fixed and the data was the confound: arithmetic appears in this corpus only
as `a+b=c`, the one format Lee et al. (arXiv 2307.03381) show a small transformer never learns.
FoNE-plus-unlearnable against BPE-plus-unlearnable could only tie.

Termination is a separate failure the representation does not touch (17% vs 21% emit `<eos>`):
three formats on identical prompts leave the answer underdetermined.

`--fone` changes the data format everywhere. Pack with `prepare_sft_math.py --fone`; a checkpoint
whose flag disagrees with the pack raises. `scripts/fone_digit_acc.py --ckpt X` scores the digit
head against its two baselines.

### Vocabulary identity

Every checkpoint is scored with the vocabulary it was trained on. `data/tokenizer.json` is rebuilt
in place; ids do not survive a rebuild and size does not identify a vocabulary. Checkpoints and
packs carry `vocab_id` (a hash of the id→token map) and `sft_math.py` refuses a mismatch. For an
older checkpoint pass `--tokenizer` / `TOKENIZER=`. Three bugs in one day came from skipping this,
the loudest a k5 SFT that trained at loss 4.77 instead of 1.28 with nothing raising.

### Synthetic data

`docs/synthetic_data_standard.md`. One distinction decides the mix weight: **anchored rephrasing**
(~30% of the mix) versus **from-scratch generation** (under 5% for sub-1B). Test: are the output's
numbers and entities a subset of its declared source. Anchored rephrasing's downstream gain at
200M is **unmeasured** — the source grid and the retracted claims live in
`facts/synthetic_data.json`.

## The harness — `python scripts/harness.py`

`check` (invariants, exit 1, in CI) · `ledger` (checkpoint, provenance, math-hard) ·
`gaps` (what is unmeasured) · `measure` (measures it) · `stages`.

### Fact store — `facts/*.json`

Measurements live here, one file per migrated section, never in prose. Required per entry:
`id`, `value`, `measured` (YYYY-MM-DD), `source` (command or artifact), `config` (non-empty),
`uncertainty`, `status` (`measured` / `recorded` / `unmeasured` / `retracted`). `unmeasured`
and `retracted` entries also need `claim`, `audit`, `refuted_by`. Optional: `unit`,
`guard_phrases` (must not reappear in AGENTS.md), `boundary` (what the measurement cannot
answer — a design limit, not uncertainty). `facts_well_formed` enforces the required fields.

### Rules, and the incident behind each

| Rule | What happened without it |
|---|---|
| A stage is done when its falsifying measurement exists, not when it produced a file | One night: three write-ups, zero runs of the metric of record. "Which checkpoint is best" was unanswerable while the conclusions read as settled |
| A check without a failing case is not a check. Every `CHECKS` entry carries `broken()`; `--selftest` asserts FAIL there | Four separately written guards shipped the same defect in one afternoon — satisfied by an empty list, a missing file, a deleted call site — and every selftest passed |
| Build the broken world by mutating a real artifact, never by hand-writing one from the check's own source | Three of six checks were dead while `--selftest` passed. `no_stale_running` read `date`; `exp.py` only ever wrote `started`, so every row hit a bare `except: continue` and the check returned PASS on zero rows with five runs three days stale. Its broken world hand-wrote `date` — both halves believed the same fiction |
| A metric without a known-answer case is not a metric | `tokenizer_report.py` reported four wrong numbers in one day (below) |
| An install probe measures teacher-forced AND free-running in the same run | `probe_procedure` scored free-running only: BOTH 0.0 → 0.0 after procedure SFT, which fits "coverage was not the constraint" and would have retired a correct path. Teacher-forced, the digit head went 21.3% → 57.2% (McNemar p=5.7e-62). The procedure was learned and does not survive the model's own rollout |
| A null landing in a pre-registered cell does not certify that cell | `docs/exp_procedure_sft.md`; its amendment is labelled as written afterwards |
| A permanent red is the same as no signal | Twice: CI red on a clean checkout at step 4, and `mix_shards_present` red because a checkout ships only `data/corpus/sample` |
| **Before running a two-arm test, name what ELSE changed with the variable, and ask whether it alone could produce the result you expect.** Then either hold it fixed or add the arm that isolates it | The textbook 36%-vs-5% ablation. `mix_v3_lowtb.json` gives the freed 31% to the real-text domains, so the 5% arm also trained on ~31% more web and wiki -- and the verdict was their held-out loss. It won by 0.097 / 0.109, about what 31% more in-domain data buys. The design guards one direction ("less textbook will of course score worse on textbook") and walks into its mirror. It answers "is web worth more per token than cosmopedia" (yes), not "is synthetic data harmful" |

### The four wrong numbers, and what catches them

Every one is a value that depends on the measurement configuration, printed without it.

| Metric | Reported | True | Cause |
|---|---|---|---|
| hanzi whole-char | 0.00% | 99.2% | Searched byte-mapped token strings for a literal hanzi — fired on every correct ByteLevel vocabulary |
| vocabulary utilised | 6.4% | 99.7% | 402 documents |
| undertrained (≤1 use) | 4.0% | 0.43% | 1.6M tokens, not 142M. A token of frequency 1e-6 appears 1.6 times in 1.6M, so a healthy Zipf tail must put percent of the vocabulary at ≤1 use |
| English fertility | 2.36 | 1.87 | Documents clipped to 2,000 characters |

`tokenizer_report.py --selftest` catches three of the four with one assertion: ten times the
text must not move a per-character or per-word ratio. A metric that does move is declared
scale-bound and carries its corpus size instead of a bare threshold. Known answers come in
pairs — a single low-hanzi case passes the broken version — and must differ by 60 points.
`sample_corpus`'s `shards` and `clip` are part of every metric's definition.

### Two failure modes specific to this file

- **`cfg_default` raises rather than returning None.** Annotating `mix = "..."` as
  `mix: str = "..."` makes it an `ast.AnnAssign`; both corpus invariants then reported SKIP
  with the text "chosen on purpose", and `check` exited 0.
- **The ledger takes names from the scores too.** `--name X` attributes a score to `ckpt_X`
  without `ckpt_X.pt` appearing in any command, so `ckpt_rl_k4` 4.1% was dropped — higher
  than the 3.6% the ledger called the best on record.

### `E2E_GPU=<idx> python scripts/test_e2e.py`

The only test of the JOINS. Every stage has a unit test; the chain had none, so the surviving
defects were between stages: a pack whose fingerprint could never equal any checkpoint's
`vocab_id` (`prepare_sft.py` hashed `str(id)` before the token), and an SFT that ran at loss
4.77 instead of 1.28 on silently reinitialised weights.

It carries one artifact through mix → tokenize → pretrain → checkpoint → load → pack → SFT →
generate and asserts:

- checkpoint `vocab_id` == tokenizer fingerprint == pack fingerprint
- cfg comes from `ck["cfg"]`
- cos(pretrained embedding, post-SFT embedding) > 0.9
- **the pretrain actually stepped**: cos(fresh init under `Cfg.seed`, checkpoint) < 0.9,
  measured −0.028 after six real steps. Without it every stage passed on a 206M random init
  with the training loop stubbed to zero iterations.

`E2E_GPU` is required and there is no CPU half. A cardless run could only re-check what
`harness.py check` covers, then exit 0 — which reads as "the chain works", and it was wired
into CI in exactly that shape. CI does not run it. It will not pick a card: the pod's GPUs are
shared.

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
- **FoNE on filtered data exists as checkpoints; the old "never combined" claim is retracted.**
  ckpt_tb36 and ckpt_tb05 carry `fone=True` on `data/mix_v3.json` (cfg verified 2026-08-29,
  `facts/fone.json`). No score for either is recorded in `runs/experiments.jsonl`, so whether
  the arithmetic effect replicates on filtered data is unmeasured here.
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

**FROZEN 2026-08-29.** A rebuild is allowed only under the three unfreeze conditions below and
invalidates every checkpoint trained on the old vocabulary (see Vocabulary identity — a size
check passes while the scores are noise). Before rebuilding, copy the live file to
`data/tokenizer_<name>.json`. `scripts/loader.py` compares fingerprints, which is what makes
this survivable.

- **Gates** — `python scripts/tokenizer_eval.py --tokenizers <paths>`: round-trip lossless and all
  256 bytes are vetoes; hanzi whole-char ≥ 0.95 is a veto; ref fertility ≤ 1.55 and never-used
  ≤ 0.01 are regression guards. `never used` excludes the by-design-unreachable entries (the
  byte-fallback alphabet, chat specials, `[NUM]`).
- **Build** with `scripts/build_tokenizer.py`: always pass `initial_alphabet=ByteLevel.alphabet()`
  (without it NUL silently drops); stratified equal-byte sample per domain; `--weights`
  rebalances, `--out` keeps a candidate out of `data/tokenizer.json`. Sample balance matters;
  sample size barely does — the binding constraint is the 32K budget, not the corpus.
- **Measure** with `scripts/tokenizer_report.py --selftest` — mandatory before believing any
  number it prints. chars/token alone is a weak proxy; TokEval (arXiv 2608.18062) is explicit
  that intrinsic metrics screen but do not rank.
- **Faster tokenizer libraries do not drop in.** Adopting one means retraining the vocabulary,
  which changes every id; re-evaluate before adopting.
- **Unfreeze conditions — three, and nothing else:**
  1. The model outgrows the fitted 12–20K optimum (arXiv 2407.13623).
  2. The corpus distribution changes materially.
  3. An extrinsic test — two pretrains differing only in the vocabulary — says a candidate is better.

Facts — fingerprint, sizes, gate values, the frontier table, the sweeps, the faster-library
evaluation: `facts/tokenizer.json`. Every entry carries its measurement config and its status;
a number without them does not land.

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
