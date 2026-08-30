# Data Provenance — raw pretrain sources

Raw jsonl sources consumed by `scripts/build_domains.sh`. Frozen files are the
single source of truth: **verify, don't re-download** — re-fetching introduces
distribution drift and a fresh contamination risk against the eval holdout.

Regenerate a frozen file ONLY if the local copy AND the pod copy are both lost;
in that case the new fetcher must be sha-compared against these records.

## Frozen on pod (2026-08-27, verified by aupai-01)

These five have NO reproduction script (git history has no producer; they were
one-time transfers). Provenance is unknown/partial — the user was asked to
confirm origins. Files live on the pod at /work/aupai/data/.

| file | domain | size (pod) | sha256 (pod, 2026-08-27) | provenance |
|---|---|---|---|---|
| pretrain_full.jsonl | web | 9.3 GB | 230525ecda660c238c3401b36b53295002f3fb5ce7a01aacb626ce3eebeedc76 | unknown (superset of skypile etc. per build_corpus.py comment; measured 2026-08-26) |
| cosmopedia_extra.jsonl | en | 685 MB | 77aabceceb9f323346b6aa21c9502745e1e1693bb542dd58603f14d6e026a145 | likely ModelScope OpenCSG/Chinese-Cosmopedia extra shards (unconfirmed) |
| en_textbook.jsonl | en | 122 MB | 6a85bfeebfa59b772afa0a2d7584abb8d0d847f7c197697d6b8f917330ecd73f | unknown |
| code_filtered.jsonl | code | 225 MB | 76dc76cf94858f1846cadb98bbef9b201232b7aba0d66c6fa465043c530a1948 | unknown ("filtered" step unrecorded; upstream gen: datagen/gen_code.py → data/synthetic/code_python_zh.jsonl) |
| en_math_text.jsonl | math | 307 MB | fa21ac063d8aa96347b508ddce3b6506d6706761504d768254142feb025c3ac7 | unknown |

These are the pod files' current content hashes — the verify-not-refetch anchor. A
re-derived file must match one of these (or the divergence be explained) before it
replaces the frozen copy.

## Present locally + reproducible (fetchers exist or being added)

| file | domain | sha256 | rows | producer |
|---|---|---|---|---|
| alpaca_gpt4_zh.jsonl | chat | 93819e69830d9eb050e58c342230f3e1986a2e3cd07c3d1a075abb9ddcb6251d | 52,049 | scripts/fetch_chat_data.py (HuggingFaceH4/alpaca_gpt4_data_zh, sha-verified) |
| coig.jsonl | chat | cdcac3f1d310c0dd8bb6cf5ee63a4b2a99d3386e098cead4985d7e962a8a10f6 | 163,443 | scripts/fetch_chat_data.py (BAAI/COIG instructions config; normalizer TBD — diff raw dump vs frozen file on first fetch) |
| school_math_r1_zh.jsonl | math | c8f6a7cce2e4c0b76711919a99767aa435a5ce6b509da722ffcb750d42124834 | 223,423 | scripts/fetch_math_data.py belle branch (pod sha identical, verified 2026-08-27; known 3.6% tail_answer gold bug, see docs/lessons/review_2026-08-26.md #2) |

## External SFT-math candidates surveyed 2026-08-28 (short-solution search)

Looking for Chinese math with answers in the eval's own length band (60-132 chars;
math_hard_eval_1k is median 85 / p90 132). Downloaded through `HF_ENDPOINT=https://hf-mirror.com`
— huggingface.co itself is unreachable from the pod, the mirror is not.

| repo | rows | answer median | in 60-132ch band | verdict |
|---|---|---|---|---|
| ALmonster/MathInstruct-Chinese | 256,294 | 146 | 94,577 (37%) | only real candidate; multiple-choice format ("选项：(A)…答案是E"), machine-translated from English MathInstruct, Indian units (卢比/便士), and some reasoning chains skip a step. Needs option-stripping + \boxed{} conversion; broken chains are not repairable. sha256 997403a204bfecf1a7aab333c5359067680ae58a38363ae9f566ac0d1cde93cd |
| Azure99/blossom-math-v4 | 10,000 | 352 | 253 (3%) | rejected — long-CoT, verbose hedging prose |
| swulling/gsm8k_chinese | 7,473 | 257 | 716 (10%) | rejected — content is English despite the name |
| zake7749/kyara-chinese-math-sft-s0-30K | — | — | — | unavailable: GatedRepoError |

Conclusion: public Chinese math SFT data is overwhelmingly long-CoT, which is the opposite of what
this project needs. The in-repo alternative (`mathbank/`, seeded generators with `vet_programs.py`
answer verification) is the only source that controls length, difficulty and correctness at once.

## math_short_v8 — generated 2026-08-28 (the answer to the short-solution gap)

`cd mathbank && python3 run_math_short.py 100000 ../data/synthetic/math_short_v8.jsonl --ratios 0,0,0.6,0.4 --seed 28`

97,771 rows (L3 57,771 + L4 40,000; L3 stalls at 57,771 — 509 programs x the 150
instance cap). sha256 7e45bc95d0aa3226823a7a493a4df525a611ee7287712a7e28bec4ed217830e8

Built after the external survey above found nothing usable. Every property is
matched to `data/synthetic/math_hard_eval_1k.jsonl` rather than inherited:

| property | v8 | eval | old sft_k4.pt |
|---|---|---|---|
| answer length median / p90 | 88 / 136 | 85 / 132 | 156 / 377 |
| level mix | 100% L3/L4 | 100% L3/L4 | unlabelled |
| answer correctness | program-verified | — | unverified |
| forward references | 0.00% | — | — |
| duplicate instructions | 0 | — | — |

Generator reject rate is 23.3%, up from 2.9%, because `verify()` now also rejects
forward references — a step citing a value no earlier step produced. Those were
9.8% of v7-era rows and every equation in them is arithmetically true, so the old
numeric check passed them; see commit ebd731a.

## Schema contract (consumed by datagen/build_corpus.py)

- pretrain/text sources: `{"content": ...}` (or `{"text": ...}`)
- QA/chat/math sources: `{"instruction": ..., "output": ...}` → rendered as 问：/答：

## Cleanup 2026-08-28

The vocabulary was rebuilt on 2026-08-28, so every packed `.pt` from before that
date holds token ids that no longer mean anything. All 8 (7.1 GB) were deleted and
`data/sft/sft_v8_fone.pt` was packed against the new vocab. Text jsonl is unaffected:
it carries no ids.

Deleted with it, 708 MB of `data/sft/*.jsonl` intermediates — sft_dedup, sft_mixed,
sft_mixed_v2, sft_clean, sft_mixed_clean, sft_tagged, sft_expanded, short_all. They
were stages of the pipeline that produced sft_k4.pt, which measured HARMFUL (k5 base
51.2% → sft_k5 44.8% on math-500, p=0.043), and every one is regenerable from
scripts/fetch_sft_data.py plus scripts/make_mixed.py. Kept: the network-sourced
downloads (fable5_cot, gsm8k_zh, qwq_mmlu, reasoning, sft_all, sft_all_v2).

In data/rl: rlvr_math_clean.jsonl (a second clean pass differing from rlvr_clean by
21 rows) and probe_gens.jsonl (21 MB of raw band-probe generations whose distilled
result is instance_rates + program_rates + rl_band). Kept rlvr_math.jsonl, which the
trainer reads, and rlvr_clean.jsonl, which run_pipeline.sh reads.

Also deleted: math_short_v1 / v2 / v4 / sol_v1 (43 MB). Their exact bytes cannot be
reproduced, because no command or seed was recorded before this file existed, but
nothing needs those bytes. v8 matches the eval on both axes and they do not:

| | rows | answer length median / p90 | levels |
|---|---|---|---|
| math_hard_eval_1k | 1,032 | 85 / 132 | 100% L3+L4 |
| v8 | 97,771 | 88 / 136 | 100% L3+L4 |
| v4 | 76,677 | 64 / 118 | 32% L1/L2 |
| v2 | 72,000 | 58 / 110 | 33% L1/L2 |
| v1 | 12,382 | 83 / 174 | unlabelled |

The one thing they held alone is L1/L2 easy problems, which the eval contains none
of and which `run_math_short.py --ratios` regenerates on demand (its default mix,
0.15/0.35/0.35/0.15, is the shape v2 and v4 have). Record the command and seed for
every future batch anyway — that is what made this call cheap to make.

## Related

- Synthetic math: data/synthetic/math_short_v8.jsonl, reproducible from the seeded
  command above.
- Synthetic code/knowledge: data/synthetic/{code_python_zh,knowledge_qa_zh}.jsonl
  via datagen/gen_code.py + gen_knowledge2.py (seeded, zero external deps).
- Eval holdout filter: scripts/holdout.py — every fetcher must exclude it.

## Eval resolution and the seed trap — measured 2026-08-28

**A new seed is not a new batch.** `run_math_short.py` seeds each draw from
`f"{seed}-{level}-{program}-{draw}"`, which makes a run reproducible but does
nothing about repetition: dedup is per-run, and each program draws from a bounded
instance space. Generating 1,031 rows at seed 99 against math_short_v8's seed 28
returned **294 identical questions and 86.4% shared templates**. The generator now
takes `--exclude <glob>`, which loads earlier batches into the dedup set; the same
run then collided **0** times. Rows also carry `program_id`, and the run prints an
`EFFSIZE` line.

**Rows are not independent observations.** Greedy pass/fail correlates inside a
program: over 1,298 programs x 8 instances (`data/rl/instance_rates.jsonl`),
ICC = 0.296 and 62.6% of programs are all-or-none. An accuracy over N rows in
K programs is worth

    n_eff = K·m / (1 + (m−1)·0.296),  m = N/K

which ceilings at K/0.296 no matter how many instances each program generates.

**math_hard_eval_1k holds up.** It has 899 distinct number-templates and 486
distinct solution skeletons over 1,032 rows, so n_eff is 774-989 and the 95%
half-width at a 3% pass rate is **±1.06% to ±1.20%**. It also shares only 0.3% of
templates with math_short_v8 against 86.4% between two same-bank batches, so it is
program-disjoint from training. Its generator predates the program bank and is
gone, which is why that disjointness is luck rather than design.

**Expanding it from the current bank does not work.** Any new rows would come from
the bank the training data comes from, at 86.4% template overlap. Reserving
programs instead caps at ±1.03% for 309 of them — no better than today, and 309
programs removed from training. `mathbank/split_bank.py` implements the split and
records the arithmetic; it is deliberately not applied.

**What actually buys resolution:** eval-only programs, roughly 312 for ±1.03%,
517 for ±0.80%, and 1,178 for ±0.53%. The whole L3/L4 bank is 943.

## SFT real-math data work — 2026-08-28 (aupai-3b)

### 1. Eval contamination
Bigram-Jaccard scan (self-validated; methods verified to recall hand-built
near-dups and reject unrelated pairs) of `real_math_filtered` (132,205 rows)
against both eval sets at Jaccard-thresholded similarity:

- `math_hard_eval_1k` is **clean**: top-1 Jaccard median 0.156 / p90 0.263 /
  max 0.538; no true duplicate. Consistent with its prior program-disjointness.
- `math_test_500` carries near-duplicates of Belle/mxode training rows. Under the
  operational definition (top-1 Jaccard ≥ 0.8 AND gold answer equal), 51/500 =
  10.2% memorizable; the same-answer share falls sharply across thresholds
  (87% at ≥0.9, 63% at 0.8-0.9, 40% at 0.5-0.8, ~12% baseline below 0.5), so
  0.8 is a real knee. Consequence: math-500's absolute value is inflated ~10pt,
  but cross-checkpoint comparison holds for equal Belle exposure. Pair examples:
  EVAL「小明有5本书，小华有3本书，一共有几本？」≈ TRAIN「小明有3本书，小华有5本
  书…」(Jaccard≈1.0) — same problem, operands swapped. Full pair file:
  pod `data/sft/contam_out/near_{test,hard}.jsonl`.
  Once measured, contamination was also *removed*: the 540 training rows within
  Jaccard 0.5 of any eval question were deleted from the training set (0.47% of
  clean), using **no same-answer constraint** — for contamination the conservative
  direction is the opposite of the one dedup needs (dedup avoids deleting useful
  re-numbered practice questions, so it adds the same-answer constraint; scrubbing
  prefers to over-remove rather than leave anything memorizable, so it drops the
  constraint and takes Jaccard 0.5). Result: `real_math_clean_scrubbed.jsonl`,
  114,908 rows, which is the file packed for SFT. math_hard had 0 such rows.

### 2. mxode 91% drop + near-dup dedup
Reproduced from raw jsonl locally (four-step, exact): length band 60-132 cuts
mxode 211,988→38,700 (**81.7%** — mxode solutions are verbose, median 188 chars);
eqcheck removes only ~50 more; global cross-source md5 dedup cuts 38,700→18,494
(**-52%**). So mxode survives at 8.7% because it is long-solution AND heavily
framing-overlapping with belle, not because it is broken.

Near-dup dedup, global at Jaccard ≥ 0.8 **with a same-answer hard constraint**
(numpy MinHash 96/16/6 LSH ~99% recall for ≥0.8 pairs; exact Jaccard confirm;
greedy keep-first; `source` preserved). The answer constraint is essential:
bare 0.8 also deletes same-topic *rephrased-number* pairs with DIFFERENT answers
(763/2,589 cross-source ≥0.8 candidates are belle↔mxode re-numbered variants of
a different problem, e.g. belle "4 candies + 2" vs mxode "6 candies + 4", ans
6≠10) — deleting those loses difficulty diversity. Chained: intra-source + cross
-source at 0.8 same-answer. Results: 132,205 → 118,567 (intra-source) → 117,098
(cross, -1,469 all same-answer) → **115,448** after dropping 1,650 stem-restated
rows (belle 96,000 / mxode 15,388 / gsm8k 4,060). Dropped rows verified genuine
near-dups (100% have a ≥0.8 same-answer same-source partner). Files on pod:
`real_math_dedup.jsonl`, `real_math_dedup_x.jsonl`, `real_math_clean.jsonl`.

### 3. Quality screening beyond arithmetic
Four detectors were built and hand-self-validated, then measured on the real
distribution — and the real-data misclassification rates DO NOT support using
them to delete data:

- answer-markers (checker1/2): flags 42% of rows but **93% are false**
  (the prevalent "prose restates the answer, then a same-valued \boxed" is a
  normal training pattern, not a defect). Only a *different* second \boxed is
  harmful, and that is rare.
- stem-restatement (checker3, 1.5%): ~2/3 genuine whole-question parroting —
  the one defensible deletion, in line with the short-CoT direction.
- inline equation (checker4, 3.9%): real but a formatting style, not an error
  (eqcheck already verifies the arithmetic); deleting it discards diversity.

Recommendation: do NOT bulk-delete on checker1/2/4; drop checker3 restatements
(~1.5%) only — applied to reach the 115,448 final clean. The "distinct boxed
answers" check self-validated 5/5 and fired on just 15 rows (0.013%), all of
which manual review showed to be mid-working \boxed emphasis on multi-part
questions (extract_boxed still resolves the last one correctly), i.e. **zero
genuine answer-conflict rows** — not a real failure mode to filter here.

### 4. External data (survey, no download)
Surveyed the public Chinese-instruction landscape (datasets-server metadata +
≤400-row samples each, no full download). No clean native-zh quality source
exists: the high-volume zh instruction corpora (BelleGroup/train_{0.5M,1M,2M}_CN,
shibing624/alpaca-zh) are all English-instruction machine translations — more
breadth/volume, not better quality; the native-zh ones (BAAI/COIG already local,
evol-instruct-zh, moss-003) are long-CoT, matching the already-rejected pattern.
Only worthwhile download if the goal is volume: **BelleGroup/train_2M_CN** (or
1M), ~19-24% of answers in the 60-132 band, standard instruction/input/output
columns. GB-scale, needs sha after download, requires aupai-fb/user approval.
### 5. Packed for the k6 A/B

Two SFT packs differing only in where the math comes from; base and general
replay are identical, so an eval gap between them is attributable.

| | sft_v8_fone.pt | sft_real_fone.pt |
|---|---|---|
| math | math_short_v8, 97,771 synthetic rows over 24,452 templates | real_math_clean_scrubbed, 114,908 human-written rows |
| general replay | alpaca_gpt4_zh 52,049 | alpaca_gpt4_zh 52,049 |
| packed rows | 4,342 | 4,676 |
| loss tokens | 12.21M (68.7%) | 13.80M (72.0%) |
| [NUM] share | 11.32% | 7.54% |

Both packed with `prepare_sft_math.py --fone`, which a FoNE base requires: it has
only ever seen a number as one [NUM] carrying a Fourier value.

### 6. Hand sampling: Belle is 38.7% defective, and every automated check passed it

Five independent random samples of `real_math_clean_scrubbed`, read row by row.

| source | sampled | defective | share of the set |
|---|---|---|---|
| belle | 31 | **12 = 38.7%** (95% CI 22-56%) | 83.5% (96,000 rows) |
| mxode | 17 | 0 | 13.3% |
| gsm8k | 18 | 0 (95% upper bound 8%) | 3.5% |

Fisher exact p = 2.87e-05. That puts 21,000-54,000 defective rows inside Belle.

Every defect class leaves the arithmetic *inside the equations* correct, so
`eqcheck.py` and the four format detectors of section 3 pass all of them:

- **`\boxed` contradicts the solution's own conclusion** (5 of 32): "8 × 6 = 48,
  so the answer checks out" → `\boxed{48}` where the answer is 8; "10 × 5 = 50,
  the school has 50 basketballs" → `\boxed{10}`; "10-7=3, 3 are left" →
  `\boxed{10}`. The distinct-boxed detector cannot see these — there is only one
  `\boxed`, and the contradiction is between the prose and it.
- **the solution answers a different question** (3 of 32): with 4 yuan for a
  2-yuan purchase, asked how much more is needed, it answers 2 — the change, not
  the shortfall.
- **the problem itself is broken** (3 of 32): "150 yuan of savings earning 2 yuan
  a month, how many months" is unanswerable; the solution invents an equation with
  an undefined variable and asserts x=25.
- **the answer is in the wrong form** (1 of 32): asked for a fraction, `\boxed{20}`.

A string-level prose-vs-boxed detector, calibrated against 12 hand labels (2 true
positives, 0 false positives, 1 false negative), finds only **1.1%** at scale.
The gap to 38.7% is real, not a threshold artifact: in most contradictions the
prose's last number equals the boxed value and the boxed value is still the wrong
answer to the question. Only a semantic comparison catches those.

**This re-attributes the harmful-SFT finding.** sft_k4's damage (k5 base 51.2% →
44.8% on math-500, p=0.043) was recorded above as answer-length mismatch, 156
against the eval's 85. sft_k4 was built from Belle, so its defect rate is the more
likely cause and the length is correlated with it. Section "math_short_v8" keeps
the length table because matching the eval's length is still right; it is no
longer the explanation.

**Method note.** Section 3's detectors were self-validated on hand-built positive
and negative cases, which is what caught their 93% false-positive rate. The same
validation gave a 38.7% false-negative rate here, because the constructed cases
were format-level and Belle's defects are semantic. Self-validation bounds false
positives; only hand sampling of the real distribution bounds false negatives.

### 7. Three SFT arms for k6

| | math source | packed rows | loss tokens |
|---|---|---|---|
| sft_v8_fone.pt | 97,771 synthetic, program-verified | 4,342 | 12.21M |
| sft_real_fone.pt | 114,908 real, Belle included | 4,676 | 13.80M |
| sft_mix_fone.pt | v8 + 19,336 real without Belle | 4,834 | 13.57M |

General replay is alpaca_gpt4_zh 52,049 in all three, and the base is the same, so
a gap between them is attributable to the math source.

### 8. What failure looks like at 3.6% (hand-read, 2026-08-28)

Reading ckpt_sft_k5_ctrl's generations rather than its score. The correct ones are
correct for real -- three-step 工程 and 最大公因数 problems with sound derivations,
not lucky guesses:

    甲乙合作10天，甲乙效率比2:3，甲单独几天？
    合作效率 = 1 ÷ 10 = 1/10 → 份数和 = 2 + 3 = 5 → 每份 = 1/50
    甲的效率 = 2 × 1/50 = 1/25 → 1 ÷ (1/25) = 25   ✓

Two failure clusters, and neither is what the FoNE work assumed:

**Arithmetic slips inside a correct structure.** 1/10 − 1/15 = 1/15 (it is 1/30);
1/8 + 1/6 + 1/12 = 1/3 (it is 3/8); 120 × 30/100 = 72 (it is 36). The setup is
right and the number is wrong.

**Template over-generalization.** Every failure reaches for 效率 = 1 ÷ n, including
問題 about trains, boats and circular tracks where it means nothing. The model
produces well-formed lines with no semantic content -- one invented 罚款 (a fine)
in a river-current problem.

The obvious mechanism -- FoNE cannot represent a fraction, since 1/15 splits into
two separate numbers -- **is not supported**. Fractions appear in 20.2% of gold
solutions and 51.1% of the model's generations, but accuracy barely differs:
4.2% with a fraction against 3.5% without. Fractions are over-produced, not
selectively fatal.

So the constraint is problem-type generalization, not arithmetic. That is
consistent with FoNE's null result and it points somewhere else: the training mix
teaches a few templates well and the eval asks for many. `mathbank` has 943 L3/L4
programs; a model that has memorized 943 shapes and meets a 944th has nothing to
fall back on.

## Domain blocks — mix_scale domains (pod, stamped 2026-08-29, re-stamped 2026-08-30)

Each block is the rebuild contract from docs/standards/corpus_rebuild.md. The
fingerprint is `scripts/corpus_fingerprint.py`'s value (name+size+sha256 of the
first/last 64KB of the shards — content-based and transfer-invariant since
2026-08-30; the former name+size+mtime scheme red on every podput/rsync while
bytes stayed identical); `harness check` corpus_fp_matches compares it to the
live directory. Bytes are `du -sb` on the pod. web_hq is the lost domain: its fineweb2 bytes
were gone from /work, /data00 and data/raw on 2026-08-30, and no build command
was recorded — the gap this section exists to close.

### textbook

- Result: fingerprint: 3f237c5191cb8571, 920,692 docs, 8,119,279,450 bytes
- Build: unrecorded (pre-0830v1; reconstruct from git history before any rebuild)

### wiki

- Result: fingerprint: 44eb88458cfb3ed5, 212,413 docs, 910,063,480 bytes
- Build: unrecorded (pre-0830v1; reconstruct from git history before any rebuild)

### en

- Result: fingerprint: 9f37cb35219bb617, 117,532 docs, 811,935,014 bytes
- Build: `scripts/build_domains.sh` (en): `build_corpus.py --domain en --filters light --target_tokens 1e9 --no_near_dedup --source jsonl:data/cosmopedia_extra.jsonl --source jsonl:data/en_textbook.jsonl`
- Fetch: frozen files above (sha256 anchors 2026-08-27)

### math

- Result: fingerprint: e1e86aa0b594c5ed, 74,158 docs, 335,034,730 bytes
- Build: `scripts/build_domains.sh` (math): school_math_r1_zh + en_math_text at `--target_tokens 8e8`, then synthetic math_short_v* at `--target_tokens 2e8` (near-dedup on the synthetic pass only)
- Fetch: frozen files above; math_short_v* generated by `mathbank/run_math_short.py` (see its block)

### code

- Result: fingerprint: 5bb775eae9029669, 199,976 docs, 232,707,019 bytes
- Build: `scripts/build_domains.sh` (code): `build_corpus.py --domain code --filters light --target_tokens 1e9 --no_near_dedup --source jsonl:data/code_filtered.jsonl`
- Fetch: frozen file above

### chat

- Result: fingerprint: 080fdd63401b4ff9, 160,414 docs, 163,274,317 bytes
- Build: `scripts/build_domains.sh` (chat): `build_corpus.py --domain chat --filters light --target_tokens 1e9 --no_near_dedup --source jsonl:data/coig.jsonl --source jsonl:data/alpaca_gpt4_zh.jsonl`
- Fetch: frozen files above

### web_hq

- Result: fingerprint: 30838d423348b2e5, 1,366,324 docs, 5,914,966,151 bytes, 1.434B
  tokens (2026-08-30)
- Build: `datagen/build_corpus.py` built the `web` domain (moved wholesale into `web_hq`
  on 2026-08-30; that directory no longer exists) with the full garbage chain
  (pass1+pass2+pass3, `AUPAI_NO_GARBAGE` unset) and `reject_holdout` per document. The
  523 LAMBADA-zh holdout documents were then carved out by `remove_holdout2.py`, leaving
  4,731,988 documents. `web_hq` is the first 62 shards of that post-removal `web`, moved
  wholesale — not a selection of the best N, and **the quality cut was not applied**.
- Deviation, on purpose: the quality-head cut that the pre-reset `web_hq` recipe called
  for is skipped. The six budget points therefore run on uncut corpus and become the F
  arm of the quality-cut experiment (`docs/lessons/quality_ab_design.md`); the W arm runs
  the cut separately. This is experiment design, not an omission.
- Fetch: fineweb2 via `HF_ENDPOINT=https://hf-mirror.com`; raw parquet at
  `/data00/fw2raw` (19GB, 4 files). The pre-0830v1 shards were lost; this is the rebuild.
- Holdout verification: two independent sources agree. aupai-3b carved 523 documents by
  deterministic hash against `lambada_zh_src.jsonl`; lessons-b0 independently confirmed
  the 523 ids match its own set and that a streaming scan of all 4,731,988 documents
  finds zero. Order was exclusion first, fingerprint last.
- Stamp: `build_corpus_stats.json` was written post-hoc from the live directory on
  2026-08-30, not by `build_corpus.py` at build time — the build-time stats file did not
  survive the move from `web`. The file says so in its own `stamped_at_build: false`.
  The stamp still does its forward job: any later change to the directory fails
  `corpus_fp_matches`.
- Known property: 15.2% of documents are predominantly Traditional Chinese (scout
  measurement, 9,000 documents from the first 3 shards; character-level 14.81%). The
  tokenizer costs 1.164x more tokens per character on Traditional-heavy documents
  (corpus-level) and 1.33-1.43x on controlled minimal pairs. Overall this is about
  +2.5% tokens for the same text — recorded, not corrected. lessons-44 owns the
  full-corpus value.
