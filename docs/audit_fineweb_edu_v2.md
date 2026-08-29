# Audit: opencsg/chinese-fineweb-edu-v2 and opencsg/Fineweb-Edu-Chinese-V2.1

Measured 2026-08-29. Scope: whether either enters the mix, and at what cut.

## Decision

| target | verdict | why |
|---|---|---|
| `opencsg/chinese-fineweb-edu-v2` (V2) | **REJECT** | its `score` column is the only cut it offers and that column does not rank quality: Spearman **+0.074** against our own head (n=6,020) and NON-monotonic, with its **top decile the worst of the ten**. Superseded by V2.1, with which it partly overlaps. |
| `opencsg/Fineweb-Edu-Chinese-V2.1`, band `4_5` | **KEEP** | the band directory IS monotonic and separates hard: 28.5 / 58.1 / **83.2%** of `2_3` / `3_4` / `4_5` score above our `web_hq` median. `4_5` passes our own filters at **89.5% ± 1.1** (n=2,918) against `web_hq`'s own source at 18% by hand. |
| V2.1 bands `2_3`, `3_4` | **CUT** | `2_3` sits below `web_hq` and is rejected. `3_4` is at parity with `web_hq` and is a second tier only if `4_5`'s 22.9B tokens prove insufficient; it needs a per-document score pass first. |

**Executable predicate**

```
path matches  4_5/*.parquet                                   # 17,790,513 docs, 25.6B tokens
  -> full-width punctuation normalisation (see "ASCII punctuation" below)
  -> python datagen/build_corpus.py --domain web_edu \
       --source jsonl:<...> --filters web --host_cap 0        # 89.5% +/- 1.1 (n=2,918)
  -> data/corpus/web_edu/
```

Yield **22.9B tokens** measured with `data/tokenizer.json`. As a fraction of the 420B headline
figure this brief quoted for V2: **10.0%** of the V2-equivalent 229.5B, or **2.5%** of V2.1's
915B. Domain name `web_edu`. Proposed weight below.

V2.1 is **ungated, Apache-2.0, not deprecated** (last modified 2026-01-28, 29,286 files,
2,416 GB). It supersedes V2 and it is the thing to audit; V2's card says so and the measurements
here agree.

## Sampling design

| what | design | n |
|---|---|---|
| V2 | ALL 301 row groups of shard `00000`, 20 rows drawn at random per group | 6,020 |
| V2 contamination, self-dedup | every row of shard `00000` | 300,271 |
| V2.1 `2_3` | 4 shards drawn at random of 9,767, all row groups, 600/group | 15,000 |
| V2.1 `3_4` | 5 shards of 9,767 | 6,400 |
| V2.1 `4_5` | 12 shards of 9,745 | 2,918 |
| reference: our `web_hq` | 50 rows per shard, all 124 shards | 6,200 |
| reference: our `textbook` | 80 rows per shard, all 79 shards | 6,320 |

**One shard of V2's 625 was available locally, so nothing here is a cross-shard measurement of
V2.** Shard-stratification within V2.1 is real (shards drawn at random) but thin: 4-12 of 9,767.

**The brief's premise that V2's `source` is {CCI3, IndustryCorpus2} is wrong, and wrong in the
way AGENTS.md warns about.** Rows are blocked by source inside a file — row groups 0-49 are 100%
CCI3 — so reading in order sees two sources. Reading all 301 row groups finds **eight**:

| source | share of V2 shard | our score, median |
|---|---|---|
| IndustryCorpus2 | 25.4% | -1.055 |
| CCI3 | 18.6% | -0.840 |
| TeleChat | 15.1% | -0.912 |
| ChineseWebText | 12.2% | -0.702 |
| wanjuan | 11.5% | -0.851 |
| SkyPile | 10.3% | -0.744 |
| WuDao | 4.7% | -1.064 |
| MiChao | 2.1% | -1.132 |

Source is a shard-level constant in V2.1 too, so the source mix reported per band there rests on
4-12 shards and is not a reliable estimate of the band's composition.

## 1. Contamination — clean

`scripts/holdout.is_holdout` (1,532 hashes from `math_test_500` + `math_hard_eval_1k`), whole
document and first 60 lines, over **every row of V2 shard 00000**:

```
300,271 docs scanned, 0 eval questions found (0.00000%)
```

Rule-of-three upper bound 0.001%. Not run on V2.1 (no full shard held locally); **run it before
ingest** — that is finding #1 of `docs/review_2026-08-26.md`.

## 2. Near-duplication

| against | measurement | result |
|---|---|---|
| our `web_hq` (1,126,846 docs, all 124 shards) | bottom-k=128 MinHash, char-8-gram, max Jaccard per candidate | J>=0.8: **0.083% ± 0.023** (5 / 6,020) |
| | | J>=0.5: 0.216% ± 0.037; median J 0.008 |
| within V2 shard `00000` | same sketch, 16-band LSH, all 300,271 docs | J>=0.8: **0.162%** (486 docs, 3,003 pairs) |
| within V2 shard `00000`, exact | whitespace-normalised hash | 13 / 300,271 = 0.004% |
| V2 vs V2.1 | exact text, 24,318 V2.1 docs against 1/625 of V2 | 3 hits |

The web_hq job is checked by a control asserted in-run (a document finds itself at J=1.000, its
80% truncation at 0.662) — an earlier run of it reported J=0.000 everywhere while silently
reading zero web documents, because `web_hq` rows key on `content` and not `text`.

**Overlap with our existing corpus is negligible: ingesting this is new information, not a
re-copy of fineweb2.** Within-candidate near-duplication is also low, but only *within* a shard;
cross-shard duplication is unmeasured and is the one that would inflate an epoch count.

The V2-vs-V2.1 probe puts 3 exact hits where full containment predicts ~9 (24,318 docs, each with
~0.1% chance of landing in the one V2 shard held). That is roughly a third overlap, 95% Poisson
interval on 3 hits **7%-95%** — too weak to quote, strong enough to say: **do not ingest both.**

## 3. Our own filters — `build_corpus.py --dry`

| corpus | kept | top rejects |
|---|---|---|
| V2 (n=6,020) | **80.8%** | not_zh 9.6, unfinished 3.3, garbage_topic 2.4, nav_menu 1.1 |
| V2.1 `2_3` (n=15,000) | **78.2%** | unfinished 12.4, garbage_topic 2.6, boilerplate 2.1, not_zh 1.7 |
| V2.1 `3_4` (n=6,400) | **84.3%** | unfinished 7.6, not_zh 2.1, garbage_topic 2.1 |
| V2.1 `4_5` (n=2,918) | **89.5%** | garbage_topic 4.9, not_zh 2.0, bad_bytes 0.9 |

Monotonic in the band. `near_dup` here is meaningless — the MinHash set is seeded only with the
sample itself, so it saw 1-5 hits and is not the number in section 2.

## 4. Traditional Chinese — absent, no conversion pass needed

Definition used: a document counts as traditional when >=2% of its hanzi are codepoints that
change under `data/t2s_table.json` (3,553 entries, single-codepoint opencc).

| corpus | traditional docs | traditional share of all hanzi |
|---|---|---|
| V2 | 0.41% ± 0.17 (n=5,583) | 0.11% |
| V2.1 `2_3` | 0.34% ± 0.09 (n=14,990) | 0.09% |
| V2.1 `3_4` | 0.28% ± 0.13 (n=6,398) | 0.08% |
| V2.1 `4_5` | 0.14% ± 0.13 (n=2,918) | 0.05% |

Against fineweb2's recorded 59.4%. The 59.4% was measured with a definition that was not
recorded, so the two are not strictly comparable, but a 400x gap does not turn on the threshold.
`scripts/t2s_corpus.py` is not needed for this source.

Compression, `data/tokenizer.json` (fingerprint `0bce3584bc24f255`), 2,000 documents each:
V2 **1.737** chars/token, V2.1 `2_3` 1.516, `3_4` 1.550, `4_5` 1.636 — all above the 1.45 the
web domain reached after t2s conversion.

## 5. ASCII punctuation — the one defect that needs a preprocessing pass

Fraction of hanzi-majority documents where ASCII `,` `.` outnumber full-width `，` `。`:

| corpus | rate |
|---|---|
| our `web_hq` | **1.89%** (n=6,200) |
| V2, all | 34.9% ± 1.3 (n=5,570) |
| V2 / IndustryCorpus2 | **100.0%** (n=1,091) |
| V2 / TeleChat | 64.4% ± 3.1 (n=905) |
| V2 / WuDao | 55.0% ± 5.8 (n=280) |
| V2 / CCI3, wanjuan, ChineseWebText, SkyPile, MiChao | 0.8%-5.2% |
| V2.1 `2_3` | 87.1% ± 0.5 (n=14,983) |
| V2.1 `3_4` | 67.8% ± 1.1 (n=6,396) |
| V2.1 `4_5` | 64.2% ± 1.7 (n=2,917) |

Content is unaffected; the sentence-boundary token is. Our vocabulary was trained on a corpus
that is 98% full-width, and 64% of the incoming band is not. This is a `str.translate` away and
must run before `build_corpus.py`, not after — several of our filters count punctuation.

## 6. Score deciles against our own judgement — the centre of the audit

Distilled head (`data/quality_head.pt` on `ckpt_k5_clean_0827.pt` mean hidden state, AUC 0.823
against hand labels). Its domain of validity is **web pages judged by the 27B**, which is what
these are. The known out-of-domain artefact reproduces exactly as a control: `textbook`
(cosmopedia, from-scratch synthesis) lands at median **-1.66**, below raw web, which is not
credible and is why it is a control and not a result.

| population | n | our median | >= web_hq median (-0.971) | >= web_hq p75 (-0.794) |
|---|---|---|---|---|
| our `web_hq` | 6,200 | -0.97 | 50.0% (by definition) | 25.0% |
| our `textbook` | 6,320 | -1.66 | 3.0% ± 0.4 | 0.4% ± 0.2 |
| V2 | 6,020 | -0.87 | 59.1% ± 1.2 | 41.6% ± 1.2 |
| V2.1 `2_3` | 15,000 | -1.14 | 28.5% ± 0.7 | 12.1% ± 0.5 |
| V2.1 `3_4` | 6,400 | -0.89 | 58.1% ± 1.2 | 39.0% ± 1.2 |
| V2.1 `4_5` | 2,918 | **-0.58** | **83.2% ± 1.4** | **71.3% ± 1.6** |

**V2's own score is not a threshold.** Spearman against our head +0.074 (n=6,020), and our mean
by their decile is non-monotonic with the top decile the worst:

| their decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| our mean | -0.978 | -0.960 | -0.932 | -0.963 | -0.902 | -0.859 | -0.873 | -0.877 | -0.885 | **-0.971** |

This is the `opencsg/Fineweb-Edu-Chinese` finding (52 / 66 / 59% usable, top band dirtiest)
happening a second time in the same family. It is a shape, not an accident.

**V2.1's band directory is a different object and it works.** Our mean by band: `2_3` -1.159,
`3_4` -0.912, `4_5` -0.645; Spearman across the three bands **+0.466** (n=24,318). Inside a band
the continuous score is again worthless or inverted: `2_3` +0.172, `3_4` +0.305, `4_5` **-0.105**.
**Cut on the directory. Never on the number inside it.**

## 7. Thirty documents read by hand — V2

Stratified 3 per score decile. Judged unusable when SEO/ads, machine-translation garble, spliced
unrelated fragments, misinformation stated as fact, boilerplate-dominated, or heavily corrupted.

**23 / 30 usable = 76.7% ± 15.1 (n=30).** Against 18% for the raw fineweb2 web read the same way.
The 7 rejects: one spliced 3C-certification/workplace-safety document, one "8 foods that kill
cancer cells", one Descartes summary spliced onto an unrelated book review, one article beginning
mid-sentence with its antecedents missing, one long-division page carrying false arithmetic
(`35568+78=456`, `818÷60≈13.63` next to `约等于100`), one context-free chemistry fragment, and one
machine translation. Rejects fell in deciles 3-7 and none in 0-2 or 8-9, but at 3 documents per
decile that is noise, not a shape.

The worst, and the kind our filters do not catch (decile 7, their score 0.756, source CCI3) — a
zh-TW article run through vocabulary substitution until "flying" became "in-flight" everywhere:

> 事实上，虽然四轴飞行器在无人机领域有数广泛应用，但依然相比之下比不上大大自然"生产"的**飞行中**生物 …
> 蝙蝠的**飞行中**能力比鸟类无法仿效得多 … 需要通过类似于"曲肘"或"刷腕"的动作**掌控**向前或向后**飞行中**

Second, false content in fluent prose, which no register-based filter can see (decile 3, 0.663,
IndustryCorpus2):

> 8大能杀死癌细胞的食物 (1)茄子:"霜打茄子"是好药 … 曾有试验从茄子中提取的一种无毒物质,用于治疗胃癌,子宫颈癌等收到良效

Third, the top of the range, to show what the band buys (decile 8, 0.802, WuDao):

> 每年的12月是东帝汶人独立建国血泪史最难忘的时刻。1975年11月葡萄牙殖民政府撤离,东帝汶在11月28日宣布独立建国,
> 短短9天之后,印度尼西亚就派兵入侵占领血腥屠杀,时间持续了24年之久

Ten more read from V2.1 `4_5`: 7 clean informative prose (family-violence law, metallurgy history,
brain organoids, Eocene artiodactyl fossils, glioblastoma immunology, dairy genetics, environmental
economics), 3 defective — one two-column PDF extraction interleaved line by line, one exam-question
splice, one legal-QA page with repeated headings. n=10; ±28pt. **The hand-read agrees with the
score distribution's ordering, so the head is inside its domain here.**

## 8. What a full ingest costs

The pod cannot reach Hugging Face. Measured today from inside the container: `pypi.org` 200,
`huggingface.co` and `hf-mirror.com` both time out. **Every byte has to be pulled to the laptop
and pushed over `tn`.** Measured push rate **1.16 MB/s** (32 MB in 28.9 s), and a 124 MB push
dropped the connection at 6.5 MB, so transfers must be chunked below ~60 MB.

| | V2.1 `4_5` (recommended) | V2.1 `3_4` | V2 whole |
|---|---|---|---|
| source bytes | 74.3 GB | 857 GB | 607 GB |
| documents | 17,790,513 | 289,975,835 | 187,669,375 |
| tokens, `data/tokenizer.json` | **25.6B** | 329B | 230B |
| tokens after our filters | **22.9B** | 277B | 186B |
| card's own token claim | 46B | 530B | 420B |
| download to laptop @ 8.7 MB/s | 2.4 h | 27 h | 19 h |
| push to pod @ 1.16 MB/s | **17.8 h**, ~1,300 chunks | 205 h | 145 h |
| filtered jsonl on pod | ~110 GB | ~1.2 TB | ~1.0 TB |
| tokenization @ 2.3M tok/s | 2.8 h | 33 h | 23 h |
| one pretrain epoch, 8xH20 @ 680K tok/s | 9.4 h | 113 h | 76 h |

`/work` has 1.2 TB free. **`4_5` fits; `3_4` and V2 do not fit as jsonl at all.** Filter and gzip
on the laptop before pushing; that is also what makes the chunk count tolerable.

Every card token figure is about 1.7-1.8x ours, consistently across all four rows — their
tokenizer, not a discrepancy in the data.

## 9. Proposed weight

Corpus v3 is 3.3B tokens at **16.5 tokens/parameter**; `web_edu` at 22.9B alone is 7x the whole
current corpus and 16x `web_hq`'s 1.42B. Proposal for the next mix, ~25B total:

| domain | v3 weight | proposed | note |
|---|---|---|---|
| `web_edu` (new) | — | **0.55** | real web, filtered; the anchor at scale |
| `web_hq` | 0.42 | 0.18 | keep, unchanged pool, ~2 epochs |
| `textbook` | 0.36 | **0.05** | from-scratch synthesis; `docs/synthetic_data_standard.md` puts the sub-1B ceiling at 5% and v3's 36% is above it |
| `wiki` | 0.055 | 0.05 | |
| `math` | 0.045 | 0.07 | |
| `chat` | 0.02 | 0.03 | |
| `en` | 0.065 | 0.05 | |
| `code` | 0.035 | 0.02 | |

**This table is a proposal and no mix file was edited.** The `textbook` number in particular is
blocked: criterion 6 (`ckpt_tb36` vs `ckpt_tb05` on a held-out web/wiki/math loss) has not
reported, and `docs/synthetic_data_standard.md` says not to write a from-scratch weight before it
does. `web_edu` at 0.55 puts real human web text back at the plurality, which is the property v3
was designed around.

## The falsifying experiment

Two pretrains at equal token budget, identical seed and card count, differing only in whether the
new tokens come from `web_edu` or from repeating `web_hq` — compared on held-out loss over
web / wiki / math and on math-hard. Cost about 20 GPU-hours at 3B tokens each. It fails this
decision if `web_edu` does not beat the repeat, which would mean the 22.9B buys epochs and not
information. `web_edu` beating `textbook` on the same design is the second arm and is the one
that would justify moving the 0.36.

## What could not be measured

- **Cross-shard duplication in either dataset.** One V2 shard of 625 and 21 V2.1 shards of 29,286
  were held. Within-shard near-duplication is 0.162%; the cross-shard number could be anything
  and it is what decides whether 22.9B tokens is really 22.9B.
- **Contamination in V2.1.** Only V2's shard was scanned. Run `is_holdout` over every ingested
  `4_5` shard before it enters a mix.
- **Whether the source composition per V2.1 band is what 4-12 shards say.** Source is a shard-level
  constant, so the per-band source table has an effective n of the shard count, not the row count.
- **V2 vs V2.1 overlap to better than a Poisson interval of 7%-95%.** Settling it needs one full
  V2.1 shard against several full V2 shards, about 2 GB of download and an hour of MinHash.
- **Whether the 27B on its binary rubric agrees with the distilled head on this source.** The
  cross-source comparison in AGENTS.md required one judge on one rubric; only the head was run
  here, and it was run inside its declared domain (web pages).
- **Anything about the `2_3` band beyond 4 shards**, which is where 61% of V2.1's tokens are.
