# Audit — `opencsg/chinese-cosmopedia` (mix domain `textbook`, weight 0.36)

Run 2026-08-29. Protocol: `.claude/agents/data-auditor.md`. No mix was edited.

## Verdict

**CUT.** The domain stays; the weight and the ingest predicate change.

| | now | proposed |
|---|---|---|
| mix weight | 0.36 (1.19B of 3.3B tokens) | **0.12** (0.40B tokens) |
| ingest filter | `--filters light` (length/bytes/holdout only) | `--filters web`, garbage patterns disabled |
| `data_format` admitted | all 5 | `college textbook`, `middle school textbook`, `wikihow` |
| per-document cut | none | distilled score >= -1.60 |
| epochs | 1 | 1 |

Predicate, executable:

```python
data_format in {"college textbook", "middle school textbook", "wikihow"}
and reject_reason_without_garbage_patterns(text) is None
and quality_head_score(text) >= -1.60
```

Measured yield of that predicate: **33.8% of documents (±0.60pt, n=24,000), 35.0% of
characters**. Yield before it: 99.95% of documents (`--filters light`, n=4,000). Pool after
the cut is ~1.6B tokens from the 8 parquets already fetched, so 0.40B is under one epoch.

## Sampling design

| measurement | population | draw | n | CI half-width |
|---|---|---|---|---|
| contamination, exact dup, band counts, leak rate | shard 00000 | census | 269,551 | — |
| our filters, traditional, specificity | shard 00000 | uniform random rows | 4,000 | 1.2pt |
| distilled score, predicate yield | shards 00000–00003 (the 4 that fed `data/corpus/textbook`) | 6,000 random rows per shard | 24,000 | 0.6pt |
| hand read | the 4,000-row draw, stratified over 5 formats × 3 score bands | — | 40 | 14.5pt |

Shards 00004–00061 were not sampled: only 00000–00003 were ingested, so they are not in the mix.
Documents were never truncated.

## 1. Contamination — clean

**0 eval questions in 269,551 documents** (`scripts/holdout.py`, full text plus the first 60
lines of each). 95% upper bound 1.1e-5. The existing `scripts/scan_contamination.py` read
60,000 rows of the first 2 parquets in sorted order; this is the census and it agrees.

Exact duplicates inside the shard: 0/269,551.

## 2. Our own filters

The corpus on the pod was ingested with `--filters light`, which checks length, byte validity
and the eval holdout and nothing else. `data/corpus/textbook/build_corpus_stats.json` records
920,692 kept, 52 rejected. The claim in `CLAUDE.md` that textbook is "100% pass through our own
filters against web's 18% by hand" describes the light filter, not the web chain.

Under the full `web` chain, n=4,000:

| reason | rate |
|---|---|
| kept | 80.7% ±1.2 |
| not_zh | 7.3% |
| garbage_topic | 5.8% |
| symbols | 3.9% |
| boilerplate | 1.3% |
| dup_lines | 0.8% |
| urls, bad_bytes | 0.1% |

`garbage_topic` is a false positive here. The patterns in `filters/pass*_garbage.py` were
written against gambling and adult web SEO and fire on 游戏 / 足球 / 运动员; the documents they
reject are about para-athletes, ichneumon wasps, dinosaurs and game design. Excluding them the
pass rate is **86.6% ±1.1** (n=4,000), confirmed at **87.2% ±0.4** on the 24,000-row pod draw.

**The pod checkout does not contain `filters/pass{1,2,3}_garbage.py`.** `build_corpus.py` loads
them with `os.path.exists` and sets `GARBAGE = None` when they are absent, so any corpus built
on the pod silently skips that filter and the local and pod runs of the same command return
different pass rates. That is how the 80.7 / 87.2 discrepancy was found.

Per band (n=4,000, web chain including garbage patterns):

| `data_format` | n | kept |
|---|---|---|
| normal story | 893 | 95.7% |
| middle school textbook | 1,146 | 83.3% |
| preschool story | 427 | 82.4% |
| college textbook | 781 | 73.6% |
| wikihow | 753 | 65.3% |

| `source` | n | kept |
|---|---|---|
| wiki | 43 | 93.0% |
| knowledge qa | 596 | 90.3% |
| baike | 2,814 | 86.0% |
| blog | 547 | 42.4% |

`blog` fails almost entirely on `not_zh`: its documents are code tutorials (PHP, Go, C) where
Chinese is under 60% of non-whitespace. Published-score decile has no relation to pass rate —
0.776 to 0.848 across all ten, no monotone trend.

## 3. Near-duplication — nothing found

MinHash LSH (128 perms, 16 bands — `datagen/build_corpus.MinHashLSH`):

| test | hits | n | 95% upper bound |
|---|---|---|---|
| cosmopedia vs 18,000-doc `web_hq` + `wiki` sample | 0 | 8,000 | 0.038% |
| cosmopedia vs cosmopedia | 0 | 2,000 | 0.15% |
| exact key, inside shard 00000 | 0 | 269,551 | 1.1e-5 |

The cross-corpus test is sample against sample. A zero bounds the rate at which a cosmopedia
document collides with an 18,000-document slice of `web_hq` + `wiki`, not with all 1.42B tokens
of it. A full pass needs the whole index, which `build_corpus.py` builds anyway on ingest.

## 4. Is it anchored, and to what — the load-bearing question

`docs/synthetic_data_standard.md` records this source as from-scratch generation, "no checkable
relationship to its own `source`", on the strength of two sampled titles. That is wrong.

**2.97% of documents (8,003 / 269,551, ±0.06pt) name the seed document in their own text** —
「网页摘录」「给定的摘录」「上述文本」「原文中」. The content next to those references is the
seed's, not a topic's:

- 「在本单元摘录中，主芯片组为 Intel G31」
- 「摘录中提到，六安大市场·三期拥有八年的品牌美誉 … 2.5 万平方米小商品综合市场和 3 万平方米的家具、灯饰市场」
- 「根据网页摘录，庆云县交通局的主要职责包括 …」
- 「在上述摘录中，变量 `a` 被赋值为 100」 followed by `var a int = 100`

Per format: middle school textbook 4.4%, wikihow 4.4%, preschool story 2.9%, college textbook
2.8%, normal story 0.03%. The generator is handed a document. 2.97% is the rate at which the
prompt leaks into the output, not the rate of anchoring — it identifies the pipeline for all of
it, and the story formats hide it because narrative has no place to put the reference.

The hand read agrees: 魏守雷's fight record (1987-05-05, 177cm, bench 135kg, losses to ANDY
SOUWER and 宍戸大树 by year), 三重县's 5,776.56 km² and 25th rank, 张邦炜《宋代婚姻家族史论》
人民出版社 2003年12月 506页, 梳山村's 193 mu of cropland and 1,206 mu of forest. Those numbers
came from a baike or wiki page, not from a language model's prior.

**But the anchoring is diluted, and that is measurable.** Checkable-fact markers per 1,000
characters, same regexes on all three corpora:

| marker | cosmopedia (n=12,000) | `web_hq` (n=12,000) | `wiki` (n=6,000) | cosmo / web |
|---|---|---|---|---|
| year `19xx年`/`20xx年` | 0.197 | 1.093 | 5.895 | **0.18x** |
| year + month | 0.032 | 0.299 | 1.965 | **0.11x** |
| percentage | 0.092 | 0.757 | 0.312 | **0.12x** |
| number + unit | 0.224 | 1.032 | 0.856 | **0.22x** |
| 《title》 | 0.583 | 0.704 | 2.149 | 0.83x |
| any number | 6.78 | 12.97 | 23.42 | 0.52x |
| 本单元/本教程/我们将 | 0.392 | 0.034 | 0.002 | 11.4x |

Numbers survive at half the rate of filtered web, but the numbers that carry a *fact* — dates,
percentages, quantities with units — survive at one fifth to one ninth. Most of cosmopedia's
digits are section numbering. The one marker it leads on by an order of magnitude is
pedagogical scaffolding.

So it is neither of the two categories the literature measures. It is seeded on a document and
then rewritten into a register that discards most of the document's specifics and adds framing.
Per `docs/synthetic_data_standard.md`, the safe share is ~30% for anchored rephrasing (itself an
interpolation to 200M, on Pile perplexity, not accuracy) and under 5% for from-scratch. **This
sits between them and the correct weight cannot be read off either number.** The three
measurements below set it instead.

A direct subset test against retrieved seeds was attempted on the 12,169 rows with
`source == "wiki"`, matching against our 212,413-document zh-Wikipedia corpus by title. Only 89
matched and most of those matches are spurious (extracted "topics" like 中国, 概念, 介绍), so the
resulting number-overlap figure (0.139 in seed vs 0.167 against a random article, n=70) measures
the matcher, not the data. It is reported as failed, not as evidence.

## 5. The domain named `textbook` is 32.1% fiction

| `data_format` | share of shard | tutorial markers | dialogue | child markers |
|---|---|---|---|---|
| middle school textbook | 30.2% | 24.3% | 6.7% | 1.8% |
| normal story | 21.5% | 0.0% | 95.4% | 2.6% |
| college textbook | 20.0% | 81.3% | 4.4% | 0.5% |
| wikihow | 17.7% | 62.4% | 5.8% | 0.6% |
| preschool story | 10.6% | 49.3% | 37.7% | 31.8% |

`preschool story` (28,516 rows) was missed by the earlier read of this dataset, which recorded
four formats. Its label is wrong about half the time: **49.3% ±0.58 carry tutorial markers**
(课程单元 / 教程标题 / 步骤 N / 本单元) and only 31.8% carry any child-story marker. Three of the
four `preschool story` documents in the hand read are a professional-certification exam guide, a
wikihow about Mie Prefecture, and a reading guide to a Song-dynasty academic monograph.

`normal story` and `preschool story` together are 32.1% of the domain. Whatever the seed
contributes, a generated short story about invented people is not textbook prose and should not
be weighted as if it were.

## 6. Distilled scorer

`CKPT=ckpt_k5_clean_0827.pt TOKENIZER=data/tokenizer_k5.json datagen/train_quality_head.py
--score`, the logistic head on the frozen 200M's mean hidden state.

Deciles:

| corpus | n | d1 | d2 | d3 | d4 | median | d6 | d7 | d8 | d9 |
|---|---|---|---|---|---|---|---|---|---|---|
| cosmopedia | 24,000 | -2.23 | -2.06 | -1.92 | -1.79 | **-1.665** | -1.55 | -1.43 | -1.30 | -1.14 |
| `web_hq` | 12,000 | -1.17 | -1.12 | -1.08 | -1.03 | **-0.972** | -0.91 | -0.83 | -0.73 | -0.61 |
| `wiki` (zh Wikipedia) | 6,000 | -2.06 | -1.91 | -1.78 | -1.66 | **-1.552** | -1.42 | -1.26 | -1.09 | -0.89 |

**Do not use this to rank cosmopedia against web.** It puts zh-Wikipedia below filtered web
crawl and level with cosmopedia. Wikipedia is not worse than a filtered CC crawl on any
educational rubric, so the head's ordering does not transfer off the web pages the 27B labelled.
3.16% of cosmopedia sits above `web_hq`'s median, and that number means nothing.

**Within cosmopedia it is usable, and it agrees with the hand read.** Per format:

| `data_format` | n | median | d1 | d9 |
|---|---|---|---|---|
| wikihow | 4,360 | -1.486 | -1.92 | -1.09 |
| middle school textbook | 7,059 | -1.488 | -1.91 | -1.06 |
| college textbook | 4,899 | -1.505 | -1.86 | -1.11 |
| preschool story | 2,505 | -1.977 | -2.52 | -1.43 |
| normal story | 5,177 | -2.125 | -2.34 | -1.89 |

The head separates the three expository formats from the two story formats by 0.5–0.6 with
almost no overlap, which is the same split the hand read produced. That is why the predicate
uses it as a within-source threshold and not as a cross-source comparison.

Per `source`: blog -1.261, wiki -1.178, knowledge qa -1.595, baike -1.776.

Per published-score decile (n≈2,400 each):

| published decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| published mean | .773 | .801 | .817 | .830 | .839 | .848 | .856 | .864 | .873 | .889 |
| distilled median | -1.88 | -1.79 | -1.77 | -1.69 | -1.63 | -1.63 | -1.60 | -1.58 | -1.55 | **-1.58** |

**Spearman(published, distilled) = +0.208, n=24,000** — the same as the +0.198 this project
already measured on a different opencsg source. The top published decile is not the best by our
judgement; decile 8 is. The published `score` is not a usable cut. Its full range is 0.750–0.989
with mean 0.839, and it splits the formats by only 0.022 (college textbook 0.848 vs normal story
0.826) while our head splits them by 0.62.

## 7. Traditional Chinese — negligible

0.275% ±0.16 of documents carry more than 0.5% traditional-only characters (n=4,000); 0.005% of
all hanzi are traditional-only. No conversion pass is needed. Compare fineweb2's Chinese slice at
59.4%.

## 8. Forty documents read by hand

Stratified over 5 formats × 3 published-score bands. Each classified on what it is:

| | count | rate (±14.5pt) |
|---|---|---|
| expository, real subject, carries specific checkable content | 13 | 32.5% |
| generic exposition, definitions and taxonomy only, nothing checkable | 9 | 22.5% |
| fiction: invented characters and plot | 12 | 30.0% |
| carries a confidently stated false claim | 6 | 15.0% |
| `data_format` label does not match the content | 3 | 7.5% |

Categories overlap: 5 of the 12 fiction documents wrap real anchored facts (a fighter's
competition record, a soybean cultivar's trial data) in an invented plot, and one of the six
false-claim documents is also fiction.

n=40 resolves these to ±11–15pt, which is enough to say 32.5% is not 82% and not 18%, and not
enough to separate 15% from 25% on the false-claim rate. Resolving that to ±5pt needs n≈196.

For contrast, hand-reading 180 random `web` documents found 18% worth training on. Cosmopedia's
expository half is better than raw web and well short of what "textbook" implies.

**Worst document read** (`college textbook`, published score 0.771, index 156905) — fabricated
culinary history and medical claims, stated without hedging:

> 椰子燕窝是一种具有悠久历史的客家菜肴 … 椰子燕窝的起源可以追溯到几百年前的清朝时期 …
> 而燕窝，则作为一种珍贵的滋补品 … 于是，椰子燕窝这一独特的菜肴便应运而生 … 因此，椰子燕窝
> 在传统医学中也被广泛用于治疗各种疾病，如咳嗽、哮喘、皮肤干燥等。

Hakka cuisine is inland and mountainous; coconut and bird's nest are not its ingredients, there
is no Qing-dynasty Hakka dish by this name, and the therapeutic claims are invented. Nothing in
the filter chain or in either quality score can see this: it is well-formed, on-register,
correctly punctuated Chinese prose.

**Second** (`college textbook`, 0.875, index 239309) — false physiology presented as mechanism:

> 坐姿侧展式可以加强腿部肌肉的弹性，减少大腿部位的多余脂肪 … 此动作有助于增强脊柱的弹性，
> 提高颈椎的力量，从而减轻或消除颈椎、腰、背疼痛。

Spot fat reduction does not occur. This is the highest-scoring band of the published score.

**Third** (`normal story`, 0.863, index 137685) — the seed blog post's advertising copy survives
the rewrite as plot:

> 在一次偶然的机会中，他发现了一款价格实惠的物理短网课。虽然只有短短八节课，但性价比极高，
> 且难度适中，非常适合他这样的高一高二学生。李明决定试试这个网课 …

The generator did not remove the seed's commercial intent; it gave it a narrator.

## 9. Why 0.12

| evidence | direction |
|---|---|
| checkable-fact density 0.11x–0.22x filtered web per character | down |
| 32.1% of the domain is fiction under a `textbook` name | down |
| 15% ±11 of documents carry a confidently false claim no filter can see | down |
| `preschool story` label wrong ~49% of the time | down |
| ingested with `--filters light`; 13.4% would fail the web chain | down |
| SmolLM2 uses Cosmopedia at ~11% against real web | 0.11 |
| seeded on a real document (2.97% leak proves it), so not the <5% from-scratch case | up |
| expository half hand-reads 32.5% usable against raw web's 18% | up |
| pool is not binding: ~1.6B tokens survive the predicate from 8 of 62 parquets | neutral |

0.12 after the predicate is 0.40B tokens of expository, filtered, above-median text. The current
0.36 is 1.19B tokens of everything, half of which is either fiction or below the source's own
median.

The 0.24 of freed weight cannot simply go to `web_hq`: its filtered pool is 1.42B tokens and
`data/mix_v3.json` already runs it at `epochs 2` with a 1.05x actual repeat. Spending the freed
budget means a larger filtered-web pull, not a larger multiplier on the pool that exists.

## Falsifying experiment

Two pretrains identical but for the textbook share (0.36 vs 0.12), scored on **held-out** web /
wiki / math shards that are in neither mix. If 0.36 wins there, this decision is wrong. Criterion
6 in `docs/synthetic_data_standard.md` stages exactly this as `ckpt_tb36` / `ckpt_tb05`; both have
500 steps but the logged val is a training-shard prefix, so the measurement does not exist yet.
Cost: one scoring pass over fixed holdout shards with the two existing checkpoints — hours, not a
retrain.

## What could not be measured

1. **Anchoring as a subset test.** The title matcher paired 89 of 12,169 `wiki`-source rows and
   most of those pairings are wrong. Doing it properly needs a retrieval index over the seed
   corpora (baike is 70% of rows and we do not hold it at all), not a title dictionary.
2. **Whether the seeds are eval-contaminated.** The outputs are clean at 0/269,551. The seed
   pages were never scanned because we do not have them.
3. **Shards 00004–00061.** Not sampled; not in the mix. Every rate here is measured on the four
   that are, and a different quarter of a 62-parquet dataset may not match.
4. **The false-claim rate to a useful precision.** 15% ±11 at n=40. n≈196 for ±5pt, and it has to
   be done by hand — the distilled head, the published score, and the filter chain all rate the
   worst document read as ordinary.
5. **Cross-source quality on one rubric.** The 27B read unfiltered web at 21.8% educational and
   cosmopedia at 59.3%. Those are the numbers on record and they were not re-run here; the
   distilled head cannot substitute for them, as section 6 shows.
6. **Near-duplication at full scale.** Zero at 8,000 × 18,000; the full cross-product was not run.

## Artifacts

On the pod: `data/_audit/{cosmo,web_hq,wiki}.jsonl` (the samples), `s_{cosmo,web,wiki}.npy` (the
distilled scores, in file order), `density.json`, `scores.json`, `anchor.json` (the failed match),
`runs/audit_{sample,anchor,dup,dup2}.log`, and the scripts `scripts/_audit_{sample,anchor,score}.py`.
Local: shard `00000.parquet` under the session scratchpad.
