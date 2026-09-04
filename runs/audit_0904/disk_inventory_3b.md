# Disk inventory — /work/aupai/data + /data00 (3b partial, 2026-09-04)

Partial draft for tilerl to merge into `runs/audit_0904/disk_inventory.md`. Read-only;
no deletes. Sizes/mtime from `du -sb` + `stat` on the pod (output read, not hand-typed).

## data/ top-level (total 664.18 GB)

| path | bytes | files | mtime (UTC) | owner/use | referenced by | disposition |
|---|---|---|---|---|---|---|
| data/corpus | 362.49G | 4209 | 09-03 21:01 | the mix-named domains + dead dirs (see C3) | the 5 mixes + facts | keep (=C3 list for dead dirs) |
| data/raw | 281.75G | 1563 | 09-03 14:59 | source manifests + fetched zh sources | `datagen/fetch_corpus.py` SOURCES + facts | keep |
| data/sft | 11.85G | 42 | 09-03 06:38 | SFT packs / expanded | sft_math / prepare_sft sources | keep |
| data/hf | 2.38G | 29 | 09-03 13:43 | HF model weights: pythia-160m, Qwen2.5-0.5B, SmolLM2 | **unreferenced** — literal `data/hf` cited nowhere in datagen/scripts/eval/facts; the live Pythia control is `data/controls/pythia-160m-step2000` (scripts/sft_hf_control.py:9), so data/hf/pythia-160m is a stale duplicate | **deletion candidate** (next broadcast) |
| data/cci | 2.00G | 3 | 08-29 15:30 | cot_synthesis_math-high scratch | **unreferenced** — `data/cci` cited nowhere in datagen/scripts/eval/facts | **deletion candidate** (next broadcast) |
| data/mix | 1.89G | 12 | 08-24 05:39 | legacy pre-0830v1 mixing jsonl (mixed_v3, pretrain_v4, cosmopedia...) | `datagen/make_mixed.py:14` PRETRAIN_PATH=data/mix/mixed_v3.jsonl (legacy script) | keep (legacy) |
| data/_audit | 289.54M | 9 | 08-29 15:15 | audit scratch (pre-0904) | — | archive |
| data/controls | 377.15M | 4 | 09-02 15:05 | control-arm heldout / sft | `scripts/sft_hf_control.py:9`, `cont.heldout_in_pretrain_corpus` | keep |
| data/math | 323.36M | 5 | 08-26 16:44 | math batch cards | cont facts | keep |
| data/workbatch | 299.62M | 5 | 08-26 08:24 | sft workbatch | `datagen/prep_math_data.py:12` (writer of school_math_train/gsm8k_zh/coig into it) | keep |
| data/synthetic | 229.52M | 20 | 08-30 04:13 | eval golds (math_hard_v2 etc.) | cont.math_hard_v2, holdout REGISTRY | keep |
| data/vocab_sweep | 27.74M | 9 | 08-29 11:44 | tokenizer-vocab sweep | `scripts/tokenizer_eval.py:12` reads data/vocab_sweep/v16384.json | keep |
| data/rl | 85.24M | 9 | 09-03 14:31 | RL task-data trial | 3b-9 (parked) | keep (parked) |
| data/code_supply | 0 | 2 | 08-30 14:47 | code supply counts | — | archive |
| data/_quarantine | 39.86M | 1 | 08-25 17:20 | quarantined corrupt file | — | archive |
| data/eval | 153.61M | 448 | 09-04 05:11 | eval golds + offline preds | REGISTRY + facts | keep |

## data/corpus per-domain — the 11 mix-named (from 0904 corpus_data) + dead

Mix-named (keep, stamps canonical): code_rp1t 23G, code_py_starcoder 28G, code_py_rp1t 1.5G,
math_owm_stage2 21G, en_c4_stage2 8.1G, cot 1.3G, zh_web (~21G? flag: no kept/kept_tokens),
textbook_30b 7.6G, wiki_chat (~0.3G), chatml 0.16G, chat_qa 0.16G.

Dead / no mix (reference C3 — cleanup.jsonl broadcast, 24h then delete, stamps stay, ~84 GB):
24 `web_cci3_p*` dirs, 115 loose `batch_*.jsonl`. Real `ls data/corpus` (2026-09-04): chat, chatml,
chat_qa, code, code_dedup08, code_py_rp1t, code_py_starcoder, code_rp1t, code_rp1t_rest, cot,
cot_open_thoughts, cot_seed, en, en_c4, en_c4_30b, en_c4_stage2, math, math_owm, math_owm_stage2,
math_seed, rp1t_arxiv_papers, sample, textbook, textbook_30b, web_cci3_p0..p23, web_hq, wiki,
wiki_chat, zh_web (+ 115 loose batch_*.jsonl). Superseded pre-stage2 variants actually present:
`math_seed`, `math`, `cot_seed`, `wiki` (plain), `textbook` (plain), plus `code`, `chat`,
`code_rp1t_rest` (0-shard stub) — all mix-unreferenced, candidates for a later broadcast.

## data/raw per-source (281.75G)

From earlier: rp1t_github 30G, ms_starcoder_py 22G, cot_open_thoughts 1.1G, skywork_or1 785M,
hf_numma(+jsonl) 2.5G, hf_om2 empty, fineweb2_cmn, fable5_2m/sft/traces, gsm8k_chinese,
hf_finemath_4plus, ms_finemath_4plus, mathinstruct_zh, kyara-zh-math-sft, rp1t_arxiv, rp1t_c4,
plus manifests. Keep; provenance retained.

## /data00 token caches (247.80 GB, 22 caches)

Each main `.pt` has `.seed` / `.srcfp` / `.vocab` markers (derived-artifact identity).

| cache | GB | mtime | referenced by |
|---|---|---|---|
| tokens_zh_web.pt | 85.17 | 08-31 | mix zh_web |
| tokens_code_py_starcoder.pt | 35.15 | 09-01 | mix code_py_starcoder |
| tokens_code_rp1t.pt | 30.28 | 08-31 | mix code_rp1t (30b) |
| tokens_math_owm_stage2.pt | 26.11 | 08-31 | mix math_owm_stage2 |
| tokens_en_c4.pt | 19.24 | 08-31 | en_c4 corpus (pre-stage2) |
| tokens_math_owm.pt | 16.14 | 08-31 | math_owm (pre-stage2) |
| tokens_en_c4_stage2.pt | 9.60 | 08-31 | mix en_c4_stage2 |
| tokens_textbook_30b / textbook | 6.44×2 | 08-31/09-01 | mix textbook_30b |
| tokens_web_hq / wiki_chat / wiki / en / math_seed / math / code / chatml / chat_qa / chat / sample / cot / code_py_rp1t | 0.01-5.74 | 08-31..09-01 | mix + superseded |

Note: `tokens_code_rp1t.pt` (30.28G) exists but code_rp1t is 30b-only — its cache held for a
run that hasn't consumed it yet (mix_30b not running). `tokens_en_c4.pt`/`tokens_math_owm.pt`
are the pre-stage2 caches; the stage2 caches coexist. Which cache is CURRENT for mix_30b_stage2
is a mix-build question (flag for tilerl).

## Owner/reference note

- C3's 84GB broadcast (web_cci3 + batch) referenced directly from `runs/audit_0904/cleanup.jsonl`.
- Supply/facts: `facts/corpus_supply.json#cs.*`, `facts/contamination.json#cont.*`.
- Others to fill: corpus per-domain exact bytes, raw full per-source bytes, the flagged
  pre-stage2 vs stage2 cache-resolution, and data/raw + data/hf + data/mix exact contents.
## Refinement (2026-09-04, exact bytes)

- corpus/zh_web = **89.85G** (909 shards — largest domain by bytes AND tokens).
- The 24 web_cci3_p* dead dirs = **81.9G** (the C3 broadcast, cleanup.jsonl).
- raw/cci3_hq = **108.03G** — the raw source for zh_web/web_cci3; provenance retained.
- wiki_chat 1.07G vs wiki 0.91G: the 30b mix names wiki_chat; plain wiki is a separate (superseded?) dir.
- Other large raw: fineweb2_cmn 33.9G, rp1t_github 31.5G, rp1t_c4 28.1G, ms_starcoder_py 23.2G, hf_finemath_4plus 18.4G, rp1t_arxiv 14.3G, ms_om2 12.6G.
- The packet-depth-1 `du -sb corpus/*/` here undercounts (156G over the listed dirs) vs the full-walk 362G because the largest code dirs (code_rp1t 23G, code_py_starcoder 28G, math_owm_stage2 21G, en_c4 17G, en_c4_stage2 8.1G) sit at depth-1 of the sub-corpora; the authoritative corpus dir set/bytes is the full walk in the top-level table + the corpus_data audit.
