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
| school_math_r1_zh.jsonl | math | c8f6a7cce2e4c0b76711919a99767aa435a5ce6b509da722ffcb750d42124834 | 223,423 | scripts/fetch_math_data.py belle branch (pod sha identical, verified 2026-08-27; known 3.6% tail_answer gold bug, see REVIEW_2026-08-26.md #2) |

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

## Related

- Synthetic math: data/synthetic/math_short_v*.jsonl — fully reproducible via
  mathbank/ (seeded), see mathbank/README or mathbank/run_math_short.py.
- Synthetic code/knowledge: data/synthetic/{code_python_zh,knowledge_qa_zh}.jsonl
  via datagen/gen_code.py + gen_knowledge2.py (seeded, zero external deps).
- Eval holdout filter: scripts/holdout.py — every fetcher must exclude it.
