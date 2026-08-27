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
| alpaca_gpt4_zh.jsonl | chat | 93819e69830d9eb050e58c342230f3e1986a2e3cd07c3d1a075abb9ddcb6251d | 52,049 | TODO: scripts/fetch_chat_data.py (HuggingFaceH4/alpaca_gpt4_data_zh, verify sha match) |
| coig.jsonl | chat | cdcac3f1d310c0dd8bb6cf5ee63a4b2a99d3386e098cead4985d7e962a8a10f6 | 163,443 | TODO: scripts/fetch_chat_data.py (coig/coig subset, verify sha match) |
| school_math_r1_zh.jsonl | math | c8f6a7cce2e4c0b76711919a99767aa435a5ce6b509da722ffcb750d42124834 | 223,423 | scripts/fetch_math_data.py belle branch (known 3.6% tail_answer gold bug, see REVIEW_2026-08-26.md #2) |
| school_math_r1_zh.jsonl — pod copy differs? | | | | compare before re-deriving |

## Schema contract (consumed by datagen/build_corpus.py)

- pretrain/text sources: `{"content": ...}` (or `{"text": ...}`)
- QA/chat/math sources: `{"instruction": ..., "output": ...}` → rendered as 问：/答：

## Related

- Synthetic math: data/synthetic/math_short_v*.jsonl — fully reproducible via
  mathbank/ (seeded), see mathbank/README or mathbank/run_math_short.py.
- Synthetic code/knowledge: data/synthetic/{code_python_zh,knowledge_qa_zh}.jsonl
  via datagen/gen_code.py + gen_knowledge2.py (seeded, zero external deps).
- Eval holdout filter: scripts/holdout.py — every fetcher must exclude it.
