# Locked English hand-read set — 300 candidates (2026-09-16, 66)

STATUS: **candidates sampled, NOT yet labelled.** Do not edit rows by hand; labelling
appends to `teacher_labels` and hand-reading to `hand_read`.

One common document id shared by three scorers of the same English natural-language
corpus (`en_c4_stage2_dc`): 0e fineweb-edu AUC, 3b English PPL, and the 66 L3
multi-dim teacher rubric. That is why `doc_id` is the content hash, not url/row.

## Files

- `en_locked_300.jsonl` — 300 rows, columns:
  `sample_id`(=doc_id), `language`, `length_band`, `source`, `url`, `content`,
  `teacher_labels`(null, filled by the rubric run), `hand_read`(null, filled by humans).
- `en_locked_300.jsonl.manifest.json` — source shard sha256, seed, per-stratum
  populations, length bins, `doc_id=content`, `only_bands=[l,m,s]`.

## Sampling

- `datagen/l3_stratified_sample.py --doc-id content --only-bands s,m,l
  --per-stratum 100 --seed 20260916` over `en_c4_000.jsonl` (one shard; populations
  s=16152 / m=13748 / l=7640 all >> 100, so 100 per band is a uniform-within-band
  reservoir). Deliberately covers shorter bands — s 400-1199, m 1200-2999, l
  3000-7999 chars — unlike L3 code, which clusters at l/xl.
- `sample_id` = sha256(content utf-8)[:16] = `datagen/score_ledger.content_doc_id`
  (schema #389). Verified: 300 unique, every id re-hashes to its content.

## Next (not done here)

1. When the A3B teacher path is ready, run `datagen/l3_label_pilot.py
   --backend openai` over these rows, then write rows to the frozen score ledger
   (`doc_id=sample_id`, scorer_name `l3-rubric`, rubric_dims the four 1-5 scores).
2. Draw ~60 rows for human hand-read to check teacher agreement before any scaling.
3. Code-domain English lock set (~100) is a SEPARATE draw, deferred until ae's code
   audit table is final, to avoid conflicting with rule changes there.
