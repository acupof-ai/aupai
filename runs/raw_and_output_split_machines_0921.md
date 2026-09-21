---
question: Which _dc gate domains have their raw and their built output on different machines, so that one machine holds the only surviving copy of something?
status: measured
source: measured 2026-09-21 on digest (/data00/home/chenkailun.c/aupai-cimap/*) and the pod (container /work = host /data00/aupai_work); du/ls per directory, sha256 on the tokenizer
---

# Raw and output on different machines — the four instances

The rule: **when a `_dc` domain's raw and its built output are not on the same machine, the
machine holding the raw holds the only source that domain can be rebuilt from. Deleting it
needs a named instruction, the same standing order that governs every other deletion here.**

A domain whose raw and output sit on the same machine is **not** an instance of this rule — it
can always be rebuilt in place, at the cost of compute alone.

## The rule needs a second column: can the raw be re-fetched?

Rebuildability is not the same as "the raw exists". A raw that can be re-fetched from a live
upstream is a cost; a raw that cannot is a single point of failure.

**These are two independent columns, and collapsing them into one flips the safety ranking.**
An earlier revision of this table carried only "raw re-fetchable?", which put
`code_py_starcoder_dc` at "yes" — reading as the safest row — when it is the **most** exposed:
its raw is already gone, so the 28 G built output is the only copy of those bytes. Meanwhile
`en_c4_stage2_dc` reads "no" and is in fact holding 26 G of irreplaceable raw. **The question
an operator needs answered is "what do I hold right now", not "could I get it again".**

| domain | **holds raw now?** | **raw re-fetchable?** | **output is the only copy?** | delete permission |
|---|---|---|---|---|
| `en_c4_stage2_dc` | **yes** — digest `wt-3b/data/raw/rp1t_c4`, 32 jsonl / 26 G | **no** — `data.together.xyz` 403 at the host level (re-measured 2026-09-21), no HF mirror for RedPajama-1T c4; the 32 files are the only bytes consistent with the historical corpus fingerprint | no | **raw: named instruction only** |
| `code_py_starcoder_dc` | **no** — digest `wt-3b/data/raw/` holds `ms_starcoder_py_manifest.txt` (59 lines), `ms_starcoder_py_sha256.txt` and `ms_starcoder_fetch.log` reading `FETCH_DONE rc=0` / `MANIFEST_DONE 59 files`; no parquet on digest or pod | yes — `www.modelscope.cn` and `hf-mirror.com` both 200/rc=0 from digest, measured 2026-09-21; the manifest names all 59 files | **yes** — digest `wt-3b/data/corpus/code_py_starcoder_dc`, **282 shards + `build_corpus_stats.json`** (283 directory entries) / 28 G | **output: named instruction only** |
| `cot_dc` | yes — digest `wt-0e-cot/data/raw/hf_numma` 1.2 G + `hf_numma_jsonl` 1.3 G | not re-verified 2026-09-21 | no | **raw: named instruction only** |
| `code_ultra_l2_dc` / `code_ultra_l3_noexec_dc` / `math_owm_stage2_dc` | yes — pod `data/raw/ultradata`, **as-of 2026-09-21T08:31Z: 153 G (L2 119/119 done + L3 33/147 in flight) and growing**, `hf_finemath_4plus` 18 G (64/64) | yes | no | raw and output on the same machine — **not an instance** |

## Every count says what it counted

A row in this table may not write "N shards" without naming the filter that produced N. The
corpus directories hold more than shards, and the extras are not removable by a `.jsonl` glob:

| directory | `ls *.jsonl` | shards | `holdout_slice_*.jsonl` | `build_corpus_stats.json` | entries |
|---|---|---|---|---|---|
| `code_py_starcoder_dc` | 282 | **282** | 0 | 1 | 283 |
| `en_c4_stage2_dc` | **243** | **242** | 1 (65 B, `{"rule_fp":"74b34c96d67013ba","n":0}`) | 1 | 244 |
| `cot_dc` | 13 | **13** | 0 | 1 | 14 |
| `math_owm_stage2` | 334 | **333** | 1 | 1 | — |

**The `holdout_slice_*.jsonl` is the case that forces this**: it *is* a `.jsonl`, so
`ls *.jsonl | wc -l` counts it, and it is the frozen holdout basis — 65 bytes of metadata, not
corpus content. A copy loop that globs `*.jsonl` moves it (correct — the domain is not valid
without it) while a count that globs `*.jsonl` miscounts it. **The two operations want
different filters on the same directory**, which is how `en_c4_stage2_dc` reads as 243 shards
when it holds 242.

**This one is not newly discovered, and the correction matters.** `scripts/filter_gate_domains.py:274`
already carries the assertion, with the measurement:

> THE HOLDOUT SLICE IS NOT A SHARD. A --phase build leaves holdout_slice_{phase}.jsonl in the
> source dir beside its shards, and a bare `*.jsonl` glob counts it. Measured on the 2026-09-19
> en_c4_stage2 rebuild: shards_total 243 and rows_scanned 11,309,623 against a real 242 shards /
> 11,309,622 documents — both off by exactly this file.

Its selftest writes a `holdout_slice_p.jsonl` into a fixture and asserts neither count moves, so
the fix is tested rather than merely commented. **The reason it is recorded here anyway is that
the fix lives in that driver, and this page describes the counts as an operator would compute
them** — which is where the same glob is reachable again. Attributed so a reader does not take it
for a new finding, and does not have to rediscover the test.

Same shape across this round: a directory-listing count is not a count of the thing the sentence
is about, and the missing filter is invisible because the number looks right.

**The rule, with both columns:** a domain where **the output is the only copy** needs a named
instruction to delete the output; a domain where **the raw cannot be re-fetched** needs one to
delete the raw. **The union of the two columns is the protected set** — using either alone
loses `code_py_starcoder_dc`, whose protection comes from the first column while its "yes" sits
in the second.

## Three things this table exists to stop

1. **"The raw is somewhere" is not "the domain can be rebuilt."** `code_py_starcoder_dc` has a
   manifest, a sha256 list and a success log, and no data. Anyone reading only the manifest
   directory would conclude the raw is available.
2. **"The raw is re-fetchable" is not "I hold a copy."** That is the error this page's own
   first revision made: one column labelled "re-fetchable" read as a safety column, and it
   ranked the most exposed domain as the safest. Ask what is held, then ask whether it could
   be obtained again — in that order, as two answers.
3. **The machine that runs the build is not automatically the machine that holds the source.**
   Three domains were built on digest while the pod is where training reads from, so the
   transfer exists for a reason that the build location alone does not reveal.

## The four instances, restated as an instruction

- `rp1t_c4` (32 jsonl, digest): **do not delete.** Already covered by the standing order on
  these files; recorded here so the rule has its instance.
- `code_py_starcoder_dc` raw: **already absent** — this is recorded as the one instance where
  the raw is gone, and the built 28 G digest output is consequently the only copy. A future
  cleanup that treats the built output as regenerable would be wrong on this domain.
- `cot_dc` raw (`hf_numma`, `hf_numma_jsonl`): two worktree-local raw dirs on digest; deleting
  either removes the only cot raw copy.
- `en_c4_stage2_dc` raw: digest-only, unrecoverable from upstream — the strictest case, since
  neither a re-fetch nor a rebuild elsewhere is possible.
