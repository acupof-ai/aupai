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
upstream is a cost; a raw that cannot is a single point of failure. Both belong in the record,
because the second is the one an operator must not clear.

| domain | raw | built output | raw re-fetchable? | delete permission |
|---|---|---|---|---|
| `en_c4_stage2_dc` | digest, `wt-3b/data/raw/rp1t_c4`, **32 jsonl / 26 G** | digest, `wt-3b/data/corpus/en_c4_stage2_dc`, 244 shards / 24 G | **NO** — `data.together.xyz` 403 at the host level, no HF mirror for RedPajama-1T c4 (`facts/`); the 32 files are the only bytes consistent with the historical corpus fingerprint | **named instruction only** |
| `code_py_starcoder_dc` | **GONE.** digest `wt-3b/data/raw/` holds only `ms_starcoder_py_manifest.txt` (59 lines), `ms_starcoder_py_sha256.txt` and `ms_starcoder_fetch.log`; the fetch log says `FETCH_DONE rc=0` / `MANIFEST_DONE 59 files`, so the parquet were fetched and later removed. No parquet on digest or the pod. | digest, `wt-3b/data/corpus/code_py_starcoder_dc`, 283 shards / 28 G | **yes** — `www.modelscope.cn` and `hf-mirror.com` both answer 200/rc=0 from digest, measured 2026-09-21; the manifest names all 59 files | **the built output is the only copy of this domain's bytes** — named instruction only |
| `cot_dc` | digest, `wt-0e-cot/data/raw/hf_numma` 1.2 G + `hf_numma_jsonl` 1.3 G | digest, `wt-0e-cot/data/corpus/cot_dc`, 14 shards / 1.3 G | upstream is hf_numma (NuminaMath-CoT); not re-verified 2026-09-21 | raw and output are both on digest but in **different worktrees** — see below |
| `code_ultra_l2_dc` / `code_ultra_l3_noexec_dc` / `math_owm_stage2_dc` | **pod**, `data/raw/ultradata` 133 G (L2 119/119) + L3 in flight; `hf_finemath_4plus` 18 G (64/64) | **pod**, `data/corpus/` | ultra: openbmb/UltraData + mirrors; finemath: hf-mirror | raw and output on the same machine — **not an instance** |

## Two things this table exists to stop

1. **"The raw is somewhere" is not "the domain can be rebuilt."** `code_py_starcoder_dc` has a
   manifest, a sha256 list and a success log, and no data. Anyone reading only the manifest
   directory would conclude the raw is available.
2. **The machine that runs the build is not automatically the machine that holds the source.**
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
