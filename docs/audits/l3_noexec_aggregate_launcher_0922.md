---
question: What did the pod-only L3 aggregate launcher do, and what does its text prove about the gate that was skipped?
status: measured
source: pod:/work/aupai/l3_agg_only.sh (md5 f92da55c33a3a9d038dd48c8c137267a, 1723 B, 34 lines); /work/aupai/runs/l3_agg_only.log; friction rows 2026-09-21 16:22 + 16:28
---

# The L3 noexec aggregate launcher, as text

`code_ultra_l3_noexec_dc` reached the gate mix (weight 0.3146, the largest domain) without a
byte-identity proof. The launcher that produced it exists only on the pod, so it is recorded
here as TEXT and deliberately not collected into `scripts/` (ruling ee 2026-09-21): a
copy-pasteable script that skips a gate is a shortcut handed to the next person.

```
pod:/work/aupai/l3_agg_only.sh   1723 B   34 lines   md5 f92da55c33a3a9d038dd48c8c137267a
  (md5 read twice on the pod, identical; the local copy used to write this section hashes the same)
```

## The text

```bash
#!/bin/bash
# L3 aggregate only (run after l3_groups_only.sh produced 37 stats_s*.json and
# CPU freed). Non-reentrant: deletes tagged jsonl as it goes. 0e 2026-09-21.
set -u
cd /work/aupai || exit 9
OUT=data/corpus/code_ultra_l3_noexec
FINAL=data/corpus/code_ultra_l3_noexec_dc
AGW=${AGW:-20}
echo "L3_AGG_START $(date -u +%FT%TZ) agg_workers=$AGW df_before: $(df -B1 /work|awk 'NR==2{print $4}')"
[ "$(ls "$OUT"/stats_s*.json 2>/dev/null | wc -l)" -eq 37 ] || { echo "NOT_37_STATS"; exit 1; }
mkdir -p "$FINAL"
env PYTHONPATH=/work/aupai CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=1 \
  python3 -u datagen/ultradata_shards.py \
  --level L3 --aggregate 'stats_s*.json' \
  --out "$OUT" --final-out "$FINAL" \
  --tokenizer data/tokenizer.json --agg-workers "$AGW" \
  > runs/ultra_groups/l3_aggregate.log 2>&1
arc=$?
if [ $arc -ne 0 ] || [ ! -s "$FINAL/build_corpus_stats.json" ]; then
  echo "AGGREGATE_FAILED rc=$arc $(date -u +%FT%TZ)"; tail -20 runs/ultra_groups/l3_aggregate.log; exit 1
fi
python3 - "$FINAL" <<'PY'
import json, os, sys
f=sys.argv[1]; s=json.load(open(f+"/build_corpus_stats.json"))
assert s.get("domain")=="code_ultra_l3_noexec_dc", f"domain={s.get('domain')}"
jl=[x for x in os.listdir(f) if x.endswith(".jsonl")]
assert jl and all(x.startswith("code_ultra_l3_noexec_") for x in jl), "bad prefix"
print("STAMP_OK domain=code_ultra_l3_noexec_dc prefix=code_ultra_l3_noexec_",
      "n_shards", s.get("n_shards"), "kept", s.get("kept"),
      "total_rows", s.get("total_rows"), "decontam_fp", s.get("decontam_fp"))
print("NOTE kept_tokens NOT in stamp; get exact count via scripts/count_dir.py",
      f, "24")
PY
echo "L3_AGG_DONE $(date -u +%FT%TZ) df_after: $(df -B1 /work|awk 'NR==2{print $4}')"
```

## What the text establishes

**It runs the aggregate directly, with no gate and no proof.** It asserts only the count of
inputs it expects (37) and post-checks the produced stamp. It never calls
`scripts/aggregate_l3_noexec_gated.sh` and never calls
`scripts/proof_aggregate_identity.py`, which is why no `proof_aggregate_identity.out` exists
for this domain.

**Its own post-check is weaker than it reads.** The `STAMP_OK` line prints `n_shards`, `kept`,
`total_rows` and `decontam_fp` — none of which is the token count the mix consumes — and the
script says so: `kept_tokens NOT in stamp`. The exact count was taken separately
(`26,699,254,040` measured; `fact` `cs` entry for the domain).

**It is honest about being non-reentrant**: `Non-reentrant: deletes tagged jsonl as it goes`,
and the header names its prerequisite (`run after l3_groups_only.sh produced 37 stats_s*.json`).
That deletion is what makes the byte-identity proof impossible after the fact — by the time
anyone looked, `data/corpus/code_ultra_l3_noexec/` held 37 `stats_s*.json` and zero
`*.jsonl` / `*.clean` / `*.sigs` / `*.ngdrop`. The removal is unconditional in the driver
(`datagen/ultradata_shards.py:453-457`, a `finally` backstop that ignores
`release_intermediates`), so the loss is a property of the driver, not of this launcher.

## What it does not establish

It says nothing about whether the aggregate's output equals what the pre-sidecar driver
would have produced. That equality is the thing the skipped proof exists to test, and it is
now unmeasured for this domain unless the units are re-converted from raw
(`data/raw/ultradata/UltraData-Code-L3-*-of-00147.parquet`, 147 files, 268 GB, present).

The counts it printed (`n_shards 940`, `kept 15,309,833`, `total_rows 20,750,202`,
`decontam_fp 302aa793f7067462`) are consistent with the landed stamp, which is the only
positive statement available.

## Verification for a second reader

`md5 f92da55c33a3a9d038dd48c8c137267a` — re-read the pod file and compare; do not compare
against a copied file's bytes without hashing both.
