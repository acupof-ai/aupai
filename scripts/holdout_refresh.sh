#!/usr/bin/env bash
# ONE idempotent entry point for the chain that used to be four hand steps:
#
#   register eval sets -> recompute the holdout fingerprint -> rebuild the packs that carry
#   holdout_fp -> accept (holdout selftest, ladder selftest, harness check)
#
# WHY IT EXISTS. Registering one evaluation set changes REGISTRY, which changes the fingerprint
# every pack stamps as `holdout_fp`. Done by hand on 2026-10-09 that took four separate
# discoveries -- podput refusing a tracked file, the fingerprint header, the pack stamp, the
# repack -- and each one failed in a different place. A chain nobody can run in one command is
# a chain that gets half-run.
#
# IDEMPOTENT: every step is a test-then-act. Re-running after a success changes nothing and
# exits 0; re-running after a failure resumes at the step that failed.
#
# FAIL-CLOSED: `set -euo pipefail`, and the fingerprint step refuses when a registered file's
# sha1 disagrees with REGISTRY_SHA1 rather than trusting a number.
#
#   bash scripts/holdout_refresh.sh --check          # verify only, change nothing (CI-safe)
#   bash scripts/holdout_refresh.sh                  # extend + accept locally
#   bash scripts/holdout_refresh.sh --packs          # also rebuild the packs, ON THE POD
#
# The pack step needs the pod. It is opt-in because it rewrites multi-MB artifacts and takes
# minutes; the local steps are what a commit needs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

CHECK_ONLY=0
DO_PACKS=0
for a in "$@"; do
  case "$a" in
    --check|--selftest) CHECK_ONLY=1 ;;   # --selftest is the hook's spelling; same thing
    --packs) DO_PACKS=1 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

say () { printf '%s\n' "$*"; }

# --- 1. every registered file that is present must hash as recorded -------------------------
# A file that is ABSENT is not a failure here: 6 of the 22 registered files exist on no machine
# this repo can reach, which is why datagen/holdout.py prints "trusting the committed" instead
# of refusing. Absent is reported, never silently skipped.
say "== step 1/4: registered files hash as recorded"
python3 - <<'PY'
import hashlib, os, sys
sys.path.insert(0, ".")
from datagen import holdout as h
bad, absent, ok = [], [], 0
for k, e in sorted(h.REGISTRY.items()):
    p = e["path"]
    if not os.path.exists(p):
        absent.append(k)
        continue
    got = hashlib.sha1(open(p, "rb").read()).hexdigest()
    if got != h.REGISTRY_SHA1[k]:
        bad.append((k, p, h.REGISTRY_SHA1[k][:12], got[:12]))
    else:
        ok += 1
print(f"   {ok} file(s) match, {len(absent)} absent here, {len(bad)} MISMATCH")
if absent:
    print(f"   absent (trusting the committed fingerprint): {', '.join(absent)}")
if bad:
    for k, p, want, got in bad:
        print(f"   FAIL {k}: {p} records {want} but hashes {got}", file=sys.stderr)
    sys.exit(1)
PY

# --- 2. the hash set is the committed body PLUS the rows of any registered file not in it ----
# Additive on purpose. A full regeneration needs all 22 files; 6 are unreachable, and
# datagen/holdout.py's own writer SKIPS a missing file with a printed note -- running it here
# would silently SHRINK the exclusion set, which is the defect its docstring names.
say "== step 2/4: extend data/eval/holdout_hashes.txt (additive, idempotent)"
if [ "$CHECK_ONLY" = "1" ]; then
  say "   --check: reporting only"
fi
python3 - "$CHECK_ONLY" <<'PY'
import os, sys
sys.path.insert(0, ".")
from datagen import holdout as h

check_only = sys.argv[1] == "1"
path = "data/eval/holdout_hashes.txt"
raw = open(path, encoding="utf-8").read().splitlines()
body = {l for l in raw if l and not l.startswith("#")}
before = len(body)

added = {}
for name, e in sorted(h.REGISTRY.items()):
    if e.get("question_field") is None or not os.path.exists(e["path"]):
        continue
    got = {h.qhash(v) for v in h._iter_questions(e["path"], e["question_field"],
                                                 e.get("format", "jsonl")) if v is not None}
    new = got - body
    if new:
        added[name] = len(new)
    body |= got

want_fp = h._fingerprint_from_recorded()
head_fp = next((l.split(":", 1)[1] for l in raw if l.startswith("# fp:")), None)
print(f"   rows {before} -> {len(body)} (+{len(body) - before}); fingerprint {head_fp} -> {want_fp}")
if added:
    print(f"   contributed: {', '.join(f'{k}+{v}' for k, v in added.items())}")
if check_only:
    if head_fp != want_fp or len(body) != before:
        print("   FAIL --check: the hash set is stale; run without --check", file=sys.stderr)
        sys.exit(1)
    print("   --check: up to date")
    sys.exit(0)
if head_fp == want_fp and len(body) == before:
    print("   already up to date, nothing written")
    sys.exit(0)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write("\n".join([f"# fp:{want_fp}", f"# n:{len(body)}"] + sorted(body)) + "\n")
os.replace(tmp, path)
print(f"   wrote {path}: fp {want_fp}, n {len(body)}")
PY

# --- 3. the packs that stamp holdout_fp must be rebuilt against the new file -----------------
# sft_pack_holdout only ever inspects data/sft/sft_all.pt, so a stale C1 pack passes every
# harness check while carrying an old fingerprint. The rebuild is therefore explicit here.
if [ "$DO_PACKS" = "1" ]; then
  say "== step 3/4: rebuild the packs that stamp holdout_fp (on the pod)"
  # podput refuses a tracked file by design; the builder is tracked, so it goes via pod_push.
  bash scripts/pod_push.sh data/sft/stage_c/format_stop/build.py
  pod "cd /work/aupai && bash data/sft/stage_c/format_stop/build.sh v1 v2" 2>&1 | tail -3
  pod "cd /work/aupai && python3 -c \"
import torch, glob
for p in sorted(glob.glob('data/sft/stage_c/format_stop/format_stop_v*.pt')):
    d = torch.load(p, map_location='cpu', weights_only=False)
    print(p, 'holdout_fp', d.get('holdout_fp'))
\"" 2>&1 | tail -4
else
  say "== step 3/4: pack rebuild SKIPPED (pass --packs; it runs on the pod)"
fi

# --- 4. acceptance --------------------------------------------------------------------------
# Skipped under --check/--selftest: the ladder selftest's e6 control world reads the host's gate
# sets and exits 3 where they are absent, so verification-only mode stops after step 2 and the
# full acceptance run stays a deliberate, data-present action.
if [ "$CHECK_ONLY" = "1" ]; then
  say "== step 4/4: acceptance SKIPPED (--check/--selftest verifies steps 1-2 only)"
  say "== holdout_refresh.sh done"
  exit 0
fi
say "== step 4/4: acceptance"
python3 datagen/holdout.py --selftest 2>&1 | tail -1
python3 eval/eval_ladder.py --selftest 2>&1 | tail -1
python3 scripts/harness.py check 2>&1 | grep -E "invariant\(s\) FAILED|FAIL of|^[0-9]+ (FAIL|invariant)" | tail -2
say "== holdout_refresh.sh done"
