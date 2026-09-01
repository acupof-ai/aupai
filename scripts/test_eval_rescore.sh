#!/usr/bin/env bash
# Does FORCE=1 let a second score of the same checkpoint through, where the first
# refused? The live failure: step15000 scored at 03:34, rescored at 04:31, and every
# generative metric errored at ArtifactExists.
#
# No GPU: the shard writer is stubbed. What is under test is the SHELL contract --
# does the wrapper pass FORCE/RUN to the writer and to the merge -- not the model.
# The stub calls the REAL open_artifact, so the refusal exercised is the production
# one rather than a re-implementation that would share this probe's assumptions.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
D=/tmp/de_rs
rm -rf "$D"
mkdir -p "$D/data/eval" "$D/eval" "$D/scripts"
cp "$REPO/eval/eval_math.sh" "$D/eval/"
cp "$REPO/scripts/eval_artifacts.py" "$D/scripts/"
printf '{}\n{}\n{}\n{}\n' > "$D/data/eval/math_test_500.jsonl"   # EXPECTED=4

cat > "$D/eval/math_zh.py" <<'PYEOF'
import argparse, json, os, sys
sys.path.insert(0, "scripts")
from eval_artifacts import open_artifact
p = argparse.ArgumentParser()
for f in ("--ckpt", "--tokenizer", "--run", "--data"):
    p.add_argument(f)
p.add_argument("--shards", type=int, default=1)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--force", action="store_true")
a = p.parse_args()
path = f"data/eval/preds_{os.path.basename(a.ckpt)}" + (f".{a.shard}" if a.shards > 1 else "") + ".jsonl"
with open_artifact(path, force=a.force, run=a.run) as f:
    for _ in range(4 // a.shards):
        f.write(json.dumps({"ok": 1, "gen": "x"}) + "\n")
print("math-500: stub")
PYEOF

echo 'exit 0' > "$D/scripts/assert_vocab.sh"
# The REAL device mapper, copied not stubbed. A hand-written stub would share this
# probe's assumptions about what _devs.sh does, and device_set_honoured correctly
# FAILed on the version that wrote its own -- the check reading this file could not
# tell a test fixture from a launcher that escapes its lane.
cp "$REPO/eval/_devs.sh" "$D/eval/_devs.sh"

cd "$D"
fail=0

set +e
bash eval/eval_math.sh ck.pt 1 > o1.txt 2>&1; r1=$?
bash eval/eval_math.sh ck.pt 1 > o2.txt 2>&1; r2=$?
FORCE=1 bash eval/eval_math.sh ck.pt 1 > o3.txt 2>&1; r3=$?
RUN=jitter2 bash eval/eval_math.sh ck.pt 1 > o4.txt 2>&1; r4=$?
RUN=r2 bash eval/eval_math.sh ck.pt 2 > o5.txt 2>&1; r5=$?
set -e

[ "$r1" = 0 ] || { echo "FAIL: first score exited $r1"; cat o1.txt; fail=1; }
[ "$r2" != 0 ] || { echo "FAIL: a bare rescore must still refuse -- that guard is the point"; fail=1; }
grep -q ArtifactExists o2.txt || { echo "FAIL: rescore refused for the wrong reason"; cat o2.txt; fail=1; }
[ "$r3" = 0 ] || { echo "FAIL: FORCE=1 rescore still blocked (the live bug)"; cat o3.txt; fail=1; }
[ "$r4" = 0 ] || { echo "FAIL: RUN=<name> rescore blocked"; cat o4.txt; fail=1; }
[ -f data/eval/preds_ck.pt.jitter2.jsonl ] || { echo "FAIL: RUN did not version the merged path"; ls data/eval; fail=1; }
[ "$r5" = 0 ] || { echo "FAIL: sharded RUN run blocked"; cat o5.txt; fail=1; }
[ -f data/eval/preds_ck.pt.r2.jsonl ] || { echo "FAIL: sharded merge did not version"; ls data/eval; fail=1; }
[ "$(wc -l < data/eval/preds_ck.pt.r2.jsonl | tr -d ' ')" = 4 ] || { echo "FAIL: sharded merge row count"; fail=1; }

[ $fail = 0 ] && echo "OK: refuses bare, allows FORCE=1 and RUN=<name>, versions shard and merge"
exit $fail
