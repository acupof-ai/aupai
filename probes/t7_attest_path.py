"""P5: does cited_artifacts_attested reject an attestation of the WRONG path?"""
import json, os, sys, tempfile, shutil, hashlib
sys.path.insert(0, "scripts")
import harness

root = tempfile.mkdtemp()
os.makedirs(f"{root}/facts"); os.makedirs(f"{root}/runs")
# a fact citing preds_A with hash H
H = hashlib.sha256(b"the A bytes").hexdigest()
json.dump({"facts": [{"id": "eff.x", "value": "cites data/eval/preds_A.jsonl", "measured": "2026-09-01",
                      "source": "data/eval/preds_A.jsonl", "config": {"a": 1},
                      "uncertainty": "u", "status": "measured", "artifact_sha256": H}]},
          open(f"{root}/facts/efficiency.json", "w"))
# the writer attested the WRONG PATH with the right bytes
with open(f"{root}/runs/artifact_refs.jsonl", "w") as f:
    f.write(json.dumps({"path": "data/eval/preds_TOTALLY_DIFFERENT.jsonl", "sha256": H,
                        "bytes": 11, "rows": 1, "written_at": "2026-09-01 08:00:00"}) + "\n")
s, msg = harness.check_cited_artifacts_attested(root)
print(f"wrong-path attestation -> [{s}] {msg}")
print("VERDICT:", "DEFECT -- accepts a wrong-path attestation" if s == harness.PASS else "correctly rejects")
shutil.rmtree(root)
