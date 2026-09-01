#!/usr/bin/env python3
"""_domain_seqs must read shards and only shards.

THE INCIDENT (2026-09-01). datagen began writing holdout_slice_<domain>.jsonl into the
corpus directory it describes. train.py globbed data/corpus/<domain>/*.jsonl as shards,
so the slice's header row {"phase":..., "rule_fp":..., "n":0} reached _jsonl_content and
died on ["content"]. Four domains carried one and code_py_starcoder was among them, so
the 20B run would have died before step 0.

Neither side was a bug: the slice belongs beside its corpus, and globbing *.jsonl was
right for two years. The DIRECTION OF THE DEFAULT is the defect -- a blacklist reads an
unknown new file as data, a whitelist ignores it.

So this test does not check "is the slice excluded". It checks the property: a file that
is not named like a shard is not read, WHATEVER it is called. A test that named the slice
would pass while the next new artifact repeated the incident.

    python3 scripts/test_shard_glob.py --selftest
"""
import os
import re
import subprocess
import sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                      text=True).stdout.strip() or os.path.dirname(
                          os.path.dirname(os.path.abspath(__file__)))


def _selection(train_py, names):
    """Which of `names` train.py's shard rule would read, using ITS OWN source.

    The rule is extracted from train.py rather than restated here. A copy of the
    predicate in the test passes against a train.py that no longer has it -- which is
    the shape of defect this whole file exists for.
    """
    src = open(train_py, encoding="utf-8").read()
    ns = {"re": re}
    m = re.search(r"^NON_SHARD_JSONL = \{.*?\n\}", src, re.S | re.M)
    exec(m.group(0), ns) if m else ns.setdefault("NON_SHARD_JSONL", set())
    m = re.search(r"^SHARD_RE = re\.compile\(.*?\)$", src, re.M)
    if m:
        exec(m.group(0), ns)
    keep = []
    for n in names:
        if ns.get("SHARD_RE") and not ns["SHARD_RE"].search(n):
            continue
        if n in ns["NON_SHARD_JSONL"]:
            continue
        keep.append(n)
    return keep


def main():
    train_py = os.path.join(ROOT, "train.py")
    shards = ["chat_qa_000.jsonl", "chat_qa_001.jsonl", "part_000000_012.jsonl",
              "00000_003.jsonl", "math_owm_206.jsonl", "code_py_rp1t_015.jsonl"]
    # Every one of these sat in a real corpus dir on the pod, or is the shape of the
    # next thing to land there. None is named like a shard.
    not_shards = ["holdout_slice_chat_qa.jsonl",       # the incident
                  "holdout_slice_code_py_starcoder.jsonl",
                  "build_corpus_stats.json.jsonl",
                  "manifest.jsonl", "labels.jsonl", "README.jsonl",
                  "some_future_sidecar.jsonl"]        # the file nobody has written yet
    got = _selection(train_py, shards + not_shards)

    missing = [s for s in shards if s not in got]
    assert not missing, f"real shards were REJECTED: {missing}"

    leaked = [n for n in not_shards if n in got]
    assert not leaked, (
        f"non-shard file(s) would be read as corpus data: {leaked}. "
        "_jsonl_content takes ['content'] from every row, so this is a KeyError before "
        "step 0 -- the 2026-09-01 incident.")

    # The blacklist must survive: two sample/ files ARE named like shards and must still
    # be excluded, so the whitelist cannot simply replace it.
    assert "cci3_audit_400.jsonl" not in _selection(train_py, ["cci3_audit_400.jsonl"]), \
        "NON_SHARD_JSONL stopped being applied; a name-shaped non-shard is back in"

    print(f"selftest OK ({len(shards)} shards kept, {len(not_shards)} non-shards refused)")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))
