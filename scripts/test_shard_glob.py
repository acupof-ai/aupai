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
import glob
import os
import re
import shutil
import textwrap
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
    for pat in (r"^SHARD_RE = re\.compile\(.*?\)$", r"^NON_SHARD_RE = re\.compile\(.*?\)$"):
        m = re.search(pat, src, re.M)
        if m:
            exec(m.group(0), ns)
    keep, unknown = [], []
    for n in names:
        if n in ns["NON_SHARD_JSONL"] or (ns.get("NON_SHARD_RE") and ns["NON_SHARD_RE"].search(n)):
            continue
        if ns.get("SHARD_RE") is None or ns["SHARD_RE"].search(n):
            keep.append(n)
        else:
            unknown.append(n)
    return keep, unknown


def _run_block(train_py, filenames):
    """Execute _domain_seqs' shard-selection block over a temp dir holding `filenames`.

    The block is lifted from train.py source and run with the module-level names it
    needs. Importing train.py is not an option here (torch, CUDA); re-implementing the
    branching is what let two mutations through. This runs the shipped lines.
    """
    import tempfile
    src = open(train_py, encoding="utf-8").read()
    blk = re.search(r"\n(    seen = sorted\(glob\.glob.*?)\n    same_vocab", src, re.S)
    assert blk, "the shard-selection block is gone or was rewritten; re-read train.py"
    body = textwrap.dedent(blk.group(1))
    ns = {"re": re, "os": os, "glob": glob}
    for pat in (r"^NON_SHARD_JSONL = \{.*?\n\}", r"^SHARD_RE = re\.compile\(.*?\)$",
                r"^NON_SHARD_RE = re\.compile\(.*?\)$"):
        m = re.search(pat, src, re.S | re.M)
        if m:
            exec(m.group(0), ns)
    ns.setdefault("NON_SHARD_RE", re.compile(r"^__absent__"))
    d = tempfile.mkdtemp(prefix="shardglob")
    os.makedirs(os.path.join(d, "corpus", "probe"))
    for n in filenames:
        open(os.path.join(d, "corpus", "probe", n), "w").close()
    ns.update(DATA=d, domain="probe")
    out = {"shards": [], "raised": None}
    try:
        exec(body, ns)
        out["shards"] = sorted(os.path.basename(x) for x in ns["shards"])
    except SystemExit as e:
        out["raised"] = str(e)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


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
    got, unknown = _selection(train_py, shards + not_shards)

    missing = [s for s in shards if s not in got]
    assert not missing, f"real shards were REJECTED: {missing}"

    leaked = [n for n in not_shards if n in got]
    assert not leaked, (
        f"non-shard file(s) would be read as corpus data: {leaked}. "
        "_jsonl_content takes ['content'] from every row, so this is a KeyError before "
        "step 0 -- the 2026-09-01 incident.")

    # The blacklist must survive: two sample/ files ARE named like shards and must still
    # be excluded, so the whitelist cannot simply replace it.
    assert "cci3_audit_400.jsonl" not in _selection(train_py, ["cci3_audit_400.jsonl"])[0], \
        "NON_SHARD_JSONL stopped being applied; a name-shaped non-shard is back in"

    # THIRD BRANCH, and it is the one that keeps the original author's property: a file
    # that is neither a shard nor a KNOWN non-shard must be reported, never silently
    # dropped. Silence here means a misnamed real shard disappears from training and
    # nothing says so -- the expensive failure. The known slices must NOT be reported.
    assert "some_future_sidecar.jsonl" in unknown, (
        "an unrecognised .jsonl was skipped silently; a misnamed shard would vanish "
        "from the training data with no message (train.py:96's reason for the blacklist)")
    assert "holdout_slice_chat_qa.jsonl" not in unknown, \
        "a known non-shard would stop the run; NON_SHARD_RE is not being applied"

    # sample/ is real and its shards are batch_NNN -- a <domain>_* prefix rule would
    # take mix_sample.json (what test_e2e.py reads) to zero shards. Measured on the pod:
    # 299 of its 301 files match the suffix, the 2 that do not are the label files.
    assert _selection(train_py, ["batch_001.jsonl"])[0] == ["batch_001.jsonl"], \
        "sample/'s batch_NNN shards were rejected; e2e would read an empty corpus"

    # THE REFUSAL ITSELF -- by EXECUTING the shipped block against a real directory,
    # not by reading it. Two earlier versions of this test grepped for `if unknown:` and
    # for the word REFUSING, and BOTH passed a mutation that removed the line filling
    # `unknown` while leaving the raise in place. Text inspection cannot see whether a
    # branch is reachable; running it can.
    run = _run_block(train_py, ["chat_qa_000.jsonl", "holdout_slice_chat_qa.jsonl"])
    assert run["shards"] == ["chat_qa_000.jsonl"], f"shard selection wrong: {run}"
    assert run["raised"] is None, f"a known non-shard stopped the run: {run['raised']}"

    run = _run_block(train_py, ["chat_qa_000.jsonl", "some_future_sidecar.jsonl"])
    assert run["raised"] is not None, (
        "an unrecognised .jsonl did NOT stop the run. It is being skipped silently, so a "
        "misnamed real shard vanishes from training with no message -- train.py:96's "
        "reason for the blacklist, discarded.")
    assert "some_future_sidecar.jsonl" in run["raised"], \
        f"the refusal does not name the offending file: {run['raised']}"

    # THE SAME COMMIT MUST DECIDE THE SAME WAY ON EVERY MACHINE. Every tracked .jsonj
    # under data/corpus/ is checked against the live rule: one that is neither a shard
    # nor a known non-shard stops the run HERE and not on the pod (or the reverse) the
    # moment the two filesystems differ. data/corpus/sample/code_rp1t_handread50.jsonl
    # was exactly that -- tracked, present on the Mac, absent from the pod, so e2e
    # passed there and raised SystemExit here from one commit (e1, 2026-09-01). It is
    # audit evidence, so it moved to data/eval/ rather than joining the blacklist:
    # every name added there makes it harder to see what is real corpus.
    tracked = subprocess.run(["git", "ls-files", "data/corpus"], capture_output=True,
                             text=True, cwd=ROOT).stdout.split()
    names = sorted({os.path.basename(t) for t in tracked if t.endswith(".jsonl")})
    if names:
        _, unknown_tracked = _selection(train_py, names)
        assert not unknown_tracked, (
            f"tracked .jsonl file(s) under data/corpus/ are neither shards nor known "
            f"non-shards: {unknown_tracked}. _domain_seqs stops the run on these, so "
            f"this commit behaves differently on a machine that has them.")

    print(f"selftest OK ({len(shards)} shards kept, {len(not_shards)} non-shards refused, "
          f"unknown files refused by _domain_seqs)")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))
