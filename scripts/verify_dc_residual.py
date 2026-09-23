"""Output-side residual rescan of the gate _dc corpora against the CURRENT gate.

The 0911 pass over these domains (runs/dc_residual_verify_0911.json, fact
cont.gate_dc_residual_0911) was run by scripts that lived in runs/ and were never
tracked; they are gone from both machines (checked 2026-09-24 through ~/bin/pod).
This is a rewrite, and it is tracked for that reason.

It answers one question: does any KEPT row still carry a HumanEval/MBPP 13-gram,
judged against a gate whose bytes are the CURRENT pins. The gate is loaded through
Decontaminator.load_default(root) -- the same entry both writers use
(scripts/filter_gate_domains.py:58, datagen/ultradata_shards.py:253) -- so the rescan
asks the filter's question rather than a re-implementation of it.

Read-only: no writes to any corpus directory, no token caches, no GPU.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

DOMAINS = [
    "code_ultra_l2_dc",
    "code_ultra_l3_noexec_dc",
    "code_py_starcoder_dc",
    "math_owm_stage2_dc",
    "en_c4_stage2_dc",
    "cot_dc",
]

_WORKER = {}


def _decon(root):
    d = _WORKER.get(root)
    if d is None:
        sys.path.insert(0, root)
        from filters.decontam_ngram import Decontaminator

        d = Decontaminator.load_default(root)
        _WORKER[root] = d
    return d


def _scan_shard(args, decon=None):
    """Return (rows, hits, sample) for one shard.

    `content` is the field both writers hand to hit(): the non-ultra writer
    (scripts/filter_gate_domains.py:71 `json.loads(s).get("content", "")`) and the
    ultra writer (datagen/ultradata_shards.py:301 `decon.hit(rec["content"])`).

    `decon` is injectable so the selftest can drive this exact read path with a
    synthetic gate on a machine without the gitignored benchmark files.
    """
    p, root = args
    if decon is None:
        decon = _decon(root)
    rows = hits = 0
    sample = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            rows += 1
            try:
                c = json.loads(s).get("content", "")
            except json.JSONDecodeError:
                c = ""
            h = decon.hit(c)
            if h:
                hits += 1
                if len(sample) < 5:
                    sample.append({"shard": os.path.basename(p), "row": rows, **h})
    return rows, hits, sample


def _dump(out, path):
    """Write the result after EVERY domain, not once at the end.

    A full pass is ~2h of pod IO across ~300GB; writing only at the end means an
    interrupt at 90% loses 100% of it (the restartability check's incident: a
    two-hour scoring job killed at 50% lost everything). Each domain is
    independent, so the partial file is a valid partial answer.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)


def _selftest(root=None):
    """Known-answer cases driven through _scan_shard itself.

    A zero over a corpus is only informative if the instrument can see a hit. The
    0911 pass left no such control behind, so a reader had to take its zeros on
    trust. These cases assert the read path: an injected gate solution is flagged
    exactly once and at the right row, and clean rows are not.

    The gate is SYNTHETIC when the gitignored benchmark files are absent (CI), the
    same split filters/decontam_ngram._selftest uses; the real-gate cases run on the
    pod as end-to-end confirmation.
    """
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from filters.decontam_ngram import Decontaminator, ngrams

    sol = (
        "if n == 0:\n        return 0\n    if n == 1:\n        return 1\n    return fib(n - 1) + fib(n - 2)\n"
    )
    synth = Decontaminator({"synth:fib": {"solution": ngrams(sol)}})
    clean = "def load_config(path):\n    return open(path).read()\n"
    prose = "the quick brown fox jumps over the lazy dog repeatedly today"
    injected = "def fib(n):\n" + sol

    def fresh(tmp, name, contents, decon):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8") as fh:
            for c in contents:
                fh.write(json.dumps({"content": c, "source": "x", "url": "y"}) + "\n")
        return _scan_shard((p, root), decon)

    # (1) the field CONTRACT: _scan_shard must hand hit() the row's `content`
    # value. Tested with a recording decon, because the return tuple cannot tell
    # this apart from "found the hit but did not count it" -- both mutations yield
    # exactly (3, 0, []) -- so no assertion on rows/hits/sample can separate them.
    # Only observing the call separates them, and this is the boundary that
    # matters: a scanner reading the wrong field returns a clean zero over any
    # corpus, which is indistinguishable from a decontaminated one.
    class _Spy:
        def __init__(self, inner):
            self.inner, self.seen = inner, []

        def hit(self, c):
            self.seen.append(c)
            return self.inner.hit(c)

    tmp = tempfile.mkdtemp()
    contents = [clean, injected, prose]
    spy = _Spy(synth)
    rows, hits, sample = fresh(tmp, "knownanswer_000.jsonl", contents, spy)
    assert spy.seen == contents, (
        f"hit() must be handed each row's `content` value in order; got {[c[:24] for c in spy.seen]}"
    )
    assert rows == 3, f"expected 3 rows, read {rows}"

    # (1b) and a found hit is COUNTED and sampled
    assert hits == 1, f"expected exactly the injected hit, got {hits}"
    assert sample, "a counted hit must produce a sample entry"
    assert sample[0]["row"] == 2, f"hit must be on row 2, got {sample[0]['row']}"
    assert sample[0]["problem"] == "synth:fib" and sample[0]["part"] == "solution", sample[0]

    # (2) a clean shard reads as zero -- the negative control every zero rests on
    assert fresh(tmp, "clean_000.jsonl", [clean, prose], synth)[:2] == (2, 0)

    # (3) a blank line is skipped, not counted as a row (the writers' rule)
    p3 = os.path.join(tmp, "blank_000.jsonl")
    with open(p3, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"content": clean}) + "\n\n\n")
    assert _scan_shard((p3, root), synth)[:2] == (1, 0), "blank lines must not count as rows"

    # (4) a row with no `content` key is scanned, not skipped
    p4 = os.path.join(tmp, "nokey_000.jsonl")
    with open(p4, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"source": "x"}) + "\n")
    assert _scan_shard((p4, root), synth)[:2] == (1, 0)

    # (5) the shard enumeration returns EVERY shard, not a prefix. N is the whole
    # content of the claim "0 hits over N shards": a main() that scanned one shard
    # per domain would report the same clean zero with nothing objecting. Checked
    # against a synthetic domain dir rather than the real corpus, so it runs in CI.
    droot = tempfile.mkdtemp()
    dd = os.path.join(droot, "data", "corpus", "synth_dc")
    os.makedirs(dd)
    names = [f"synth_{i:03d}.jsonl" for i in range(7)]
    for n in names:
        with open(os.path.join(dd, n), "w", encoding="utf-8") as fh:
            fh.write("{}\n")
    # a non-shard file in the same directory must not enter the population
    with open(os.path.join(dd, "build_corpus_stats.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    got = domain_shards(droot, "synth_dc")
    assert len(got) == len(names), f"enumeration must return all {len(names)} shards, got {len(got)}"
    assert [os.path.basename(g) for g in got] == names, "enumeration must be sorted and complete"
    assert all(g.endswith(".jsonl") for g in got), "non-shard files must not be scanned"

    # (6) the REAL gate, where the benchmark files exist (pod only)
    if os.path.exists(os.path.join(root, "data/eval/humaneval/humaneval_164.jsonl")):
        real = Decontaminator.load_default(root)
        with open(os.path.join(root, "data/eval/humaneval/humaneval_164.jsonl")) as fh:
            he = json.loads(fh.readline())
        assert real.hit(he["canonical_solution"]), "real gate must flag an HE solution"
        tmp2 = tempfile.mkdtemp()
        r, h, s = fresh(tmp2, "he_000.jsonl", [clean, he["canonical_solution"]], real)
        assert (r, h) == (2, 1), f"real gate: expected (2,1), got {(r, h)}"
        assert s[0]["problem"].startswith("humaneval:"), s[0]
        print("selftest: 7 cases pass (synthetic read path, enumeration, + REAL gate end-to-end)")
    else:
        print("selftest: 6 cases pass (synthetic only; benchmark files absent)")


def domain_shards(root, dom):
    """Every .jsonl in the domain directory, sorted. The rescan's N.

    Extracted from main() because N is the whole content of the claim: "0 hits
    over N shards" says nothing if N is a truncated population, and nothing else
    asserts it. A main() that scanned only the first shard would report a clean
    zero with no assertion objecting.
    """
    d = os.path.join(root, "data/corpus", dom)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/aupai")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="runs/dc_residual_current.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest(a.root if os.path.isdir(a.root) else None)

    sys.path.insert(0, a.root)
    from filters.decontam_ngram import decontam_fp

    he = os.path.join(a.root, "data/eval/humaneval/humaneval_164.jsonl")
    mb = os.path.join(a.root, "data/eval/mbpp_holdouts.jsonl")
    gate = {"decontam_fp": decontam_fp(he, mb)}
    for p in (he, mb):
        with open(p, "rb") as bf:
            gate[os.path.basename(p)] = hashlib.sha256(bf.read()).hexdigest()
    with open(os.path.join(a.root, "filters/decontam_ngram.py"), "rb") as bf:
        gate["module_sha256"] = hashlib.sha256(bf.read()).hexdigest()
    print("gate:", json.dumps(gate), flush=True)

    out = {"measured": "2026-09-24", "gate": gate, "per_domain": {}}
    for dom in a.domains.split(","):
        shards = domain_shards(a.root, dom)
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_scan_shard, [(p, a.root) for p in shards]))
        rows = sum(r for r, _, _ in res)
        hits = sum(h for _, h, _ in res)
        sample = [s for _, _, ss in res for s in ss][:20]
        out["per_domain"][dom] = {
            "shards_rescanned": len(shards),
            "rows_rescanned": rows,
            "residual_hits": hits,
            "sample_hits": sample,
        }
        out["shards_total_rescanned"] = sum(v["shards_rescanned"] for v in out["per_domain"].values())
        out["rows_total_rescanned"] = sum(v["rows_rescanned"] for v in out["per_domain"].values())
        out["residual_hits_total"] = sum(v["residual_hits"] for v in out["per_domain"].values())
        _dump(out, a.out)
        print(f"{dom}: shards={len(shards)} rows={rows} hits={hits}", flush=True)

    print("wrote", a.out, "hits_total =", out["residual_hits_total"])


if __name__ == "__main__":
    main()
