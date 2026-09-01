"""Does eval/domain_loss.py's "head" actually land in train.py's val slice? It does not.

domain_loss.py:47 selects `sorted(files)[0]` and calls it "the head, matching train.py's val
split". train.py shuffles the concatenated documents (random.Random(_sample_seed()).shuffle,
:1560) and THEN slices val = seqs[:n_val] (:1751). The shuffle is between the two, so the
alphabetically-first shard's first rows are scattered uniformly across the pool and the
correspondence the comment asserts does not exist.

This replicates the shuffle on document INDICES -- no tokenizer, no GPU, seconds -- and reports
what fraction of the scored documents land in the val prefix, against what pure chance predicts.
Equal means the scored set is a uniform sample of the training pool.

    python3 t64_val_head_overlap.py <corpus_root> <domain> [sample_seed]
    python3 t64_val_head_overlap.py --selftest
"""
import glob
import json
import os
import random
import sys

SCORED = 4000   # eval/domain_loss.py:33 HOLDOUT_ROWS
VAL_MAX = 5000  # train.py:225 Cfg.val_rows_max, which binds on every real domain


def count_docs(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("text") or o.get("content"):
                n += 1
    return n


def overlap(root, domain, seed=0, scored=SCORED, val_rows=VAL_MAX):
    shards = sorted(p for p in glob.glob(os.path.join(root, domain, "*.jsonl"))
                    if "build_corpus_stats" not in os.path.basename(p))
    counts = [count_docs(p) for p in shards]
    total = sum(counts)
    head_n = min(scored, counts[0]) if counts else 0
    idx = list(range(total))
    random.Random(seed).shuffle(idx)  # the same call train.py makes on the text list
    pos = [0] * total
    for newpos, orig in enumerate(idx):
        pos[orig] = newpos
    in_val = sum(1 for d in range(head_n) if pos[d] < val_rows)
    return {"domain": domain, "shards": len(shards), "total_docs": total,
            "first_shard_docs": counts[0] if counts else 0, "scored_docs": head_n,
            "scored_landing_in_val": in_val,
            "frac_in_val": round(in_val / max(head_n, 1), 5),
            "expected_if_uniform": round(val_rows / max(total, 1), 5)}


def selftest():
    """Known answer: with NO shuffle the head is entirely inside val; with one it is not.

    A probe that reports "the head is not in val" is worthless if it would report that for a
    pipeline where the head IS in val -- so the green case has to be exercised too.
    """
    d = os.path.join(__import__("tempfile").mkdtemp(), "kat")
    os.makedirs(d)
    rows = [{"text": f"doc {i}"} for i in range(1000)]
    with open(os.path.join(d, "kat_000.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    root = os.path.dirname(d)

    # identity "shuffle": seed irrelevant, we pass val_rows >= scored so every scored doc is in
    got = overlap(root, "kat", scored=100, val_rows=1000)
    assert got["frac_in_val"] == 1.0, got

    # a real shuffle over 1000 docs with a 100-row val: chance is 0.1, and the measured
    # fraction must sit near it rather than at 1.0
    got = overlap(root, "kat", seed=0, scored=100, val_rows=100)
    assert got["expected_if_uniform"] == 0.1, got
    assert abs(got["frac_in_val"] - 0.1) < 0.12, got
    print(f"selftest OK: no-shuffle head is 100% in val; shuffled head is "
          f"{got['frac_in_val']:.2f} against a 0.10 chance floor")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(json.dumps(overlap(sys.argv[1], sys.argv[2],
                                 int(sys.argv[3]) if len(sys.argv) > 3 else 0)))
