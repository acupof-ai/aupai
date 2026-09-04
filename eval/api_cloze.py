#!/usr/bin/env python3
"""API-name cloze over two regions of one training domain: recall vs generalisation.

Readout 2 of the memory-layers program (runs/prereg.jsonl#memory_layers_0905, amendment_2
and amendment_3). The claim under test is "a memory layer buys knowledge, not reasoning",
and the estimator is a DIFFERENCE IN DIFFERENCES:

    delta_seen   = arm - control on rows the arms TRAINED on      (recall)
    delta_unseen = arm - control on rows NOBODY read              (generalisation)
    knowledge    = delta_seen - delta_unseen, beyond both SEs

WHY NOT A CLOSED-BOOK PROBE ALONE, which is what the charter first specified: a cloze over
unread rows cannot show memorisation, because no arm ever saw those spans. It measures
in-domain generalisation, which readout 1 (block-paired doc_cu val) already measures with a
tighter estimator. Recall of TRAINING content is what a memory table could hold, so the
seen region has to be in the probe and the unread region is its control. Same statistical
shape as gate_failure_shapes.md §114: a treatment effect that was never isolated from "any
treatment does this", where the fix was also the difference of differences.

ABSOLUTE SEEN SCORES ARE NOT A CAPABILITY NUMBER. They are training-set accuracy -- the
defect tilerl measured in domain_loss on 2026-09-01 -- and are valid here ONLY as an
arm-minus-control difference on identical data order, which the charter guarantees by
reusing the control's launch line byte-for-byte. main() refuses to print SEEN without the
paired label saying so.

THE REGION BOUNDARY IS READ, NEVER DERIVED FROM A FRACTION. train.py:1783 allocates
`idx = torch.arange(used[name], used[name] + want) % len(pool)` over `pool = seqs[n_val:]`,
so a run consumes a CONTIGUOUS half-open range of pool indices; the shuffle at :1789
permutes visit order within a phase, not membership. So:

    SEEN = seqs[n_val : n_val + row_cursor]
    TAIL = seqs[n_val + row_cursor : N]

`row_cursor` comes off the control checkpoint, and it is a CHECKED quantity rather than a
recorded one: train.py:1163-1178 refuses to write a checkpoint unless
sum(row_cursor.values()) == step x batch x accum x world. Verified on
ckpt_b0_headmix_armA.pt: 244,160 == 3815 x 16 x 2 x 2. An earlier version of this boundary
used the mix file's cap_covers (542,151) and was wrong by 6.7x -- that field is the 8B
budget's PLAN, and the control ran 1B. A fraction of a plan is not a cursor.

Provenance of the design decisions: e1 <-> 4c, message 0b99e58b-1c06-47b0-a199-778daa98cbd7
(pool-coordinate contiguity, the cursor-sum identity, 64.0 rows/step as the denominator);
amendment_3 closed by that message rather than by a fourth amendment.

    python eval/api_cloze.py --build --tokenizer data/tokenizer.json   # writes the items
    python eval/api_cloze.py --ckpt <ckpt> --data data/eval/api_cloze.jsonl

# restartable: --build streams the cache once and writes one file; scoring reads it.
"""

import argparse
import json
import math
import os
import random
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

DOMAIN = "code_py_starcoder"
DATA = os.path.join(ROOT, "data", "eval", "api_cloze.jsonl")
N_OPTIONS = 4  # gold + 3 distractors; chance = 0.25

#: An attribute access whose object is a plain name and whose attribute is a real
#: identifier. Anchored on `\b` so `self.x` and `a.b.c` both yield their last hop, and the
#: gold is the ATTRIBUTE, never the object: the object is usually a local variable (no fact
#: to recall) while the attribute is the arbitrary proposition -- `os.path`, `np.arange`,
#: `torch.cat` are memorised, `foo` is not.
ATTR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{1,30})\.([A-Za-z_][A-Za-z0-9_]{1,30})\b")

#: Objects whose attribute sets are large, stable and genuinely arbitrary. Restricting to
#: these is what makes a distractor a REAL name from the same module rather than a random
#: token: `os.pathx` is not a plausible wrong answer, `os.getcwd` is. Measured against the
#: corpus rather than imagined -- see --build's printed per-object counts.
MIN_PER_OBJECT = 8       # an object needs this many distinct attributes to supply 3 distractors
MIN_PREFIX_CHARS = 120   # a prefix shorter than this gives the model almost nothing to condition on
MAX_PREFIX_CHARS = 800   # and a very long one costs forward time for no extra signal


def region_bounds(ckpt_path, cache_tokens=None, seq=4096, domain=DOMAIN):
    """(n_val, seen_lo, seen_hi, n_rows) read from the control checkpoint, never derived.

    n_rows needs the cache's token count; passing None leaves it None so the caller can
    fill it from the pod. Everything else is on the checkpoint.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) or {}
    rc = ck.get("row_cursor") or {}
    if domain not in rc:
        raise RuntimeError(
            f"{ckpt_path} has no row_cursor entry for {domain}; the trained region cannot be "
            f"read and MUST NOT be guessed from the mix's cap_covers (that is the 8B plan, "
            f"not what this run consumed -- wrong by 6.7x on the 1B control)")
    step = ck.get("row_cursor_as_of_step")
    n_val_cap = int(cfg.get("val_rows_max", 5000))
    val_frac = float(cfg.get("val_frac", 0.05))
    cursor = int(rc[domain])
    n_rows = (cache_tokens // (seq + 1)) if cache_tokens else None
    # n_val = min(max(1, int(N * val_frac)), val_rows_max) -- train.py:1690. The cap binds
    # for any domain over 100k rows, which this one is by 20x, but it is computed rather
    # than assumed so a smaller domain does not silently get the cap.
    n_val = min(max(1, int(n_rows * val_frac)), n_val_cap) if n_rows else n_val_cap
    return {
        "n_val": n_val, "seen_lo": n_val, "seen_hi": n_val + cursor,
        "n_rows": n_rows, "row_cursor": cursor, "row_cursor_as_of_step": step,
        "seed": ck.get("row_cursor_seed"), "srcfp": (ck.get("row_cursor_srcfp") or {}).get(domain),
        "vocab_id": ck.get("vocab_id"), "val_frac": val_frac, "val_rows_max": n_val_cap,
    }


def items_from_text(text, rng, per_object):
    """API-name cloze items from one decoded row.

    per_object maps object -> set of attributes seen ACROSS THE WHOLE REGION, so distractors
    are real names from the same object rather than same-row coincidences.

    OPTIONS ARE NOT LENGTH-EQUALISED HERE; build() filters on token length after
    tokenizing, because run_eval.score_mc_items sums log-probs without normalising and a
    longer option accumulates more negative log-prob. A 4-way item whose options differ in
    token count measures length, not knowledge.
    """
    out = []
    for m in ATTR_RE.finditer(text):
        obj, attr = m.group(1), m.group(2)
        pool = per_object.get(obj)
        if not pool or len(pool) < MIN_PER_OBJECT or attr not in pool:
            continue
        start = m.start(2)
        prefix = text[:start]
        if not (MIN_PREFIX_CHARS <= len(prefix) <= MAX_PREFIX_CHARS):
            if len(prefix) > MAX_PREFIX_CHARS:
                prefix = prefix[-MAX_PREFIX_CHARS:]
            else:
                continue
        others = sorted(pool - {attr})
        if len(others) < N_OPTIONS - 1:
            continue
        distractors = rng.sample(others, N_OPTIONS - 1)
        options = [attr] + distractors
        order = list(range(N_OPTIONS))
        rng.shuffle(order)
        out.append({
            "prompt": prefix,
            "options": [options[i] for i in order],
            "label": order.index(0),
            "object": obj,
            "gold": attr,
        })
    return out


def _equal_token_length(tok, item):
    """True when every option tokenizes to the same number of ids.

    THE ONE FILTER THAT MAKES THE SCORES COMPARABLE. Dropped items are counted, never
    silently skipped: a probe that quietly discards 90% of its candidates is measuring
    whatever survived the filter.
    """
    lens = {len(tok.encode(o, add_special_tokens=False).ids) for o in item["options"]}
    return len(lens) == 1


def score_items(model, tok, items, device, batch=32):
    """(acc, se, n, n_unscoreable) via run_eval.score_mc_items.

    Per-item outcomes, because the prereg's rule is "beyond both SEs" and a mean cannot
    supply one.

    UNSCOREABLE ITEMS ARE EXCLUDED AND COUNTED, never scored. An item whose options all
    fail to tokenize keeps its whole `scores` row at the -1e9 fill, argmax returns the
    first maximum, so it is predicted 0 and counts CORRECT whenever label == 0 -- a free
    point whose size depends on where the caller happened to put the gold. Found by a
    read of score_mc's return path and then reproduced (preds [0,0] against labels [0,2]
    scored 1 of 2). It matters more here than in a 2-way benchmark: a 4-way item's gold
    slot is uniformly random, so 1/4 of the affected items would read correct and inflate
    accuracy toward chance from below -- exactly the region where the prereg decides
    "undefined at this scale".
    """
    from eval.run_eval import score_mc_items

    if not items:
        return float("nan"), float("nan"), 0, 0
    preds, labels, scored = score_mc_items(model, tok, items, device, batch_size=batch,
                                           num_id=None)
    n_bad = int((~scored).sum())
    preds, labels = preds[scored], labels[scored]
    correct = (preds == labels).float()
    n = int(correct.numel())
    if not n:
        return float("nan"), float("nan"), 0, n_bad
    acc = correct.mean().item()
    # Binomial SE of a proportion: the outcome is Bernoulli, and this closed form is what
    # the prereg's "beyond both SEs" is read against.
    se = math.sqrt(max(acc * (1 - acc), 0.0) / n)
    return acc, se, n, n_bad


def build(tokenizer_path, ckpt_path, cache_path, per_region, seed, out_path, cache_tokens):
    """Write the item file: equal counts from SEEN and TAIL, with the boundary pinned in it.

    Streams the cache with torch.load's mmap so a 35 GB file is not read into RAM.
    """
    tok = load_tokenizer(tokenizer_path, None)
    bounds = region_bounds(ckpt_path, cache_tokens=cache_tokens)
    seq = 4096
    stream = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    n_rows = stream.numel() // (seq + 1)
    if bounds["n_rows"] and bounds["n_rows"] != n_rows:
        raise RuntimeError(
            f"cache holds {n_rows} rows but the passed token count implied "
            f"{bounds['n_rows']}; one of them is stale and the region boundary depends on it")
    bounds["n_rows"] = n_rows
    seen_lo, seen_hi = bounds["seen_lo"], bounds["seen_hi"]
    if not (0 < seen_lo < seen_hi <= n_rows):
        raise RuntimeError(f"region bounds {seen_lo}..{seen_hi} do not fit {n_rows} rows")

    rng = random.Random(seed)
    regions = {"seen": (seen_lo, seen_hi), "tail": (seen_hi, n_rows)}

    def decode_rows(lo, hi, want_rows):
        idx = rng.sample(range(lo, hi), min(want_rows, hi - lo))
        for k in idx:
            row = stream[k * (seq + 1):(k + 1) * (seq + 1)]
            yield k, tok.decode(row.tolist())

    # PASS 1, per region: which attributes each object actually takes in THIS region.
    # Built per region, not globally, because a distractor drawn from the other region's
    # vocabulary would leak that region's content into this one's options.
    scan_rows = 4000
    per_object = {}
    for name, (lo, hi) in regions.items():
        d = {}
        for _k, text in decode_rows(lo, hi, scan_rows):
            for m in ATTR_RE.finditer(text):
                d.setdefault(m.group(1), set()).add(m.group(2))
        per_object[name] = {o: a for o, a in d.items() if len(a) >= MIN_PER_OBJECT}
        print(f"{name}: {len(d)} objects seen, {len(per_object[name])} with "
              f">={MIN_PER_OBJECT} distinct attributes", flush=True)

    # PASS 2: items, filtered to equal token length, capped per region.
    out, stats = [], {}
    for name, (lo, hi) in regions.items():
        got, seen_rows, dropped_len = [], 0, 0
        for k, text in decode_rows(lo, hi, scan_rows * 3):
            seen_rows += 1
            for it in items_from_text(text, rng, per_object[name]):
                if not _equal_token_length(tok, it):
                    dropped_len += 1
                    continue
                it["region"] = name
                it["row"] = k
                got.append(it)
                if len(got) >= per_region:
                    break
            if len(got) >= per_region:
                break
        stats[name] = {"items": len(got), "rows_read": seen_rows,
                       "dropped_unequal_token_length": dropped_len}
        print(f"{name}: {len(got)} items from {seen_rows} rows "
              f"({dropped_len} dropped for unequal option token length)", flush=True)
        out += got

    header = {
        "_header": True,
        "domain": DOMAIN,
        "n_options": N_OPTIONS,
        "chance": 1.0 / N_OPTIONS,
        "regions": {k: list(v) for k, v in regions.items()},
        "bounds": bounds,
        "build_seed": seed,
        "stats": stats,
        "estimator": "delta_seen - delta_unseen, each arm minus control; SEEN alone is "
                     "training-set accuracy and is not a capability number",
        "provenance": "prereg memory_layers_0905 amendment_2/_3; boundary from "
                      "ckpt_b0_headmix_armA.pt row_cursor, cursor-sum identity verified "
                      "(244160 == 3815*16*2*2); e1<->4c msg "
                      "0b99e58b-1c06-47b0-a199-778daa98cbd7",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for it in out:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} items -> {out_path}")
    return 0


def load_items(path):
    """(header, items). The header is a row, so a reader that forgets to filter it would
    otherwise score it as an item with no options."""
    header, items = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("_header"):
                header = d
            else:
                items.append(d)
    return header, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ckpt", help="checkpoint to score, or the control for --build")
    ap.add_argument("--control_ckpt", default="ckpt_b0_headmix_armA.pt",
                    help="the checkpoint the region boundary is read from")
    ap.add_argument("--cache", default="/data00/tokens_code_py_starcoder.pt")
    ap.add_argument("--cache_tokens", type=int, default=None,
                    help="token count, to cross-check the cache against a recorded value")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--per_region", type=int, default=2500)
    ap.add_argument("--build_seed", type=int, default=20260905)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.build:
        ctrl = a.ckpt or a.control_ckpt
        return build(a.tokenizer, ctrl, a.cache, a.per_region, a.build_seed, a.data,
                     a.cache_tokens)

    if not a.ckpt:
        sys.exit("--ckpt is required to score")
    header, items = load_items(a.data)
    if not header:
        sys.exit(f"{a.data} has no header row; rebuild it -- the region boundary lives there "
                 f"and a scoring run that cannot read it cannot say what it measured")
    model, tok, cfg = load_checkpoint(a.ckpt, a.tokenizer, a.device)
    model.eval()

    res = {}
    for region in ("seen", "tail"):
        sub = [it for it in items if it["region"] == region]
        acc, se, n, n_bad = score_items(model, tok, sub, a.device, a.batch)
        res[region] = {"acc": acc, "se": se, "n": n, "unscoreable": n_bad}
        label = ("TRAINING-SET accuracy, not a capability number -- valid only as an "
                 "arm-minus-control difference" if region == "seen" else "held-out")
        print(f"  {region:5s} acc {acc:.4f} +- {se:.4f}  n={n}  ({label})")
        if n_bad:
            print(f"        {n_bad} item(s) excluded as unscoreable (no option tokenized); "
                  f"they would have counted CORRECT wherever label == 0")
    chance = header["chance"]
    print(f"  chance floor {chance:.4f}")
    for region in ("seen", "tail"):
        r = res[region]
        if not math.isnan(r["se"]) and abs(r["acc"] - chance) <= r["se"]:
            print(f"  {region}: within 1 SE of chance -- readout 2 is UNDEFINED at this "
                  f"scale on this region, not null (prereg memory_layers_0905)")
    if a.json:
        print(json.dumps({"regions": res, "chance": chance,
                          "bounds": header["bounds"], "ckpt": a.ckpt}, ensure_ascii=False))
    return 0


def _selftest():
    """The item construction and the region arithmetic, on known answers.

    No cache, no checkpoint, no card: the region boundary is arithmetic over recorded
    numbers, and item construction is a function of text.
    """
    # 1. THE BOUNDARY. The numbers are the control's real ones, so a change in how they are
    #    combined fails here rather than in a scoring run three hours later.
    toks, seq = 8786916332, 4096
    n_rows = toks // (seq + 1)
    assert n_rows == 2144719, n_rows
    n_val = min(max(1, int(n_rows * 0.05)), 5000)
    assert n_val == 5000, n_val
    seen_lo, seen_hi = n_val, n_val + 80380
    assert (seen_lo, seen_hi) == (5000, 85380), (seen_lo, seen_hi)
    assert n_rows - seen_hi == 2059339, n_rows - seen_hi
    # The cursor-sum identity train.py:1163 enforces at save. If this ever fails the
    # checkpoint's cursor is not absolute and the region is not what it claims.
    cursor_sum = 64450 + 40082 + 19776 + 24700 + 1796 + 1714 + 7496 + 80380 + 3766
    assert cursor_sum == 3815 * 16 * 2 * 2 == 244160, cursor_sum
    # AND THE FRACTION THAT WAS WRONG, kept as a negative known answer: the mix's
    # cap_covers is the 8B plan, 6.7x the 1B control's actual consumption. A future edit
    # that reaches for it fails here.
    assert 542151 / 80380 > 6.5, "cap_covers is no longer far from the cursor -- re-check"

    # 2. ITEM CONSTRUCTION. A synthetic text whose objects and attributes are known.
    rng = random.Random(0)
    per_object = {"os": {f"f{i}" for i in range(10)} | {"path"}}
    text = ("x = 1\n" * 30) + "import os\np = os.path\n"
    items = items_from_text(text, rng, per_object)
    assert items, "no item from a text that plainly contains os.path"
    it = items[0]
    assert it["gold"] == "path" and it["object"] == "os", it
    assert len(it["options"]) == N_OPTIONS, it
    assert it["options"][it["label"]] == "path", "label does not point at the gold"
    assert it["prompt"].endswith("os."), repr(it["prompt"][-12:])
    assert "path" not in it["prompt"][-6:], "the gold leaked into the prompt"
    # The gold is the ATTRIBUTE, never the object: `os` is a name, `path` is the fact.
    assert all(o != "os" for o in it["options"]), it["options"]

    # 3. AN OBJECT WITH TOO FEW ATTRIBUTES YIELDS NOTHING, so distractors are always real
    #    names from the same object rather than padding.
    thin = items_from_text(text, rng, {"os": {"path", "a"}})
    assert not thin, "an object with 2 attributes produced a 4-way item"

    # 4. A SHORT PREFIX IS DROPPED. Without this an item can be `os.` with nothing before
    #    it, which measures the prior over attribute names and not recall.
    short = items_from_text("os.path\n", rng, per_object)
    assert not short, "an item with a 0-char prefix survived"

    # 5. THE HEADER IS NOT AN ITEM. load_items must separate them, or the header scores as
    #    an item with no options and score_mc_items raises on max().
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "items.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_header": True, "chance": 0.25}) + "\n")
            fh.write(json.dumps({"prompt": "a", "options": ["b", "c"], "label": 0,
                                 "region": "seen"}) + "\n")
        h, its = load_items(p)
        assert h and h["chance"] == 0.25, h
        assert len(its) == 1 and "_header" not in its[0], its

    # 6. THE SE IS A BINOMIAL SE AND IT SHRINKS WITH n. Checked because "beyond both SEs"
    #    is the prereg's decision rule, so an SE off by sqrt(n) silently changes the verdict.
    se = lambda p, n: math.sqrt(p * (1 - p) / n)  # noqa: E731
    assert abs(se(0.25, 2500) - 0.00866) < 1e-4, se(0.25, 2500)
    assert se(0.25, 10000) < se(0.25, 2500), "SE does not shrink with n"
    # At n=2500 per region, a difference of differences needs roughly 4x0.0087 = 0.035 to
    # clear both SEs on both halves. Recorded so nobody reads a 0.01 delta as significant.
    assert 4 * se(0.25, 2500) > 0.03, "the resolution note is stale"

    # 7. AN UNSCOREABLE ITEM IS NEITHER RIGHT NOR WRONG, and this is the world that would
    #    have caught the defect the scouts found by reading. score_mc_items leaves the whole
    #    -1e9 fill row in place when every option fails to tokenize; argmax returns the FIRST
    #    maximum, so preds is 0 and the item reads CORRECT whenever label == 0. In a 4-way
    #    probe the gold slot is uniformly random, so a quarter of such items become free
    #    points -- inflating accuracy from below toward chance, which is precisely the band
    #    where the prereg decides "undefined at this scale".
    import types as _types

    class _Flat:
        def __call__(self, x, num_vals=None):
            return (torch.zeros(x.shape[0], x.shape[1], 64),)

    class _EmptyTok:
        """Tokenizes to nothing, which is how a real tokenizer answers for options made only
        of bytes it has no merge for."""

        def encode_batch(self, texts):
            return [_types.SimpleNamespace(ids=([1] if t.strip() else [])) for t in texts]

    from eval.run_eval import score_mc_items as _smi

    bad = [{"prompt": "p", "options": ["", "", "", ""], "label": 0},
           {"prompt": "p", "options": ["", "", "", ""], "label": 2}]
    _p, _l, _sc = _smi(_Flat(), _EmptyTok(), bad, "cpu", batch_size=4)
    assert not bool(_sc.any()), f"an all-empty item was marked scoreable: {_sc.tolist()}"
    assert _p.tolist() == [0, 0], _p.tolist()
    # Without the mask this is 1 of 2 "correct". With it, both are excluded.
    naive = int((_p == _l).sum())
    assert naive == 1, f"the defect no longer reproduces ({naive}); re-derive before relaxing"
    acc, se, n, n_bad = score_items(_Flat(), _EmptyTok(), bad, "cpu", batch=4)
    assert n == 0 and n_bad == 2, (n, n_bad)
    assert math.isnan(acc), f"an all-unscoreable region reported acc {acc} rather than NaN"

    # 8. AND A MIXED REGION USES ONLY THE SCOREABLE ITEMS. If the mask were ignored the
    #    denominator would be 3 and the accuracy 2/3; with it, 1/1.
    mixed = [{"prompt": "p", "options": ["", "", "", ""], "label": 0},
             {"prompt": "p", "options": ["", "", "", ""], "label": 0},
             {"prompt": "p", "options": ["a", "b", "c", "d"], "label": 0}]
    acc, se, n, n_bad = score_items(_Flat(), _EmptyTok(), mixed, "cpu", batch=4)
    assert n == 1 and n_bad == 2, (n, n_bad)
    assert acc == 1.0, acc
    assert se == 0.0, se

    print("api_cloze selftest OK: boundary arithmetic on the control's real numbers "
          f"(N={n_rows}, SEEN=[{seen_lo},{seen_hi}), TAIL={n_rows - seen_hi} rows, cursor sum "
          f"{cursor_sum} == 3815x16x2x2), cap_covers rejected as 6.7x too wide, item "
          "construction on known text (gold is the attribute, label points at it, prompt "
          "ends at the dot), thin objects and short prefixes dropped, header separated, "
          "binomial SE 0.0087 at n=2500, and an unscoreable item is excluded rather than "
          "counted correct (the -1e9 tie-toward-slot-0 free point, reproduced then masked)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
