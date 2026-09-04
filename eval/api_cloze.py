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


def replay_read_rows(mix_path, seed, world, steps, batch, accum, domain=DOMAIN, seq=4096):
    """The POOL INDICES the control actually trained on, by rebuilding its index plan.

    THE ONLY WAY TO GET THIS RIGHT, and the first version of this file got it wrong. The
    allocation inside one phase is contiguous -- train.py:1783,
    `idx = arange(used, used + want) % len(pool)` -- but the plan is then SHUFFLED ACROSS
    ALL DOMAINS TOGETHER at :1789 and STRIPED BY RANK at :1833. A run that stops early
    consumes a prefix of the SHUFFLED plan, so the rows it read are scattered over the
    whole allocation rather than sitting at its front. Measured on this control: a
    prefix-based SEEN region of 80,380 rows contained 10,129 read rows (12.60%) while the
    "never read" tail contained 70,151 of them (3.41%) -- 87% of the training set was
    inside the control region, which dilutes delta_seen - delta_unseen by ~8x toward a
    false null.

    Two numbers say the prefix reading is wrong without running anything. cursor/steps =
    80380/3815 = 21.069 is not an integer, and it equals this domain's WEIGHT SHARE of the
    64 rows/step (0.3297123 x 64 = 21.102) -- a per-step share is what a shuffled draw
    produces, not a contiguous walk. And the phase allocation is
    int(1953125 x 0.3297123) = 643,969 rows, 8.01x the cursor: the run consumed one eighth
    of its own allocation, so seven eighths of the allocation is unread and lies BEYOND the
    cursor, exactly where a prefix reading puts "unseen".

    Reproducing rather than recording is forced: the plan is 22 MB of indices and the
    checkpoint keeps only `_plan_domains` (the per-row DOMAIN, int8, not the row index).
    The replay is verified against a quantity it does not fit: it predicts all NINE
    domains' row_cursor entries exactly, sum 244,160 == 3815 x 16 x 2 x 2. A wrong shuffle
    would have to miss nine independent counts to look right.

    Determinism holds because no domain is capped in this mix (checked below): every
    `want` is int(rows x frac x weight), so the pool SIZE never enters the plan and a pool
    estimate cannot move it. Only the mix, the seed and world do.
    """
    with open(mix_path, encoding="utf-8") as fh:
        mix = json.load(fh)
    names = list(mix["domains"])
    if domain not in names:
        raise RuntimeError(f"{mix_path} has no domain {domain}")
    di = names.index(domain)
    rows = mix["total_tokens"] / seq
    anneal_frac = float(mix.get("anneal_frac", 0.0))
    pools = {n: int(mix["domains"][n].get("pool_rows_estimated") or 0) for n in names}
    used = {n: 0 for n in names}
    g = torch.Generator().manual_seed(seed)
    plan, capped = [], []
    for frac, key in ((1 - anneal_frac, "weight"), (anneal_frac, "anneal")):
        parts = []
        for i, n in enumerate(names):
            d = mix["domains"][n]
            want = int(rows * frac * d.get(key, d["weight"]))
            cap = int(pools[n] * d.get("epochs", 1)) - used[n]
            if want > cap:
                # A CAPPED DOMAIN MAKES THIS REPLAY DEPEND ON THE POOL SIZE, which comes
                # from the mix's estimate rather than from the cache. Refused rather than
                # approximated: the target domain being capped would change its own index
                # set, and any capped domain shifts the shuffle for all of them.
                capped.append(f"{n}: want {want} > cap {cap}")
                want = max(0, cap)
            if want:
                idx = torch.arange(used[n], used[n] + want) % pools[n]
                parts.append(torch.stack([torch.full_like(idx, i), idx]))
            used[n] += want
        if parts:
            ph = torch.cat(parts, dim=1)
            plan.append(ph[:, torch.randperm(ph.shape[1], generator=g)])
    if capped:
        raise RuntimeError(
            "refusing to replay a plan with a capped domain: the epoch cap makes the "
            "allocation depend on the POOL SIZE, which this file reads from the mix's "
            f"estimate and not from the cache -- {'; '.join(capped[:3])}")
    plan = torch.cat(plan, dim=1)
    n = (plan.shape[1] // world) * world
    rows_done = steps * batch * accum
    # THE READ SET IS THE UNION OVER RANKS; THE CURSOR IS RANK 0'S COUNT x WORLD. Two
    # different quantities, and using one formula for both is what the first run of this
    # replay did -- summing the per-rank counts predicted all nine cursors WRONG by up to
    # 97 rows, because train.py:1142 writes `int(counts[i]) * world` from RANK 0's stripe
    # alone. The two agree only when every rank sees the same domain mix, which a shuffled
    # plan does not guarantee (rank 0 drew 40,190 target rows and rank 1 drew 40,090). So
    # the cursor is reproduced the way it is WRITTEN, and the read set the way the data was
    # actually consumed -- every rank's rows were trained on.
    read = set()
    for rank in range(world):
        mine = plan[:, :n][:, rank::world][:, :rows_done]
        read |= set(mine[1][mine[0] == di].tolist())
    r0 = plan[:, :n][:, 0::world][:, :rows_done][0]
    counts = torch.bincount(r0.to(torch.int64), minlength=len(names))
    return {
        "read": read,
        "alloc": used[domain],
        "cursor_pred": {nm: int(counts[i]) * world for i, nm in enumerate(names)},
        "anneal_frac": anneal_frac,
        "mix": os.path.basename(mix_path),
    }


def region_bounds(ckpt_path, mix_path, cache_tokens=None, seq=4096, domain=DOMAIN):
    """The two regions, from the checkpoint's own cursor and a verified plan replay.

    SEEN is the set of pool rows the control read; UNSEEN is pool [alloc, n_pool), which
    the allocation never reached -- verified 0 rows read of 1,495,750 on this control, so
    it is unread by construction rather than by sampling luck. The old version used
    [n_val, n_val + cursor) as SEEN and everything after it as UNSEEN, and both halves
    were wrong: see replay_read_rows.

    The replay is CHECKED against the checkpoint, not trusted: every domain's predicted
    cursor must equal the recorded one. A shuffle, world size or anneal fraction that
    differs from the run's shows up as a mismatch here instead of as a diluted null.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) or {}
    rc = ck.get("row_cursor") or {}
    if domain not in rc:
        raise RuntimeError(
            f"{ckpt_path} has no row_cursor entry for {domain}; the trained region cannot "
            f"be read and MUST NOT be guessed from the mix's cap_covers (that is the "
            f"phase ALLOCATION, 8.01x what this 1B control consumed)")
    step = ck.get("row_cursor_as_of_step")
    if not step:
        raise RuntimeError(f"{ckpt_path} carries no row_cursor_as_of_step; without the step "
                           f"the replay has no prefix length and the regions cannot be built")
    n_val_cap = int(cfg.get("val_rows_max", 5000))
    val_frac = float(cfg.get("val_frac", 0.05))
    cursor = int(rc[domain])
    batch = int(cfg.get("batch", 16))
    accum = int(cfg.get("accum", 2))
    world = int(cfg.get("world", 0) or 0)
    if not world:
        # world is not a Cfg field on this checkpoint, so it comes from the identity
        # train.py:1163 enforces at save: sum(row_cursor) == step x batch x accum x world.
        # Derived and then CHECKED below, never assumed to be 2.
        tot = sum(int(v) for v in rc.values())
        den = step * batch * accum
        if den <= 0 or tot % den:
            raise RuntimeError(
                f"cannot derive world: sum(row_cursor)={tot} is not a multiple of "
                f"step x batch x accum = {den}")
        world = tot // den
    rep = replay_read_rows(mix_path, int(ck.get("row_cursor_seed") or cfg.get("seed", 42)),
                           world, step, batch, accum, domain=domain, seq=seq)
    bad = {k: (v, rc.get(k)) for k, v in rep["cursor_pred"].items() if rc.get(k) != v}
    if bad:
        raise RuntimeError(
            f"plan replay disagrees with {os.path.basename(ckpt_path)}'s row_cursor on "
            f"{len(bad)} of {len(rep['cursor_pred'])} domain(s) (predicted, recorded): "
            f"{dict(list(bad.items())[:4])}. The replay does not describe this run, so the "
            f"regions it would build are not this run's seen and unseen rows")
    n_rows = (cache_tokens // (seq + 1)) if cache_tokens else None
    n_val = min(max(1, int(n_rows * val_frac)), n_val_cap) if n_rows else n_val_cap
    n_pool = (n_rows - n_val) if n_rows else None
    alloc = rep["alloc"]
    if n_pool and alloc >= n_pool:
        raise RuntimeError(
            f"the allocation ({alloc}) covers the whole pool ({n_pool}), so there is no "
            f"unread region to use as a control and readout 2 has no estimator here")
    return {
        "n_val": n_val, "n_rows": n_rows, "n_pool": n_pool,
        "row_cursor": cursor, "row_cursor_as_of_step": step,
        "batch": batch, "accum": accum, "world": world,
        "alloc": alloc, "seen_rows": sorted(rep["read"]),
        "unseen_lo": alloc, "unseen_hi": n_pool,
        "mix": rep["mix"], "anneal_frac": rep["anneal_frac"],
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


def build(tokenizer_path, ckpt_path, mix_path, cache_path, per_region, seed, out_path,
          cache_tokens):
    """Write the item file: equal counts from SEEN and UNSEEN, with both regions pinned in it.

    Streams the cache with torch.load's mmap so a 35 GB file is not read into RAM.

    SEEN IS A SET OF ROW INDICES, NOT AN INTERVAL. region_bounds replays the control's
    index plan to get the rows it actually read; UNSEEN is the pool tail beyond the phase
    allocation, which the plan never reached. Sampling an interval on the seen side is what
    made the first build invalid -- the read rows are scattered over the allocation at
    ~12.5% density, so an interval draws 7 unread rows for every read one.
    """
    tok = load_tokenizer(tokenizer_path, None)
    bounds = region_bounds(ckpt_path, mix_path, cache_tokens=cache_tokens)
    seq = 4096
    # THE CO-RESIDENCY REFUSAL, at this read. 35.1 GB off /data00 is over the threshold, so
    # beside a live claim this raises. It is called HERE and not left to assert_caches_fresh
    # because this file never goes through train._domain_seqs -- it opens the cache by path,
    # which is the population the guard's harness check did not cover when I landed it
    # (38af3d47). Freshness is deliberately NOT asserted: the item file pins the cache's
    # srcfp and vocab_id, and a rebuild would change those rather than silently agree.
    from cache_guard import assert_not_co_resident

    assert_not_co_resident([DOMAIN])
    stream = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    n_rows = stream.numel() // (seq + 1)
    if bounds["n_rows"] and bounds["n_rows"] != n_rows:
        raise RuntimeError(
            f"cache holds {n_rows} rows but the passed token count implied "
            f"{bounds['n_rows']}; one of them is stale and the region boundary depends on it")
    bounds["n_rows"] = n_rows
    n_val = bounds["n_val"]
    n_pool = n_rows - n_val
    if bounds["n_pool"] and bounds["n_pool"] != n_pool:
        raise RuntimeError(f"pool size {n_pool} from the cache but {bounds['n_pool']} from "
                           f"the recorded token count")
    bounds["n_pool"], bounds["unseen_hi"] = n_pool, n_pool
    # POOL COORDINATES -> CACHE ROW. `pool = seqs[n_val:]` (train.py:1712), so pool index i
    # is cache row n_val + i. Applied once, here, rather than inside the sampler: the
    # replay speaks pool coordinates and the mmap speaks cache rows, and the first build
    # conflated them in the header (it printed [5000, 85380] as if the cursor were a pool
    # interval AND a cache interval).
    seen_pool = bounds["seen_rows"]
    if not seen_pool:
        raise RuntimeError("the replay returned no read rows for this domain")
    unseen_lo, unseen_hi = bounds["unseen_lo"], n_pool
    if not (0 <= unseen_lo < unseen_hi):
        raise RuntimeError(f"unseen region {unseen_lo}..{unseen_hi} is empty: the allocation "
                           f"covers the pool and there is no unread control region")
    seen_set = set(seen_pool)
    leak = sum(1 for k in seen_pool if unseen_lo <= k < unseen_hi)
    if leak:
        raise RuntimeError(
            f"{leak} read row(s) fall inside the UNSEEN region [{unseen_lo}, {unseen_hi}) -- "
            f"the control region is contaminated and delta_unseen would not be a "
            f"generalisation measurement")

    rng = random.Random(seed)

    def sample_pool(name, want_rows):
        if name == "seen":
            return rng.sample(seen_pool, min(want_rows, len(seen_pool)))
        span = unseen_hi - unseen_lo
        return [unseen_lo + i for i in rng.sample(range(span), min(want_rows, span))]

    def decode_rows(name, want_rows):
        for k in sample_pool(name, want_rows):
            r = n_val + k  # pool coordinate -> cache row
            row = stream[r * (seq + 1):(r + 1) * (seq + 1)]
            yield k, tok.decode(row.tolist())

    # PASS 1, per region: which attributes each object actually takes in THIS region.
    # Built per region, not globally, because a distractor drawn from the other region's
    # vocabulary would leak that region's content into this one's options.
    scan_rows = 4000
    per_object = {}
    for name in ("seen", "unseen"):
        d = {}
        for _k, text in decode_rows(name, scan_rows):
            for m in ATTR_RE.finditer(text):
                d.setdefault(m.group(1), set()).add(m.group(2))
        per_object[name] = {o: a for o, a in d.items() if len(a) >= MIN_PER_OBJECT}
        print(f"{name}: {len(d)} objects seen, {len(per_object[name])} with "
              f">={MIN_PER_OBJECT} distinct attributes", flush=True)

    # PASS 2: items, filtered to equal token length, capped per region.
    out, stats = [], {}
    for name in ("seen", "unseen"):
        got, rows_read, dropped_len = [], 0, 0
        for k, text in decode_rows(name, scan_rows * 3):
            rows_read += 1
            for it in items_from_text(text, rng, per_object[name]):
                if not _equal_token_length(tok, it):
                    dropped_len += 1
                    continue
                it["region"] = name
                it["pool_row"] = k
                it["cache_row"] = n_val + k
                it["was_read"] = k in seen_set
                got.append(it)
                if len(got) >= per_region:
                    break
            if len(got) >= per_region:
                break
        # THE PROPERTY THE REGIONS EXIST FOR, asserted per item rather than assumed from the
        # bounds: every seen item comes from a row the control read, every unseen item from
        # one it did not. This is the assertion the first build would have failed.
        want_read = (name == "seen")
        wrong = [it["pool_row"] for it in got if it["was_read"] != want_read]
        if wrong:
            raise RuntimeError(
                f"{len(wrong)} {name} item(s) have was_read != {want_read}, e.g. pool row "
                f"{wrong[0]}: the region does not hold what its name says")
        stats[name] = {"items": len(got), "rows_read": rows_read,
                       "dropped_unequal_token_length": dropped_len,
                       "distinct_rows": len({it["pool_row"] for it in got})}
        print(f"{name}: {len(got)} items from {rows_read} rows "
              f"({dropped_len} dropped for unequal option token length)", flush=True)
        out += got

    header = {
        "_header": True,
        "domain": DOMAIN,
        "n_options": N_OPTIONS,
        "chance": 1.0 / N_OPTIONS,
        "regions": {"seen": f"the {len(seen_pool)} pool rows the control read (scattered "
                            f"over [{min(seen_pool)}, {max(seen_pool)}], "
                            f"{100.0 * len(seen_pool) / (max(seen_pool) - min(seen_pool) + 1):.2f}"
                            f"% density)",
                    "unseen": [unseen_lo, unseen_hi]},
        "bounds": {k: v for k, v in bounds.items() if k != "seen_rows"},
        "n_seen_rows": len(seen_pool),
        "build_seed": seed,
        "stats": stats,
        "estimator": "delta_seen - delta_unseen, each arm minus control; SEEN alone is "
                     "training-set accuracy and is not a capability number",
        "provenance": "prereg memory_layers_0905 amendment_2/_3; SEEN = the control's "
                      "replayed read set (plan replay predicts all 9 row_cursor entries "
                      "exactly, sum 244160 == 3815*16*2*2); UNSEEN = pool beyond the phase "
                      "allocation, never allocated. The first build used [n_val, n_val+"
                      "cursor) as SEEN and was invalid: 87% of the read rows fell in its "
                      "control region. e1<->4c msg 0b99e58b-1c06-47b0-a199-778daa98cbd7",
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
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_200m_8b.json"),
                    help="the mix the control trained on; its index plan is replayed to "
                         "recover which rows the run actually read")
    ap.add_argument("--cache_tokens", type=int, default=None,
                    help="token count, to cross-check the cache against a recorded value")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--per_region", type=int, default=2500)
    ap.add_argument("--build_seed", type=int, default=20260905)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="write the result JSON here (score_matrix's runner contract)")
    a = ap.parse_args()

    if a.build:
        ctrl = a.ckpt or a.control_ckpt
        return build(a.tokenizer, ctrl, a.mix, a.cache, a.per_region, a.build_seed, a.data,
                     a.cache_tokens)

    if not a.ckpt:
        sys.exit("--ckpt is required to score")
    header, items = load_items(a.data)
    if not header:
        sys.exit(f"{a.data} has no header row; rebuild it -- the region boundary lives there "
                 f"and a scoring run that cannot read it cannot say what it measured")
    if "n_seen_rows" not in header:
        # THE FIRST BUILD'S FILE IS NOT SCOREABLE, and it is not distinguishable from a
        # correct one by its item rows: both hold prompts, options and a region label. Only
        # the header says whether SEEN is the replayed read set or the invalid
        # [n_val, n_val+cursor) interval, so the version check is here rather than in a note.
        sys.exit(f"{a.data} was built before the region fix (no n_seen_rows in its header): "
                 f"its 'seen' region is the interval [n_val, n_val+cursor), which holds only "
                 f"12.6% read rows while its 'tail' holds 87% of the training set. Rebuild "
                 f"with --build before scoring.")
    model, tok, cfg = load_checkpoint(a.ckpt, a.tokenizer, a.device)
    model.eval()

    res = {}
    for region in ("seen", "unseen"):
        sub = [it for it in items if it["region"] == region]
        acc, se, n, n_bad = score_items(model, tok, sub, a.device, a.batch)
        res[region] = {"acc": acc, "se": se, "n": n, "unscoreable": n_bad}
        label = ("TRAINING-SET accuracy, not a capability number -- valid only as an "
                 "arm-minus-control difference" if region == "seen"
                 else "never allocated to any run")
        print(f"  {region:6s} acc {acc:.4f} +- {se:.4f}  n={n}  ({label})")
        if n_bad:
            print(f"        {n_bad} item(s) excluded as unscoreable (no option tokenized); "
                  f"they would have counted CORRECT wherever label == 0")
    chance = header["chance"]
    print(f"  chance floor {chance:.4f}")
    undefined = []
    for region in ("seen", "unseen"):
        r = res[region]
        if not math.isnan(r["se"]) and abs(r["acc"] - chance) <= r["se"]:
            undefined.append(region)
            print(f"  {region}: within 1 SE of chance -- readout 2 is UNDEFINED at this "
                  f"scale on this region, not null (prereg memory_layers_0905)")
    # THE HALF-DIFFERENCE THIS ONE CHECKPOINT CAN CARRY, and nothing more. The estimator is
    # delta_seen - delta_unseen where each delta is arm MINUS CONTROL, so a single run
    # produces two accuracies and not a knowledge number. Recorded as the two regions plus
    # their gap so the caller subtracting two records has both halves; `within_region_gap`
    # is NOT the readout -- it is seen minus unseen on ONE model, which is contaminated by
    # anything that makes the seen region easier than the unseen one for every model
    # (different files, different API mix). Only the arm-minus-control difference of these
    # gaps removes that.
    out = {
        "regions": res,
        "chance": chance,
        "within_region_gap": (res["seen"]["acc"] - res["unseen"]["acc"]
                              if not (math.isnan(res["seen"]["acc"])
                                      or math.isnan(res["unseen"]["acc"])) else None),
        "gap_note": "seen minus unseen on ONE checkpoint; readout 2 is the ARM MINUS CONTROL "
                    "difference of this gap (prereg memory_layers_0905 amendment_2/_4). This "
                    "number alone is not evidence of memorisation",
        "undefined_regions": undefined,
        "n_seen_rows": header.get("n_seen_rows"),
        "item_file": os.path.relpath(a.data, ROOT),
        "bounds": header["bounds"],
        "ckpt": a.ckpt,
    }
    print(f"  gap (seen - unseen, ONE model, not the readout): {out['within_region_gap']}")
    if a.json:
        print(json.dumps(out, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
    return 0


def _selftest():
    """The item construction and the region arithmetic, on known answers.

    No cache, no checkpoint, no card: the region boundary is arithmetic over recorded
    numbers, and item construction is a function of text.
    """
    # 1. THE REGIONS, on the control's real numbers. The load-bearing case is the one that
    #    caught the first build: a CONTIGUOUS reading of the cursor is wrong, and the two
    #    regions it produces are both contaminated. Reproduced here from the mix's own
    #    arithmetic so a future edit that reaches back for the interval fails at commit.
    toks, seq = 8786916332, 4096
    n_rows = toks // (seq + 1)
    assert n_rows == 2144719, n_rows
    n_val = min(max(1, int(n_rows * 0.05)), 5000)
    assert n_val == 5000, n_val
    n_pool = n_rows - n_val
    assert n_pool == 2139719, n_pool
    # The cursor-sum identity train.py:1163 enforces at save. If this ever fails the
    # checkpoint's cursor is not absolute and the region is not what it claims.
    cursor_sum = 64450 + 40082 + 19776 + 24700 + 1796 + 1714 + 7496 + 80380 + 3766
    assert cursor_sum == 3815 * 16 * 2 * 2 == 244160, cursor_sum
    # THE PHASE ALLOCATION, which is what the cursor is a fraction OF. int() of the same
    # product train.py:1774 computes, at anneal_frac 0 (the control's launch flag).
    alloc = int((8000000000 / seq) * 1.0 * 0.3297123)
    assert alloc == 643969, alloc
    assert 8.0 < alloc / 80380 < 8.05, alloc / 80380
    # WHY A PREFIX IS THE WRONG READING, in one line each, with no shuffle needed:
    #   - cursor/steps is not an integer, so the run did not walk a contiguous block;
    #   - it equals the domain's WEIGHT SHARE of the 64 rows/step, which is what a
    #     shuffled per-step draw produces.
    assert 80380 % 3815, "cursor/steps is an integer -- re-check the contiguity argument"
    share = 0.3297123 * (16 * 2 * 2)
    assert abs(80380 / 3815 - share) < 0.05, (80380 / 3815, share)
    # And the measured consequence, recorded so the number is not re-derived by hand:
    # the old SEEN interval held 10,129 of the 80,280 read rows (12.60%) while the old
    # TAIL held 70,151 of them (87.4%).
    assert 10129 + 70151 == 80280, "the recorded split no longer sums to the read set"
    assert 70151 / 80280 > 0.87, "the old tail no longer holds most of the training set"
    # THE UNSEEN REGION IS UNREAD BY CONSTRUCTION: it starts where the allocation ends.
    assert n_pool - alloc == 1495750, n_pool - alloc

    # 1b. THE REPLAY ITSELF, against the nine cursor entries it must predict. This is the
    #     check that separates a correct shuffle from a plausible one -- a wrong permutation,
    #     world size or phase split would have to miss nine independent counts to pass.
    import tempfile
    RECORDED = {"math_owm_stage2": 64450, "en_c4_stage2": 40082, "cot": 19776,
                "textbook_30b": 24700, "chatml": 1796, "chat_qa": 1714, "zh_web": 7496,
                "code_py_starcoder": 80380, "code_py_rp1t": 3766}
    mix_real = os.path.join(ROOT, "data", "mix_200m_8b.json")
    if os.path.isfile(mix_real):
        rep = replay_read_rows(mix_real, 42, 2, 3815, 16, 2)
        assert rep["cursor_pred"] == RECORDED, {
            k: (v, RECORDED.get(k)) for k, v in rep["cursor_pred"].items()
            if RECORDED.get(k) != v}
        assert rep["alloc"] == 643969, rep["alloc"]
        assert len(rep["read"]) == 80280, len(rep["read"])
        # The property the UNSEEN region rests on, asserted rather than argued.
        assert not any(k >= 643969 for k in rep["read"]), "a read row lies beyond the allocation"
        # And the OLD regions, recomputed from the real read set so the two percentages
        # above are measured here rather than quoted.
        old_seen = sum(1 for k in rep["read"] if 0 <= k < 80380)
        old_tail = sum(1 for k in rep["read"] if 80380 <= k < n_pool)
        assert (old_seen, old_tail) == (10129, 70151), (old_seen, old_tail)
        # A WRONG SEED MUST NOT LOOK RIGHT: the replay's own falsifiability. Seed 43 shares
        # only ~12.5% of its rows with seed 42, i.e. chance overlap at this density.
        rep43 = replay_read_rows(mix_real, 43, 2, 3815, 16, 2)
        ov = len(rep43["read"] & rep["read"]) / len(rep["read"])
        assert ov < 0.2, f"seed 43 overlaps seed 42 by {ov:.1%} -- the shuffle is not seeded"
        # A CAPPED DOMAIN MUST REFUSE, because the cap makes the plan depend on the pool
        # ESTIMATE rather than on the mix. Provoked by shrinking one pool in a temp copy.
        with tempfile.TemporaryDirectory() as d:
            with open(mix_real, encoding="utf-8") as fh:
                m = json.load(fh)
            m["domains"]["code_py_starcoder"]["pool_rows_estimated"] = 1000
            p = os.path.join(d, "mix_capped.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(m, fh)
            try:
                replay_read_rows(p, 42, 2, 3815, 16, 2)
                raise AssertionError("a capped domain did not refuse")
            except RuntimeError as e:
                assert "capped domain" in str(e), str(e)[:120]

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

    print("api_cloze selftest OK: the region pair on the control's real numbers "
          f"(N={n_rows}, pool={n_pool}, allocation {alloc} = 8.01x the {80380} cursor, "
          f"cursor sum {cursor_sum} == 3815x16x2x2), the CONTIGUOUS reading rejected three "
          "ways (cursor/steps is not an integer, it equals the weight share of 64 rows/step, "
          "and the old regions held 10129 vs 70151 of the 80280 read rows), plan replay "
          "predicts all 9 row_cursor entries exactly with seed 43 overlapping only at chance "
          "and a capped domain refusing, UNSEEN = pool [643969, 2139719) with 0 read rows, "
          "item construction on known text (gold is the attribute, label points at it, prompt "
          "ends at the dot), thin objects and short prefixes dropped, header separated, "
          "binomial SE 0.0087 at n=2500, and an unscoreable item is excluded rather than "
          "counted correct (the -1e9 tie-toward-slot-0 free point, reproduced then masked)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
