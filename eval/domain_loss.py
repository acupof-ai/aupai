#!/usr/bin/env python3
"""Held-out loss for one checkpoint, reported per domain.

train.py reports a single combined figure, and on a mix that is 49.6% textbook that
figure is largely a textbook figure. Per domain it becomes two things at once: the
backbone of a score matrix that works on a base checkpoint, and the only way to price
a domain -- "is a CCI3 token worth a web_hq token" is this number and nothing else.

Runs on a checkpoint after the fact, so the six 0830v1 budget points can be filled in
without retraining. Scores with the vocabulary the checkpoint was trained on.

    python eval/domain_loss.py --ckpt ckpt_0830v1_3.24b.pt [--mix data/mix_scale_3.24b.json]
    python eval/domain_loss.py --ckpt A.pt --ckpt B.pt --json runs/domain_loss.json
"""

import argparse
import hashlib
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.loader import EOS_ID, load_checkpoint, load_tokenizer  # noqa: E402

# WRONG SINCE THE SHUFFLE, corrected 2026-09-01 (tilerl measured it, de fixed it).
#
# The reasoning below was right when it was written: train.py held out the HEAD of each
# domain, so the head of the first shard was unseen by construction. train.py now
# SHUFFLES the packed sequences (Random(Cfg.seed).shuffle) and then slices val off the
# shuffled order, so the head of the alphabetically-first raw shard has no relationship
# to val at all. Measured on cot: 0.625% of the docs this scored land in val, against
# 0.587% expected by pure chance -- indistinguishable. Every per-domain nat this has
# produced, in every readout, is TRAINING-SET loss.
#
# The old comment is kept above the fix because it explains why nobody caught it: the
# rationale is sound, internally consistent, and cites a real line. It stopped being
# true when a different file changed, and a comment cannot notice that.
#
# The fix reconstructs val the way train.py does -- through the same token cache, the
# same seeded shuffle, the same val_frac and val_rows_max -- which is what eval/ppl.py
# already did with zero callers. Sequences, not raw source lines: the shuffle operates
# on packed sequences, so there is no way back to "which lines were val".
#
# Original reasoning, retained:
# train.py:1187 holds out the HEAD of each domain -- seqs[:n_val] is validation, seqs[n_val:]
# is the training pool -- so the head is unseen for EVERY budget by construction. Scoring the
# tail instead reads the training pool, and reads more of it the larger the budget: at epoch cap
# 1 the 3.24B run consumes essentially the whole pool while the 0.2B run barely touches it. That
# hands larger budgets an easier test set and inflates exactly the deltas this script exists to
# report. Score the head.
HOLDOUT_ROWS = 4000  # source lines read from the first shard, packed then truncated to SEQ_CAP
SEQ_CAP = 64  # sequences per domain: 64 x 4096 = 262K tokens, enough for +-0.01 nat

# Probe for the --selftest pre-flight: real structured text, small enough to ship, long
# enough that shuffling it must score far worse. The metric is what is tested, not the
# cache path, so domain text is unnecessary.
_SELFT_PROBE = [
    "The sum of the first n positive integers is n(n+1)/2; the proof pairs the smallest "
    "with the largest, and each pair sums to n+1.",
    "A prime number has exactly two divisors: one and itself. Two is the only even prime, "
    "and every composite number factors into primes in exactly one way, up to ordering.",
    "The derivative of x squared is two x. At x equals three the tangent line has slope "
    "six, and the area under the curve from zero to three is nine.",
    "Training loss decreases when the model assigns more probability to the observed "
    "token; shuffling the tokens destroys the structure the probability is built from.",
    "A cache stamped with one vocabulary must not be read by a process scoring another: "
    "every id would be valid, in range, and wrong, and no error would fire.",
    "The mean of batch means is not the mean of the tokens, because batches carry "
    "different token counts; the accumulator must sum and divide once at the end.",
]


def val_seqs(domain, tok, cap=SEQ_CAP):
    """The rows train.py actually holds out for this domain, as packed sequences.

    Goes through train._domain_seqs, so it shares the token cache (and its vocab_id and
    .srcfp guards) and the seeded shuffle. Returns None when the domain has no shards --
    absent is not zero, and a domain that cannot be scored must be skipped rather than
    contribute a number.

    The freshness guard is HERE rather than in each caller, because _domain_seqs does not
    only read the cache -- it rebuilds and re-stamps one whose stamps disagree, and an
    eval has no business doing that to a cache a training run is reading (fb, 2026-09-02:
    ppl.py on card 7 was two minutes from retokenizing all nine of the live 20B run's
    domains). Two callers reach _domain_seqs through this function; a guard each of them
    had to remember is a guard one of them eventually will not.
    """
    import train

    from cache_guard import assert_caches_fresh

    assert_caches_fresh([domain])
    seqs = train._domain_seqs(domain, tok, True, False)
    seqs = seqs[0] if train.Cfg.fone else seqs
    if seqs is None or not len(seqs):
        return None
    n_val = min(max(1, int(len(seqs) * train.Cfg.val_frac)), train.Cfg.val_rows_max)
    return seqs[:n_val][:cap].long()


def domain_files(mix_path, root):
    mix = json.load(open(mix_path, encoding="utf-8"))
    out = {}
    for name in mix["domains"]:
        d = os.path.join(root, "data", "corpus", name)
        files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []
        if not files:
            print(f"  {name}: no shards under {d} -- SKIPPED, not scored as zero", flush=True)
            continue
        out[name] = os.path.join(d, files[0])  # the head, matching train.py's val split
    return out


def head_texts(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(line)
            if len(rows) >= n:
                break
    texts = []
    for line in rows:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("content") or d.get("text") or ""
        if t:
            texts.append(t)
    return texts


def seqs_fp(rows):
    """sha1 of the exact val SEQUENCES a score was computed over, first 16 hex.

    Same contract as head_fp and the same refusal on the readout side; the input is
    token ids rather than text because val_seqs returns rows train.py already packed.
    Hashing ids is if anything stricter -- two texts that tokenize identically are the
    same input to the model, and a vocabulary change moves the hash, which is correct:
    a score taken under a different vocabulary is not comparable either.
    """
    import hashlib

    return hashlib.sha1(rows.cpu().numpy().tobytes()).hexdigest()[:16]


def head_fp(texts):
    """sha1 of the exact text this score was computed over, first 16 hex.

    Not the corpus directory's fingerprint and not the domain NAME: the readout's
    head guard compared names, so two roles scored on different bytes under the
    same name would have differenced silently, and the guard only ever worked
    because stage 2 happened to RENAME the domains it rebuilt (b0, 2026-09-01).
    Hashing the scored text closes both directions -- same name different bytes
    refuses, and a rename over identical bytes becomes readable.

    It is also for the human reader. A person differencing two rows out of
    score_matrix.jsonl by hand had nothing in the record telling them the rows
    were incomparable; they had to go find a separate stats file. That is how
    this defect was found, by making exactly that mistake."""
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@torch.no_grad()
def domain_loss(model, tok, texts, seq, device, cap=SEQ_CAP, bs=4):
    """Mean next-token CE over packed held-out text. Packing matches training, so the
    number is on the same scale as the val figure train.py prints."""
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids + [EOS_ID])
    n = (len(ids) - 1) // seq
    if n == 0:
        return None, 0
    n = min(n, cap)
    x = torch.tensor(ids[: n * seq], dtype=torch.long).view(n, seq).to(device)
    y = torch.tensor(ids[1 : n * seq + 1], dtype=torch.long).view(n, seq).to(device)
    return _ce(model, x, y, bs)


# The pair above has one and this did not, and this is the one score_matrix calls
# (score_matrix.py:147). Scoring lrprobe_0.85 then built a backward graph it never used
# and asked for 94.65 GiB on a card someone else was holding -- read at the time as the
# training run exhausting memory, which it was not (fb, 2026-09-02).
@torch.no_grad()
def domain_loss_seqs(model, rows, device, bs=4, per_row=False):
    """Mean next-token CE over rows train.py actually held out (from val_seqs).

    Separate from domain_loss because the input is different in kind: val_seqs returns
    sequences train.py already packed and shuffled, and re-deriving text from them to
    feed the packing path would reorder exactly what makes them val.

    per_row=True adds a third return value: one (ce_sum, n_tokens) per row, in input order.
    b0-23 pairs on those rows -- the BLOCK is the pairing unit, not the domain.
    """
    if rows is None or not len(rows):
        return (None, 0, []) if per_row else (None, 0)
    x = rows[:, :-1].to(device)
    y = rows[:, 1:].to(device)
    return _ce(model, x, y, bs, per_row=per_row)


def _ce(model, x, y, bs, per_row=False):
    """Sum CE over (x, y) in batches; the loss and the token count it averages over.

    per_row=True ALSO returns [(row_ce_sum, row_tokens), ...] in input order, one entry per
    sequence. That list is what a per-block paired statistic needs and what the scalar return
    destroys: reduction="sum" over the whole batch adds every row's CE together, so two runs
    scored on the same 512 sequences yield one number each and n=1, while the pairing that is
    actually available is n=512. Per-DOMAIN pairing (paired_stats) has n=9 and its sd measures
    cross-domain consistency; neither is the sampling error of the delta.

    The rows are the pairing unit, so they must be comparable across runs: same sequences in
    the same order. val_seqs is deterministic given a vocabulary and a seed, and head_fp
    already asserts that -- per_row does not weaken it, it depends on it.
    """
    tot = cnt = 0.0
    rows = [] if per_row else None
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x[i : i + bs])
        if isinstance(logits, tuple):
            logits = logits[0]
        yb = y[i : i + bs]
        flat = logits.float().view(-1, logits.shape[-1])
        # THE REPORTED SCALAR IS STILL THE FLAT reduction="sum", BIT FOR BIT. Summing per row
        # and then adding the rows gives a different fp32 accumulation order: measured 7e-8
        # relative on a 6x9 fixture. That is far below the round(4) the record stores, but the
        # scalar here is what eff.l9_branch_split_p200m's numbers were taken with, and a
        # measurement path must not shift because a NEW output was added beside it. So the
        # per-row pass is an addition, not a replacement.
        tot += float(torch.nn.functional.cross_entropy(flat, yb.reshape(-1), reduction="sum"))
        if per_row:
            # reduction="none" then sum along the token axis: the per-row sum and the per-row
            # token count, so a caller can form either a per-row mean or a token-weighted one.
            # Both are needed -- they are different quantities on unequal-length rows, which is
            # exactly the fixture case that tells a correct implementation from a plausible one.
            per = torch.nn.functional.cross_entropy(
                flat, yb.reshape(-1), reduction="none").view(yb.shape).sum(-1)
            ntok = yb.shape[-1]
            rows += [(float(v), int(ntok)) for v in per]
        cnt += yb.numel()
    if per_row:
        return tot / cnt, int(cnt), rows
    return tot / cnt, int(cnt)


def selftest(model, tok, texts, seq, device):
    """Three known answers. A metric without one is not a metric -- four numbers in this
    repo were wrong for a day because nobody had a case where they must fail."""
    import random

    real, _ = domain_loss(model, tok, texts, seq, device)
    if real is None:
        # A None here means the probe packed to 0 rows; name it, do not let it reach
        # the format string below (an empty float is a crash, not a verdict).
        raise SystemExit(f"probe packs to 0 rows at seq {seq} -- nothing to score")

    # 1. SHUFFLED text must score far worse. Same tokens, same length, no structure:
    #    a scorer that reads its input at all cannot be indifferent to this.
    ids = [i for t in texts for i in tok.encode(t).ids]
    random.Random(0).shuffle(ids)
    shuf = ["".join(tok.decode([i]) for i in ids[: len(ids)])]
    bad, _ = domain_loss(model, tok, shuf, seq, device)

    # 2. The accumulator: THE SAME sequences regrouped into different batches must give the
    #    same per-token mean. A sum divided by a token count is invariant; a mean of batch
    #    means is not, and that is the bug this catches.
    #    Two earlier versions of this assertion were vacuous. `texts * 10` is truncated by the
    #    cap back to the same sequences. `texts * 2` with a doubled cap reads FURTHER INTO THE
    #    SAME STREAM, so it compares two different samples and fails on sampling noise -- a
    #    check that fires on correct code is as useless as one that never fires.
    a, _ = domain_loss(model, tok, texts, seq, device, cap=16, bs=4)
    b, _ = domain_loss(model, tok, texts, seq, device, cap=16, bs=1)

    ok = True
    print(f"  selftest real {real:.4f} | shuffled {bad:.4f} | bs4 {a:.4f} | bs1 {b:.4f}")
    if bad - real < 1.0:
        print(f"  FAIL shuffled text scores {bad - real:+.3f} vs real; must be much worse")
        ok = False
    if abs(b - a) > 1e-3:
        print(
            f"  FAIL regrouping the same sequences moved the mean by {b - a:+.5f}; the "
            "accumulator is batch-dependent"
        )
        ok = False
    print("  selftest " + ("OK" if ok else "FAILED"))
    return ok


def paired_stats(rows_by_ckpt, a_name, b_name, c_name=None):
    """Per-domain paired differences between checkpoints scored on IDENTICAL val rows.

    b0-18. The unweighted mean is compared against 0.24 nat (ds.seed_variance_0p2b), which is the
    seed-to-seed spread of a WHOLE RUN. A perturbation of one checkpoint measured against itself
    has no seed in it, so that bar is the wrong noise model for it: b0-16's rescale moved 0.0487
    nat and read "bounded" against a bar built for a quantity it does not contain.

    Pairing is legitimate here only because every checkpoint is scored on the same rows. That is
    ASSERTED, not assumed -- each row carries head_fp (sha1 of the exact val sequences), and a
    mismatch means the two scores are over different inputs, in which case a per-domain difference
    is not a paired difference and the sd below is meaningless. val_seqs is deterministic given a
    vocabulary and a seed, so a mismatch is a real event (a rebuilt cache, a vocab change), not a
    theoretical one.

    WITH A CONTROL ARM (c_name), THE STATISTIC IS (B-A)-(C-A), NOT B-A. b0-16 measured B-A as 9/9
    positive and nearly called that a result -- but C-A is also 9/9 positive, because ANY rescale
    of trained weights hurts. B-A > 0 therefore does not isolate the tensor under test; the
    control's own damage has to be subtracted. Without c_name this returns B-A and says so, which
    is correct for an A/B of two independently trained arms (no shared perturbation to subtract).

    Returns a dict with per-domain differences, mean, sd, t, sign counts and a one-sided sign-test
    p. t uses the paired sd over n domains: this is a WITHIN-run statistic about consistency across
    domains, NOT a claim about reseeding, and the caller must not read it as one.
    """
    import math

    names = [n for n in (a_name, b_name, c_name) if n]
    missing = [n for n in names if n not in rows_by_ckpt]
    if missing:
        raise KeyError(f"paired_stats needs rows for {missing}; got {sorted(rows_by_ckpt)}")
    doms = [sorted(rows_by_ckpt[n]["domains"]) for n in names]
    if len({tuple(d) for d in doms}) != 1:
        raise ValueError(
            "the checkpoints do not share a domain set: "
            + "; ".join(f"{n} has {len(d)}" for n, d in zip(names, doms, strict=True))
            + ". A paired difference needs the same domains on both sides, and taking the "
              "intersection silently would change which quantity the mean is over.")
    dom_list = doms[0]

    fp_mismatch = []
    for dom in dom_list:
        fps = {n: rows_by_ckpt[n]["domains"][dom].get("head_fp") for n in names}
        if len(set(fps.values())) != 1 or None in fps.values():
            fp_mismatch.append((dom, fps))
    if fp_mismatch:
        detail = "; ".join(f"{d}: " + ", ".join(f"{n}={v}" for n, v in fps.items())
                           for d, fps in fp_mismatch[:3])
        raise ValueError(
            f"{len(fp_mismatch)} domain(s) were scored on DIFFERENT val rows, so these are not "
            f"paired measurements and the paired sd would be nonsense: {detail}. head_fp is the "
            f"sha1 of the exact sequences each score ran over; val_seqs is deterministic, so a "
            f"mismatch means the token cache was rebuilt or the vocabulary moved between scores.")

    def loss(n, dom):
        return rows_by_ckpt[n]["domains"][dom]["loss"]

    if c_name:
        diffs = {d: (loss(b_name, d) - loss(a_name, d)) - (loss(c_name, d) - loss(a_name, d))
                 for d in dom_list}
        stat = f"(B-A)-(C-A) with B={b_name}, C={c_name}, A={a_name}"
    else:
        diffs = {d: loss(b_name, d) - loss(a_name, d) for d in dom_list}
        stat = f"B-A with B={b_name}, A={a_name}"

    vals = [diffs[d] for d in dom_list]
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    t = mean / (sd / math.sqrt(n)) if sd else None
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    # One-sided sign test at p=0.5 per domain, on the majority direction. Reported alongside t
    # because it assumes only independence and a direction, not normality of nine points.
    k = max(pos, neg)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"statistic": stat, "domains": dom_list, "diffs": diffs, "n": n,
            "mean": mean, "sd": sd, "t": t, "positive": pos, "negative": neg,
            "sign_test_p_one_sided": p}


def print_paired(ps, bar=0.24):
    print(f"\n=== paired per-domain differences: {ps['statistic']} ===")
    for d in ps["domains"]:
        print(f"  {d:22s} {ps['diffs'][d]:+9.4f}")
    t = f"{ps['t']:+.2f}" if ps["t"] is not None else "n/a (zero spread)"
    print(f"  {'mean':22s} {ps['mean']:+9.4f}   sd {ps['sd']:.4f}  t {t}")
    print(f"  same sign: {ps['positive']}/{ps['n']} positive, {ps['negative']}/{ps['n']} negative;"
          f" sign-test one-sided p {ps['sign_test_p_one_sided']:.4f}")
    print(f"  against the {bar} nat bar (ds.seed_variance_0p2b): "
          f"{100 * abs(ps['mean']) / bar:.0f}% of it. THAT BAR IS SEED SPREAD ACROSS WHOLE RUNS; "
          f"a perturbation of one checkpoint against itself contains no seed, so the bar is the "
          f"wrong noise model for it and the paired sd above is the right one.")
    if ps["mean"] == 0.0 and ps["sd"] == 0.0:
        print("  ALL DIFFERENCES EXACTLY ZERO. What that proves depends on WHERE the two sides "
              "came from. Two SEPARATE scoring runs reading 0.0 is evidence the eval is "
              "deterministic. The same row compared with itself (--arms X X on one artifact) is "
              "an IDENTITY -- it tests the arithmetic and the pairing plumbing, and it cannot "
              "detect nondeterminism at all, because there is only one measurement in it. Do not "
              "quote the second as 'the eval is deterministic'.")


def _paired_selftest():
    """Known answers for --paired, with NO card and NO model.

    Runs against real b0-16 numbers rather than round fixtures, because the point of this mode is
    that a new statistic's first run on new data reads as correct whatever it prints. These nine
    values have an answer computed independently by e1 during b0-16's review, which makes them the
    only input that can catch an implementation that is merely plausible.

    Every check below was verified red on its own broken version.
    """
    fails = []
    # The nine per-domain (B-A)-(C-A) values from runs/b0_16_l9_rescale.json, and their known
    # statistics: mean +0.0422, sd 0.0225, t 5.62, 9/9 same sign, sign-test p 0.0020.
    A = {"chat_qa": 3.0, "chatml": 3.0, "code_py_rp1t": 3.0, "code_py_starcoder": 3.0,
         "cot": 3.0, "en_c4_stage2": 3.0, "math_owm_stage2": 3.0, "textbook_30b": 3.0,
         "zh_web": 3.0}
    BA = {"chat_qa": 0.0838, "chatml": 0.0703, "code_py_rp1t": 0.0247,
          "code_py_starcoder": 0.0261, "cot": 0.0266, "en_c4_stage2": 0.0469,
          "math_owm_stage2": 0.0351, "textbook_30b": 0.0512, "zh_web": 0.0730}
    CA = {"chat_qa": 0.0055, "chatml": 0.0041, "code_py_rp1t": 0.0067,
          "code_py_starcoder": 0.0061, "cot": 0.0029, "en_c4_stage2": 0.0059,
          "math_owm_stage2": 0.0075, "textbook_30b": 0.0118, "zh_web": 0.0077}

    def mk(base, delta, fp="same"):
        return {"domains": {d: {"loss": round(base[d] + delta.get(d, 0.0), 4),
                                "head_fp": fp if fp != "per" else f"fp_{d}"}
                            for d in base}}

    rows = {"A": mk(A, {}), "B": mk(A, BA), "C": mk(A, CA)}

    # 1. THE CONTROL MUST BE SUBTRACTED. (B-A)-(C-A) is the statistic; B-A alone was b0-16's
    #    near-miss, since C-A is also 9/9 positive (any rescale of trained weights hurts).
    ps = paired_stats(rows, "A", "B", "C")
    for lbl, got, want, tol in (("mean", ps["mean"], 0.0422, 5e-4),
                                ("sd", ps["sd"], 0.0225, 5e-4),
                                ("t", ps["t"], 5.62, 0.02),
                                ("p", ps["sign_test_p_one_sided"], 1 / 512, 1e-9)):
        if got is None or abs(got - want) > tol:
            fails.append(f"(B-A)-(C-A) {lbl} is {got}, and b0-16's independently computed answer "
                         f"is {want} -- this artifact is the only input with a known answer, so a "
                         f"mismatch here is the implementation, not the data")
    if ps["positive"] != 9:
        fails.append(f"(B-A)-(C-A) is {ps['positive']}/9 positive, expected 9/9")

    # 2. WITHOUT A CONTROL IT MUST BE B-A, and must differ. If the two modes agreed, the control
    #    would be doing nothing and b0-16's whole correction would be undone silently.
    ps2 = paired_stats(rows, "A", "B")
    if abs(ps2["mean"] - 0.0486) > 5e-4:
        fails.append(f"B-A mean is {ps2['mean']:.4f}, expected 0.0486 (the fact's unpaired figure)")
    if abs(ps2["mean"] - ps["mean"]) < 1e-6:
        fails.append("B-A and (B-A)-(C-A) returned the SAME mean, so the control arm is being "
                     "ignored -- the exact defect b0-16's review corrected")

    # 3. A-vs-A IS AN IDENTITY, and must read exactly zero. Not proof of determinism (one
    #    measurement cannot disagree with itself), but a nonzero here is broken arithmetic.
    ps3 = paired_stats(rows, "A", "A")
    if ps3["mean"] != 0.0 or ps3["sd"] != 0.0:
        fails.append(f"A-vs-A read mean {ps3['mean']}, sd {ps3['sd']}; must be exactly 0.0")

    # 4. MISMATCHED val ROWS MUST REFUSE, by ValueError naming head_fp. Without this the paired sd
    #    is computed over scores taken on different inputs, which is not a paired difference at
    #    all -- and nothing about the output would look wrong.
    bad = {"A": rows["A"], "B": mk(A, BA, fp="per")}
    try:
        paired_stats(bad, "A", "B")
        fails.append("differing head_fp was accepted; the two sides were scored on DIFFERENT val "
                     "rows and the paired sd is then meaningless, with no visible symptom")
    except ValueError as e:
        if "head_fp" not in str(e):
            fails.append(f"head_fp mismatch raised ValueError without naming head_fp: {e!r}")
    except Exception as e:                                       # noqa: BLE001
        fails.append(f"head_fp mismatch raised {type(e).__name__}, not ValueError: {e!r} -- a "
                     f"crash reports 'the tool is broken' where the truth is 'these scores are "
                     f"not comparable'")

    # 5. A DIFFERENT DOMAIN SET MUST REFUSE WITH ValueError, not crash downstream. Taking the
    #    intersection would change which domains the mean is over while still printing a mean.
    #    DELETING the explicit check does NOT make this red on its own: the head_fp loop below it
    #    hits the missing domain and raises KeyError, which an `except ValueError` alone would
    #    read as the guard working. So the type is asserted -- a KeyError says "the tool broke",
    #    the ValueError says "these two checkpoints are not comparable", and only the second tells
    #    the operator what to do. Third time this shape has appeared today (l9_branch_probe's
    #    control check, head_path_rows' vocab ordering, and here).
    short = {"domains": {d: v for d, v in rows["B"]["domains"].items() if d != "zh_web"}}
    try:
        paired_stats({"A": rows["A"], "B": short}, "A", "B")
        fails.append("a checkpoint missing a domain was accepted; the mean would silently be over "
                     "8 domains while the other side has 9")
    except ValueError as e:
        if "domain set" not in str(e):
            fails.append(f"domain-set mismatch raised the wrong ValueError: {e!r}")
    except Exception as e:                                       # noqa: BLE001
        fails.append(f"domain-set mismatch raised {type(e).__name__}, not ValueError: {e!r}. The "
                     f"explicit domain-set check is gone or unreachable -- the head_fp loop is "
                     f"crashing on the missing domain instead, which reports a broken tool where "
                     f"the truth is two checkpoints that cannot be paired")

    # 6. THE SIGN TEST MUST BE ONE-SIDED ON THE MAJORITY: 9/9 gives C(9,9)/2^9 = 1/512.
    #    My first version of this check asserted 2/512, which is the TWO-sided figure -- and the
    #    code was right while the check was wrong, so it went red on a correct implementation. The
    #    fact quotes 0.0020, which is 1/512 = 0.001953 rounded, and that is what pins it.
    want_p = 1 / 512
    if abs(ps["sign_test_p_one_sided"] - want_p) > 1e-9:
        fails.append(f"sign-test p for 9/9 is {ps['sign_test_p_one_sided']}, expected {want_p} = "
                     f"C(9,9)/2^9 one-sided (b0-16 quotes 0.0020, which is this rounded). 2/512 "
                     f"would be the two-sided value.")
    # And a split must NOT read as significant, or the test is not testing.
    mixed = mk(A, {d: (0.05 if i % 2 else -0.05) for i, d in enumerate(A)})
    ps4 = paired_stats({"A": rows["A"], "B": mixed}, "A", "B")
    if ps4["sign_test_p_one_sided"] < 0.05:
        fails.append(f"a 5/4 sign split gave p {ps4['sign_test_p_one_sided']:.4f} < 0.05; the sign "
                     f"test is not discriminating")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("domain_loss --paired selftest OK: (B-A)-(C-A) reproduces b0-16's independently computed "
          "answer to the digit (mean +0.0422, sd 0.0225, t 5.62, 9/9, p 0.0020 = 1/512 exactly), which is "
          "the only input with an answer known by other means -- a new statistic's first run on "
          "new data reads as correct whatever it prints. B-A without a control gives 0.0486 (the "
          "fact's unpaired figure) and must DIFFER from the controlled statistic, or the control "
          "arm is being ignored and b0-16's correction is silently undone. A-vs-A is exactly 0.0, "
          "and is labelled an identity rather than proof of determinism. Mismatched head_fp and "
          "mismatched domain sets both REFUSE with ValueError naming the cause, because a paired "
          "sd over scores taken on different inputs has no visible symptom. And the sign test is "
          "checked in both directions: 9/9 gives 1/512, a 5/4 split does not read as significant (my first version of that check asserted 2/512, the TWO-sided value, and went red on correct code).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    # NOT required=True. --paired_selftest and --paired_from need no checkpoint at all -- they
    # read rows somebody already scored -- and argparse would reject them before main() could say
    # so. The requirement is enforced below, per mode, where it is actually true.
    ap.add_argument("--ckpt", action="append")
    ap.add_argument("--mix", default=os.path.join(ROOT, "data/mix_scale_3.24b.json"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/tokenizer.json"))
    ap.add_argument("--json", help="append one record per checkpoint here")
    ap.add_argument("--selftest", action="store_true", help="known answers; run before believing any number")
    ap.add_argument("--paired_selftest", action="store_true",
                    help="b0-18's known answers for --paired. NO CARD, NO MODEL, NO MIX -- it "
                         "checks the statistic against b0-16's independently computed figures, so "
                         "it runs on a fresh tree while every card is busy (unlike --selftest, "
                         "which needs a card for the KDA kernel).")
    ap.add_argument("--paired", action="store_true",
                    help="b0-18: report per-domain PAIRED differences between the scored "
                         "checkpoints, with the paired sd. Needs >=2 --ckpt, or --paired_from to "
                         "read rows already scored. Asserts every checkpoint saw identical val "
                         "rows (head_fp) rather than assuming it.")
    ap.add_argument("--paired_from",
                    help="JSONL of rows already scored (this script's or l9_branch_probe's --out) "
                         "to compute paired differences from WITHOUT a card. The b0-16 rescale "
                         "artifact is the one input with an answer known by other means, which is "
                         "why it is the first test of this mode: a new statistic's first run on "
                         "new data reads as correct whatever it prints.")
    ap.add_argument("--arms", nargs="+",
                    help="which ckpt names from --paired_from to use, as A B [C]. C is a CONTROL "
                         "arm: with it the statistic becomes (B-A)-(C-A), because B-A alone does "
                         "not isolate B when any perturbation of trained weights hurts.")
    a = ap.parse_args()

    if a.paired_selftest:
        sys.exit(_paired_selftest())

    if not a.ckpt and not a.paired_from:
        ap.error('--ckpt is required unless --paired_selftest or --paired_from is used')

    if a.paired_from:
        # NO CARD, NO MODEL. Reads rows somebody already scored, so the known-answer test costs
        # nothing and can run while a lane job holds every card.
        rows_by_ckpt = {}
        with open(a.paired_from, encoding="utf-8") as fh:
            for ln in fh:                      # JSONL -- these files carry a .json name and are
                ln = ln.strip()                # NOT json.load-able (b0_16_l9.json, b0_16_weights.json)
                if not ln:
                    continue
                r = json.loads(ln)
                if "domains" in r:
                    rows_by_ckpt[r.get("ckpt", f"row{len(rows_by_ckpt)}")] = r
        if not rows_by_ckpt:
            sys.exit(f"{a.paired_from} has no rows with a 'domains' key -- wrong file, or it is "
                     f"one of the weights-only artifacts, which carry no per-domain losses")
        if not a.arms or not (2 <= len(a.arms) <= 3):
            sys.exit(f"--paired_from needs --arms A B [C]; available: {sorted(rows_by_ckpt)}")
        ps = paired_stats(rows_by_ckpt, *a.arms)
        print_paired(ps)
        if a.json:
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps(ps, ensure_ascii=False) + "\n")
            print(f"appended paired record to {a.json}")
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cache_guard import set_vocab_id

    if a.selftest:
        # Pre-flight known answers, before the mix and cache machinery: this path needs
        # only the checkpoint, so it runs on a fresh tree -- but it DOES need one card:
        # the KDA kernel has no CPU path (triton driver error with no visible device).
        # _mix_for and the cache build below used to run first, so --selftest paid for
        # a full mix it never scored.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = load_checkpoint(a.ckpt[0], device=device, dtype=torch.bfloat16)
        set_vocab_id(cfg)
        tok = load_tokenizer(a.tokenizer, cfg)
        model.eval()
        # 256, not cfg.seq: the probe is ~400 tokens, so at the training seq (4096) it
        # packs to ZERO rows and domain_loss returns (None, 0) -- the pre-flight would
        # crash in the format string. Repeated x4 so the bs4/bs1 check has multiple rows.
        # The metric is what is tested, not the training-length path.
        ok = selftest(model, tok, _SELFT_PROBE * 4, 256, device)
        sys.exit(0 if ok else "selftest failed -- the metric is not measuring")

    # The mix is a property of the CHECKPOINT, and this CLI defaulted to the ladder's -- so
    # scoring a non-ladder checkpoint read domain rows for domains it never trained on, and
    # cache_guard refused, correctly, since the seqs fingerprint belongs to another corpus. Fixed
    # in score_matrix.py first (3415e9e); this is the same defect in the standalone entry point,
    # found by 44 reviewing that commit.
    #
    # IMPORTED, not reimplemented. _mix_for carries five known answers built on real torch.save
    # files, and its subtlety is that "--mix was named" cannot be read from a.mix (argparse fills
    # the default either way) -- only from sys.argv. A second copy here would be a second thing
    # to keep correct, and the copy that drifts is the one nobody runs the selftest for.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_matrix import _mix_for

    mix_given = any(x == "--mix" or x.startswith("--mix=") for x in sys.argv[1:])
    mix_path = _mix_for(a.ckpt[0], a.mix, explicit=mix_given)
    # One mix for the batch, from the FIRST checkpoint, and it is named out loud: --ckpt takes
    # several, the domain caches are read once for all of them, and two checkpoints from
    # different mixes cannot share one table of per-domain rows. Scoring them together against
    # the first one's mix silently is what this whole fix is about.
    if len(a.ckpt) > 1 and not mix_given:
        print(f"note: {len(a.ckpt)} checkpoints, all scored against "
              f"{os.path.basename(mix_path)} (from {os.path.basename(a.ckpt[0])}); pass --mix to "
              f"override", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = domain_files(mix_path, ROOT)
    assert files, f"{mix_path} named domains but none have shards -- nothing to score"
    cache = {name: head_texts(p, HOLDOUT_ROWS) for name, p in files.items()}

    out = []
    for ck_path in a.ckpt:
        # bf16: the MLA path goes through FlashAttention, which refuses fp32 outright.
        model, cfg = load_checkpoint(ck_path, device=device, dtype=torch.bfloat16)
        # train.VOCAB_ID for cache_guard, which val_seqs hits per domain below. Without
        # this the module global stays None, every cache stamp reads as a mismatch with
        # an empty right side, and the error says "cache dirty" when the process simply
        # has no fingerprint (score_matrix.py:1086; fb 2026-09-02, caught on ppl.py).
        set_vocab_id(cfg)
        # load_tokenizer cross-checks size and vocab_id against this cfg and raises on a
        # mismatch. Scoring with the wrong vocabulary is silent noise, so it is checked
        # per checkpoint rather than once for the batch.
        tok = load_tokenizer(a.tokenizer, cfg)
        model.eval()
        seq = getattr(cfg, "seq", 4096)  # cfg is a SimpleNamespace, not a dict
        row = {"ckpt": os.path.basename(ck_path), "domains": {}}
        print(f"\n{os.path.basename(ck_path)}  (vocab {getattr(cfg, 'vocab', '?')}, seq {seq})", flush=True)
        for name in cache:
            # val, not the shard head: the head stopped being val when train.py started
            # shuffling before slicing (0.625% overlap against 0.587% by chance). Both
            # this CLI and score_matrix's metric go through val_seqs, or the two would
            # disagree while reporting the same metric name.
            rows = val_seqs(name, tok)
            if rows is None:
                loss, ntok, per = None, 0, []
            else:
                loss, ntok, per = domain_loss_seqs(model, rows, device, per_row=True)
            if loss is None:
                print(f"  {name:10s} too few tokens to score -- SKIPPED", flush=True)
                continue
            row["domains"][name] = {"loss": round(loss, 4), "tokens": ntok,
                                    "head_fp": seqs_fp(rows), "split": "val"}
            # b0-23 pairs on BLOCKS, so the per-row numbers are recorded unconditionally
            # rather than behind a flag. A flag would mean the record that N2 needs depends
            # on someone having remembered it at scoring time, and a re-score costs a card.
            # Full precision, not round(4): these are summed over hundreds of blocks before
            # anything is reported, and rounding each one first puts noise into the SE that
            # the measurement does not have.
            row["domains"][name]["blocks"] = [{"ce_sum": ce, "n_tokens": nt} for ce, nt in per]
            print(f"  {name:10s} {loss:.4f}   ({ntok:,} tok)", flush=True)
        vals = [d["loss"] for d in row["domains"].values()]
        row["unweighted_mean"] = round(sum(vals) / len(vals), 4)
        # NOT the mix-weighted figure train.py prints: an unweighted mean asks "how good
        # across domains", the weighted one asks "how good on this mix". Reporting the
        # weighted one alone is what let a 49.6%-textbook mix read as a model result.
        print(f"  {'MEAN':10s} {row['unweighted_mean']:.4f}   (unweighted across domains)", flush=True)
        out.append(row)
        del model
        torch.cuda.empty_cache()

    if a.json:
        with open(a.json, "a", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nappended {len(out)} record(s) to {a.json}")


if __name__ == "__main__":
    main()
