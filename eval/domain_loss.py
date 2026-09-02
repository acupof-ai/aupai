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
def domain_loss_seqs(model, rows, device, bs=4):
    """Mean next-token CE over rows train.py actually held out (from val_seqs).

    Separate from domain_loss because the input is different in kind: val_seqs returns
    sequences train.py already packed and shuffled, and re-deriving text from them to
    feed the packing path would reorder exactly what makes them val.
    """
    if rows is None or not len(rows):
        return None, 0
    x = rows[:, :-1].to(device)
    y = rows[:, 1:].to(device)
    return _ce(model, x, y, bs)


def _ce(model, x, y, bs):
    """Sum CE over (x, y) in batches; the loss and the token count it averages over."""
    tot = cnt = 0.0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x[i : i + bs])
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), y[i : i + bs].reshape(-1), reduction="sum"
        )
        tot += loss.item()
        cnt += y[i : i + bs].numel()
    return tot / cnt, int(cnt)


def selftest(model, tok, texts, seq, device):
    """Three known answers. A metric without one is not a metric -- four numbers in this
    repo were wrong for a day because nobody had a case where they must fail."""
    import random

    real, _ = domain_loss(model, tok, texts, seq, device)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--mix", default=os.path.join(ROOT, "data/mix_scale_3.24b.json"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/tokenizer.json"))
    ap.add_argument("--json", help="append one record per checkpoint here")
    ap.add_argument("--selftest", action="store_true", help="known answers; run before believing any number")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cache_guard import set_vocab_id

    if a.selftest:
        # Pre-flight known answers, before the mix and cache machinery: this path needs
        # only the checkpoint, so it runs on a fresh tree. _mix_for and the cache build
        # below used to run first, so --selftest paid for a full mix it never scored.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = load_checkpoint(a.ckpt[0], device=device, dtype=torch.bfloat16)
        set_vocab_id(cfg)
        tok = load_tokenizer(a.tokenizer, cfg)
        model.eval()
        ok = selftest(model, tok, _SELFT_PROBE, getattr(cfg, "seq", 4096), device)
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
            loss, ntok = domain_loss_seqs(model, rows, device) if rows is not None else (None, 0)
            if loss is None:
                print(f"  {name:10s} too few tokens to score -- SKIPPED", flush=True)
                continue
            row["domains"][name] = {"loss": round(loss, 4), "tokens": ntok,
                                    "head_fp": seqs_fp(rows), "split": "val"}
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
