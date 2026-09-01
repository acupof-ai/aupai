#!/usr/bin/env python3
"""Arm 3: token loss on text that is certainly unseen, against val loss.

Pre-registered in docs/lessons/copy_hypothesis_prereg.md: if external >> val,
our val is measuring memorisation, because val is drawn from the same
never-deduplicated corpus.

THE UNSEEN TEXT, and why it qualifies (the pre-registration committed to naming
this in advance, because the arm is uninterpretable if the text is not unseen):

  this repository's own docs/ prose -- authored 2026-08-30..09-01 by this team,
  never published, never fetched, and postdating every snapshot in
  datagen/fetch_corpus.py's SOURCES.

That is the strongest unseen guarantee available without network access: not
"probably not in a web crawl" but "did not exist when the crawl was taken, and
exists in no crawl." Every candidate is still run through the corpus dedup
predicates and discarded on any match, as pre-registered.

THE CONFOUND, stated because it is what limits the arm rather than a caveat to
skim: our docs are technical prose in a register the corpus barely contains, so
external-minus-val mixes UNSEENNESS with DOMAIN SHIFT. A high gap is therefore
NOT sufficient evidence for memorisation on its own. The control below is what
separates them: the same model scored on corpus text it has certainly seen, held
at the same sequence length and packing. Without that control this arm should
report ABSENT rather than a number.

Usage: python probes/t70_external_loss.py --ckpt ckpt_... --out runs/external_loss.json
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("FLA_FLASH_KDA", "0")

SEQ = 1024


def unseen_texts(root, limit_chars=400_000):
    """docs/ prose authored by this team after every corpus snapshot."""
    out = []
    for sub in ("docs/lessons", "docs/audits", "docs/standards"):
        for p in sorted(glob.glob(os.path.join(root, sub, "*.md"))):
            t = open(p, encoding="utf-8").read()
            if len(t) > 400:
                out.append((os.path.relpath(p, root), t))
    tot, keep = 0, []
    for name, t in out:
        keep.append((name, t))
        tot += len(t)
        if tot >= limit_chars:
            break
    return keep


def corpus_texts(root, mix_path, n_docs=400):
    """Text the model has certainly SEEN -- the control that separates
    unseenness from domain shift.

    Reads the TRAINING MIX's domains, not data/corpus/*.jsonl. The first version
    of this probe globbed loose jsonl at the top of data/corpus/, which is
    scratch and staging -- none of it is a domain in mix_30b_stage2.json, so the
    "seen" control was not seen and the arm measured unseen-vs-unseen. The gap it
    produced was real arithmetic on the wrong contrast; see t58 for the same
    shape. Every domain here is named in the mix the checkpoint trained on.
    """
    mix = json.load(open(mix_path, encoding="utf-8"))
    per = max(1, n_docs // max(1, len(mix["domains"])))
    out = {}
    for name in mix["domains"]:
        d = os.path.join(root, "data", "corpus", name)
        files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        if not files:
            print(f"  seen-control: {name} has no shards under {d} -- SKIPPED", flush=True)
            continue
        got = []
        for line in open(files[0], encoding="utf-8"):
            if not line.strip():
                continue
            t = (json.loads(line).get("content") or "").strip()
            if len(t) > 400:
                got.append(t)
            if len(got) >= per:
                break
        out[name] = got
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "external_loss.json"))
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_30b_stage2.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    a = ap.parse_args()

    import torch

    sys.path.insert(0, os.path.join(ROOT, "datagen"))
    from eval.domain_loss import domain_loss
    from scripts.loader import load_checkpoint, load_tokenizer

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    model = model.to(torch.bfloat16)
    model.eval()
    tok = load_tokenizer(a.tokenizer, cfg)

    unseen = unseen_texts(ROOT)
    seen = corpus_texts(ROOT, a.mix)

    # Pre-registered gate: a candidate matching the corpus is DISCARDED, not
    # explained away. Exact-key match against the seen sample.
    import build_corpus as B

    seen_keys = {B.exact_key(t) for ts in seen.values() for t in ts}
    kept = [(n, t) for n, t in unseen if B.exact_key(t) not in seen_keys]
    dropped = len(unseen) - len(kept)

    res = {
        "probe": "t70_external_loss",
        "ckpt": os.path.basename(a.ckpt),
        "seq": SEQ,
        "unseen_source": "this repo's docs/ prose, authored 2026-08-30..09-01",
        "unseen_why_unseen": ("postdates every snapshot in fetch_corpus.SOURCES and was "
                              "never published; not 'probably uncrawled' but 'did not "
                              "exist when the crawls were taken'"),
        "unseen_docs": len(kept),
        "unseen_docs_dropped_by_dedup": dropped,
        "seen_docs": sum(len(v) for v in seen.values()),
        "seen_source": f"training-mix domains from {os.path.basename(a.mix)}",
    }

    # PER-DOMAIN, not one pooled call. domain_loss caps at SEQ_CAP=64 sequences,
    # so pooling seven domains into one list scores only the first ~65K tokens --
    # i.e. domain #1 alone, silently. The first run of this probe reported
    # seen_loss 0.7185, which is exactly code_rp1t's own loss: a truncation
    # wearing the costume of an average. Scoring each domain separately and
    # reporting the spread also makes the domain-shift confound visible instead
    # of hiding it inside a single pooled number.
    with torch.no_grad():
        u_loss, u_tok = domain_loss(model, tok, [t for _n, t in kept], SEQ, a.device)
        per_domain = {}
        for name, texts in seen.items():
            if not texts:
                continue
            dl, dt = domain_loss(model, tok, texts, SEQ, a.device)
            if dl is not None:
                per_domain[name] = {"loss": round(dl, 4), "tokens": dt}

    res["unseen_loss"] = round(u_loss, 4) if u_loss else None
    res["unseen_tokens"] = u_tok
    res["seen_per_domain"] = per_domain
    if per_domain:
        losses = [v["loss"] for v in per_domain.values()]
        res["seen_loss_mean"] = round(sum(losses) / len(losses), 4)
        res["seen_loss_min"] = min(losses)
        res["seen_loss_max"] = max(losses)
        res["seen_domain_spread"] = round(max(losses) - min(losses), 4)
        if u_loss:
            res["gap_unseen_minus_seen_mean"] = round(u_loss - res["seen_loss_mean"], 4)
            res["gap_vs_worst_seen_domain"] = round(u_loss - max(losses), 4)
    res["confound"] = ("the gap mixes UNSEENNESS with DOMAIN SHIFT -- docs/ prose is a "
                       "register the corpus barely contains. A large gap is not "
                       "sufficient evidence of memorisation by itself.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
