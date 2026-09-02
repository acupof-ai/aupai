#!/usr/bin/env python3
# restartable: one JSON line per domain is appended to --preds as it is scored, and a rerun with
# the same --preds skips domains already there. An interrupt costs the current domain only.
"""Per-domain BPB on the SAME held-out bytes, for a model with any tokenizer.

    python3 eval/domain_bpb.py --ckpt <ckpt.pt> --mix data/mix_200m_4b.json
    python3 eval/domain_bpb.py --ckpt <hf-dir> --hf --mix data/mix_200m_4b.json
    python3 eval/domain_bpb.py --selftest

WHY THIS EXISTS AND domain_loss.py DOES NOT COVER IT. domain_loss.py reports nats per TOKEN over
rows that come out of the token cache as OUR ids (val_seqs -> train._domain_seqs). Both halves
block a control comparison: the control model cannot consume our ids, and a per-token figure is
not comparable across tokenizers anyway -- the same text is a different token count per side, so
whichever tokenizer packs the corpus more tightly wins for a reason unrelated to modelling.

This script fixes both by moving the unit to BYTES and the interface to TEXT: our held-out ids
are DECODED back to text, and each arm re-encodes that text with its own tokenizer. The bytes
scored are then identical on both sides by construction, and bits/byte is comparable.

WHAT THE DECODE COSTS, stated because it is a real boundary and not a rounding detail. Decoding
ids -> text is not guaranteed byte-exact against the original corpus: a lossy merge or a
normalisation step would show up as a difference. So the script MEASURES the round-trip instead
of assuming it: re-encoding the decoded text must reproduce the ids it came from, and the
fraction that does is reported as `roundtrip_exact`. Below a threshold it refuses rather than
publishing a number over bytes that are not the held-out bytes.

domain_loss.py IS NOT MODIFIED. It is a frozen path, its nats/token figure is what every existing
record and threshold is denominated in, and changing its unit would silently redefine a metric
whose history is in runs/score_matrix.jsonl. This is a second reading, not a replacement.
"""

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

os.environ.setdefault("FLA_FLASH_KDA", "0")

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
# A round trip below this is a refusal. Not 1.0: a single unlucky row must not void a domain,
# and the exact figure is reported either way so a reader never has to trust the threshold.
MIN_ROUNDTRIP = 0.98


class OursModel:
    def __init__(self, ckpt, tok_path, device):
        from scripts.loader import load_checkpoint, load_tokenizer  # noqa: PLC0415

        self.model, self.cfg = load_checkpoint(ckpt, device=device, dtype=torch.bfloat16)
        self.tok = load_tokenizer(tok_path, self.cfg)
        self.device = device
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def logprobs(self, ids):
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.model(x)
        lg = (out[0] if isinstance(out, tuple) else out)[0].float()
        return torch.log_softmax(lg, -1)


class HFModel:
    def __init__(self, path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
        self.model.eval()
        self.device = device
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def encode(self, s):
        return self.tok.encode(s)

    def logprobs(self, ids):
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            return torch.log_softmax(self.model(x).logits[0].float(), -1)


def text_bpb(m, text, max_ctx=2048):
    """Bits per UTF-8 byte of `text`, every token after the first contributing loss.

    The FIRST token carries no loss and its bytes are excluded from the divisor: nothing
    predicts it, so charging its bytes to the model would make the figure depend on how many
    chunks the text was split into. Returns (bits, bytes_scored) or (None, reason).
    """
    ids = m.encode(text)
    if len(ids) < 2:
        return None, f"text encodes to {len(ids)} token(s); at least 2 are needed to score one"
    ids = ids[:max_ctx]
    lp = m.logprobs(ids)
    total = 0.0
    for j in range(1, len(ids)):
        total -= float(lp[j - 1, ids[j]])
    # The bytes SCORED are the bytes of the text the scored tokens cover, which is the whole
    # text minus whatever the first token spans. Measured by decoding, not assumed.
    first = m.tok.decode([ids[0]]) if hasattr(m.tok, "decode") else ""
    scored_bytes = len(text.encode("utf-8")) - len(first.encode("utf-8"))
    if scored_bytes <= 0:
        return None, f"first token spans the whole text ({len(text)} chars); nothing to score"
    return (total / math.log(2), scored_bytes), None


def roundtrip_fraction(tok, rows, decode, encode):
    """Fraction of rows whose decode->encode reproduces the original ids exactly.

    This is the number that says whether "the same bytes" is true. It is measured rather than
    asserted because a lossy merge or a normalisation step in either direction would otherwise
    turn into a silent difference in what the two arms were scored on.
    """
    ok = 0
    for r in rows:
        ids = [int(t) for t in r]
        if encode(decode(ids)) == ids:
            ok += 1
    return ok / max(1, len(rows))


def _selftest():
    # BPB arithmetic against a known answer: uniform over V, one token per byte, so every
    # scored token costs exactly log2(V) bits and the figure is log2(V) bits/byte.
    class UniformTok:
        def __init__(self, v):
            self.v = v

        def decode(self, ids):
            return bytes(i % 256 for i in ids).decode("utf-8", "replace")

    class UniformModel:
        def __init__(self, v):
            self.v = v
            self.tok = UniformTok(v)

        def encode(self, s):
            return [b for b in s.encode("utf-8")]

        def logprobs(self, ids):
            return torch.full((len(ids), self.v), -math.log(self.v))

    for v in (256, 4096):
        m = UniformModel(v)
        (bits, nbytes), err = text_bpb(m, "abcdefgh")
        assert err is None, err
        # 8 bytes, first excluded -> 7 scored tokens, 7 scored bytes
        assert nbytes == 7, nbytes
        assert abs(bits / nbytes - math.log2(v)) < 1e-6, (bits / nbytes, math.log2(v))

    # THE FIRST TOKEN MUST NOT BE CHARGED. Splitting the same text into two chunks and summing
    # must give the same bits/byte as scoring it whole -- that only holds if each chunk's
    # unscored first token is excluded from both numerator and denominator.
    m = UniformModel(256)
    (b_all, n_all), _ = text_bpb(m, "abcdefgh")
    (b1, n1), _ = text_bpb(m, "abcd")
    (b2, n2), _ = text_bpb(m, "efgh")
    assert abs((b1 + b2) / (n1 + n2) - b_all / n_all) < 1e-9, \
        f"chunking changed bits/byte: {(b1 + b2) / (n1 + n2)} vs {b_all / n_all}"

    # BYTES, NOT CHARACTERS -- and this case is what tells them apart. Every string above is
    # ASCII, where len(s) == len(s.encode()), so a divisor of characters passes all of it. A
    # 3-byte-per-character string is the only input that separates the two, and the earlier
    # version of this selftest did not have one: `len(text) - len(first)` was green.
    (b_cjk, n_cjk), err = text_bpb(m, "\u4e2d\u6587\u6d4b\u8bd5")   # 4 chars, 12 bytes
    assert err is None, err
    assert n_cjk == 9, (f"scored bytes {n_cjk}, expected 9 (12 bytes minus the first "
                        f"character's 3). If it is 3, the divisor is COUNTING CHARACTERS.")

    # A text too short to score REFUSES rather than returning 0.
    out, err = text_bpb(m, "")
    assert out is None and "at least 2" in err, (out, err)

    # The round-trip measure must actually detect a lossy codec, or it certifies nothing.
    rows = [[104, 105], [106, 107]]
    assert roundtrip_fraction(None, rows, lambda i: "".join(chr(x) for x in i),
                              lambda s: [ord(c) for c in s]) == 1.0
    assert roundtrip_fraction(None, rows, lambda i: "".join(chr(x) for x in i),
                              lambda s: [ord(c) for c in s][:1]) == 0.0

    print("domain_bpb self-test OK: uniform models read exactly log2(V) bits/byte, the "
          "unscored first token enters neither numerator nor denominator (so chunking cannot "
          "change the figure), a too-short text refuses, and the round-trip measure detects a "
          "lossy codec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="our .pt, or an HF directory with --hf")
    ap.add_argument("--hf", action="store_true", help="control arm: HF format, own tokenizer")
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_200m_4b.json"))
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=64, help="held-out rows per domain")
    ap.add_argument("--max_ctx", type=int, default=2048)
    ap.add_argument("--preds", help="jsonl, appended per domain; rerun resumes from it")
    ap.add_argument("--out", help="summary json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return 0
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")

    # OUR tokenizer decodes the held-out ids to text; that text is what BOTH arms score. It is
    # loaded even on the control arm, because it is the only thing that can read the cache.
    from scripts.loader import load_tokenizer  # noqa: PLC0415
    from domain_loss import val_seqs  # noqa: PLC0415

    ours_tok = load_tokenizer(a.tokenizer, None)
    mix = json.load(open(a.mix, encoding="utf-8"))
    m = HFModel(a.ckpt, a.device) if a.hf else OursModel(a.ckpt, a.tokenizer, a.device)
    print(f"Loaded {a.ckpt}: {m.n_params / 1e6:.2f}M params | {len(mix['domains'])} domains",
          flush=True)

    done = {}
    if a.preds and os.path.exists(a.preds):
        with open(a.preds, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[r["domain"]] = r

    out, skipped = {}, {}
    for name in mix["domains"]:
        if name in done:
            r = done[name]
            (out if r.get("bpb") is not None else skipped)[name] = r.get("bpb") or r.get("skip")
            continue
        rows = val_seqs(name, ours_tok)
        if rows is None or not len(rows):
            skipped[name] = "no shards for this domain (absent, not zero)"
            continue
        rows = rows[: a.rows]
        rt = roundtrip_fraction(ours_tok, rows, ours_tok.decode,
                               lambda s: ours_tok.encode(s, add_special_tokens=False).ids)
        if rt < MIN_ROUNDTRIP:
            skipped[name] = (f"decode round-trip only {rt:.3f} < {MIN_ROUNDTRIP}: the text the "
                             f"two arms would score is not the held-out bytes")
            print(f"  {name:16} SKIPPED (round-trip {rt:.3f})", flush=True)
            continue
        bits = nbytes = 0.0
        errs = 0
        for r in rows:
            text = ours_tok.decode([int(t) for t in r])
            res, err = text_bpb(m, text, a.max_ctx)
            if err:
                errs += 1
                continue
            bits += res[0]
            nbytes += res[1]
        if nbytes <= 0:
            skipped[name] = f"no row produced scored bytes ({errs} errors)"
            continue
        rec = {"domain": name, "bpb": bits / nbytes, "scored_bytes": int(nbytes),
               "n_rows": len(rows), "roundtrip_exact": rt, "row_errors": errs}
        out[name] = rec["bpb"]
        if a.preds:
            with open(a.preds, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {name:16} {rec['bpb']:.4f} bits/byte over {int(nbytes):,} bytes "
              f"(round-trip {rt:.3f})", flush=True)

    if not out:
        print("REFUSING: no domain produced a number")
        return 1
    result = {
        "ckpt": a.ckpt, "hf": a.hf, "mix": os.path.basename(a.mix),
        "n_params": m.n_params,
        "per_domain_bpb": out,
        "unweighted_mean_bpb": sum(out.values()) / len(out),
        "n_domains": len(out), "n_domains_total": len(mix["domains"]),
        "skipped": skipped,
        "reading": "bits per UTF-8 byte of the held-out rows train.py holds out, decoded to "
                   "text so both arms score the SAME bytes with their own tokenizers",
        "boundary": "NOT comparable to domain_loss.py's numbers: that metric is nats per TOKEN "
                    "and every existing record and threshold is in those units. This is a "
                    "second reading for the cross-tokenizer comparison, not a replacement. The "
                    "mean is unweighted over the domains that produced a number; `skipped` "
                    "names the rest with reasons, since a mean over a silently smaller set is "
                    "a different metric wearing the same name.",
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
