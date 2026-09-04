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
# THERE IS NO THRESHOLD, because the right check is exact. MIN_ROUNDTRIP = 0.98 was here and it
# gated on ID stability -- whether decode->encode reproduces the same ids -- while the property this
# metric needs is BYTE identity: that both arms score the same text. The two come apart in exactly
# one place, a row cut mid-UTF-8-character: the fragment decodes to U+FFFD, which re-encodes as the
# three tokens of U+FFFD, so the ids differ while the text does not.
#
# Measured before the change (audit_0904 E19/C12, 576 val rows over all 9 domains of mix_200m_4b):
#   id round-trip  0.9826  -- below 0.98 for zh_web (0.9375), cot and chatml (0.9688)
#   TEXT identity  1.0000  -- every row, every domain, no exceptions
# So the old gate skipped all 9 domains, `out` came out empty, and the metric printed REFUSING and
# exited 1 for every checkpoint ever scored -- it has never produced a number. Only 6 of 576 rows
# carry a boundary U+FFFD at all; the other misses were the same fragment at the row END.
# Unpacked plain text round-trips exactly by ids too (5/5 known-answer cases in _selftest), so the
# tokenizer is not lossy and there was never a fraction to tune.


class OursModel:
    def __init__(self, ckpt, tok_path, device, cu_path="cu_none"):
        from scripts.loader import load_checkpoint, load_tokenizer  # noqa: PLC0415

        self.model, self.cfg = load_checkpoint(ckpt, device=device, dtype=torch.bfloat16)
        self.tok = load_tokenizer(tok_path, self.cfg)
        self.device = device
        # cu_path="doc_cu" passes the document mask, the same spelling and the same helper as
        # eval/domain_loss.py:245. This metric decodes a WHOLE PACKED ROW to one text at :272 and
        # scores it as one undivided sequence, so without the mask attention reads across the
        # document boundaries inside that row while training used doc_cu_seqlens (audit_0904 E10).
        # HFModel takes no cu: a foreign model has no such argument, which is why the control arm
        # is unaffected and why a doc_cu run compares our arm against its own cu_none run, not
        # against the control.
        if cu_path not in ("cu_none", "doc_cu"):
            raise ValueError(f"cu_path must be 'doc_cu' or 'cu_none', got {cu_path!r}")
        self.cu_path = cu_path
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def logprobs(self, ids):
        x = torch.tensor([ids], device=self.device)
        cu = None
        if self.cu_path == "doc_cu":
            from train import doc_cu_seqlens  # noqa: PLC0415  (CPU callers need no train)
            from scripts.loader import EOS_ID  # noqa: PLC0415
            cu = doc_cu_seqlens(x, EOS_ID)
        with torch.no_grad():
            out = self.model(x, cu=cu) if cu is not None else self.model(x)
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


def text_identity_misses(rows, decode, encode):
    """Rows whose decoded text does NOT survive a decode->encode->decode round trip.

    THE PROPERTY IS BYTE IDENTITY, NOT ID STABILITY. Both arms score `decode(ids)`; what has to
    hold is that this text is what the codec really represents, i.e. that re-encoding it and
    decoding again gives the same text back. A lossy merge or a normalisation step in either
    direction changes the text and is caught here.

    Returns the list of offending row indices -- empty means exact, and there is nothing to
    threshold. The predecessor asked `encode(decode(ids)) == ids` and failed on every row cut
    mid-UTF-8-character, where the ids legitimately differ (U+FFFD is one token in the row and
    three when re-encoded) while the text is identical. See the MIN_ROUNDTRIP note above.
    """
    bad = []
    for i, r in enumerate(rows):
        text = decode([int(t) for t in r])
        if decode(encode(text)) != text:
            bad.append(i)
    return bad


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

    # THE GATE MUST DETECT A LOSSY CODEC AND MUST NOT FIRE ON A MERE RE-SPLIT. Both halves, or it
    # certifies nothing: the predecessor had only the first half and rejected all 9 domains.
    rows = [[104, 105], [106, 107]]
    ident = lambda i: "".join(chr(x) for x in i)          # noqa: E731
    back = lambda s: [ord(c) for c in s]                  # noqa: E731
    assert text_identity_misses(rows, ident, back) == [], "an exact codec was reported lossy"
    # Lossy: encode drops a character, so the text does not come back.
    assert text_identity_misses(rows, ident, lambda s: back(s)[:1]) == [0, 1], \
        "a codec that loses a character was certified as exact"
    # RE-SPLIT ONLY: same text, different ids -- the U+FFFD case that voided every real domain.
    # `encode` returns different ids than the row held, but decode of them gives the same text,
    # so this must NOT be flagged. An id-equality gate fails here and that was the whole defect.
    resplit = lambda s: [ord(c) for c in s] + [0]         # noqa: E731
    ident0 = lambda i: "".join(chr(x) for x in i if x != 0)  # noqa: E731
    assert text_identity_misses(rows, ident0, resplit) == [], \
        "a re-split that preserves the text was flagged; that is the MIN_ROUNDTRIP defect"
    # KNOWN ANSWERS on the real tokenizer, if it is present: unpacked plain text must survive
    # exactly. These five are the cases C12 measured (E19); a tokenizer that fails any of them is
    # lossy and no threshold could rescue the metric.
    if os.path.exists(TOK_PATH):
        from scripts.loader import load_tokenizer  # noqa: PLC0415
        _t = load_tokenizer(TOK_PATH, None)
        _d = lambda ids: _t.decode(ids, skip_special_tokens=False)      # noqa: E731
        _e = lambda x: _t.encode(x, add_special_tokens=False).ids       # noqa: E731
        for text in ("The quick brown fox jumps over the lazy dog.",
                     "def add(a, b):\n    return a + b\n",
                     "\u4e2d\u6587\u6bb5\u843d\uff0c\u5e26\u6807\u70b9\u3002 3.14159 \u548c 2026-09-04\u3002",
                     "x = 1e-9 * 42 // 7 % 3",
                     "mixed \u4e2d\u82f1 text with 12345 numbers"):
            assert _d(_e(text)) == text, f"tokenizer lost text on a known-answer case: {text!r}"

    print("domain_bpb self-test OK: uniform models read exactly log2(V) bits/byte, the "
          "unscored first token enters neither numerator nor denominator (so chunking cannot "
          "change the figure), a too-short text refuses, the gate detects a "
          "lossy codec while ignoring a pure re-split, and five known-answer texts survive "
          "the real tokenizer exactly")


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
    ap.add_argument("--cu_path", choices=["cu_none", "doc_cu"], default="cu_none",
                    help="doc_cu passes the document mask to our arm's forward; cu_none is "
                         "what every published row was taken with (E10). Ignored under --hf: "
                         "a foreign model has no cu argument.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return 0
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")

    # OUR tokenizer decodes the held-out ids to text; that text is what BOTH arms score. It is
    # loaded even on the control arm, because it is the only thing that can read the cache.
    from scripts.loader import load_tokenizer, vocab_fingerprint  # noqa: PLC0415
    from domain_loss import val_seqs  # noqa: PLC0415
    import train  # noqa: PLC0415

    ours_tok = load_tokenizer(a.tokenizer, None)
    mix = json.load(open(a.mix, encoding="utf-8"))
    # REFUSE BEFORE LOADING, not after: loading a 438M checkpoint to then exit wastes the load,
    # and a refusal that runs after the work is a refusal nobody sees in time.
    if a.hf and a.cu_path != "cu_none":
        sys.exit("REFUSING: --cu_path is meaningless under --hf; a foreign model takes no cu "
                 "argument, so this run would silently be a cu_none run under a doc_cu name.")
    m = (HFModel(a.ckpt, a.device) if a.hf
         else OursModel(a.ckpt, a.tokenizer, a.device, cu_path=a.cu_path))
    # val_seqs -> _domain_seqs compares every cache stamp against train.VOCAB_ID, which starts
    # None and is set only by train.build_tokenizer -- which no eval calls. Without this the
    # guard reports "cache dirty" when the process simply has no fingerprint, and that is why
    # this file had never produced a value: all three of its rows in runs/score_matrix.jsonl
    # are that error, one of them the --hf control.
    #
    # FROM THE TOKENIZER, NOT FROM A CHECKPOINT'S cfg. Both arms score the same decoded rows, so
    # BOTH must reach val_seqs -- the control has no cfg and no vocab_id, so a cfg-derived
    # fingerprint (set_vocab_id, which domain_loss.py:624 uses because it has a checkpoint per
    # arm) can only ever cover our own arm and leaves the control hitting the guard. The rows are
    # ours_tok's regardless of which model scores them, so ours_tok is the right source for
    # their fingerprint; scripts/test_domain_loss_val.py:103 does the same for the same reason.
    # load_tokenizer already refused a tokenizer disagreeing with our checkpoint's vocab_id.
    train.VOCAB_ID = vocab_fingerprint(ours_tok)
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
        # skip_special_tokens=False: the delimiter is part of the held-out bytes. The default drops
        # it (EOS decodes to ""), which is what made the id-based predecessor fail on every packed
        # row -- see the MIN_ROUNDTRIP note at the top.
        _dec = lambda ids: ours_tok.decode(ids, skip_special_tokens=False)  # noqa: E731
        _enc = lambda s: ours_tok.encode(s, add_special_tokens=False).ids   # noqa: E731
        # PER ROW, and the offenders are dropped rather than the domain. A row whose text does not
        # survive the codec cannot be scored on "the same bytes", but one such row is no reason to
        # void the other 63 -- the count is reported so a silent shrink is impossible.
        bad = text_identity_misses(rows, _dec, _enc)
        if bad:
            keep = [r for i, r in enumerate(rows) if i not in set(bad)]
            skipped[f"{name}#rows"] = (f"{len(bad)} of {len(rows)} row(s) dropped: decoded text "
                                       f"does not survive decode->encode->decode, so the two arms "
                                       f"would not score the same bytes")
            print(f"  {name:16} dropped {len(bad)}/{len(rows)} row(s) on text identity", flush=True)
            rows = keep
        if not len(rows):
            skipped[name] = "every row failed text identity; nothing left to score"
            print(f"  {name:16} SKIPPED (no row survived text identity)", flush=True)
            continue
        bits = nbytes = 0.0
        errs = 0
        for r in rows:
            # THE SAME decode THE GATE CHECKED. Using the default here while the gate used
            # skip_special_tokens=False would score text the gate never cleared.
            text = _dec([int(t) for t in r])
            res, err = text_bpb(m, text, a.max_ctx)
            if err:
                errs += 1
                continue
            bits += res[0]
            nbytes += res[1]
        if nbytes <= 0:
            skipped[name] = f"no row produced scored bytes ({errs} errors)"
            continue
        # text_identity_dropped, not a round-trip FRACTION: the gate is exact now, so what a
        # reader needs is how many rows it removed, which is 0 on every domain measured so far.
        rec = {"domain": name, "bpb": bits / nbytes, "scored_bytes": int(nbytes),
               "n_rows": len(rows), "text_identity_dropped": len(bad), "row_errors": errs}
        out[name] = rec["bpb"]
        if a.preds:
            with open(a.preds, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {name:16} {rec['bpb']:.4f} bits/byte over {int(nbytes):,} bytes "
              f"({len(bad)} row(s) dropped on text identity)", flush=True)

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
        # THE RECORD SAYS WHICH PATH. Every other field is identical between a cu_none run
        # and a doc_cu run of the same checkpoint, so a reader cannot otherwise tell, and
        # 0 of 60 published score_matrix rows said (E1/E5).
        "cu_path": "n/a (hf)" if a.hf else a.cu_path,
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
