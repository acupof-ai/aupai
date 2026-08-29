#!/usr/bin/env python3
"""FoNE — Fourier Number Embedding (arXiv:2502.09741).

A number is one token, not a BPE fragment. The tokenizer's own segmentation of
digits is frequency-driven and carries no place value: this vocab splits 1640 as
16|40 but 3200 as 3|200, so a carry rule learned for one number does not transfer
to the next. FoNE sidesteps the tokenizer: every number becomes a single [NUM]
token whose embedding is computed from the *value*, two dimensions per digit.

    phi(x, T) = (cos(2*pi*x/T), sin(2*pi*x/T))               # Definition 3.1

Digit i occupies dims [2i, 2i+1] as phi(d_i, 10): the pair sits on a circle where
one step of the digit is one tenth of a turn, so reading it back is a nearest-angle
lookup over the ten points phi(0..9, 10) (Definition 3.7). See encode() for why
this is written per-digit rather than as the paper's phi(x, 10^i).

Digits are recovered from the hidden state the same way, which makes the number
head ten-way per digit instead of vocab-wide (Definition 3.6).
"""

import math
import re

import torch

# Numbers as they appear in the corpus: optional sign, digits, optional decimals.
# Fractions (1/3) are two numbers around a separator, which is what the corpus
# writes and what the eval's grader parses.
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

INT_DIGITS = 6  # 10^0 .. 10^5   -> values up to 999999
FRAC_DIGITS = 2  # 10^-1, 10^-2  -> two decimal places
NUM_DIMS = 2 * (INT_DIGITS + FRAC_DIGITS)


def encode(values, m=INT_DIGITS, n=FRAC_DIGITS, dtype=torch.float32):
    """Values -> (len(values), 2*(m+n)) Fourier features, digit i in dims [2i, 2i+1].

    The paper writes this as phi(x, 10^i) (Definition 3.2), whose angle at period
    10^(i+1) is 2*pi*d_i/10 PLUS 2*pi*(lower digits)/10^(i+1). That carry term is
    up to nine tenths of a digit step, so the encoder and the nearest-angle decoder
    disagree: x=7 lands its tens pair closer to the point for 1 than for 0, and 7
    reads back as 17. Encoding each digit at its own angle, phi(d_i, 10), removes
    the cross-talk and is exactly what the decoder looks for. It is the same
    quantity -- phi has period 10 in its first argument, so phi(floor(x/10^i), 10)
    == phi(d_i, 10) -- just without the neighbouring digits leaking in.
    """
    d = digits_of(values, m, n).to(torch.float64)  # (N, m+n), least significant first
    ang = 2 * math.pi * d / 10.0
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).reshape(d.shape[0], -1).to(dtype)


def encode_tensor(x, m=INT_DIGITS, n=FRAC_DIGITS):
    """Batched encode for the training graph: (...) values -> (..., 2*(m+n)) features.

    Same quantity as encode(), written in torch on the input's own device and kept
    differentiable-shaped. Values are data, not parameters, so no gradient flows
    back through the digit extraction -- the learnable part is num_proj downstream.
    """
    # float64 throughout: the scaled integer reaches 10^(m+n) = 10^8, past float32's
    # exact range of 2^24, so a float32 path would return wrong digits for large
    # numbers. MPS has no float64 at all, so that device computes on CPU and comes
    # back -- correctness first, and the tensor is one row per [NUM].
    dev = x.device
    if dev.type == "mps":
        x = x.cpu()
    scaled = torch.round(x.double().abs() * (10.0**n))
    place = 10.0 ** torch.arange(m + n, dtype=torch.float64, device=x.device)
    d = torch.remainder(torch.div(scaled.unsqueeze(-1), place, rounding_mode="floor"), 10)
    ang = 2 * math.pi * d / 10.0
    out = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(-2)
    return out.to(x.dtype).to(dev)  # cast first: float64 cannot cross to MPS


def digit_targets(x, m=INT_DIGITS, n=FRAC_DIGITS):
    """Values -> (..., m+n) integer digit labels, on the input's own device.

    The tensor-side twin of digits_of(), for building cross-entropy targets inside
    the training loop without a round trip through python lists.
    """
    dev = x.device
    if dev.type == "mps":  # no float64 on MPS; see encode_tensor
        x = x.cpu()
    scaled = torch.round(x.double().abs() * (10.0**n))
    place = 10.0 ** torch.arange(m + n, dtype=torch.float64, device=x.device)
    out = torch.remainder(torch.div(scaled.unsqueeze(-1), place, rounding_mode="floor"), 10).long()
    return out.to(dev)


def digit_basis(dtype=torch.float32):
    """The ten reference points phi(j, 10) for j = 0..9 — the decoding dictionary."""
    j = torch.arange(10, dtype=torch.float64)
    ang = 2 * math.pi * j / 10.0
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).to(dtype)  # (10, 2)


def digit_logits(h, m=INT_DIGITS, n=FRAC_DIGITS):
    """Hidden state -> (..., m+n, 10) per-digit logits. Definitions 3.6 / 3.7.

    Reads the first 2*(m+n) dimensions as (digit, [cos, sin]) pairs and scores each
    against the ten digit points. Cross-entropy over this is the number loss;
    argmax over it is the prediction.
    """
    d = m + n
    pairs = h[..., : 2 * d].reshape(*h.shape[:-1], d, 2)
    return pairs @ digit_basis(h.dtype).to(h.device).T  # (..., d, 10)


def digits_of(values, m=INT_DIGITS, n=FRAC_DIGITS):
    """Values -> (len(values), m+n) integer digit targets, least significant first.

    Digit i of the scaled integer round(x * 10^n); negatives use |x|, since the
    sign is not one of the periods and rides on the surrounding text.
    """
    x = torch.as_tensor(values, dtype=torch.float64).abs().reshape(-1)
    scaled = torch.round(x * (10.0**n))
    out = torch.empty(x.shape[0], m + n, dtype=torch.long)
    for i in range(m + n):
        out[:, i] = torch.remainder(torch.div(scaled, 10.0**i, rounding_mode="floor"), 10).long()
    return out


def decode(logits, n=FRAC_DIGITS):
    """Per-digit logits -> the numbers they encode."""
    d = logits.argmax(-1)  # (..., m+n)
    dev = d.device
    if dev.type == "mps":  # no float64 on MPS; see encode_tensor
        d = d.cpu()
    place = 10.0 ** torch.arange(d.shape[-1], dtype=torch.float64, device=d.device)
    return ((d.double() * place).sum(-1) / (10.0**n)).float().to(dev)


def encode_prompts(texts, tok, num_id):
    """Texts -> ([ids], [values]) per text, one value per position rather than per number.

    encode_text stores values compactly; a sampler wants them spread over the
    positions so it can hand the model a (B, T) tensor.
    """
    pieces, vals = encode_text(texts, tok, num_id)
    ids_out, val_out, k = [], [], 0
    for p in pieces:
        ids, dense = p.tolist(), []
        for t in ids:
            dense.append(float(vals[k]) if t == num_id else 0.0)
            k += t == num_id
        ids_out.append(ids)
        val_out.append(dense)
    return ids_out, val_out


def render(v, n=FRAC_DIGITS):
    """A decoded value -> the text a number would have been written as.

    The code always carries n fractional digits, so 36 comes back as 36.0 and has
    to lose them again; 3.5 keeps the one it needs.
    """
    s = f"{v:.{n}f}".rstrip("0").rstrip(".")
    return s or "0"


def decode_text(ids, vals, tok, num_id):
    """Generated ids plus their [NUM] values -> text, numbers written back in.

    tok.decode() alone would emit the [NUM] token itself and lose the number, so
    the stream is decoded in runs between [NUM] positions.
    """
    out, run, k = [], [], 0
    for t in ids:
        if t == num_id:
            out.append(tok.decode(run))
            out.append(render(float(vals[k])))
            run, k = [], k + 1
        else:
            run.append(t)
    out.append(tok.decode(run))
    return "".join(out)


def generate_texts(model, tok, cfg, prompts, steps, device="cuda:0", temperature=0.0, batch=60):
    """prompts -> texts, FoNE or not. The ONE place the [NUM] value channel is threaded.

    The encode -> generate_batch -> decode triple with a `fone` branch was copy-pasted into
    four files, and the batching fix landed in only one of them: scripts/probe_arith.py still
    fed generate_batch a single prompt at a time, 180 sequential decodes at 31% GPU, ~15
    minutes where the work is ~20 seconds.

    Chunked because generate_batch allocates (B, T) up front and grows it a column per step;
    every other caller chunks too."""
    from train import generate_batch

    fone_on = bool(getattr(cfg, "fone", False))
    out = []
    for s in range(0, len(prompts), batch):
        chunk = prompts[s : s + batch]
        if fone_on:
            ids, vals = encode_prompts(chunk, tok, cfg.num_id)
            gen, gv = generate_batch(model, ids, steps, device, temperature, vals)
            out += [decode_text(g, v, tok, cfg.num_id) for g, v in zip(gen, gv, strict=True)]
        else:
            ids = [tok.encode(p, add_special_tokens=False).ids for p in chunk]
            out += [tok.decode(g) for g in generate_batch(model, ids, steps, device, temperature)]
    return out


def split_numbers(text):
    """Text -> (segments, values): the text with numbers cut out, and their values.

    len(segments) == len(values) + 1; re-joining segments around [NUM] rebuilds the
    original string, so this is lossless for the round trip.
    """
    segs, vals, last = [], [], 0
    for mo in NUM_RE.finditer(text):
        segs.append(text[last : mo.start()])
        vals.append(float(mo.group(0)))
        last = mo.end()
    segs.append(text[last:])
    return segs, vals


NUM_TOKEN = "[NUM]"


def encode_text(texts, tok, num_id):
    """Texts -> (ids int32, values float32), numbers replaced by one [NUM] token each.

    Values are stored compactly, in stream order: the k-th [NUM] in ids takes the
    k-th value. Positions need no index of their own -- `ids == num_id` recovers
    them, and a cumsum over that mask maps any slice of the stream back into this
    array. At web-text number density that is a few percent of the id stream,
    against the 100% a value-per-position array would cost.

    Magnitudes at or above 10^INT_DIGITS cannot be represented; those numbers keep
    their ordinary BPE tokens instead of becoming a [NUM] that would silently mean
    something else.

    Every segment of every document goes through ONE encode_batch call. Tokenizing
    per document instead costs a python-level call per document, which at corpus
    scale is the difference between minutes and hours.
    """
    import numpy as np

    batch_fn = getattr(tok, "encode_batch_fast", tok.encode_batch)
    limit = 10**INT_DIGITS
    flat, bounds, vals = [], [], []
    for text in texts:
        segs, nums = _split_bounded(text, limit)
        flat.extend(segs)
        bounds.append(len(segs))
        vals.extend(nums)
    enc = batch_fn(flat)

    pieces, at = [], 0
    for k in bounds:
        row = []
        for j in range(k):
            row.extend(enc[at + j].ids)
            if j < k - 1:  # one [NUM] between consecutive segments
                row.append(num_id)
        at += k
        pieces.append(np.asarray(row, dtype=np.int32))
    return pieces, np.asarray(vals, dtype=np.float32)


def _split_bounded(text, limit):
    """split_numbers, but numbers too large to encode stay inside the text segments.

    A number at or above the limit has no representable Fourier code, so it is left
    where it is and keeps its ordinary BPE tokens; only the representable ones become
    [NUM]. len(segments) == len(values) + 1 still holds, which is what lets the caller
    put exactly one [NUM] between consecutive segments.
    """
    segs, vals, last = [], [], 0
    for mo in NUM_RE.finditer(text):
        v = float(mo.group(0))
        if abs(v) >= limit:
            continue  # leave it in the surrounding segment
        segs.append(text[last : mo.start()])
        vals.append(v)
        last = mo.end()
    segs.append(text[last:])
    return segs, vals


def _selftest():
    """Round-trip and boundary checks. Run: python fone.py"""
    vals = [0, 7, 42, 152, 1640, 3200, 6142, 999999, 3.14, 0.75, 12.5, -8]
    emb = encode(vals)
    assert emb.shape == (len(vals), NUM_DIMS), emb.shape

    # A number's embedding must recover its own digits.
    got = decode(digit_logits(emb))
    for v, g in zip(vals, got.tolist()):
        assert abs(abs(v) - g) < 1e-6, f"round trip {v} -> {g}"

    # The pairs that differ by one in a digit must be distinguishable — this is the
    # property the BPE vocab destroys (152 -> 15|2 shares no structure with 153).
    for a, b in [(152, 153), (1640, 1641), (3200, 3201), (99, 100)]:
        assert not torch.allclose(encode([a]), encode([b])), f"{a} vs {b} collide"

    # Digit targets line up with the decoder's own reading.
    tgt = digits_of(vals)
    assert torch.equal(digit_logits(encode(vals)).argmax(-1), tgt), "targets != argmax"

    # Text splitting is lossless.
    t = "200 × 8/10 = 160元，再减20元，实付140元"
    segs, nums = split_numbers(t)
    assert nums == [200.0, 8.0, 10.0, 160.0, 20.0, 140.0], nums
    rebuilt = segs[0] + "".join(f"{v:g}{s}" for v, s in zip(nums, segs[1:]))
    assert rebuilt == t, rebuilt

    # Values beyond the representable range wrap rather than crash; encode_text
    # keeps them out by leaving oversized numbers as ordinary text.
    assert encode([10**INT_DIGITS]).shape == (1, NUM_DIMS)

    # The training-graph encoder must agree with the preprocessing one exactly,
    # or the model sees a different code than the data was built with.
    bt = torch.tensor(vals, dtype=torch.float32)
    assert torch.allclose(encode_tensor(bt), encode(vals), atol=1e-6), "encode_tensor != encode"
    b2 = encode_tensor(bt.reshape(3, 4))  # batched shape is preserved
    assert b2.shape == (3, 4, NUM_DIMS), b2.shape
    assert torch.allclose(b2.reshape(-1, NUM_DIMS), encode(vals), atol=1e-6)

    # Text -> (ids, values) against the real tokenizer, if it is available.
    import os

    tok_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tokenizer.json")
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer

        tk = Tokenizer.from_file(tok_path)
        nid = tk.token_to_id(NUM_TOKEN)
        if nid is None:  # tokenizer not upgraded yet: stand in with a free id
            nid = tk.get_vocab_size()
        ids, v = encode_text([t, "没有数字的一句话", "超大数 12345678 保持原样"], tk, nid)
        assert (ids[0] == nid).sum() == 6, (ids[0] == nid).sum()
        assert list(v[:6]) == [200, 8, 10, 160, 20, 140], v[:6]
        assert (ids[1] == nid).sum() == 0, "no numbers -> no [NUM]"
        assert (ids[2] == nid).sum() == 0, "oversized number must stay as text"
        # the k-th [NUM] takes the k-th value: cumsum over the mask is the index
        mask = ids[0] == nid
        assert mask.sum() == len(v) - 0, "value count matches [NUM] count"
        print(f"  encode_text OK — {len(v)} values across 3 docs, [NUM] id {nid}")

    print(
        f"fone selftest OK — {NUM_DIMS} dims "
        f"({INT_DIGITS} integer + {FRAC_DIGITS} decimal digits), {len(vals)} values round-tripped"
    )


if __name__ == "__main__":
    _selftest()
