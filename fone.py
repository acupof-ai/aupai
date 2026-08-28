#!/usr/bin/env python3
"""FoNE — Fourier Number Embedding (arXiv:2502.09741).

A number is one token, not a BPE fragment. The tokenizer's own segmentation of
digits is frequency-driven and carries no place value: this vocab splits 1640 as
16|40 but 3200 as 3|200, so a carry rule learned for one number does not transfer
to the next. FoNE sidesteps the tokenizer: every number becomes a single [NUM]
token whose embedding is computed from the *value*, two dimensions per digit.

    phi(x, T)      = (cos(2*pi*x/T), sin(2*pi*x/T))          # Definition 3.1
    FoNE(x, m, n)  = [phi(x, 10^-n+1); ...; phi(x, 10^m)]     # Definition 3.2

phi(x, 10^(i+1)) depends on x only through x mod 10^(i+1), and its angle advances
by exactly 2*pi/10 per unit of the i-th digit — so the pair (cos, sin) at period
10^(i+1) *is* that digit on a circle, and reading it back is a nearest-angle
lookup over the ten digit points (Definition 3.7).

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


def periods(m=INT_DIGITS, n=FRAC_DIGITS):
    """The periods T_i = 10^i, ordered least significant digit first."""
    return [10.0 ** (i + 1) for i in range(-n, m)]


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
    place = 10.0 ** torch.arange(d.shape[-1], dtype=torch.float64, device=d.device)
    return (d.double() * place).sum(-1) / (10.0**n)


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

    # Values beyond the representable range wrap rather than crash; the caller is
    # responsible for keeping magnitudes under 10^INT_DIGITS.
    assert encode([10**INT_DIGITS]).shape == (1, NUM_DIMS)

    print(
        f"fone selftest OK — {NUM_DIMS} dims "
        f"({INT_DIGITS} integer + {FRAC_DIGITS} decimal digits), {len(vals)} values round-tripped"
    )


if __name__ == "__main__":
    _selftest()
