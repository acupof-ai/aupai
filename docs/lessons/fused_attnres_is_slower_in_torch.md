---
question: Why did fusing AttnRes into one node make it 2.2x slower, when the fusion provably removed half the reads?
status: measured
source: /tmp/attnres_bench.py on H20 card 5 (isolated, imports nothing from the tree); torch profiler; algorithms/attnres_fused.py
---

# The integer got better and the clock got worse

The fused AttnRes node removes the double read. That is not in doubt — `add_`
per step goes from `n(n+1)` to `n(n+1)/2`, measured at L=2/3/4/12, exactly
2.00× at every depth, agreeing with b0's closed form and with the ablation.

It is also **2.2× slower**.

| L | eager ms | fused ms | speedup | eager GiB | fused GiB |
|---|---|---|---|---|---|
| 4 | 35.5 | 80.3 | **0.44×** | 0.74 | 1.07 |
| 8 | 120.6 | 268.3 | 0.45× | 1.26 | 1.83 |
| 12 | 255.9 | 567.1 | 0.45× | 1.78 | 2.60 |

Predicted +2.8% of wall. Measured −124%. **Wrong in direction, not magnitude.**

## The ledger that closes it

Profiler, one call at n=25: eager 5.00 ms in 128 kernel launches, fused 14.28 ms
in 504.

```
saved:  v reads 50 -> 25                        -0.84 GB
spent:  fp32 accumulator, rescaled per source   +1.68 GB
net:                                            +0.84 GB
```

The online softmax rescales the whole `[B,T,D]` accumulator by
`exp(m_old - m_new)` every time a new source raises the running max. In a Triton
kernel that accumulator lives in SMEM: rescaling is a register operation and only
the final store reaches HBM. **In torch every rescale is a full HBM round trip**,
and the accumulator is fp32 — a hard constraint, bf16 fails parity by four orders
— so it is twice the width of eager's bf16 `out` as well.

## What this actually invalidates

Not the design. The sequencing argument — mine, and it was wrong in a specific
way worth naming.

I chose "wire it up, A/B, then write the kernel" because the torch version would
**independently price the double read**. It cannot. The mechanism that makes one
pass possible (online-softmax rescaling) is the same mechanism that makes it slow
here. They are two faces of one design choice, not two separately measurable
quantities, and no A/B of this implementation can separate them.

**Triton is not a faster version of the same thing. It is the premise the design
rests on.** A fusion whose benefit is "fewer passes over the input" only pays when
the thing you carry between passes stays in fast memory; carried through HBM, the
carrying costs more than the passes saved.

## The counter-example that makes the earlier rule concrete

b0's rule: a countable integer and a throughput number answer different
questions, so do not mix them. **This is the case where they point opposite
ways.** `add_` at exactly 2.00× says the double read is gone — true, verified
three independent ways. Wall time at 0.45× says the implementation is worse —
also true. Neither number is wrong and neither is evidence about the other.

Had the integer been treated as a proxy for speed, the flag would have shipped
on strong-looking evidence.

## Rule

**Before pricing a change with an A/B, ask whether the mechanism under test can
be isolated from the mechanism that implements it.** If the same design choice
produces both the benefit and the cost, the A/B measures their sum and tells you
nothing about either. Name the two quantities and check they are separable —
"this will independently answer X" is a claim that can be false.

And: **a negative result with a closed ledger is worth more than a positive
result without one.** This one says exactly where the benefit lives (SMEM), which
is what makes the next step obvious instead of speculative.
