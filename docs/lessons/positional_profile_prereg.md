---
question: "is the free-running loss immediate at the first step off a correct prefix, or gradual accumulation"
status: open
source: "e1, 2026-09-01, written BEFORE runs/positional_profile.json existed"
---

# Pre-registration: immediate or gradual

> **`status: open` means pre-registered.** `runs/positional_profile.json` is
> absent as of this commit.

## Why this exists

Two mechanisms produce the same free-running collapse (~0.23 against 0.727
teacher-forced) and imply different work:

| mechanism | signature | what it implies |
|---|---|---|
| **gradual compounding** | agreement decays smoothly with distance from the prompt | a long-horizon problem; better search or shorter generations help |
| **immediate off-distribution** | agreement drops sharply within the first few tokens, then decays slowly from an already-low level | the conditional distribution is calibrated on gold prefixes and wrong on its own from step one; search does not help |

I have been telling the compounding story. It may be wrong, and this
distinguishes them on data I control.

**Provenance note:** a ruling reached me citing 0.343 in the first eight tokens
decaying to 0.078, attributed to me. I did not produce those numbers and cannot
find them in `facts/`, `runs/`, `docs/`, `probes/` on either branch or on the
pod. I have flagged that separately. **This pre-registration deliberately fixes
its bands without reference to them**, so that if they turn out to be real this
is an independent replication, and if they do not, nothing here inherits them.

## What is measured

`ckpt_pretrain_30b_s2.pt.step24000`, 100 code + 100 math prompts, greedy,
`rep_stop=False`, teacher-forced ranks as the baseline. Agreement binned by
distance from the prompt: tokens 1–8, 9–16, 17–32, 33–64, 65–128, 129–192.

Free-running agreement per bin, against the teacher-forced top-1 rate on the
same positions.

## Falsification, fixed now

Let **B1** = free-running agreement in tokens 1–8, **TF** = 0.727 (code).

| observation | reading |
|---|---|
| B1 ≥ 0.55 and later bins decay smoothly toward ~0.1 | **gradual compounding** — my story stands |
| B1 ≤ 0.40, and the drop from TF to B1 exceeds the drop from B1 to the last bin | **immediate off-distribution** — the first step off a gold prefix costs more than the entire remaining horizon |
| B1 between 0.40 and 0.55 | partial; report the curve, no verdict |

The second row's second clause is the load-bearing one: **"immediate" means the
initial drop is larger than the subsequent decay**, not merely that bin 1 is
below TF. Bin 1 is below TF under either mechanism.

## The limit, sized rather than named

The ±4 lesson applies directly, so:

- **positional bins are not independent measurements.** Once a generation
  diverges, every later bin is scored on a trajectory already off the gold, so
  late-bin agreement is not "per-position accuracy at position 100" — it is
  "accuracy given ~100 tokens of prior divergence". The profile shows **where
  the loss begins**, not how accurate the model is at depth.
- **bin 1 is the only bin conditioned on a mostly-correct prefix**, which makes
  it the only bin comparable to the teacher-forced number at all. Comparisons
  of later bins against TF are not like-for-like and I will not make them.
- alignment: bins are computed on the **shift-aligned** sequence (search to
  ±150), not the naive one, because the naive alignment is the instrument that
  reported 0.0 where the truth was 0.23.

## What I will report

The six-bin curve per set, TF on the same positions, and which row it lands in.
**Not a verdict** — fb rules.
