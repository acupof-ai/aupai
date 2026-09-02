---
question: Which nanochat / modded-nanogpt speedrun techniques does train.py use, which are missing, and which of ours deviate from both references?
status: recorded
source: modded-nanogpt README records 1-89 and train_gpt.py (raw.githubusercontent.com, 2026-09-02); nanochat README, dev/LEADERBOARD.md, dev/LOG.md, nanochat/gpt.py, optim.py, scripts/base_train.py; train.py and model.py at main f5417d4
---

# Speedrun techniques vs train.py (2026-09-02)

Reference points. modded-nanogpt track 1: 124M, 8xH100, val 3.28, 45 min (2024-05) to 1.23 min
(record 89, 2026-07). nanochat time-to-GPT-2: 3.04 h to 1.65 h in six leaderboard rows. Our model is
a KDA + gated-MLA hybrid with NoPE, so attention-window and RoPE items do not transfer; everything
else does. Gains quoted are the record's own claim, at the reference's scale.

## Present and matching a reference

| technique | reference value | ours | cite |
|---|---|---|---|
| Muon, Nesterov, 5 orthogonalization steps | both | same | train.py:494-548 |
| Polar Express coefficients | MN #38 (-10 steps), NC default | same 5 tuples | train.py:485-491 |
| Cautious weight decay on Muon | MN #43 (-40 steps), NC "solid" | `mask=(g*w)>=0` | train.py:547 |
| Muon momentum warmup | MN 0.85->0.95 over 300; NC 0.85->0.97 over 400 | 0.85->0.95 over 150 | train.py:782 |
| Muon params = 2D matrices only; embed, head, scalars on AdamW | both | same routing | train.py:721-733 |
| Embedding AdamW lr 0.1, betas (0.8, 0.995), wd 0.001 | NC wte lr 0.2*(d/768)^-0.5 = 0.17, same betas and wd | 0.1 | train.py:251-253 |
| QK-norm | MN #5, NC | MLA rms_norm q,k; KDA in-kernel l2 | model.py:167, :132 |
| Logit softcap 15 | MN #18, NC | 15 | model.py:63 |
| Attention output gate | MN #28 sparse gate (-50 steps) | MLA output sigmoid gate | model.py:194 |
| Vocab padded to a multiple of 16/64 | MN #5, NC | 32784->32832, +13.9% e2e | train.py:160-166 |
| Document masking with packed varlen attention | MN #12/#29 | flash varlen + KDA cu_seqlens | model.py:169, :127 |
| fp8 tensorwise on all eligible Linear, head excluded | NC run 2 (+17% tok/s at d26) | torchao e4m3 tensorwise, `_fp8_ok` | train.py:436-475 |
| Fused CE without a logits tensor | MN #37/#60 | Liger FLCE outside compile | train.py:2348 |
| torch.compile dynamic=False, expandable_segments | both | same, no fullgraph | train.py:2220, :13 |
| Warmdown fraction 0.65 to 0.05 floor | NC 0.65 linear to 0.05 | Cfg default 0.65 **cosine** to 0.05 | train.py:205-206, :1748 |
| Cosine WD-to-zero on Muon only | NC | linear-to-zero on Muon only | train.py:783 |

## Absent, applicable to our architecture

| technique | reference gain | applies to us how | cost to try |
|---|---|---|---|
| Value embeddings, gated, alternating layers | MN #14 (4.66->4.41 min) plus #55/#63/#65; NC LOG "models love VEs", halves optimal tokens:params | token-indexed table added to V of MLA layers (and KDA v) with a 3*sigmoid gate on 12 residual dims | 1 A/B, 500 steps, model.py |
| Shape-based Muon lr `max(1, rows/cols)^0.5` | MN and NC both | our Muon update has no shape factor; tall w13 [3072x2,1024] and MLA projections under-stepped relative to square | one line, 1 A/B |
| NorMuon variance reduction (beta2 0.9) | MN #41/#42; NC "small, kept" | after orthogonalization | ~15 lines, same A/B as above |
| Muon momentum cooldown to 0.85 in the last 50 steps | MN #39 | schedule only | trivial, same A/B |
| Drop gradient clipping | NC LOG: 0.25-2.0 all within noise, all-reduce costs ~2%; MN none | MEASURED (de, cfd09fa): clipping is 1.50 ms/step GPU, 0.092%, and its `.item()` sync lands inside windows where the GPU is 99.2% busy, so it costs no step time here | loss-side A/B only, low priority |
| Batch-size ramp 1/3 -> 2/3 -> 1 with lr x(B ratio)^0.5-0.6 | MN #46 (-65 steps, -1.8 s); NC tried, small gain, not merged | schedule | 1 A/B |
| Sequence-length curriculum 896 -> 2048 | MN #72 | seq 4096 fixed | 1 A/B |
| Untied head with its own lr | NC lm_head lr 0.004*(d/768)^-0.5 = 0.0035; MN ties, unties at 2/3 | ours tied, so the head trains at the embedding lr 0.1, 28x NC's head lr. b0-10 measured tok.weight growing 1.43x per 500 steps at L12 | 1 A/B: untie, head lr 0.004 |
| Smear (previous-token mix), x0 shortcut, backout, U-net skip | MN #34, #9, #40, #11; NC x0 -0.004 to -0.010 bpb | AttnRes already attends over all previous sublayer outputs, which covers x0, skip and backout functionally; smear is not covered | smear: 1 A/B; the rest: measure AttnRes-off first (R2 control) |
| Sharded optimizer with reduce_scatter | MN #6 (15.2->13.1 min), #24, #36; NC MuonAdamW | MEASURED (de, cfd09fa): the whole Muon step is 35.79 ms of 1624.25 ms, 2.2%, at 200M/4 cards; 90.68 ms of Newton-Schulz GEMM over 3 steps sat inside the `gemm` class of eff.step_class_breakdown_p200m_4card; 500M shape unmeasured | not now; 2.2% is the ceiling |
| Zero-init attention and MLP output projections | MN #5 (with QK-norm and ReLU^2, 22.3->15.2 min); NC | model.py zero-inits only `dyn[1]` and the vocab pad rows (model.py:353, :364); `o`, `w2`, MLA out are default-init | one init line, 1 A/B |
| ReLU^2 MLP | MN #5; NC: SwiGLU FLOP-matched worse at d12 and d24 | ours is a bounded tanh-GLU (K3 SiTU-GLU), untested against ReLU^2 | 1 A/B |
| Bigram hash embedding, MTP, prefix loss, paired heads, MUDD, XSA | MN #62 and later | NC reverted bigram at d25 (gain vanished in wall-clock); MTP negative in NC | not now |

## Ours that deviate from both references

| item | ours | both references | consequence |
|---|---|---|---|
| Production 200M launch `--warmdown 0.1 --anneal_frac 0` | warmdown over the last 10% of steps | NC 0.65, MN 0.60 cooldown; MN #26 gained by lengthening 0.40->0.45 | the run's final loss understates what the recipe gives; a resume-only convenience became the recipe. Every ladder point must state its warmdown |
| Cosine warmdown | cosine | linear in both | small; linear is what the references' floor values were tuned on |
| Gradient clipping 1.0 | on | off in both | 0.092% of step time here (de, cfd09fa), sync free; the open question is loss-side only |
| Tied embedding at lr 0.1 | tied, one AdamW group | NC untied, head lr 0.0035; MN tied with head-only update and transposed copy | head trains at 28x the reference head lr; candidate mechanism for the uniform 1.43x/500-step embedding growth (b0-10) |
| Muon lr 0.01 | 0.01 | NC 0.02, MN 0.023 | half; with `lr_scale` as the only knob |
| Warmup 20 steps (Cfg), 300 in the 200M launch | 20 / 300 | NC 40, MN 0 | 300 is 8% of a 3814-step run |

## What this does not say

No number above was measured on our stack. The references are 124M and 560M-1.9B transformers on
H100 with GPT-2 or 32K tokenizers; our step is 50.8% GEMM, 30.6% elementwise at the bandwidth
roofline, 10.6% KDA kernels (facts/efficiency.json#eff.step_roofline_p200m_4card), so optimizer
and schedule items move loss-per-token, not tok/s. Each row in the two tables above is one A/B at
equal tokens (`scripts/run_ablation.sh` shape, 500 steps, 4 cards) before it enters the recipe.
