"""Step-D long-horizon numeric-stability probe for the v41f training path.

Runs `v41f.train.train_step` with a `v41f.master.TrainState` (bf16 forward / fp32 master
AdamW) for N steps on each indexer/engram FORM, and reports, per form:

  layer 1 (the comparable learning signal): per-pool-cycle mean CE over a FIXED pool of
      random batches, first/last cycle delta, and same-batch wins;
  layer 2: per-point single-batch CE min/max/range (noise, not learning);
      master-vs-bf16 max/mean gap first and last; AdamW exp_avg / exp_avg_sq health;
      bit-invariance of every frozen leaf; which SIX indexer leaves moved; NaN/Inf.

Why a tiny synthetic tokenizer: uniform-random ids over the full 12800 vocab are i.i.d. and
UNLEARNABLE -- CE just random-walks near ln12800=9.46, so that setup cannot test "loss keeps
descending" (measured 2026-09-18). The 12-piece tokenizer makes the fixed pool small enough
to be memorized, so a healthy training step drives cycle-mean CE down. That MEMORIZATION is
the boundary: this probe establishes optimizer/numeric hygiene over a long CPU horizon, not
convergence or generalization, and it does not exercise checkpoint/resume (that is the
bit-exact gate in tests/v41f/test_p1_train_ckpt.py).

Usage:
    python probes/v41f_stepd_longrun.py [STEPS=160] [BATCH=1] [OUTDIR]
Writes OUTDIR/v41f_longrun_{form}.json (OUTDIR defaults to ./runs/longrun). The three forms
run as separate invocations in the shell wrapper; call one form directly:
    python probes/v41f_stepd_longrun.py 160 1 runs/longrun off
Memory: each form is one process building an 180M model + fp32 master + AdamW; on a shared
laptop gate launch on >=9 GB free (the ste differentiable attention graph peaks higher and a
memory-starved run is SIGKILLed with rc137, not a Python traceback).
"""

import dataclasses
import gc
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tests", "v41f"))  # ref_oracle sibling
sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from ref_oracle import synthetic_tokenizer  # noqa: E402
from v41f.config import v41f_small  # noqa: E402
from v41f.master import TrainState  # noqa: E402
from v41f.model import V41FModel  # noqa: E402
from v41f.train import train_step  # noqa: E402

PROBE_BF16 = "layers.0.attn.qproj.wq_b.weight"
POOL = 32
SEQ = 64
SEED = 20260918
LR = 3e-3


def cfg_for(form, tok):
    if form == "engram_on":
        c = dataclasses.replace(
            v41f_small(),
            indexer_train_mode="off",
            vocab_size=len(tok),
            engram_layer_ids=(1,),
            engram_n_heads=2,
            engram_head_dim=8,
            engram_vocab_size=20,
            engram_pad_id=2,
        )
        # with_derived_engram measures engram_compressed_vocab_size and the table rows from
        # the tokenizer itself; passing them by hand drifts from that contract.
        return c.with_derived_engram(tok)
    return v41f_small(indexer_train_mode="ste" if form == "ste" else "off", vocab_size=len(tok))


def build(cfg, tok, batch):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = V41FModel(cfg, max_batch_size=batch, max_seq_len=SEQ, tokenizer=tok)
    finally:
        torch.set_default_dtype(prev)
    return m


def run(form, steps, batch):
    torch.manual_seed(SEED)
    tok = synthetic_tokenizer()
    cfg = cfg_for(form, tok)
    model = build(cfg, tok, batch)
    st = TrainState(model, lr=LR)
    named = dict(model.named_parameters())

    g = torch.Generator().manual_seed(7)
    pool = [torch.randint(0, cfg.vocab_size, (batch, SEQ), generator=g) for _ in range(POOL)]

    frozen = sorted(n for n, p in named.items() if not p.requires_grad)
    frozen_sum0 = {n: float(named[n].float().sum()) for n in frozen}
    six = {
        f"layers.{l}.attn.indexer.{w}.weight"
        for l in cfg.index_source_layers
        for w in ("wq_b", "weights_proj")
    }
    six0 = {n: named[n].detach().float().clone() for n in six}

    losses, gaps, mv, all_loss = [], [], [], []
    t0 = time.time()
    for i in range(steps):
        loss = train_step(model, pool[i % POOL], None, state=st)
        st.refresh_bf16()
        if (i + 1) % POOL == 0:
            # ste keeps differentiable attention activations live; collect each cycle so peak
            # does not creep into jetsam on a shared laptop.
            gc.collect()
        v = float(loss)
        if not (v == v) or v in (float("inf"), float("-inf")):
            print(json.dumps({"form": form, "died_at": i, "loss": v}))
            sys.exit(1)
        all_loss.append(v)
        if i < 5 or (i + 1) % 10 == 0:
            losses.append([i + 1, v])
            with torch.no_grad():
                d = (named[PROBE_BF16].float() - st.master[PROBE_BF16].float()).abs()
                gaps.append([i + 1, float(d.max()), float(d.mean())])
        p = st.master[PROBE_BF16]
        s = st.optimizer.state.get(p)
        if s is not None and ((i + 1) % 10 == 0 or i == steps - 1):
            m_, v_ = s["exp_avg"], s["exp_avg_sq"]
            mv.append(
                [
                    i + 1,
                    float(m_.abs().max()),
                    float(v_.max()),
                    float((v_ > 0).float().mean()),
                    bool(torch.isfinite(m_).all() and torch.isfinite(v_).all()),
                ]
            )

    ncycle = steps // POOL
    cyc = [sum(all_loss[c * POOL : (c + 1) * POOL]) / POOL for c in range(ncycle)]
    first_c, last_c = all_loss[:POOL], all_loss[(ncycle - 1) * POOL : ncycle * POOL]
    wins = sum(1 for a, b in zip(first_c, last_c) if b < a)
    frozen_moved = [n for n in frozen if float(named[n].float().sum()) != frozen_sum0[n]]
    six_drift = {n: float((named[n].float() - six0[n]).abs().max()) for n in six}
    return {
        "form": form,
        "steps": steps,
        "batch": batch,
        "seq": SEQ,
        "pool": POOL,
        "lr": LR,
        "seed": SEED,
        "vocab_size": cfg.vocab_size,
        "sec": round(time.time() - t0, 1),
        "loss_first": losses[0][1],
        "loss_last": losses[-1][1],
        "loss_min": min(v for _, v in losses),
        "cycle_means": [round(x, 4) for x in cyc],
        "cycle_first_mean": round(cyc[0], 4),
        "cycle_last_mean": round(cyc[-1], 4),
        "cycle_delta": round(cyc[-1] - cyc[0], 4),
        "same_batch_wins": f"{wins}/{POOL}",
        "loss_curve": losses,
        "point_min": round(min(v for _, v in losses), 4),
        "point_max": round(max(v for _, v in losses), 4),
        "gap_first_max": gaps[0][1],
        "gap_last_max": gaps[-1][1],
        "gap_last_mean": gaps[-1][2],
        "gap_series": gaps[:: max(1, len(gaps) // 6)],
        "mv_last": mv[-1] if mv else None,
        "frozen_moved": frozen_moved,
        "n_frozen": len(frozen),
        "six_drift": {k: round(v, 8) for k, v in sorted(six_drift.items())},
        "bad_master": [n for n, p in st.master.items() if not torch.isfinite(p).all()],
        "bad_run": [n for n, p in model.named_parameters() if not torch.isfinite(p).all()],
        "nan_or_inf": False,
    }


def main(argv):
    steps = int(argv[1]) if len(argv) > 1 else 160
    batch = int(argv[2]) if len(argv) > 2 else 1
    outdir = argv[3] if len(argv) > 3 else os.path.join("runs", "longrun")
    form = argv[4] if len(argv) > 4 else "off"
    os.makedirs(outdir, exist_ok=True)
    out = run(form, steps, batch)
    path = os.path.join(outdir, f"v41f_longrun_{form}.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "form",
                    "cycle_first_mean",
                    "cycle_last_mean",
                    "cycle_delta",
                    "same_batch_wins",
                    "gap_last_max",
                    "frozen_moved",
                    "n_frozen",
                )
            }
        )
    )
    print(path)


if __name__ == "__main__":
    main(sys.argv)
