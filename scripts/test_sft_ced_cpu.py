#!/usr/bin/env python3
"""CED code-SFT smoke gate driven through the REAL sft_math.py loop on CPU.

1e's order 2026-09-25: before the post-pretraining SFT can launch, a small CED checkpoint
plus the first rows of an SFT pack must run ~20 steps in sft_math.py, loss must go DOWN,
and the completion-only mask / eos supervision must be asserted. This is that gate. It also
pins the two things the same PR added to make the user-ordered SFT recipe possible:

  * lr_decay="linear" + warmup_frac: 5% linear warmup then linear decay to ZERO (the
    pretraining cosine default must stay byte-identical),
  * Cfg.kind="sft" written into every SFT checkpoint: score_matrix.classify and the RL
    resume gate read that marker, continuation SFT is not inferable from the rest of cfg,
  * epoch-boundary checkpoints (.epoch1/.epoch2): the user order scores each epoch end.

The heavy run builds a real d512/L4/h4 CED checkpoint WITH MoE (8 experts exercises
assert_moe_matches_ckpt and the CPU sort-and-loop grouped fallback) and a synthetic pack in
the real pack schema (prompt masked -100, completion supervised, every supervised run ends
on eos id 1, pad eos), then drives sft_math.main() in a fresh subprocess: 48 rows, batch 4,
2 epochs = 24 steps, linear LR. CI has no liger/tokenizer/flash_attn; FLCE and the
tokenizer are substituted the same way test_ced_resume_cpu.py does.

    python scripts/test_sft_ced_cpu.py
Exit 0 = loss decreases, mask/eos assertions hold, kind + epoch saves correct. The
model-free --selftest pins the LR schedule and the source markers.
"""

import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NROWS, BATCH, EPOCHS, SEQ = 48, 4, 2, 64  # SEQ = model context; pack rows are SEQ+1 wide
VOCAB, EOS = 512, 1
PROMPT_LEN, COMP_LEN = 6, 8  # per packed doc; 2 docs per row

if len(sys.argv) > 3 or (len(sys.argv) == 2 and sys.argv[1] != "--selftest"):
    raise SystemExit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")


# ── worker side ────────────────────────────────────────────────────────────────
def _worker():
    torch.set_num_threads(1)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    tmp = os.environ["SFT_CED_TMP"]
    sys.path.insert(0, ROOT)
    os.chdir(tmp)

    import train  # noqa: E402
    import sft_math  # noqa: E402

    # ── small CED + MoE checkpoint ────────────────────────────────────────────
    train.Cfg.d = 512
    train.Cfg.heads = 4
    train.Cfg.layers = 4
    train.Cfg.ffn_hidden = 512
    train.Cfg.vocab = VOCAB
    train.Cfg.vocab_real = VOCAB
    train.Cfg.num_id = VOCAB - 1
    train.Cfg.seq = SEQ
    train.Cfg.csa = True
    train.Cfg.csa2 = True
    train.Cfg.ced = True
    train.Cfg.ced_enc_layers = 2
    train.Cfg.ced_kc_norm = True
    train.Cfg.rope_dims = 64
    train.Cfg.n_swa_only_layers = 0
    train.Cfg.attn_res = False
    train.Cfg.attn_every = 1  # no KDA layer (fla absent in CI); every block is CSA2
    train.Cfg.doc_mask = True
    train.Cfg.compile = False
    train.Cfg.grad_ckpt = False
    train.Cfg.fone = False
    train.Cfg.mix = None  # no corpus fingerprint scan in the CPU smoke
    train.Cfg.moe_experts = 8
    train.Cfg.moe_top_k = 3
    train.Cfg.moe_shared = 1
    train.Cfg.moe_expert_ffn = 128  # (3+1)*128 == ffn_hidden 512: active-width parity
    train.Cfg.moe_layers = "0-3"
    torch.manual_seed(42)
    model = train.HybridLM(train.Cfg)
    ckpt_path = os.path.join(tmp, "base.pt")
    train.save_checkpoint(ckpt_path, model.state_dict(), train.Cfg, "sftcedsmoke")

    # ── synthetic pack in the real schema: 2 docs per row ─────────────────────
    # Doc layout within a row: [PROMPT_LEN prompt][COMP_LEN completion incl eos],
    # twice, then eos pad to SEQ+1. labels[j] = input_ids[j] on completion positions,
    # -100 elsewhere. sft_math feeds X=ids[:,:-1], Y=labels[:,1:], so the supervised
    # eos at the completion's last input column is predicted by the step before it.
    ids = torch.full((NROWS, SEQ + 1), EOS, dtype=torch.int32)
    lab = torch.full((NROWS, SEQ + 1), -100, dtype=torch.int32)
    g = torch.Generator().manual_seed(7)
    doc = PROMPT_LEN + COMP_LEN
    for r in range(NROWS):
        body = torch.randint(10, VOCAB - 1, (2 * doc,), generator=g)
        ids[r, : 2 * doc] = body
        for d_i in range(2):
            c0 = d_i * doc + PROMPT_LEN
            ids[r, c0 + COMP_LEN - 1] = EOS  # completion ends on eos
            lab[r, c0 : c0 + COMP_LEN] = ids[r, c0 : c0 + COMP_LEN]
    # Self-check the pack before training: every supervised run ends on eos.
    for r in range(NROWS):
        m = lab[r] != -100
        starts = torch.where(m[1:] & ~m[:-1])[0] + 1
        ends = torch.where(m[:-1] & ~m[1:])[0]
        if m[0]:
            starts = torch.cat([torch.tensor([0]), starts])
        if m[-1]:
            ends = torch.cat([ends, torch.tensor([SEQ])])
        assert len(starts) == len(ends) == 2, (r, len(starts), len(ends))
        assert all(int(ids[r, b]) == EOS for b in ends.tolist()), "supervised run missing eos"
    pack_path = os.path.join(tmp, "pack.pt")
    torch.save({"input_ids": ids, "labels": lab, "vocab_id": "sftcedsmoke"}, pack_path)

    # ── CI substitutes ────────────────────────────────────────────────────────
    train.ROOT = tmp  # RunLog writes tmp/runs
    os.makedirs(os.path.join(tmp, "runs"), exist_ok=True)
    train.TOK_PATH = os.path.join(tmp, "tokenizer.json")
    train.VOCAB_ID = "sftcedsmoke"
    torch.Tensor.pin_memory = lambda self, *a, **k: self

    losses = []

    class FakeFLCE:
        def __init__(self, ignore_index=-100, softcap=15.0):
            self.ignore_index = ignore_index

        def __call__(self, weight, hidden, targets):
            t = targets.reshape(-1)
            # The mask must reach the loss: every batch has masked AND supervised positions,
            # and every batch supervises an eos. A pack builder regression that labelled the
            # eos -100 or forgot prompt masking reddens here, not in a downstream eval.
            assert (t == self.ignore_index).any(), "batch has no masked prompt positions"
            assert (t == EOS).any(), "batch supervises no eos"
            logits = hidden.float() @ weight.float().T
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), t, ignore_index=self.ignore_index
            )
            losses.append(float(loss.item()))
            return loss

    sft_math.LigerFusedLinearCrossEntropyLoss = FakeFLCE

    out_path = os.path.join(tmp, "sft_out.pt")
    old = sys.argv
    sys.argv = [
        "sft_math.py",
        "--resume",
        ckpt_path,
        "--sft_path",
        pack_path,
        "--out",
        out_path,
        "--batch",
        str(BATCH),
        "--epochs",
        str(EPOCHS),
        "--lr_scale",
        "0.5",
        "--lr_decay",
        "linear",
        "--warmup_frac",
        "0.05",
        "--no_fp8",
        "--allow_unstamped_pack",
        "--save_every",
        "1000",  # only epoch saves land
    ]
    try:
        sft_math.main()
    finally:
        sys.argv = old

    steps_per_epoch = NROWS // BATCH
    rec = {
        "losses": losses,
        "steps_per_epoch": steps_per_epoch,
        "epoch1": os.path.exists(out_path + ".epoch1"),
        "epoch2": os.path.exists(out_path + ".epoch2"),
        "final": os.path.exists(out_path),
    }
    e1 = torch.load(out_path + ".epoch1", map_location="cpu", weights_only=False)
    rec["epoch1_step"] = int(e1["step"])
    rec["kind"] = e1["cfg"].get("kind")
    rec["ced"] = (e1["cfg"].get("ced"), e1["cfg"].get("ced_enc_layers"), e1["cfg"].get("ced_kc_norm"))
    rec["vocab_id"] = e1.get("vocab_id")
    torch.save(rec, os.path.join(tmp, "rec.pt"))


def main():
    tmp = tempfile.mkdtemp(prefix="sft_ced_gate_")
    env = dict(os.environ)
    env.update(
        {
            "SFT_CED_TMP": tmp,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "TORCHDYNAMO_DISABLE": "1",
        }
    )
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--selftest", "worker"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    if r.returncode != 0:
        raise SystemExit(f"SFT CED smoke FAILED:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    rec = torch.load(os.path.join(tmp, "rec.pt"), map_location="cpu", weights_only=False)
    losses = rec["losses"]
    spe = EPOCHS * (NROWS // BATCH)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else '  <- ' + detail}")
        if not ok:
            fails.append(name)

    check(f"step count {spe} (got {len(losses)})", len(losses) == spe)
    head = sum(losses[:3]) / 3
    tail = sum(losses[-3:]) / 3
    check(
        f"loss decreases over the run ({head:.3f} -> {tail:.3f})",
        tail < head - 0.05,
        f"{head:.4f} -> {tail:.4f}",
    )
    check("epoch1 checkpoint saved", rec["epoch1"])
    check("epoch2 checkpoint saved", rec["epoch2"])
    check("final checkpoint saved", rec["final"])
    check(f"epoch1 at step {NROWS // BATCH} (got {rec['epoch1_step']})", rec["epoch1_step"] == NROWS // BATCH)
    check("cfg kind = sft marker", rec["kind"] == "sft", repr(rec["kind"]))
    check("CED cfg survives the SFT save (1, 2, 1)", rec["ced"] == (1, 2, 1), repr(rec["ced"]))
    check("vocab_id carried", rec["vocab_id"] == "sftcedsmoke", repr(rec["vocab_id"]))
    if fails:
        print(f"\nSFT CED smoke: {len(fails)} FAIL")
        return 1
    print(
        f"\nSFT CED smoke gate OK: {spe} steps, loss {head:.3f}->{tail:.3f}, "
        "mask/eos held, kind + epoch saves correct"
    )
    return 0


def _selftest():
    """Model-free: the linear schedule shape, cosine unchanged, and the source markers."""
    import types

    sys.path.insert(0, ROOT)
    from train import lr_mult

    def cfg(**kw):
        base = dict(warmup=20, warmdown=0.65, final_lr_frac=0.05, warmup_frac=None, lr_decay="cosine")
        base.update(kw)
        return types.SimpleNamespace(**base)

    # User order: 5% linear warmup, then linear to ZERO over every remaining step.
    lin = cfg(lr_decay="linear", warmup_frac=0.05)
    total = 200
    assert abs(lr_mult(0, total, lin) - 0.1) < 1e-12
    assert abs(lr_mult(9, total, lin) - 1.0) < 1e-12
    assert abs(lr_mult(10, total, lin) - 1.0) < 1e-12
    assert abs(lr_mult(total, total, lin)) < 1e-12, "linear must decay to exactly zero"
    assert 0 < lr_mult(total - 1, total, lin) < 0.01
    # Monotone non-increasing after warmup (warmup itself ramps up).
    ms = [lr_mult(s, total, lin) for s in range(total + 1)]
    wu = round(0.05 * total)
    assert all(b <= a + 1e-12 for a, b in zip(ms[wu:], ms[wu + 1 :]))
    assert ms[wu] == 1.0 and ms[0] < ms[1] <= 1.0

    # The pretraining cosine path is unchanged: absolute warmup, flat until warmdown start,
    # floor final_lr_frac at the end (old lr_mult behaviour verbatim).
    cos = cfg()
    t2 = 200
    assert abs(lr_mult(0, t2, cos) - 1 / 20) < 1e-12
    assert lr_mult(19, t2, cos) == 1.0 and lr_mult(30, t2, cos) == 1.0
    assert abs(lr_mult(t2, t2, cos) - 0.05) < 1e-12

    # Source markers the rest of the chain depends on.
    src = open(os.path.join(ROOT, "sft_math.py"), encoding="utf-8").read()
    assert 'Cfg.kind = "sft"' in src, "sft_math must stamp Cfg.kind on every save"
    assert '".epoch"' in src or ".epoch{ep + 1}" in src, "epoch-boundary save missing"
    assert "--warmup_frac" in src and "--lr_decay" in src
    # Option B wiring (1e 2026-09-25): the SFT line must cast bf16 on --no_fp8 and pass the
    # flag to Cfg so build_optimizers builds the SR path. A line deleted here is a silent revert
    # to the fp32-weights OOM path or to frozen round-to-nearest.
    assert "--stochastic_round" in src, "sft_math must expose --stochastic_round"
    assert "Cfg.stochastic_round = args.stochastic_round" in src, (
        "the SR flag must reach Cfg/build_optimizers")
    train_src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    assert "self._rounder.round(W)" in train_src, "Muon must Bernoulli-round its fp32 write"
    assert "class StochasticAdamW" in train_src, "embed/scalar groups need the SR AdamW"
    print("sft_ced_cpu selftest: linear-to-zero + cosine-unchanged + kind/epoch + SR wiring OK")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest" and sys.argv[2] == "worker":
        _worker()
    elif len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        sys.exit(main())
