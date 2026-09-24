#!/usr/bin/env python3
"""CED resume-continuity gate driven through the REAL train.py training loop on CPU.

Three properties 1e required (2026-09-24), none covered by tests/v41f/* -- those drive
v41f/master.py, never train.py's loop:

  1. A save_every checkpoint's weights, optimizer state and AdamW step agree with the step
     the file names. This is the 500-mod-1000 defect (snapshot refreshed every 200, saved
     every 500): ckpt_v41_ced_0923.pt.step18500 carried AdamW state step 18400. The
     in-process mutant below restores the two-cadence shape and this gate reds on it.
  2. Resume is transparent: train N steps uninterrupted (A) vs N/2 steps then resume from
     the save for N/2 more (B1+B2) -- per-step train loss and final weights bit-identical.
  3. The row cursor continues the data: B2 consumes exactly A's rows [N/2, N), no skip,
     no repeat, and row_cursor counts match.

Why the plan shuffle is neutralized. build_mix shuffles each phase with a fresh
Generator().manual_seed(Cfg.seed); a cursor-seeded resume subtracts the spent rows and
shuffles the REMAINDER, so perm(N - K) has no prefix relation to the uninterrupted perm(N)
-- the data order genuinely differs and bit-equality cannot hold against an unconstrained
shuffle. The shuffle's own seeding is covered by the de-7 cursor-seed machinery and
test_plan_length; this gate isolates save/resume continuity, so it replaces ONLY the
Generator-based torch.randperm with the identity. Model-init RNG (global generator) is
untouched, and every row is unique so a repeated or skipped row still fails.

Runs the real train.main() in fresh subprocesses (the actual resume shape), 19.5M d512/4
CED model, seq 32, CPU, ~1-2 min. CI installs no liger/tokenizer/flash_attn; all three are
substituted exactly as the CPU path requires. Thread count is pinned for oneDNN
determinism (same finding as the v41f ckpt gate).

    python scripts/test_ced_resume_cpu.py
Exit 0 = all three properties + the mutant reds. Exit 1 = named failure.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, HALF, BATCH, SEQ, NROWS_TR, VOCAB = 8, 4, 2, 32, 16, 512
MUT_INTERVAL = "3"  # refresh every 3, save every 4 -> the save names step 4 but holds step 3

if len(sys.argv) > 3 or (len(sys.argv) == 2 and sys.argv[1] != "--selftest"):
    raise SystemExit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")


# ── worker side ────────────────────────────────────────────────────────────────
def _worker(seg):
    """One training segment. Writes <TMP>/<seg>.pt with losses, consumed token rows, ckpt path."""
    torch.set_num_threads(1)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    tmp = os.environ["CED_RES_TMP"]
    os.environ["AUPAI_TOKEN_CACHE_DIR"] = tmp
    train_src = os.environ.get("CED_RES_TRAIN")
    # The mutated/source tree must win over ROOT on sys.path, so insert ROOT first and the
    # source dir AFTER (later insert -> earlier in sys.path).
    sys.path.insert(0, ROOT)
    if train_src:
        sys.path.insert(0, os.path.dirname(train_src))
    os.chdir(tmp)

    import train  # noqa: E402

    os.makedirs(os.path.join(tmp, "scripts"), exist_ok=True)
    for nm, body in (
        ("shape_audit.py", "raise SystemExit(0)\n"),
        ("env_fp.py", "def env_fingerprint():\n    return 'cedres-env'\n"),
    ):
        p = os.path.join(tmp, "scripts", nm)
        if not os.path.exists(p):
            open(p, "w").write(body)

    n_pool = NROWS_TR + 1  # one row held out for val (val_split_n max(1, ...))

    def fake_tokenizer(texts=None):
        class T:
            def token_to_id(self, s):
                return 1
        return T()

    def fake_seqs(domain, tok, is_main, ddp, workers=1, allow_build=False):
        # UNIQUE deterministic rows: row r token t = 10 + (37r + 101t) mod 490.
        r = torch.arange(n_pool, dtype=torch.int32).view(-1, 1)
        t = torch.arange(SEQ + 1, dtype=torch.int32).view(1, -1)
        return (10 + (37 * r + 101 * t) % 490).to(torch.int32)

    train_losses, seen_ids = [], []

    class FakeFLCE:
        def __init__(self, ignore_index=-100, softcap=15.0):
            self.sc = softcap

        def __call__(self, weight, hidden, targets):
            logits = hidden.float() @ weight.float().T
            logits = torch.tanh(logits / self.sc) * self.sc
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if torch.is_grad_enabled():  # validation runs under no_grad and is not compared
                train_losses.append(float(loss.item()))
            return loss

    RealHybridLM = train.HybridLM

    class RecModel(RealHybridLM):
        def forward(self, x, *a, **k):
            # Record only TRAIN forwards: the epoch-end validate() runs under no_grad and
            # would append the val batch, breaking the 16-row expected-data comparison.
            if torch.is_grad_enabled():
                seen_ids.append(x.detach().cpu().clone())
            return super().forward(x, *a, **k)

    def patch(t):
        t.ROOT = tmp
        t.TOK_PATH = os.path.join(tmp, "tokenizer.json")
        open(t.TOK_PATH, "w").write("{}")
        t.VOCAB_ID = "cedrestest"
        t.build_tokenizer = fake_tokenizer
        t._domain_seqs = fake_seqs
        t._assert_mix_domains = lambda names, cdir, allow_drift=False: {}
        t.LigerFusedLinearCrossEntropyLoss = FakeFLCE
        t.HybridLM = RecModel
        # Identity phase shuffle (file docstring): only the Generator-based randperm calls.
        _real_randperm = torch.randperm

        def ident_randperm(n, generator=None, *a, **k):
            return _real_randperm(n) if generator is None else torch.arange(n)

        torch.randperm = ident_randperm
        # CPU-only gate: pin_memory is a CUDA-host path and raises with no device
        # (Linux) or targets MPS (macOS). On CPU the staged buffers live in host RAM anyway.
        torch.Tensor.pin_memory = lambda self, *a, **k: self
        if seg in ("b1", "mut"):
            # b1 must run under the SAME total_steps as A (the LR schedule depends on it) yet
            # stop after the HALF checkpoint: max_steps=4 would recompute total_steps=4 and
            # diverge the cosine schedule from step warmup+1. Wrap the save so the process
            # exits cleanly right after the step4 file lands.
            real_save = t.save_checkpoint

            def save_and_exit(path, *a, **k):
                real_save(path, *a, **k)
                if path.endswith(f".step{HALF}"):
                    raise SystemExit(0)

            t.save_checkpoint = save_and_exit

    patch(train)
    mix = {
        "total_tokens": NROWS_TR * SEQ,  # budget exactly N steps * BATCH rows
        "domains": {"dom": {"weight": 1.0, "epochs": 1, "anneal": 0.0}},
    }
    json.dump(mix, open(os.path.join(tmp, "mix.json"), "w"))

    common = [
        "train.py", "--mix", "mix.json",
        "--dim", "512", "--layers", "4", "--heads", "4", "--ffn_hidden", "512",
        "--batch", str(BATCH), "--accum", "1", "--seq", str(SEQ), "--vocab", str(VOCAB),
        "--warmup", "1", "--warmdown", "0.0", "--anneal_frac", "0.0",
        "--lr_scale", "1.0", "--save_every", "4", "--no-grad_ckpt",
        "--attn_every", "1", "--csa", "--csa2",
        "--ced", "--ced_enc_layers", "2", "--ced_kc_norm",
        "--rope_dims", "64", "--n_swa_only_layers", "0", "--no-attn_res",
        "--allow_pod_drift", "--seed", "42", "--val_every", "100000",
    ]
    if seg == "a":
        argv = common + ["--name", "a", "--max_steps", str(N)]
    elif seg == "b1":
        # No --max_steps: the natural run is N steps (NROWS_TR/BATCH), so total_steps == A's;
        # patch() exits the process right after the step4 save.
        argv = common + ["--name", "b1"]
    elif seg == "b2":
        argv = common + ["--name", "b2", "--resume",
                         os.path.join(tmp, "ckpt_b1.pt.step4"), "--max_steps", str(N)]
    elif seg == "mut":
        argv = common + ["--name", "m"]
    else:
        raise SystemExit(f"unknown segment {seg}")

    old = sys.argv
    sys.argv = argv
    train.Cfg.doc_mask = False
    train.Cfg.compile = False
    train.Cfg.vocab_real = VOCAB
    train.Cfg.num_id = VOCAB - 1
    try:
        train.main()
    except SystemExit:
        # b1/mut deliberately exit at the step4 save; persist what they recorded first.
        if seg not in ("b1", "mut"):
            raise
    finally:
        sys.argv = old
    torch.save({"losses": train_losses, "ids": seen_ids},
               os.path.join(tmp, f"{seg}.rec.pt"))


# ── orchestrator side ─────────────────────────────────────────────────────────
def _run_worker(tmp, seg, train_src=None):
    env = dict(os.environ)
    env.update({
        "CED_RES_TMP": tmp,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "TORCHDYNAMO_DISABLE": "1",
    })
    if train_src:
        env["CED_RES_TRAIN"] = train_src
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest", seg],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"worker {seg} FAILED:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return torch.load(os.path.join(tmp, f"{seg}.rec.pt"), map_location="cpu", weights_only=False)


def _adamw_step(ck):
    for g in ck["opt"]:
        for st in g["state"].values():
            if "step" in st:
                v = st["step"]
                return int(v.item() if hasattr(v, "item") else v)
    raise AssertionError("no optimizer step recorded")


def _load(tmp, name, step):
    return torch.load(os.path.join(tmp, f"ckpt_{name}.pt.step{step}"),
                      map_location="cpu", weights_only=False)


def main():
    tmp = tempfile.mkdtemp(prefix="ced_resume_gate_")
    try:
        ra, rb1, rb2 = _run_worker(tmp, "a"), _run_worker(tmp, "b1"), _run_worker(tmp, "b2")
        ca4, ca8, cb4, cb8 = (_load(tmp, "a", 4), _load(tmp, "a", 8),
                              _load(tmp, "b1", 4), _load(tmp, "b2", 8))
        fails = []

        def check(name, ok, detail=""):
            print(f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else '  <- ' + detail}")
            if not ok:
                fails.append(name)

        # ── property 1: checkpoint state agrees with its named step ──────────
        check("ckpt a.step4 AdamW step == 4", _adamw_step(ca4) == 4, f"got {_adamw_step(ca4)}")
        check("ckpt a.step8 AdamW step == 8", _adamw_step(ca8) == 8, f"got {_adamw_step(ca8)}")
        check("ckpt b1.step4 AdamW step == 4", _adamw_step(cb4) == 4, f"got {_adamw_step(cb4)}")
        check("ckpt b2.step8 AdamW step == 8", _adamw_step(cb8) == 8, f"got {_adamw_step(cb8)}")
        check("ced cfg survives into b2 ckpt",
              cb8["cfg"].get("ced") == 1 and cb8["cfg"].get("ced_enc_layers") == 2
              and cb8["cfg"].get("ced_kc_norm") == 1,
              f"ced={cb8['cfg'].get('ced')} split={cb8['cfg'].get('ced_enc_layers')} "
              f"kc_norm={cb8['cfg'].get('ced_kc_norm')}")

        # ── property 2: resume is bit-transparent ────────────────────────────
        la, lb1, lb2 = ra["losses"], rb1["losses"], rb2["losses"]
        check(f"loss counts 8/4/4 (got {len(la)}/{len(lb1)}/{len(lb2)})",
              (len(la), len(lb1), len(lb2)) == (8, 4, 4))
        if (len(la), len(lb1), len(lb2)) == (8, 4, 4):
            check("b1 losses bit-equal A[0:4]", la[:4] == lb1,
                  f"A={[round(x,6) for x in la[:4]]} b1={[round(x,6) for x in lb1]}")
            check("b2 losses bit-equal A[4:8]", la[4:] == lb2,
                  f"A={[round(x,6) for x in la[4:]]} b2={[round(x,6) for x in lb2]}")

        def maxabs(s1, s2):
            ks = sorted(set(s1) & set(s2))
            return max((s1[k].float() - s2[k].float()).abs().max().item() for k in ks)

        check("weights b1.step4 bit-equal A.step4", maxabs(ca4["model"], cb4["model"]) == 0.0)
        check("weights b2.step8 bit-equal A.step8", maxabs(ca8["model"], cb8["model"]) == 0.0)

        # ── property 3: cursor continuity, no skip no repeat ─────────────────
        check("cursor a.step4 == 8 rows", ca4.get("row_cursor") == {"dom": 8},
              f"got {ca4.get('row_cursor')}")
        check("cursor b2.step8 == 16 rows", cb8.get("row_cursor") == {"dom": 16},
              f"got {cb8.get('row_cursor')}")
        ida = torch.cat([r.reshape(-1, SEQ) for r in ra["ids"]], 0)
        idb1 = torch.cat([r.reshape(-1, SEQ) for r in rb1["ids"]], 0)
        idb2 = torch.cat([r.reshape(-1, SEQ) for r in rb2["ids"]], 0)
        check("b1 consumed A's rows [0:8]", torch.equal(ida[:8], idb1),
              "resume-half-1 saw different data than the uninterrupted run")
        check("b2 consumed A's rows [8:16] exactly (no skip, no repeat)",
              torch.equal(ida[8:16], idb2) and len(torch.unique(idb2, dim=0)) == 8,
              "resume did not continue the plan where the checkpoint stopped")

        # ── mutant: old two-cadence snapshot shape must FAIL property 1 ───────
        # mdir is a symlink view of the repo with ONLY train.py replaced by a real file,
        # so `from model import ...` and friends resolve and the mutated train's own ROOT
        # (abspath of its __file__) is mdir.
        mdir = os.path.join(tmp, "mut")
        os.makedirs(mdir)
        for entry in os.listdir(ROOT):
            if entry.startswith(".git"):
                continue
            os.symlink(os.path.join(ROOT, entry), os.path.join(mdir, entry))
        src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
        anchor = "                if step % args.save_every == 0:\n"
        assert src.count(anchor) == 1, "mutant anchor (refresh guard) not unique in train.py"
        mutated = src.replace(anchor, f"                if step % {MUT_INTERVAL} == 0:\n", 1)
        mp = os.path.join(mdir, "train.py")
        os.remove(mp)
        open(mp, "w", encoding="utf-8").write(mutated)
        _run_worker(tmp, "mut", train_src=mp)
        cm = _load(tmp, "m", 4)
        check("MUTANT (refresh every 3, save every 4) writes a stale AdamW step",
              _adamw_step(cm) != 4, f"mutant ckpt step read {_adamw_step(cm)}, expected != 4")

        if fails:
            print(f"\n{len(fails)} FAIL: {fails}")
            return 1
        print("CED resume-continuity gate: checkpoint/step agreement, bit-exact resume, "
              "and cursor continuity all OK; the two-cadence mutant reds")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _selftest():
    """Model-free, <0.1s: the property-1 predicate and the mutant anchor. The heavy four-segment
    bit-exact training run is CI-only in ci.yml, like test_p1_train_resume_smoke -- a commit
    hook cannot afford four CPU trainings. This proves what a hook can prove cheaply:
      (a) the step-agreement predicate distinguishes a consistent ckpt (4==4) from a stale
          snapshot (file says 4, AdamW says 3) -- the exact 500-mod-1000 defect;
      (b) the mutant anchor (the single save_every refresh guard) exists uniquely in train.py,
          which is what the full gate's mutation replaces.
    """
    def pred(named, adamw):
        return named == adamw

    assert pred(4, 4) and pred(8, 8), "a consistent checkpoint must pass"
    assert not pred(4, 3), "a file naming step 4 holding AdamW step 3 MUST fail (the defect)"
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    anchor = "                if step % args.save_every == 0:\n"
    assert src.count(anchor) == 1, ("the refresh/save cadence guard must be unique in train.py; "
                                    f"found {src.count(anchor)}")
    import ast
    assigned = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            assigned.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert "GOOD_SAVE_INTERVAL" not in assigned, "the old independent-refresh constant must not return"
    print("ced resume gate selftest: step-agreement predicate + mutant anchor OK")


if __name__ == "__main__":
    # Bare --selftest = fast model-free gate for the commit hook. --selftest <seg> = a worker
    # segment driven by the orchestrator subprocess (the hook never passes a segment).
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    elif len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        _worker(sys.argv[2])
    else:
        sys.exit(main())
