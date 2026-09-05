#!/usr/bin/env python3
"""Per-token loss correlation between two checkpoints on the SAME held-out tokens.

Answers one design question: is the paired-BPB estimator load-bearing? Pairing helps by
sqrt(2(1-corr)) / sqrt(2), so corr is the whole lever. corr 0.95 makes the paired SE 4.5x
tighter than unpaired; corr 0.5 makes it 1.4x and the design is not worth its complexity.

CPU-ONLY BY CONSTRUCTION. --device defaults to cpu and the script refuses cuda unless
--allow_cuda is passed, because the cards belong to another team by user order (2026-09-05)
and a probe that quietly takes one is the failure this repo has paid for twice.

THE VAL ROWS ARE READ THE WAY TRAINING READ THEM, not resampled: train.py:1989 takes
val = seqs[:n_val] per domain with n_val = min(max(1, int(len(seqs) * val_frac)),
val_rows_max), a deterministic prefix. So the tokens both arms validated on are recoverable
from the caches without rebuilding the mix -- which matters twice over: rebuilding would
read 35 GB per domain (the co-residency guard's population) and would also let a seed or
cache change silently substitute different tokens for the ones the arms actually scored.
mmap reads only the prefix.
"""
import argparse
import json
import math
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

SEQ = 4096


def val_rows(domain, cache_path, val_frac, val_rows_max, seq=SEQ):
    """The exact val prefix train.py would have taken for this domain, via mmap.

    Returns (rows [n_val, seq+1] int64, n_rows_total). Reads the prefix only.
    """
    stream = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    n_rows = stream.numel() // (seq + 1)
    n_val = min(max(1, int(n_rows * val_frac)), val_rows_max)
    flat = stream[: n_val * (seq + 1)]
    return flat.view(n_val, seq + 1).long(), n_rows, n_val


@torch.no_grad()
def token_losses(model, X, Y, batch, device):
    """Per-token cross-entropy, flattened. No reduction: the correlation is per token."""
    out = []
    for j in range(0, len(X), batch):
        xb = X[j : j + batch].to(device)
        yb = Y[j : j + batch].to(device)
        logits = model(xb)
        logits = logits[0] if isinstance(logits, tuple) else logits
        ls = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none"
        )
        out.append(ls.cpu())
        print(f"    rows {j + len(xb)}/{len(X)}", flush=True)
    return torch.cat(out)


def paired_stats(la, lb, rows_per_token):
    """Correlation, the difference SD, and what each buys for the paired design.

    rows_per_token maps each token to its document (row), because the SE of a mean over
    tokens is not sqrt(var/n) when tokens within a document are correlated -- the same
    clustering correction the api_cloze probe carries.
    """
    n = la.numel()
    ma, mb = la.mean().item(), lb.mean().item()
    sa, sb = la.std(unbiased=True).item(), lb.std(unbiased=True).item()
    cov = ((la - ma) * (lb - mb)).mean().item() * n / (n - 1)
    corr = cov / (sa * sb)
    d = la - lb
    sd_diff = d.std(unbiased=True).item()
    # Clustered SE of the mean difference: group by row, average within, then SE over rows.
    k = int(rows_per_token.max().item()) + 1
    sums = torch.zeros(k, dtype=torch.float64).index_add_(
        0, rows_per_token, d.to(torch.float64)
    )
    cnts = torch.zeros(k, dtype=torch.float64).index_add_(
        0, rows_per_token, torch.ones_like(d, dtype=torch.float64)
    )
    keep = cnts > 0
    per_row = (sums[keep] / cnts[keep])
    se_cluster = (per_row.std(unbiased=True) / math.sqrt(per_row.numel())).item()
    se_naive = sd_diff / math.sqrt(n)
    sd_indep = math.sqrt(sa * sa + sb * sb)
    # sd_diff == 0 means the two arms scored every token identically -- the same weights, or
    # the same checkpoint passed twice. Pairing's gain is then unbounded, and inf is the
    # honest value: it says "no difference to resolve", where a fallback of 1.0 would read as
    # "pairing buys nothing" and a NaN would propagate silently into a design decision.
    gain = sd_indep / sd_diff if sd_diff > 0 else float("inf")
    return {
        "n_tokens": n,
        "n_rows": int(keep.sum().item()),
        "mean_a": ma,
        "mean_b": mb,
        "mean_diff": d.mean().item(),
        "sd_a": sa,
        "sd_b": sb,
        "corr": corr,
        "sd_diff": sd_diff,
        "sd_diff_if_independent": sd_indep,
        "se_diff_naive": se_naive,
        "se_diff_cluster": se_cluster,
        "deff": (se_cluster / se_naive) ** 2 if se_naive > 0 else float("nan"),
        "pairing_gain_vs_unpaired": gain,
    }


def claim_own_card(name, device):
    """Claim the card THIS process is on, from inside, after the device is really held.

    Returns (claimed: bool, message). Never raises: a probe that dies because its bookkeeping
    failed is worse than one that reports the card unclaimed.

    WHY IN HERE RATHER THAN IN THE LAUNCHER, which is the correction to my first attempt
    (2026-09-05, the card-7 run that ran unclaimed for its whole life):

    - card_claim refuses a SHELL pid by design (card_claim.py:659 `_argv0_is_shell`): the claim
      must name the process that dies WITH the job, and a shell either execs away or outlives it.
      My launcher passed its own pid and was refused ~60 times, with the refusal PRINTING the
      python pid to use instead each time.
    - card_claim also refuses a pid holding no GPU device fd (:697), and on the pod the first
      non-shell descendant appears at t=0.11s holding ZERO fds while the first fd appears at
      t=1.33s. An outside claimer has to poll across that window; this process does not have to
      guess -- it calls after torch.cuda.set_device, so the fd exists by construction.
    - A launcher claiming "its python child" has to pick one. Mine had two python descendants:
      this probe and the card_claim invocation itself, which duly appeared in card_claim's own
      "claim one of these" list.

    The card index is read back from torch rather than from an argument, so the claim names the
    card actually in use even if CUDA_VISIBLE_DEVICES maps it differently.
    """
    if not str(device).startswith("cuda"):
        return False, "not a cuda device; nothing to claim"
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from card_claim import acquire

        # The PHYSICAL index, which is what a claim names: CUDA_VISIBLE_DEVICES=7 makes
        # torch's cuda:0 the physical card 7, and claiming "0" there would protect the wrong
        # card while this job ran on an unclaimed one -- the orphan-behind-a-healthy-claim
        # shape card_claim.py:695 exists to refuse.
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        local = torch.cuda.current_device()
        if vis.strip():
            phys = [p.strip() for p in vis.split(",") if p.strip()]
            card = phys[local] if local < len(phys) else str(local)
        else:
            card = str(local)
        ok, msg = acquire(
            "e1_arm_token_corr", [int(card)], pid=os.getpid(), require_device=True,
            note="claimed from inside the probe after torch.cuda.set_device",
        )
        return ok, msg
    except Exception as e:  # noqa: BLE001 -- see the docstring: never kill the run over this
        return False, f"claim attempt raised {type(e).__name__}: {e}"


def release_own_card():
    """Release this probe's claim. Never raises, for the same reason as claim_own_card."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from card_claim import release

        return release("e1_arm_token_corr")
    except Exception as e:  # noqa: BLE001
        return False, f"release raised {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    # NOT required=True: --selftest must run without checkpoints, and argparse enforces
    # required arguments before any code sees the namespace, so a required --ckpt_a makes
    # `--selftest` exit 2 with a usage message rather than testing anything.
    ap.add_argument("--ckpt_a")
    ap.add_argument("--ckpt_b")
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_200m_8b.json"))
    ap.add_argument("--domain", default="code_py_starcoder")
    ap.add_argument("--cache", default=None, help="override the domain cache path")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--rows", type=int, default=64, help="val rows to score (cost is linear)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--val_rows_max", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow_cuda", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    if not a.ckpt_a or not a.ckpt_b:
        sys.exit("--ckpt_a and --ckpt_b are both required to measure a correlation")

    if "cuda" in a.device and not a.allow_cuda:
        sys.exit(
            "REFUSING --device cuda without --allow_cuda: the cards are assigned to another "
            "team by user order (2026-09-05) and runs/card_assignment.json is the authority, "
            "not an idle nvidia-smi row. This probe is designed to run on CPU."
        )

    cache = a.cache or f"/data00/tokens_{a.domain}.pt"
    rows, n_rows_total, n_val = val_rows(a.domain, cache, a.val_frac, a.val_rows_max)
    take = min(a.rows, len(rows))
    rows = rows[:take]
    X, Y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    # Token -> row map for the clustered SE.
    rpt = torch.arange(take).repeat_interleave(SEQ)
    print(
        f"{a.domain}: cache holds {n_rows_total} rows, train's val prefix is {n_val}, "
        f"scoring {take} rows = {take * SEQ} tokens",
        flush=True,
    )

    losses = {}
    claimed = False
    try:
        for tag, path in (("a", a.ckpt_a), ("b", a.ckpt_b)):
            print(f"  loading {path}", flush=True)
            model, cfg = load_checkpoint(path, device=a.device)
            load_tokenizer(a.tokenizer, cfg)  # cross-checks vocab_real then vocab_id
            model.eval()
            # CLAIM AFTER THE FIRST MODEL IS RESIDENT, not before: card_claim refuses a pid
            # holding no GPU device fd, and the fd appears when the weights reach the device.
            # Claiming here rather than in the launcher is the 2026-09-05 correction -- see
            # claim_own_card. Unclaimed is REPORTED, not fatal: the measurement is still valid,
            # and a probe that refuses to run because bookkeeping failed wastes the card window.
            if not claimed:
                claimed, cmsg = claim_own_card("e1_arm_token_corr", a.device)
                print(f"  card claim: {'held' if claimed else 'NOT HELD'} -- {cmsg}", flush=True)
                if not claimed:
                    print("  WARNING: running UNCLAIMED. Another job reading card_claim will see "
                          "this card as free while it is in use.", flush=True)
            print(f"  scoring {tag}", flush=True)
            losses[tag] = token_losses(model, X, Y, a.batch, a.device)
            del model
    finally:
        if claimed:
            ok, rmsg = release_own_card()
            print(f"  card release: {'ok' if ok else 'FAILED'} -- {rmsg}", flush=True)

    st = paired_stats(losses["a"], losses["b"], rpt)
    st.update(
        {
            "ckpt_a": a.ckpt_a,
            "ckpt_b": a.ckpt_b,
            "domain": a.domain,
            "rows_scored": take,
            "val_prefix_rows": n_val,
            "cache_rows": n_rows_total,
            "device": a.device,
        }
    )
    print(json.dumps(st, indent=1))
    print()
    print(f"corr = {st['corr']:.4f}")
    print(f"sd of the per-token difference = {st['sd_diff']:.4f} nat "
          f"(vs {st['sd_diff_if_independent']:.4f} if the arms were independent)")
    print(f"pairing buys {st['pairing_gain_vs_unpaired']:.2f}x on the SE")
    print(f"SE(mean diff): naive {st['se_diff_naive']:.5f}, "
          f"document-clustered {st['se_diff_cluster']:.5f}, deff {st['deff']:.1f}")
    print("The CLUSTERED SE is the one a design decision uses: tokens within a document are "
          "not independent, and the naive SE understates by sqrt(deff).")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
    return 0


def _selftest():
    torch.manual_seed(0)
    n, k = 4096, 8
    rpt = torch.arange(k).repeat_interleave(n // k)

    # 1. IDENTICAL ARMS: corr 1, zero difference, and the gain is infinite rather than a
    #    quiet 1.0 -- a design whose two arms are the same model has no difference to
    #    measure, and reporting a finite gain there would hide that.
    x = torch.randn(n).abs()
    st = paired_stats(x, x.clone(), rpt)
    assert abs(st["corr"] - 1.0) < 1e-6, st["corr"]
    assert st["sd_diff"] == 0.0, st["sd_diff"]
    assert math.isinf(st["pairing_gain_vs_unpaired"]), st["pairing_gain_vs_unpaired"]

    # 2. INDEPENDENT ARMS: pairing buys nothing (gain 1.0), which is the null the design
    #    argument must beat. Same variance both sides so the algebra is checkable by hand.
    y = torch.randn(n)
    z = torch.randn(n)
    st = paired_stats(y, z, rpt)
    assert abs(st["corr"]) < 0.06, st["corr"]
    assert abs(st["pairing_gain_vs_unpaired"] - 1.0) < 0.06, st["pairing_gain_vs_unpaired"]

    # 3. THE GAIN IS sqrt(2/(2(1-corr))) AT EQUAL VARIANCE, checked against the closed form
    #    at a known correlation rather than asserted from the definition.
    g = torch.randn(n)
    for rho in (0.5, 0.9, 0.99):
        h = rho * g + math.sqrt(1 - rho * rho) * torch.randn(n)
        st = paired_stats(g, h, rpt)
        want = 1.0 / math.sqrt(2 * (1 - st["corr"]) / 2)
        assert abs(st["pairing_gain_vs_unpaired"] - want) / want < 0.02, (rho, st)

    # 4. CLUSTERING INFLATES THE SE OF THE DIFFERENCE. A difference that is constant within
    #    a document and varies across documents has deff ~ tokens-per-document; the naive SE
    #    would divide by 4096 tokens when the design really has 8 independent units.
    per_row = torch.tensor([0.5, -0.4, 0.3, -0.2, 0.1, -0.1, 0.2, -0.3])
    d = per_row.repeat_interleave(n // k)
    base = torch.randn(n)
    st = paired_stats(base + d, base, rpt)
    assert st["deff"] > 100, f"clustering not detected: deff {st['deff']}"
    # the clustered SE must be the SD of the 8 row means over sqrt(8)
    want = (per_row.std(unbiased=True) / math.sqrt(k)).item()
    assert abs(st["se_diff_cluster"] - want) < 1e-5, (st["se_diff_cluster"], want)

    # 5. NO CLUSTERING -> deff ~ 1. The correction must not invent a design effect where
    #    the data has none, or every SE it reports is inflated.
    st = paired_stats(torch.randn(n), torch.randn(n), rpt)
    assert 0.5 < st["deff"] < 2.0, st["deff"]

    # 6. THE CUDA REFUSAL IS IN main() AND READS --allow_cuda. Checked by source, because
    #    calling main() needs two checkpoints on disk. The cards belong to another team;
    #    a probe that takes one silently is the defect, not the wrong number.
    import inspect

    src = inspect.getsource(main)
    assert 'if "cuda" in a.device and not a.allow_cuda:' in src, (
        "the CPU-only guard is gone or its condition changed"
    )
    assert "REFUSING --device cuda" in src

    # 7. val_rows TAKES A PREFIX AND CAPS IT, matching train.py:1989. Off-by-one here would
    #    score tokens the arms never validated on while still printing a correlation.
    class _Stream:
        def __init__(self, n_rows):
            self.t = torch.arange(n_rows * (SEQ + 1), dtype=torch.int32)

    real_load = torch.load
    try:
        torch.load = lambda *_, **__: _Stream(1000).t
        rows, total, n_val = val_rows("x", "unused", 0.05, 5000)
        assert total == 1000 and n_val == 50, (total, n_val)
        assert rows.shape == (50, SEQ + 1), rows.shape
        assert rows[0, 0].item() == 0 and rows[1, 0].item() == SEQ + 1
        # the cap binds before the fraction on a large domain
        torch.load = lambda *_, **__: _Stream(200000).t
        _, total, n_val = val_rows("x", "unused", 0.05, 5000)
        assert total == 200000 and n_val == 5000, (total, n_val)
        # and at least one row is kept on a tiny domain
        torch.load = lambda *_, **__: _Stream(3).t
        _, total, n_val = val_rows("x", "unused", 0.05, 5000)
        assert n_val == 1, n_val
    finally:
        torch.load = real_load

    # 8. THE CLAIM NAMES THIS PROCESS, AND THE PHYSICAL CARD. Both halves failed on 2026-09-05:
    #    the card-7 run claimed from its launcher shell, was refused ~60 times, and ran unclaimed
    #    for its whole life. card_claim refuses a shell pid (:659) and a pid holding no device fd
    #    (:697), so the only pid that can satisfy both is one that has already called set_device --
    #    this process. Checked by source plus a stubbed acquire, since a real claim needs a GPU.
    import inspect as _inspect
    import types as _types

    _src = _inspect.getsource(claim_own_card)
    assert "pid=os.getpid()" in _src, (
        "claim_own_card no longer claims THIS process. A launcher's pid is a shell, which "
        "card_claim.py:659 refuses by design, and the card then runs unclaimed."
    )
    assert "require_device=True" in _src, (
        "require_device dropped: a claim on a pid not yet on a card protects nothing, and the "
        "measured window where a pid passes every other test with 0 device fds is 0.11s-1.33s"
    )
    _msrc = _inspect.getsource(main)

    #    ORDER AND RELEASE ARE CHECKED BY RUNNING main(), not by reading its source. My first
    #    version asserted `_msrc.index("load_checkpoint(") < _msrc.index("claim_own_card(")`,
    #    and a mutant that disabled the claim entirely (`if False:`) stayed GREEN -- the text was
    #    still there in the source, in the right order, doing nothing. A source-order assertion
    #    cannot see whether a call executes. So: stub everything expensive, run main(), and record
    #    the sequence of events.
    _events = []

    def _fake_val_rows(domain, cache, vf, vm, seq=SEQ):
        return torch.zeros(2, SEQ + 1, dtype=torch.long), 1000, 50

    def _fake_load_ckpt(path, device="cpu", dtype=None, fone_ok=True):
        _events.append(("load", path))
        return types.SimpleNamespace(eval=lambda: None), types.SimpleNamespace(
            vocab=None, vocab_id=None
        )

    def _fake_losses(model, X, Y, batch, device):
        _events.append(("score", None))
        if _fake_losses.boom:
            raise RuntimeError("synthetic crash mid-scoring")
        return torch.randn(X.numel())

    _fake_losses.boom = False

    _mod = sys.modules[__name__]
    _orig = {k: getattr(_mod, k) for k in
             ("val_rows", "load_checkpoint", "load_tokenizer", "token_losses")}
    _saved_argv, _saved_env2 = sys.argv[:], os.environ.get("CUDA_VISIBLE_DEVICES")
    _saved_mod2, _real_cuda2 = sys.modules.get("card_claim"), torch.cuda
    try:
        _stub2 = types.ModuleType("card_claim")

        def _acq(name, cards, pid=None, require_device=False, note=""):
            _events.append(("claim", cards))
            return True, "stub"

        _stub2.acquire = _acq
        _stub2.release = lambda name, cards=None: (_events.append(("release", None)), (True, "ok"))[1]
        sys.modules["card_claim"] = _stub2

        class _FC:
            @staticmethod
            def current_device():
                return 0

        torch.cuda = _FC()
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        _mod.val_rows = _fake_val_rows
        _mod.load_checkpoint = _fake_load_ckpt
        _mod.load_tokenizer = lambda p, cfg: None
        _mod.token_losses = _fake_losses

        sys.argv = ["arm_token_corr.py", "--ckpt_a", "A", "--ckpt_b", "B",
                    "--device", "cuda:0", "--allow_cuda", "--rows", "2", "--batch", "1"]
        main()
        kinds = [e[0] for e in _events]
        # The claim EXECUTED, once, and after the first load.
        assert kinds.count("claim") == 1, f"claim ran {kinds.count('claim')} time(s): {kinds}"
        assert kinds.index("load") < kinds.index("claim"), kinds
        assert kinds.index("claim") < kinds.index("score"), kinds
        assert kinds[-1] == "release", f"release is not last: {kinds}"
        assert _events[kinds.index("claim")][1] == [7], _events[kinds.index("claim")][1]

        # AND THE RELEASE SURVIVES A CRASH. A release placed after the loop instead of in a
        # finally block passes every assertion above and leaks the card on any failure.
        _events.clear()
        _fake_losses.boom = True
        try:
            main()
        except RuntimeError as e:
            assert "synthetic crash" in str(e)
        else:
            raise AssertionError("the synthetic crash did not propagate")
        assert [e[0] for e in _events][-1] == "release", (
            f"a crash mid-scoring left the card CLAIMED: {[e[0] for e in _events]}"
        )
    finally:
        _fake_losses.boom = False
        for k, v in _orig.items():
            setattr(_mod, k, v)
        sys.argv = _saved_argv
        torch.cuda = _real_cuda2
        sys.modules.pop("card_claim", None)
        if _saved_mod2 is not None:
            sys.modules["card_claim"] = _saved_mod2
        if _saved_env2 is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _saved_env2

    #    The physical-index mapping, exercised rather than read: CUDA_VISIBLE_DEVICES=7 makes
    #    torch's cuda:0 the physical card 7, and claiming "0" would protect a card this job is
    #    not on while it ran on an unclaimed one.
    _seen = {}

    def _fake_acquire(name, cards, pid=None, require_device=False, note=""):
        _seen.update(name=name, cards=cards, pid=pid, require_device=require_device)
        return True, "stub"

    _saved_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    _saved_mod = sys.modules.get("card_claim")
    _real_cuda = torch.cuda
    try:
        _stub = _types.ModuleType("card_claim")
        _stub.acquire = _fake_acquire
        _stub.release = lambda name, cards=None: (True, "stub release")
        sys.modules["card_claim"] = _stub

        class _FakeCuda:
            @staticmethod
            def current_device():
                return 0

        torch.cuda = _FakeCuda()
        for vis, want in (("7", 7), ("2,5", 2), ("", 0)):
            _seen.clear()
            if vis:
                os.environ["CUDA_VISIBLE_DEVICES"] = vis
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            ok, _m = claim_own_card("t", "cuda:0")
            assert ok and _seen["cards"] == [want], (vis, _seen.get("cards"), want)
            assert _seen["pid"] == os.getpid(), _seen.get("pid")
            assert _seen["require_device"] is True
        # A CPU device claims nothing rather than claiming card 0.
        ok, msg = claim_own_card("t", "cpu")
        assert not ok and "nothing to claim" in msg, msg
    finally:
        torch.cuda = _real_cuda
        sys.modules.pop("card_claim", None)
        if _saved_mod is not None:
            sys.modules["card_claim"] = _saved_mod
        if _saved_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _saved_env

    print(
        "arm_corr selftest OK: identical arms give corr 1 and an infinite gain, independent "
        "arms give gain 1.00, the gain matches 1/sqrt(1-corr) at rho 0.5/0.9/0.99, a "
        "within-document difference is caught as deff>100 with the clustered SE equal to the "
        "row means' SE over sqrt(k), unclustered data keeps deff ~1, the val prefix matches "
        "train.py:1989 on three domain sizes (fraction, cap, and the 1-row floor), and the "
        "CPU-only refusal is present in main(). CLAIM: it names os.getpid() with "
        "require_device, sits AFTER the first checkpoint load (no device fd exists before it), "
        "releases in a finally, and maps torch's local index to the PHYSICAL card (CVD=7 claims "
        "7, CVD=2,5 claims 2, unset claims 0) while a cpu device claims nothing"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
