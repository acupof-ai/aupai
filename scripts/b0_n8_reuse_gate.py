#!/usr/bin/env python3
"""Can ckpt_b0_sd_unlooped.pt serve as the N8 A/B's "current" arm, or must both arms run fresh?

THE CONDITION (6e's, and the right one): reusing a pre-flag checkpoint as the unfixed arm is only
sound if the new code with conv_doc_isolated OFF computes what the old code computed -- BITWISE, on
the path the A/B will score on. Not within a tolerance: a tolerance-passing difference is exactly
how a changed compiled graph hides, and the whole point of the reuse is that the current arm's
numbers are the ones already measured.

WHY NOT CITE e1's OWN GATE. 28ae5917's table reports "cu=None, both settings bitwise identical,
0.0". That is a real check and it is not this one. The A/B will score WITH cu (6e ruled
domain_loss passes doc cu), and the flag's branch condition is `self.conv_doc_isolated and cu is
not None` -- so at cu=None the new branch cannot be reached by construction and the check is
vacuous for the reuse question. The reachable-but-off case is flag OFF with cu PASSED, which is
what this measures.

THE REFERENCE IS THE RECORD, NOT A RECOMPUTE. runs/b0_sd_blocks_cu.jsonl was written by
scripts/b0_sd_cu_rescore.py under the PRE-flag code (pod sync 01103bf0, before 28ae5917 landed).
Comparing today's forward against those stored per-block ce_sum values asks the question that
matters -- does the checkpoint still score as it scored -- rather than comparing new code to
itself, which would pass regardless.

VERDICT IS ONE OF TWO THINGS, no middle: every block bit-identical -> reuse; any difference at all
-> both arms fresh, and the size of the difference is reported so the decision is auditable.

USAGE
    CUDA_VISIBLE_DEVICES=5 python3 scripts/b0_n8_reuse_gate.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT = "ckpt_b0_sd_unlooped.pt"
REF = os.path.join(ROOT, "runs", "b0_sd_blocks_cu.jsonl")
REF_KEY = "ckpt_b0_sd_unlooped.pt#cu"


def main():
    import torch  # noqa: PLC0415
    from cache_guard import set_vocab_id  # noqa: PLC0415
    from domain_loss import val_seqs  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    import model as M  # noqa: PLC0415
    from scripts.loader import load_checkpoint  # noqa: PLC0415
    from train import doc_cu_seqlens  # noqa: PLC0415

    ref = None
    with open(REF, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                if r.get("ckpt") == REF_KEY:
                    ref = r
    if ref is None:
        raise SystemExit(f"REFUSING: {REF} has no row for {REF_KEY}; there is nothing to compare "
                         f"against and a self-comparison would pass regardless.")

    mdl, cfg = load_checkpoint(CKPT, dtype=torch.bfloat16)
    mdl = mdl.cuda().eval()
    set_vocab_id(cfg)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    eos = tok.token_to_id("<eos>")

    # THE FLAG MUST BE OFF ON EVERY KDA MIXER, checked rather than assumed from the cfg default.
    # loader.py pins it False for a checkpoint whose cfg lacks the key; this asserts the pin
    # actually reached the modules, because a default that did not propagate would make this whole
    # gate measure the wrong configuration and still print a number.
    flags = [getattr(b.mixer, "conv_doc_isolated", "ABSENT") for b in mdl.blocks
             if not isinstance(b.mixer, M.GatedMLA)]
    print(f"{CKPT}: cfg.conv_doc_isolated={getattr(cfg, 'conv_doc_isolated', '<absent>')!r}, "
          f"per-KDA-mixer flags={set(map(repr, flags))}")
    if set(flags) != {False}:
        raise SystemExit(f"REFUSING: the KDA mixers do not all read conv_doc_isolated False "
                         f"({flags}). This gate would then be measuring the fixed path against a "
                         f"pre-fix record, which is a different question.")
    if not hasattr(mdl.blocks[0].mixer, "conv_doc_isolated"):
        raise SystemExit("REFUSING: the loaded model has no conv_doc_isolated attribute, so this "
                         "pod is running PRE-flag code and the gate would compare the old path to "
                         "itself -- a guaranteed pass that proves nothing.")

    print("\n== flag OFF, cu PASSED, against the stored pre-flag per-block ce_sum values")
    print(f"  {'domain':20s} {'blocks':>6s} {'exact':>6s} {'max|d|':>12s} {'max rel':>10s}")
    worst, worst_rel, worst_at, n_tot, n_exact = 0.0, 0.0, "", 0, 0
    for dom, dref in ref["domains"].items():
        if not isinstance(dref, dict) or "blocks" not in dref:
            continue
        rows = val_seqs(dom, tok)
        if rows is None:
            raise SystemExit(f"REFUSING: {dom} has a reference row but no val rows now, so the "
                             f"comparison would silently cover fewer blocks than the record.")
        x, y = rows[:, :-1].cuda(), rows[:, 1:].cuda()
        if len(x) != len(dref["blocks"]):
            raise SystemExit(f"REFUSING: {dom} has {len(x)} rows now against {len(dref['blocks'])} "
                             f"in the record -- different sequences, not a bitwise question.")
        dmax = drel = 0.0
        exact = 0
        with torch.no_grad():
            for i in range(len(x)):
                xi, yi = x[i:i + 1], y[i:i + 1]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = mdl(xi, cu=doc_cu_seqlens(xi, eos))
                lg = lg[0] if isinstance(lg, tuple) else lg
                ce = float(torch.nn.functional.cross_entropy(
                    lg.float().view(-1, lg.shape[-1]), yi.reshape(-1), reduction="sum"))
                old = dref["blocks"][i]["ce_sum"]
                d = abs(ce - old)
                if d == 0.0:
                    exact += 1
                dmax = max(dmax, d)
                drel = max(drel, d / abs(old) if old else 0.0)
        n_tot += len(x)
        n_exact += exact
        if dmax > worst:
            worst, worst_at = dmax, dom
        worst_rel = max(worst_rel, drel)
        print(f"  {dom:20s} {len(x):6d} {exact:6d} {dmax:12.6e} {drel:10.2e}")

    print(f"\n  {n_exact}/{n_tot} blocks bit-identical; worst |diff| {worst:.6e} "
          f"({worst_at or 'none'}), worst relative {worst_rel:.2e}")
    if n_exact == n_tot:
        print("  VERDICT: REUSE ckpt_b0_sd_unlooped.pt as the N8 current arm. Every block "
              "reproduces its pre-flag value exactly, so the flag-off path is the path that "
              "produced the record and the A/B's two arms differ only in the flag.")
        return 0
    print(f"  VERDICT: BOTH ARMS FRESH. {n_tot - n_exact} of {n_tot} blocks differ, worst "
          f"{worst:.6e} absolute / {worst_rel:.2e} relative. The size does not matter: a checkpoint "
          f"that no longer scores as it scored cannot stand in for the unfixed arm, because the "
          f"A/B's delta would then contain the code change as well as the flag.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
