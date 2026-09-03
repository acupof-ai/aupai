#!/usr/bin/env python3
"""Stage D's two arms rescored with document boundaries declared, in the block schema.

THE QUESTION: the looped arm beat the unlooped one by -0.030937 nat per block (n=576, t -28.50,
9/9 domains) on eval/domain_loss.py's path, which passes NO cu -- so every row was scored with
attention running across document boundaries and KDA state carrying across them. The measured
size of that artifact is -0.081780 nat pooled (scripts/b0_sd_pack_ce.py, nine domains), 7.6x
N2's own delta, and it is DOSE-DEPENDENT in documents per row. The looped arm's two largest wins
were chatml (-0.0625) and chat_qa (-0.0646), which are the two domains with 18 documents per row
and leaks of -0.281 and -0.204. So the per-domain win size tracks the per-domain leak size, and
this script asks whether the win survives when the boundaries are declared.

WHAT IT DOES NOT SETTLE: declaring cu does not isolate the documents -- e1-32 and my own
reproduction both show state crossing the boundary even with cu passed (block 0 KDA, max 3.65
against tol 0.206). So this is an upper bound on how much of the win is path-dependent, not a
clean answer. The clean answer needs e1's conv fix.

WHY A SEPARATE SCRIPT AND NOT --cu ON domain_loss.py: 6e ruled that domain_loss will pass doc cu
and assigned that one-line change to e1 after the model.py fix. Editing it here would collide with
that and would also silently change the path every published number was taken on. This script
writes the SAME row schema into a SEPARATE file, so eval/block_paired.py does the statistics and
no existing record moves.

USAGE
    CUDA_VISIBLE_DEVICES=5 python3 scripts/b0_sd_cu_rescore.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(ROOT, "runs", "b0_sd_blocks_cu.jsonl")
ARMS = [("ckpt_b0_sd_unlooped.pt", None), ("ckpt_b0_sd_looped.pt", (4, 7))]
DOMAINS = ["math_owm_stage2", "en_c4_stage2", "cot", "textbook_30b", "chatml", "chat_qa",
           "zh_web", "code_py_starcoder", "code_py_rp1t"]


def main():
    import torch  # noqa: PLC0415
    from cache_guard import set_vocab_id  # noqa: PLC0415
    from domain_loss import seqs_fp, val_seqs  # noqa: PLC0415
    from loop_wrapper import patch_body  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    from scripts.loader import load_checkpoint  # noqa: PLC0415
    from train import doc_cu_seqlens  # noqa: PLC0415

    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    written = []
    for ck, loop in ARMS:
        mdl, cfg = load_checkpoint(ck, dtype=torch.bfloat16)
        mdl = mdl.cuda().eval()
        set_vocab_id(cfg)
        if loop:
            # AFTER .eval() and on this instance, matching domain_loss.py:671 so the looped arm is
            # scored by the same code path as the unlooped one.
            patch_body(mdl, loop)
        name = ck + (f"#loop{loop[0]}-{loop[1]}" if loop else "") + "#cu"
        row = {"ckpt": name, "domains": {}, "path": "doc_cu", "loop_blocks": list(loop) if loop
               else None}
        print(f"\n{name}", flush=True)
        for dom in DOMAINS:
            rows = val_seqs(dom, tok)
            if rows is None:
                print(f"  {dom:20s} no val rows -- SKIPPED, not scored as zero", flush=True)
                continue
            x, y = rows[:, :-1].cuda(), rows[:, 1:].cuda()
            blocks, tot, ntok = [], 0.0, 0
            with torch.no_grad():
                for i in range(len(x)):
                    xi, yi = x[i:i + 1], y[i:i + 1]
                    # THE ONE DIFFERENCE FROM domain_loss.py:229, which calls model(x) with no cu.
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = mdl(xi, cu=doc_cu_seqlens(xi, eos))
                    lg = lg[0] if isinstance(lg, tuple) else lg
                    ce = float(torch.nn.functional.cross_entropy(
                        lg.float().view(-1, lg.shape[-1]), yi.reshape(-1), reduction="sum"))
                    blocks.append({"ce_sum": ce, "n_tokens": int(yi.numel())})
                    tot += ce
                    ntok += int(yi.numel())
            # head_fp from the SAME val_seqs rows, so block_paired's identity check can still
            # refuse a pairing across different sequences. The rows are unchanged by cu; only the
            # forward is.
            row["domains"][dom] = {"loss": round(tot / ntok, 4), "tokens": ntok,
                                   "head_fp": seqs_fp(rows), "split": "val", "blocks": blocks}
            print(f"  {dom:20s} {tot / ntok:.4f}   ({ntok:,} tok, {len(blocks)} blocks)", flush=True)
        # THE FILE IS APPEND-ONLY, folded by key downstream, like every other runs/*.jsonl.
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.append(name)
        # A ROW WITHOUT BLOCKS IS THE FAILURE THIS WHOLE MEASUREMENT DEPENDS ON NOT HAVING
        # (score_matrix.py:236 shipped exactly that for weeks). Checked here rather than
        # discovered in block_paired.
        nb = sum(len(v["blocks"]) for v in row["domains"].values())
        if nb != 64 * len(row["domains"]):
            raise SystemExit(f"REFUSING: {name} wrote {nb} blocks over {len(row['domains'])} "
                             f"domains; 64 per domain expected. The pairing would be on a "
                             f"different n than the cu=None run and not comparable.")
        del mdl
        torch.cuda.empty_cache()
    print(f"\nwrote {len(written)} row(s) to {OUT}: {written}")
    print("pair with: python3 eval/block_paired.py --from runs/b0_sd_blocks_cu.jsonl --arms "
          f"'{written[0]}' '{written[1]}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
