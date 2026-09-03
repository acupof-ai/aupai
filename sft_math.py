#!/usr/bin/env python3
"""Stage-2 math SFT: sft.py plus --out (never overwrites ckpt_sft.pt) and FoNE digit loss.

Usage: torchrun --nproc_per_node=6 sft_math.py --resume ckpt_sft.pt \
  --sft_path data/sft/sft_math.pt --out ckpt_sft_math.pt --epochs 2 --lr_scale 0.05
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import re
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP

import fone
from train import (
    Cfg,
    HybridLM,
    RunLog,
    SOFTCAP,
    build_optimizers,
    convert_to_fp8_compute,
    ddp_even_len,
    doc_cu_seqlens,
    opt_snapshot,
    save_checkpoint,
    set_schedule,
    setup_ddp,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SFT_DATA = os.path.join(ROOT, "data", "sft", "sft_all.pt")
EOS_ID = 1  # <eos> id in data/tokenizer.json
SAVE_INTERVAL = 200
LOG_INTERVAL = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="pretrained checkpoint path")
    parser.add_argument("--sft_path", default=SFT_DATA)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_scale", type=float, default=0.1, help="SFT LR = pretrain LR x scale")
    parser.add_argument("--no_fp8", action="store_true")
    # Spelled --no-grad_ckpt (hyphen) to match train.py, whose BooleanOptionalAction
    # generates that form (ead2d2b). Two entry points spelling the same switch differently
    # is a trap a person walks into once per script; the underscore form is kept for one
    # version and prints a deprecation so nothing in flight breaks silently.
    parser.add_argument(
        "--grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="activation checkpointing; --no-grad_ckpt disables it "
             "(FP8 backward goes NaN without it, so it defaults ON here)",
    )
    parser.add_argument(
        "--no_grad_ckpt",
        action="store_true",
        help=argparse.SUPPRESS,  # deprecated spelling, one version only
    )
    parser.add_argument("--save_every", type=int, default=SAVE_INTERVAL,
                        help="mid-run checkpoint interval; each one now carries optimizer state, "
                             "so a later extension resumes on one curve instead of restarting "
                             "Adam moments and the LR schedule. Was a module constant fixed at "
                             f"{SAVE_INTERVAL}, which is why N7 Stage B's 250-step arms landed "
                             "their only mid-run save at 200 and could not be extended from 250.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--stop_after", type=int, default=None,
                        help="stop after N steps WITHOUT shortening the schedule. --max_steps "
                             "also feeds total_steps (line below) and lr_mult reads total, so "
                             "it moves warmdown's start and gives a DIFFERENT lr curve than the "
                             "run you are reproducing a prefix of. How much different depends "
                             "on the resumed cfg: at warmup 20 / warmdown 0.65 the step-40 "
                             "multiplier differs 18.7x between total=1024 and total=40, while "
                             "at warmup 300 / warmdown 0.1 (what ckpt_p200m_4b_0902.pt carries) "
                             "step 40 is still inside warmup and both give 40/300 -- so the "
                             "hazard is real but its size is not knowable from the flag alone. "
                             "Use this whenever the intent is a prefix.")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "ckpt_sft_math.pt"),
        help="output checkpoint path (default: ckpt_sft_math.pt)",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="override the base's vocabulary fingerprint, for a checkpoint saved before "
        "train.py started recording it (print it with scripts/ckpt_info.py)",
    )
    parser.add_argument(
        "--allow_unstamped_pack",
        action="store_true",
        help="train on a pack that carries no holdout_fp. Records 'holdout status unknown' "
             "against the run: this run's eval numbers cannot be read as holdout-clean, and that "
             "belongs in the exp row. Without this flag an unstamped pack REFUSES, because an "
             "unverified pack and a verified one were otherwise the same state to this launcher.",
    )
    parser.add_argument(
        "--loop",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        help="N7 Stage B: TRAIN with blocks LO..HI run twice (eval/loop_wrapper.py, AttnRes "
             "option 3). Patched on raw_model BEFORE torch.compile and before DDP wraps it, so "
             "the loop is inside the traced graph rather than around it. Stage A measured this "
             "same loop applied at inference only to weights trained to be visited once: worse "
             "on all three rulers (humaneval BPB +0.0273, domain_loss +0.1166 nat, 1.64x "
             "latency). Stage B asks the different question of whether weights TRAINED under the "
             "loop recover that -- so a Stage B arm must never be compared against a Stage A "
             "number, only against its own unlooped arm at the same step.",
    )
    args = parser.parse_args()
    if args.stop_after and args.max_steps:
        parser.error("--stop_after and --max_steps together are ambiguous: --max_steps also "
                     "shortens total_steps (and therefore the LR schedule) while --stop_after "
                     "does not. Pass exactly one.")

    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    for k, v in ck.get("cfg", {}).items():
        setattr(Cfg, k, v)
    Cfg.batch = args.batch
    Cfg.epochs = args.epochs
    # Before ANY save: SAVE_INTERVAL writes .stepN checkpoints mid-run, and an interrupted
    # run's last .stepN is precisely the file someone has to identify later.
    Cfg.lr_scale = args.lr_scale
    # grad_ckpt must stay ON: FP8 e4m3 backward goes NaN without it.
    if args.no_grad_ckpt:
        print("WARNING --no_grad_ckpt is deprecated; use --no-grad_ckpt (hyphen), the "
              "spelling train.py uses. Honoured this once.", flush=True)
        args.grad_ckpt = False
    Cfg.grad_ckpt = args.grad_ckpt

    torch.manual_seed(Cfg.seed)
    torch.set_float32_matmul_precision("high")
    ddp, rank, world, local = setup_ddp()
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0
    runlog = (
        RunLog(re.sub(r"^ckpt_", "", os.path.splitext(os.path.basename(args.out))[0])) if is_main else print
    )
    amp = device.startswith("cuda")

    d = torch.load(args.sft_path, map_location="cpu", weights_only=True)
    X = d["input_ids"][:, :-1].long().contiguous()
    Y = d["labels"][:, 1:].long().contiguous()
    # A pack from another vocabulary trains silently at ~4x the loss: every id is
    # wrong and in range, and the sizes match.
    ck_vocab = args.vocab or ck.get("vocab_id")
    # GUARDED ON THE WRONG KEY UNTIL 2026-09-03. The condition was `"vocab" in d` while
    # prepare_sft.pack_and_save writes "vocab_id" (only the pre-2026-08 arith_* packs carry a
    # bare "vocab"). So for every pack built by the current packer the assert was skipped and
    # the run took the WARNING branch instead -- "the pack predates vocabulary fingerprinting"
    # printed about a pack that carries the fingerprint. The check whose comment says a wrong
    # vocabulary "trains silently at ~4x the loss" has therefore never once fired, and its
    # warning read as a property of the pack rather than a defect in the reader.
    pack_vocab = d.get("vocab_id", d.get("vocab"))
    if ck_vocab and pack_vocab is not None:
        assert pack_vocab == ck_vocab, (
            f"{args.sft_path} was packed against vocabulary {pack_vocab} but "
            f"{args.resume} was trained on {ck_vocab}; repack with "
            "`datagen/prepare_sft_math.py --tokenizer <the base's tokenizer.json>`"
        )
        if is_main:
            print(f"vocab_id matches: {ck_vocab}", flush=True)
    elif is_main:
        missing = "the checkpoint" if not ck_vocab else "the pack"
        print(f"WARNING {missing} predates vocabulary fingerprinting; verify by hand", flush=True)
    # A pack built against a stale holdout set may contain held-out questions.
    # Refuse, the same way a vocab_id mismatch refuses.
    #
    # AND A PACK WITH NO STAMP REFUSES TOO (6e's ruling, 2026-09-03). Until this change the
    # unstamped case printed a WARNING and proceeded, which made "stamped and verified clean" and
    # "holdout status unknown" the same state to the launcher -- the shape §140 names, where the
    # representation of success is shared with the thing that has no evidence behind it. 12 of the
    # 16 packs in data/sft/ carry no holdout_fp, so this was not a hypothetical gap. The number
    # that forced it: building data/rl/rlvr_math.jsonl the same day found 515 of 218,095 rows were
    # holdout questions (0.2361%), three of them verbatim eval text, in a pool nothing had ever
    # filtered because nothing required it.
    #
    # --allow_unstamped_pack is the escape hatch and it is LOUD: it names the pack and prints
    # "holdout status unknown" so the fact reaches the log and the exp row. An escape hatch that
    # left no trace would restore exactly the state this refusal removes.
    holdout_path = os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")
    if "holdout_fp" in d and os.path.isfile(holdout_path):
        import hashlib
        live_fp = hashlib.sha256(open(holdout_path, "rb").read()).hexdigest()[:16]
        if d["holdout_fp"] != live_fp:
            raise RuntimeError(
                f"{args.sft_path} was packed against holdout set {d['holdout_fp']}, "
                f"but the current holdout_hashes.txt is {live_fp}. The pack may contain "
                f"held-out questions. Repack with prepare_sft.py."
            )
    elif "holdout_fp" not in d:
        if not args.allow_unstamped_pack:
            raise RuntimeError(
                f"{args.sft_path} carries NO holdout_fp, so nothing can say whether it contains "
                f"held-out questions. Until 2026-09-03 this printed a warning and trained anyway, "
                f"which made an unverified pack indistinguishable from a verified one. Repack with "
                f"prepare_sft.py to stamp it, or pass --allow_unstamped_pack to train on it "
                f"deliberately -- that flag records 'holdout status unknown' against this run."
            )
        if is_main:
            print(f"UNSTAMPED PACK {args.sft_path}: holdout status unknown -- this run's eval "
                  f"numbers cannot be read as holdout-clean (--allow_unstamped_pack)", flush=True)
    elif is_main:
        # Stamped pack, but holdout_hashes.txt is missing: the stamp cannot be checked against
        # anything. Not a refusal, because the pack did record what it was built against.
        print(f"WARNING {holdout_path} missing; {args.sft_path} claims holdout_fp "
              f"{d['holdout_fp']} and nothing here can verify it", flush=True)
    assert Cfg.fone == ("values" in d), (
        f"checkpoint fone={Cfg.fone} but {args.sft_path} "
        f"{'has' if 'values' in d else 'has no'} values; repack with datagen/prepare_sft_math.py --fone"
    )
    # V feeds the embedding, W is the digit target one position later (train.py's split)
    V = d["values"][:, :-1].contiguous() if Cfg.fone else None
    W = d["values"][:, 1:].contiguous() if Cfg.fone else None
    del d
    if ddp:
        X = X[rank::world].contiguous()
        Y = Y[rank::world].contiguous()
        if Cfg.fone:
            V, W = V[rank::world].contiguous(), W[rank::world].contiguous()
    n_even = ddp_even_len(len(X), Cfg.batch, ddp)
    X, Y = X[:n_even].pin_memory(), Y[:n_even].pin_memory()
    if Cfg.fone:
        V, W = V[:n_even].pin_memory(), W[:n_even].pin_memory()
    if is_main:
        print(f"sft rows {len(X)} per rank (world {world})", flush=True)

    raw_model = HybridLM(Cfg).to(device)
    raw_model.load_state_dict(ck["model"])
    fp8 = not args.no_fp8 and amp
    if fp8:
        raw_model = raw_model.to(torch.bfloat16)
        convert_to_fp8_compute(raw_model)
    if is_main:
        from train import HAS_FA

        print(
            f"resumed {args.resume} | params {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M | "
            f"fp8 {fp8} | fa {HAS_FA} | doc_mask {Cfg.doc_mask}",
            flush=True,
        )

    optimizers = build_optimizers(raw_model, Cfg)

    if args.loop:
        # BEFORE torch.compile and before DDP: the patch replaces a bound method, and compile
        # traces whatever _body is at trace time, so patching after would either be traced around
        # or (under DDP static_graph) change the graph the buckets were built for. Also AFTER
        # build_optimizers, which walks parameters -- the loop adds no parameters, so the optimizer
        # groups are identical between the arms and that is the point.
        sys.path.insert(0, os.path.join(ROOT, "eval"))
        from loop_wrapper import patch_body

        patch_body(raw_model, tuple(args.loop))
        # ON Cfg, so save_checkpoint carries it into the final ckpt AND every .stepN. Without this
        # the looped and unlooped arms write byte-different checkpoints whose metadata is
        # identical, and six weeks later nothing but the filename says which is which -- the
        # failure this repo has already paid for with .stepN files holding earlier weights.
        Cfg.loop_blocks = list(args.loop)
        if is_main:
            print(f"LOOPED TRAINING: blocks {args.loop[0]}..{args.loop[1]} run twice "
                  f"(AttnRes option 3); grad_ckpt {Cfg.grad_ckpt}", flush=True)

    model = raw_model
    if ddp:
        model = DDP(
            model, device_ids=[local], bucket_cap_mb=100, gradient_as_bucket_view=True, static_graph=True
        )
    if Cfg.compile and amp:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        model = torch.compile(model, dynamic=False)

    good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    good_opt = [None] * len(optimizers)
    total_steps = Cfg.epochs * (len(X) // Cfg.batch)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    # --stop_after ends the run without touching total_steps, so the LR schedule is the one
    # the full run would have had. --max_steps keeps its old meaning (shorten BOTH), because
    # runs in flight pass it. _stop is whichever bound applies; the two are refused up at
    # parse time, not here, so a contradictory launch dies before loading 1.6 GB.
    _stop = args.stop_after or args.max_steps
    # THE RUN'S OWN RECORD OF WHAT IT WAS ASKED TO DO. lr_scale never reaches Cfg -- train.py
    # :848 applies it inside set_schedule as initial_lr * lr_scale * m -- so it reached no log
    # and no checkpoint, and ckpt_control_ours.pt's scale is now unrecoverable: not in its cfg,
    # not in runs/control_ours.log, not in the launch log. That checkpoint's held-out loss is
    # the divisor of every number in docs/audits/control_pythia160m_vs_ours.md, and "the
    # argparse default was 0.1" is not evidence of what ran. Printing the ARGV and the REALISED
    # per-group lr costs two lines and makes the question answerable from the log alone.
    if is_main:
        runlog("argv " + json.dumps(sys.argv[1:]))
        runlog(f"lr_scale {args.lr_scale} total_steps {total_steps} stop_after {args.stop_after} "
               f"batch {Cfg.batch} epochs {Cfg.epochs} seed {Cfg.seed}")
        set_schedule(optimizers, 0, total_steps, Cfg, args.lr_scale)
        for opt in optimizers:
            for gi, g in enumerate(opt.param_groups):
                # The realised lr at step 0, not the configured base: a reader can multiply
                # initial_lr by a scale themselves, but only the process knows which groups
                # exist and which optimizer owns them.
                runlog(f"  lr[{type(opt).__name__}:{gi}] initial {g['initial_lr']:.3g} "
                       f"-> step0 {g['lr']:.3g}")

    step = 0
    flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)
    weight = raw_model.head.weight[: raw_model.cfg.vocab]

    for ep in range(Cfg.epochs):
        model.train()
        perm = torch.randperm(len(X))
        t0 = time.time()
        for i in range(0, len(X) - Cfg.batch + 1, Cfg.batch):
            idx = perm[i : i + Cfg.batch]
            xb = X[idx].to(device, non_blocking=True)
            yb = Y[idx].to(device, non_blocking=True)
            vb = V[idx].to(device, non_blocking=True) if Cfg.fone else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                hidden, _ = model(xb, yb, doc_cu_seqlens(xb, EOS_ID) if Cfg.doc_mask else None, vb)
            B, T, D = hidden.shape
            loss = flce(weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1))
            if Cfg.fone:
                # Supervised [NUM] positions only: a prompt-masked one must not be scored
                nmask = yb == Cfg.num_id
                if nmask.any():
                    wb = W[idx].to(device, non_blocking=True)
                    loss = loss + Cfg.fone_loss_w * F.cross_entropy(
                        raw_model.num_logits(hidden[nmask].float()).reshape(-1, 10),
                        fone.digit_targets(wb[nmask]).reshape(-1),
                    )
            loss.backward()
            last = loss.item()
            grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), Cfg.clip)

            if ddp:
                flag = torch.tensor([float(math.isfinite(last) and math.isfinite(grad_norm))], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                healthy = flag.item() > 0.5
            else:
                healthy = math.isfinite(last) and math.isfinite(grad_norm)
            if not healthy:
                raw_model.load_state_dict(good_state)
                for j, opt in enumerate(optimizers):
                    if good_opt[j] is not None:
                        opt.load_state_dict(good_opt[j])
                if is_main:
                    runlog(f"step {step}/{total_steps} NaN — restored last good state")
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                step += 1
                if _stop and step >= _stop:
                    break
                continue

            set_schedule(optimizers, step, total_steps, Cfg, args.lr_scale)
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.save_every == 0:
                good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                good_opt = opt_snapshot(optimizers)
                if is_main:
                    # opt=good_opt, and the omission of it is why N7 Stage B's 250-step arms
                    # could not be extended: save_checkpoint has taken an `opt` argument all
                    # along (train.py:1024, stored verbatim as ck["opt"]), this site already
                    # held the snapshot on the line above, and not passing it wrote a
                    # checkpoint with `step` and no optimizer. A resume from that restarts
                    # Adam moments and the LR schedule, which is a new run wearing the word
                    # resume, so the 250 and 500 points could not lie on one curve.
                    #
                    # good_opt IS THE NaN-ROLLBACK BUFFER (:283, restored at :351), not an
                    # artifact built for resuming, and the two want different things: the
                    # rollback wants the last GOOD state, a resume wants the state AT this
                    # step. They coincide here only because the line above recomputes the
                    # snapshot at the save step. If the rollback ever keeps an older good
                    # state, this call starts writing an optimizer from a different step than
                    # the weights -- which `step` in the file would not reveal.
                    save_checkpoint(args.out + f".step{step}", good_state, Cfg, ck_vocab,
                                    opt=good_opt, step=step)
            if is_main and step % LOG_INTERVAL == 0:
                # The elapsed figure covers LOG_INTERVAL steps, not one: t0 resets on
                # every log line. Naming the interval is the whole fix -- read as
                # seconds-per-step it turned a 9-minute run into a 2.3-hour estimate
                # (e1, 2026-08-31). A number that does not carry its unit gets one.
                runlog(f"step {step}/{total_steps} loss {last:.3f} "
                       f"{time.time() - t0:.0f}s/{LOG_INTERVAL}steps")
                t0 = time.time()
            if _stop and step >= _stop:
                break
        if _stop and step >= _stop:
            break

    if is_main:
        save_checkpoint(args.out, raw_model.state_dict(), Cfg, ck_vocab)
        print(f"saved {args.out}", flush=True)
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
