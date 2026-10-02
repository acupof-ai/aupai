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
try:  # CUDA-only kernel; the holdout gate and argparse run before any loss is built
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None
from torch.nn.parallel import DistributedDataParallel as DDP

import fone
from train import (
    Cfg,
    RunLog,
    _softcap,
    build_model,
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

#: MoE fields that change what the weights MEAN. A checkpoint and a live Cfg that disagree
#: on any of them build a different model than the one that was trained.
MOE_KEYS = ("moe_experts", "moe_top_k", "moe_shared", "moe_expert_ffn", "moe_layers")


def refuse_v42_unsupported(args):
    """The SFT-side twin of the `REFUSING --arch v42` list train.py's main() raises before the build.

    Every flag here is INERT under v42 rather than wrong-but-working, which is the dangerous
    kind: the run starts, the loss falls, and the thing the flag names never happens.

      --fp32_master     MasterWeights.map is an argument of train.build_optimizers.
                        build_v42_optimizers takes no master map, so the masters would be built,
                        pulled into, stepped past, and master.push() would then write the
                        UNSTEPPED fp32 copies back over the weights the optimizer just moved --
                        a silent revert of every step.
      --stochastic_round  read only by train.Muon, via the `sr = bool(getattr(cfg, "stochastic_round", False))` line in train.build_optimizers. V42Muon does not read it, so
                        the flag would be recorded on the checkpoint and do nothing.
      --fone            v41f.lm.V42LM.forward raises on num_vals; there is no FoNE head.
      --loop            eval/loop_wrapper.patch_body patches HybridLM's _body.
      --prefix          the prefix-LM mask is applied to HybridLM's MLA blocks.

    v42 SFT therefore runs plain bf16 weights under the V4.1 optimizer, which is what the v42
    PRETRAIN runs -- no master, no stochastic rounding. Stated rather than implied, because
    "not supported" and "supported and silently off" are the same log line (de, 2026-09-30).
    """
    if getattr(Cfg, "arch", "hybrid") != "v42":
        return
    bad = [f for f, on in (
        ("--fp32_master", args.fp32_master),
        ("--stochastic_round", args.stochastic_round),
        ("--fone (Cfg.fone from the checkpoint)", getattr(Cfg, "fone", False)),
        ("--loop", getattr(args, "loop", None)),
        # Spelled with its parenthetical, not bare: scripts/test_sft_prefix.py:158 locates the
        # argparse action by splitting this file on the first occurrence of that flag name
        # followed by a comma, and a bare spelling here would send it to this tuple instead,
        # where it would read --lr_decay's choices as the prefix arms. Do not write the bare
        # form anywhere above the parser, comments included.
        ("--prefix (prefix-LM mask on HybridLM MLA blocks)", getattr(args, "prefix", None)),
    ) if on]
    if bad:
        raise SystemExit(
            f"REFUSING --arch v42 SFT: {'; '.join(bad)}. These are implemented against HybridLM "
            f"and are inert or actively wrong under the v41f stack -- see "
            f"sft_math.refuse_v42_unsupported. v42 SFT is bf16 weights + build_v42_optimizers.")


def build_sft_model(device):
    """The model main() fine-tunes, built from the live Cfg (which main() has already overwritten
    with the resumed checkpoint's cfg).

    build_model, not HybridLM: it dispatches on Cfg.arch and returns the v41f V42LM under
    --arch v42, rebuilt from the checkpoint's own cfg.v42_cfg. Constructing HybridLM here made a
    v42 checkpoint unloadable -- the two state dicts have zero key overlap -- so the SFT path
    could not touch the v42 line at all (de, 2026-09-30; the same defect scripts/loader.py
    carried, fixed at 9588ee2b). A function rather than one line inside main() so the CPU
    known-answer test can drive the decision instead of restating it.
    """
    return build_model(Cfg).to(device)


def build_sft_optimizers(raw_model, master=None):
    """The optimizer set for the live Cfg.arch, branching exactly as train.py's main() does at its `if Cfg.arch == "v42": ... build_v42_optimizers` site.

    train.build_optimizers routes parameters by NAME pattern against HybridLM's names
    (blocks.N.mixer.qkv, ffn.w13, ...). Not one V42LM parameter (layers.N.attn.qproj.wq_a,
    layers.N.ffn.gate.weight) matches, so under --arch v42 every weight fell through to the
    fallback group: no Muon on the backbone matrices, no Sinkhorn on embed/head, no per-group
    weight decay. Nothing raises and the run converges -- to a different model than the one the
    v42 pretrain was producing (de, 2026-09-30).
    """
    if getattr(Cfg, "arch", "hybrid") != "v42":
        return build_optimizers(raw_model, Cfg, master.map if master is not None else None)
    assert master is None, (
        "--fp32_master maps HybridLM parameters into build_optimizers; build_v42_optimizers "
        "takes no master map, so the masters would be built and never stepped from")
    from v41f.optim import build_v42_optimizers

    # Cfg.v42_lr bare, NOT * args.lr_scale: set_schedule applies lr_scale to every group as
    # initial_lr * lr_scale * m, so scaling it here would square it -- 0.1 becoming 0.01.
    return build_v42_optimizers(raw_model, raw_model.v41f_cfg, Cfg.v42_lr)


#: The v42 MoE fields that change what the weights MEAN, in v41f.config.V41FConfig's names.
V42_MOE_KEYS = ("n_routed_experts", "n_activated_experts", "n_shared_experts", "moe_inter_dim")


def _assert_v42_moe_matches_ckpt(model, ck_cfg):
    """The v42 arm of assert_moe_matches_ckpt, reading the shape where v42 keeps it.

    The hybrid arm read `ck_cfg["moe_experts"]` against `b.ffn.w13.shape[0]` and BOTH sides are
    absent under --arch v42: the launch line (runs/v42_arch_b.sh) passes no --moe_experts, so
    Cfg.moe_experts stays at its class default 0, and v41f.moe.MoE has no `w13` -- it holds
    either stacked w1/w3/w2 of shape [E, inter, dim] or an `experts` ModuleList. So `want` was 0,
    the hasattr filter matched no block, `got` was 0, and the assertion passed on 0 == 0 while
    the real stack is 64 routed / top-8 (v41f.config.v42_s24). Measured 2026-09-30: vacuous.

    Same discipline as the hybrid arm -- the count comes off a real tensor dimension, never off
    self.n_routed_experts, which only restates the config the model was built from.
    """
    v = ck_cfg.get("v42_cfg") or {}
    if not v:
        raise SystemExit(
            "REFUSING: the checkpoint says arch=v42 but carries no v42_cfg, so nothing states "
            "the MoE shape it was trained with. train.build_model writes cfg.v42_cfg on every "
            "build; a checkpoint without it predates that and cannot be verified.")
    want = int(v["n_routed_experts"])
    ffns = [b.ffn for b in getattr(model, "layers", []) if getattr(b, "ffn", None) is not None]
    counts = set()
    for f in ffns:
        if getattr(f, "w1", None) is not None and hasattr(f.w1, "shape"):
            counts.add(int(f.w1.shape[0]))          # stacked: [E, inter, dim]
        elif getattr(f, "experts", None) is not None:
            counts.add(len(f.experts))              # loop: one Expert module per routed expert
    if counts != {want}:
        raise SystemExit(
            f"REFUSING: the checkpoint was trained with n_routed_experts={want} but the model "
            f"built here has {sorted(counts) or 'no'} routed expert(s) over {len(ffns)} MoE "
            f"layer(s). cfg.v42_cfg did not reach build_model.")
    got_top = {int(f.top_k) for f in ffns}
    if got_top != {int(v["n_activated_experts"])}:
        raise SystemExit(
            f"REFUSING: the checkpoint routes top-{v['n_activated_experts']} but the model built "
            f"here routes top-{sorted(got_top)}. Routing changes no tensor shape, so "
            f"load_state_dict accepts it and the SFT trains a different model.")
    live = getattr(Cfg, "v42_cfg", None) or {}
    bad = {k: (v[k], live.get(k)) for k in V42_MOE_KEYS if k in v and live.get(k) != v[k]}
    if bad:
        raise SystemExit(
            f"REFUSING: Cfg.v42_cfg disagrees with the checkpoint on {', '.join(sorted(bad))}: "
            + "; ".join(f"{k} ckpt={c!r} live={lv!r}" for k, (c, lv) in sorted(bad.items())))
    return want


def assert_moe_matches_ckpt(model, ck_cfg):
    """The built model's MoE shape is the checkpoint's, checked before load_state_dict.

    WHY, given that :151 copies every ck["cfg"] key onto Cfg before the build. That copy is
    how "build from ck['cfg']" is implemented here, and it works -- for the keys the
    checkpoint carries. The hole is the pair it cannot cover: a key the checkpoint does NOT
    carry keeps whatever the live Cfg class holds, and a CLI flag or module-level default
    that mutates Cfg between the copy and the build wins silently. Neither is hypothetical
    for MoE, where `moe_experts` 0 is the off sentinel: a dense checkpoint saved before the
    field existed, loaded by a Cfg whose default became non-zero, builds an MoE model and
    the mismatch surfaces as a load error naming a tensor rather than a config.

    The routed count is read from w13.shape[0], a real tensor dimension, not from
    self.n_routed -- the attribute restates the config the model was built from, so
    comparing it to the config it came from is the check comparing code against itself.
    load_state_dict would catch a wrong expert COUNT on its own; it would not catch
    moe_top_k or moe_layers, which change routing and leave every tensor shape intact.
    """
    if ck_cfg.get("arch", "hybrid") == "v42":
        return _assert_v42_moe_matches_ckpt(model, ck_cfg)
    want = int(ck_cfg.get("moe_experts", 0) or 0)
    blocks = [b for b in getattr(model, "blocks", []) if hasattr(getattr(b, "ffn", None), "w13")]
    got = int(blocks[0].ffn.w13.shape[0]) if blocks else 0
    if got != want:
        raise SystemExit(
            f"REFUSING: the checkpoint was trained with moe_experts={want} but the model "
            f"built here has {got} routed expert(s) in {len(blocks)} MoE block(s). The cfg "
            f"copy at the top of main() did not take effect -- a flag or a class default "
            f"moved Cfg between the copy and HybridLM(Cfg). 0 means dense."
        )
    bad = {
        k: (ck_cfg[k], getattr(Cfg, k, None))
        for k in MOE_KEYS
        if k in ck_cfg and getattr(Cfg, k, None) != ck_cfg[k]
    }
    if bad:
        raise SystemExit(
            f"REFUSING: Cfg disagrees with the checkpoint on {', '.join(sorted(bad))}: "
            + "; ".join(f"{k} ckpt={c!r} live={live!r}" for k, (c, live) in sorted(bad.items()))
            + ". These change routing, not tensor shapes, so load_state_dict accepts them "
            "and the SFT runs a different model than the one that was pretrained."
        )
    return got




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="pretrained checkpoint path")
    parser.add_argument("--sft_path", default=SFT_DATA)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_scale", type=float, default=0.1, help="SFT LR = pretrain LR x scale")
    parser.add_argument("--lr_decay", choices=("cosine", "linear"), default="cosine",
                        help="post-warmup decay shape. linear decays to ZERO over every "
                             "remaining step (CED code-SFT user order 2026-09-25) and ignores "
                             "the resumed ckpt's warmdown; cosine keeps the pretraining shape")
    parser.add_argument("--warmup_frac", type=float, default=None,
                        help="warmup as a fraction of total steps (e.g. 0.05); overrides the "
                             "resumed ckpt's absolute warmup. None keeps absolute warmup")
    parser.add_argument("--no_fp8", action="store_true")
    parser.add_argument("--stochastic_round", action="store_true",
                        help="bf16 weights with fp32 candidate Bernoulli-rounded on write, fp32 "
                             "Muon momentum (1e option B). The model is cast bf16 even with "
                             "--no_fp8; without this flag --no_fp8 keeps fp32 weights")
    # Spelled --no-grad_ckpt (hyphen) to match train.py, whose BooleanOptionalAction
    # generates that form (6337c30). Two entry points spelling the same switch differently
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
    parser.add_argument("--fp32_master", action="store_true",
                        help="optimizer owns fp32 master copies of the bf16 weights "
                             "(train.MasterWeights). REQUIRED for bf16 SFT at small LR: a "
                             "single-step rel update ~1e-4 is below the bf16 half-ULP, so "
                             "without a master only 0.1-10%% of elements move per step (measured "
                             "2026-09-25 on the CED ckpt). Costs one fp32 copy + fp32 "
                             "optimizer state; the model, DDP and kernels stay bf16")
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
    parser.add_argument(
        "--prefix",
        choices=("p3", "p7"),
        help="N7 Stage C: TRAIN with a prefix-LM attention mask on the MLA layers of the named arm "
             "(eval/prefix_mask.py PREFIX_ARMS -- p3 = blocks 3,7,11; p7 = block 7 alone). Prompt "
             "tokens attend bidirectionally among the prompt tokens of their own DOCUMENT; response "
             "tokens stay causal, so no position ever reads a token that carries loss. Runs on the "
             "TRAINING kernel as two varlen calls -- causal=True over the documents plus "
             "causal=False over the prompt segments -- and NOT through flash_attn.cute's mask_mod, "
             "whose forward is exact on SM 9.0 and whose backward is wrong (160 of 169 gradient "
             "tensors disagree with a same-mask SDPA reference, norm ratio median 21.65; "
             "facts/efficiency.json#eff.flash_attn_cute_mask_mod_backward_wrong_sm90). "
             "BLOCK 11 ALONE IS NOT AN ARM and cannot be: prefix and causal differ only for prompt "
             "queries, prompt positions are ignore_index, and block 11 is the last block, so a "
             "changed prompt position has no path to a supervised one -- measured bitwise-identical "
             "loss, 0 of 3660 changed positions supervised. p7 works because layer 7 has KDA layers "
             "above it. p3's causal twin is ckpt_n7c_unlooped.pt and p7's is ckpt_n7c_looped.pt, "
             "same pack, seed and step count, so neither baseline is retrained. Run "
             "scripts/n7c_gates.py on the pod first; it certifies both arms' layer sets.",
    )
    parser.add_argument("--check_pack", action="store_true",
                        help="cardless dry gate: load resume ckpt and pack on CPU, run the "
                             "vocab_id/holdout/fone checks, print row counts, and exit 0 before "
                             "model build, card claim or any GPU use")
    args = parser.parse_args()
    if args.stop_after and args.max_steps:
        parser.error("--stop_after and --max_steps together are ambiguous: --max_steps also "
                     "shortens total_steps (and therefore the LR schedule) while --stop_after "
                     "does not. Pass exactly one.")

    ddp, rank, world, local = setup_ddp()
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0

    d = torch.load(args.sft_path, map_location="cpu", weights_only=True)
    X = d["input_ids"][:, :-1].long().contiguous()
    Y = d["labels"][:, 1:].long().contiguous()

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

    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    for k, v in ck.get("cfg", {}).items():
        setattr(Cfg, k, v)
    Cfg.batch = args.batch
    Cfg.epochs = args.epochs
    # Before ANY save: SAVE_INTERVAL writes .stepN checkpoints mid-run, and an interrupted
    # run's last .stepN is precisely the file someone has to identify later.
    Cfg.lr_scale = args.lr_scale
    # SFT checkpoint marker: score_matrix.classify and the RL trainer's resume gate read
    # cfg["kind"] first. Continuation-format SFT is not inferable from the rest of cfg, so
    # the marker must be written, never guessed. Set AFTER the ckpt cfg copy above.
    Cfg.kind = "sft"
    Cfg.lr_decay = args.lr_decay
    Cfg.warmup_frac = args.warmup_frac
    # grad_ckpt must stay ON: FP8 e4m3 backward goes NaN without it.
    if args.no_grad_ckpt:
        print("WARNING --no_grad_ckpt is deprecated; use --no-grad_ckpt (hyphen), the "
              "spelling train.py uses. Honoured this once.", flush=True)
        args.grad_ckpt = False
    Cfg.grad_ckpt = args.grad_ckpt

    torch.manual_seed(Cfg.seed)
    torch.set_float32_matmul_precision("high")
    runlog = (
        RunLog(re.sub(r"^ckpt_", "", os.path.splitext(os.path.basename(args.out))[0])) if is_main else print
    )
    amp = device.startswith("cuda")

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
    assert Cfg.fone == ("values" in d), (
        f"checkpoint fone={Cfg.fone} but {args.sft_path} "
        f"{'has' if 'values' in d else 'has no'} values; repack with datagen/prepare_sft_math.py --fone"
    )
    if args.check_pack:
        if is_main:
            sup = int((Y != -100).sum())
            print(f"CHECK_PACK ok: {args.sft_path} rows={Y.shape[0]} seq={Y.shape[1]} "
                  f"supervised={sup / 1e6:.2f}M vocab_id={pack_vocab} "
                  f"holdout_fp={d.get('holdout_fp')} -- accepted by {args.resume}", flush=True)
        return
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
    # pin_memory only exists with CUDA; on a CPU run (device=="cpu") it raises
    # "No CUDA GPUs are available". The CPU 2-step SFT smoke/resume path hits exactly this, so
    # guard it instead of relying on the test's pin_memory monkeypatch.
    if device.startswith("cuda"):
        X, Y = X[:n_even].pin_memory(), Y[:n_even].pin_memory()
        if Cfg.fone:
            V, W = V[:n_even].pin_memory(), W[:n_even].pin_memory()
    if is_main:
        print(f"sft rows {len(X)} per rank (world {world})", flush=True)

    # After the ck["cfg"] copy set Cfg.arch, before anything is built: an inert flag must stop
    # the run at the top, not be discovered from a checkpoint six hours later.
    refuse_v42_unsupported(args)
    raw_model = build_sft_model(device)
    assert_moe_matches_ckpt(raw_model, ck.get("cfg", {}))
    raw_model.load_state_dict(ck["model"])
    fp8 = not args.no_fp8 and amp
    # On GPU the model is ALWAYS stored bf16: fp8 casts before the float8 conversion, and the
    # --no_fp8 bf16 path must cast too -- leaving the checkpoint's fp32 weights doubles static
    # memory AND leaves the optimizer stepping fp32 weights when --fp32_master is off. With
    # --fp32_master the optimizer owns fp32 copies (MasterWeights) while the model stays bf16.
    # CPU (amp False) keeps fp32: the CPU smoke runs there and bf16 CPU kernels are not the path.
    if amp:
        raw_model = raw_model.to(torch.bfloat16)
    if fp8:
        convert_to_fp8_compute(raw_model)
    # TWO mutually-exclusive fp32-numerics paths for bf16 SFT. --fp32_master (de, #710) gives the
    # optimizer true fp32 weight copies (MasterWeights): exact for small-LR updates but OOM'd the
    # 85-GiB world-8 box at ~93 GiB on 2026-09-25. --stochastic_round (Option B, 1e 2026-09-25)
    # keeps weights bf16 and Bernoulli-rounds fp32 w+delta into them, the path that fits. Default
    # SFT launch uses Option B; --fp32_master stays for a box with more headroom or for the
    # known-answer test. train.build_optimizers asserts the two are never on together.
    master = None
    assert not (args.fp32_master and args.stochastic_round), (
        "--fp32_master and --stochastic_round are alternative fp32-numerics paths, not combined: "
        "the master replaces the fp32 weight the stochastic rounder writes from")
    if args.fp32_master:
        from train import MasterWeights
        master = MasterWeights(raw_model)
        if is_main:
            _mb = sum(m.numel() for _, m in master.pairs) * 4 / 2**30
            print(f"fp32 master weights: {_mb:.2f} GiB over {len(master.pairs)} tensors", flush=True)
    elif args.stochastic_round and amp:
        # Option B runs bf16 compute without fp8: weights must be bf16, or --no_fp8 leaves the
        # checkpoint cast fp32 and doubles static memory (the B48/B4 OOM root cause).
        raw_model = raw_model.to(torch.bfloat16)
    Cfg.stochastic_round = args.stochastic_round
    if args.stochastic_round:
        assert not fp8, "stochastic_round is the bf16 (--no_fp8) path, not the fp8 path"
        assert amp, ("stochastic_round needs CUDA bf16 compute: on CPU the model is fp32, and "
                     "Bernoulli-casting every write to bf16 then back into fp32 would quantize "
                     "every parameter. Pass it on a GPU run only")
    if is_main:
        from train import HAS_FA

        print(
            f"resumed {args.resume} | params {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M | "
            f"fp8 {fp8} | bf16_store {amp} | fp32_master {master is not None} | "
            f"fa {HAS_FA} | doc_mask {Cfg.doc_mask}",
            flush=True,
        )

    optimizers = build_sft_optimizers(raw_model, master)

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

    prefix_state = None
    if args.prefix:
        # THE MASK NEEDS PER-STEP DATA, so unlike --loop this cannot be a one-time patch: the
        # per-document prompt lengths are computed from THIS batch's labels and cu. So the setup
        # here only builds the callback and installs the layer-scoping wrapper; the aux tensor is
        # swapped per step inside the loop.
        # IMPORTED AS model_mod, not `model`: that name is the wrapped (DDP / compiled) model in
        # this function, and rebinding it here would hand torch.compile a module object.
        import model as model_mod  # noqa: PLC0415
        from eval.prefix_mask import (  # noqa: PLC0415
            PREFIX_ARMS,
            doc_prompt_lengths,
            prefix_two_call,
        )


        if not model_mod.HAS_FA:
            raise SystemExit(
                "REFUSING: HAS_FA is False, so GatedMLA takes the SDPA fallback at model.py:196 and "
                "no mask_mod is ever called. The arm would train a CAUSAL model under a prefix "
                "flag and its checkpoint would claim an intervention that never ran.")
        if not Cfg.doc_mask:
            raise SystemExit(
                "REFUSING: --prefix needs doc_mask, because model.py:189 reaches "
                "flash_attn_varlen_func -- the only entry point that takes mask_mod -- only when cu "
                "is not None. With doc_mask off every forward takes flash_attn_func at :194 and the "
                "mask is silently absent, which is how this repo's gates once printed three passes "
                "while testing nothing.")
        layers = PREFIX_ARMS[args.prefix]
        targets = []
        for li in layers:
            mixer = raw_model.blocks[li].mixer
            if not isinstance(mixer, model_mod.GatedMLA):
                raise SystemExit(
                    f"REFUSING: block {li}'s mixer is {type(mixer).__name__}, not GatedMLA. Arm "
                    f"{args.prefix} names MLA layers {list(layers)} from cfg (layers "
                    f"{Cfg.layers}, attn_every {Cfg.attn_every}); this asserts them against the "
                    "built model rather than trusting the arithmetic.")
            targets.append(mixer)
        orig_varlen = model_mod.flash_attn_varlen_func
        aux_box = [None]

        # SCOPED BY A DEPTH COUNTER RAISED INSIDE THE TARGET'S OWN forward, not by a hook.
        # model.py:191 looks flash_attn_varlen_func up at call time, so the module global is the
        # only hook -- but it is global, so the wrapper must decide which layer is calling.
        #
        # THE FIRST VERSION USED forward HOOKS to raise a flag during a target's forward, and that
        # crashed p7 with CheckpointError: "Recomputed values ... have different metadata than
        # during the forward pass". --loop turns on grad_ckpt, which RECOMPUTES the forward inside
        # backward, and hooks do not fire inside a checkpoint recompute -- so the recomputed layer 7
        # ran UNMASKED and produced activations that did not match the saved ones. --loop alone and
        # --prefix p3 alone both pass; the failure needs both, which is what located it.
        #
        # WRAPPING forward FIXES IT because the wrapper is part of the function the checkpoint
        # recomputes: the counter is raised identically in the original pass and in the recompute.
        # try/finally, so an exception inside the mixer cannot leave the counter raised and silently
        # mask every layer that runs after it.
        _depth = [0]

        # TWO CALLS, NOT A mask_mod, and this is not a style choice. mask_mod's FORWARD is exact on
        # SM 9.0 and its BACKWARD is wrong: a mask_mod bitwise-identical to causal=True in the
        # forward (1.5946985483 to ten decimals) disagrees with a same-mask SDPA reference on 160 of
        # 169 gradient tensors, norm ratio median 21.65, cosines negative -- measured in
        # scripts/n7c_grad_check.py, recorded as
        # facts/efficiency.json#eff.flash_attn_cute_mask_mod_backward_wrong_sm90. Two 500-step arms
        # diverged on it (loss climbing to 3.07 while the causal twin fell to 1.158) and every static
        # gate passed, because they all read a frozen forward. prefix_two_call decomposes the same
        # mask into causal=True over the documents plus causal=False over the prompt segments, both
        # of which the kernel gets right; the identity is asserted in prefix_mask's selftest.
        def _prefix_varlen(q, k, v, **kw):
            if aux_box[0] is None or _depth[0] == 0:
                return orig_varlen(q, k, v, **kw)
            cu_in = kw.pop("cu_seqlens_q", None)
            kw.pop("cu_seqlens_k", None)
            if cu_in is None:
                raise SystemExit(
                    "REFUSING: the prefix path needs cu_seqlens_q, and model.py:191 passes it on "
                    "the varlen path only. Without it the document boundaries are unknown and the "
                    "prompt segments cannot be built.")
            return prefix_two_call(orig_varlen, q, k, v, cu_in, aux_box[0][0], **kw)

        def _wrap(mod):
            inner = mod.forward

            def fwd(*a, **k):
                _depth[0] += 1
                try:
                    return inner(*a, **k)
                finally:
                    _depth[0] -= 1
            mod.forward = fwd

        for t in targets:
            _wrap(t)
        model_mod.flash_attn_varlen_func = _prefix_varlen
        # ON Cfg, for the same reason loop_blocks is: the prefix and causal arms write
        # byte-different checkpoints whose metadata would otherwise be identical, and a checkpoint
        # that cannot say which mask trained it is a checkpoint whose numbers cannot be attributed.
        Cfg.prefix_arm = args.prefix
        Cfg.prefix_layers = list(layers)
        prefix_state = (aux_box, doc_prompt_lengths)
        if is_main:
            print(f"PREFIX TRAINING: arm {args.prefix}, prefix-LM mask on MLA blocks "
                  f"{list(layers)}; the other MLA layers stay causal", flush=True)

    model = raw_model
    if ddp:
        model = DDP(
            model, device_ids=[local], bucket_cap_mb=100, gradient_as_bucket_view=True, static_graph=True
        )
    if Cfg.compile and amp:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        # DDPOptimizer splits the compiled graph at DDP bucket boundaries. On v42 (2026-09-29,
        # world 8) that split forward returned a hidden with NO grad_fn -- backward raised "does
        # not require grad" on the first step -- while the same model compiled single-process
        # trained, and optimize_ddp=False made the same launch train (the `_ddp_opt_default` block in
        # train.py's main(), runs/v42_arch_b_0929.step20.txt). This SFT loop has no accum, so the splitter's
        # backward/allreduce overlap is the one thing it does buy; it is given up under v42 and
        # kept under hybrid. DYNAMO_OPTIMIZE_DDP=1/0 overrides either way, same env name as
        # train.py so one variable answers for both.
        _ddp_opt_default = "0" if getattr(Cfg, "arch", "hybrid") == "v42" else "1"
        if os.environ.get("DYNAMO_OPTIMIZE_DDP", _ddp_opt_default) == "0":
            torch._dynamo.config.optimize_ddp = False
            if is_main:
                print(f"dynamo optimize_ddp=False (default {_ddp_opt_default!r} for arch "
                      f"{getattr(Cfg, 'arch', 'hybrid')})", flush=True)
        model = torch.compile(model, dynamic=False)

    # The balancer's scope, resolved ONCE from the built model. Empty on a dense or hybrid
    # checkpoint, so the call site below is a no-op there without a second arch test.
    _moe_balance_layers = (
        [b.ffn for b in raw_model.layers] if getattr(Cfg, "arch", "hybrid") == "v42" else [])
    if _moe_balance_layers and is_main:
        _g = {float(getattr(f, "gamma", 0.0)) for f in _moe_balance_layers}
        print(f"v42 expert-bias balancer: {len(_moe_balance_layers)} MoE layer(s), gamma {_g}, "
              f"once per optimizer step on rank-summed counts", flush=True)

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
    steps_per_epoch = len(X) // Cfg.batch
    # THE RUN'S OWN RECORD OF WHAT IT WAS ASKED TO DO. lr_scale never reaches Cfg -- train.py
    # :848 applies it inside set_schedule as initial_lr * lr_scale * m -- so it reached no log
    # and no checkpoint, and ckpt_control_ours.pt's scale is now unrecoverable: not in its cfg,
    # not in runs/control_ours.log, not in the launch log. That checkpoint's held-out loss is
    # the divisor of every number in docs/audits/control_pythia160m_vs_ours.md, and "the
    # argparse default was 0.1" is not evidence of what ran. Printing the ARGV and the REALISED
    # per-group lr costs two lines and makes the question answerable from the log alone.
    if is_main:
        runlog("argv " + json.dumps(sys.argv[1:]))
        runlog(f"lr_scale {args.lr_scale} lr_decay {Cfg.lr_decay} warmup_frac {Cfg.warmup_frac} "
               f"total_steps {total_steps} steps_per_epoch {steps_per_epoch} "
               f"stop_after {args.stop_after} batch {Cfg.batch} epochs {Cfg.epochs} seed {Cfg.seed}")
        for e in range(Cfg.epochs):
            runlog(f"epoch {e + 1}/{Cfg.epochs} boundary save at step {(e + 1) * steps_per_epoch} "
                   f"-> {os.path.basename(args.out)}.epoch{e + 1}")
        set_schedule(optimizers, 0, total_steps, Cfg, args.lr_scale)
        for opt in optimizers:
            for gi, g in enumerate(opt.param_groups):
                # The realised lr at step 0, not the configured base: a reader can multiply
                # initial_lr by a scale themselves, but only the process knows which groups
                # exist and which optimizer owns them.
                runlog(f"  lr[{type(opt).__name__}:{gi}] initial {g['initial_lr']:.3g} "
                       f"-> step0 {g['lr']:.3g}")

    step = 0
    weight = raw_model.head.weight[: raw_model.cfg.vocab]
    # The FUNCTION, not the SOFTCAP constant: train._softcap() returns None under --arch v42,
    # because V4.1 has no router/logit softcap. Importing the constant made SFT squash logits
    # through tanh(x/30)*30 that the v42 pretrain never applied -- a different loss surface than
    # the one the checkpoint was trained on, with nothing raising (de, 2026-09-30).
    softcap = _softcap()

    def _cpu_ce(hidden_flat, targets):
        # Liger FLCE has no CPU kernel (it raises "0 active drivers" without CUDA). The CPU
        # 2-step smoke exercises the load/pack/step path only, so materialize logits and use
        # torch CE with the same tanh softcap Liger applies (train._softcap()). 537 MiB fp32 at
        # B1/seq4096 -- fine for a smoke, never the training path.
        logits = F.linear(hidden_flat.float(), weight.float())
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return F.cross_entropy(logits, targets, ignore_index=-100)

    # Is the FLCE symbol the real CUDA-only liger kernel? The CI SFT CED gate injects a
    # CPU-capable FakeFLCE by reassigning the module symbol; distinguish by origin, never by a
    # probe call (the fake counts one loss per step, so an extra call would desync its gate).
    is_real_liger = (
        LigerFusedLinearCrossEntropyLoss is not None
        and getattr(LigerFusedLinearCrossEntropyLoss, "__module__", "").startswith("liger_kernel"))

    if device.startswith("cuda"):
        # Pinned in source position by scripts/test_sft_holdout_gate.py: this guard must stay
        # after the --check_pack return and before the loss is built.
        assert LigerFusedLinearCrossEntropyLoss is not None, (
            "the GPU SFT path builds the loss with liger_kernel; it is installed on the pod but "
            "not in the CPU image.")
        # A symbol that exists but is not the liger_kernel class (a CPU test substitute on a
        # CUDA run) must not silently take the fused-linear path.
        assert is_real_liger, "CUDA SFT requires the real liger_kernel FLCE, got a substitute"
        flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=softcap)

        def ce_loss(hidden_flat, targets):
            return flce(weight, hidden_flat.to(weight.dtype), targets)
    elif not is_real_liger and LigerFusedLinearCrossEntropyLoss is not None:
        # CPU with a CPU-capable substitute injected (scripts/test_sft_ced_cpu.py's FakeFLCE).
        flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=softcap)

        def ce_loss(hidden_flat, targets):
            return flce(weight, hidden_flat, targets)
    else:
        ce_loss = _cpu_ce
    if amp:
        torch.cuda.reset_peak_memory_stats(device)

    for ep in range(Cfg.epochs):
        model.train()
        perm = torch.randperm(len(X))
        t0 = time.time()
        for i in range(0, len(X) - Cfg.batch + 1, Cfg.batch):
            idx = perm[i : i + Cfg.batch]
            xb = X[idx].to(device, non_blocking=True)
            yb = Y[idx].to(device, non_blocking=True)
            vb = V[idx].to(device, non_blocking=True) if Cfg.fone else None
            cub = doc_cu_seqlens(xb, EOS_ID) if Cfg.doc_mask else None
            if prefix_state is not None:
                # THE AUX TENSOR IS REBUILT EVERY STEP, from THIS batch's labels and cu. The
                # per-document prompt length is the offset of each document's first supervised
                # token, and both the document boundaries and the boundary inside them change with
                # the batch -- a tensor computed once would be read against the wrong documents for
                # every step after the first, out of bounds whenever the document count grew.
                # yb is the SHIFTED labels the loss uses, so the boundary read here is the same
                # boundary the loss enforces.
                _box, _plens = prefix_state
                _box[0] = [_plens(yb, cub).to(torch.int32)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                hidden, _ = model(xb, yb, cub, vb)
            B, T, D = hidden.shape
            loss = ce_loss(hidden.reshape(-1, D), yb.reshape(-1))
            # THE v42 AUXILIARY TERM, as train.py's train step adds it beside its LigerFusedLinearCrossEntropyLoss call. It carries two things, and the
            # second is why its absence is not a small numerical difference: the per-layer MoE
            # sequence-balance loss, AND V42LM.indexer_loss, the KL that trains the sparse-
            # attention indexer (indexer_train_mode "kl"). The indexer's inputs are detached, so
            # this term is the ONLY gradient it ever receives -- omit it and the indexer stops
            # learning for the whole SFT while the CE falls normally and every log line looks
            # healthy. None on a dense/hybrid model, so the branch is the arch, not a hasattr.
            if getattr(Cfg, "arch", "hybrid") == "v42":
                _aux = raw_model.aux_loss()
                if _aux is not None:
                    loss = loss + _aux
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
            # MoE COUNTER CORRECTION, as in train.py and sft.py. grad_ckpt defaults on here (:212
            # from --grad_ckpt, and the comment at :207 records why: FP8 e4m3 backward goes NaN
            # without it), so an MoE forward runs twice per micro-batch.
            #
            # No consumer in this script today, and the counters do NOT reach the checkpoint --
            # they are persistent=False in MoEFFN.__init__, so state_dict() omits them. See the
            # longer note at the matching call in sft.py, which records the wrong reason I first
            # wrote there and the census that corrected it. This call keeps the counters exact for
            # any later reader; it is a no-op on a dense checkpoint and on a layer that did not
            # forward.
            raw_model.commit_moe_token_counts()
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
                if master is not None:
                    # The good_state load rewrote the bf16 model; the fp32 masters must follow
                    # or the next push would resurrect the pre-rollback weights.
                    master.resync()
                if is_main:
                    runlog(f"step {step}/{total_steps} NaN — restored last good state")
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                step += 1
                if _stop and step >= _stop:
                    break
                continue

            set_schedule(optimizers, step, total_steps, Cfg, args.lr_scale)
            # fp32 master: lift the (clipped) bf16 grads to the optimizer's fp32 copies before
            # the step, then write the stepped masters back to the bf16 model after it -- same
            # pull/push ordering as train.py. Without these, the optimizer would step a master
            # whose grad is still None (and the bf16 grads would leak into the next backward).
            if master is not None:
                master.pull_grads()
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            # THE AUX-LOSS-FREE BALANCER'S STEP under v42, the policy train.py's `if _moe_balance_layers:` block runs and the
            # one this SFT inherits: every MoE layer, once per optimizer step, on counts SUMMED
            # across ranks, then the counter zeroed. Named rather than left undefined --
            # Cfg.moe_bias_gamma reached V42LM.__init__ and set every layer's gamma, so the bias
            # machinery is live on the model and only its CALLER was missing here; without this
            # call expert_bias stays identically zero for the whole SFT and the sequence-balance
            # loss is the only balancer, which is a different intervention than the pretrain ran.
            #
            # After opt.step() and not on a rolled-back step: the NaN branch above `continue`s
            # before reaching here, so a dropped step leaves the bias alone and folds its load
            # into the next one that lands. Outside any is_main guard, because all_reduce is a
            # collective every rank must enter, and because expert_bias is a buffer DDP never
            # synchronises -- the ranks would diverge if only rank 0 updated it.
            if _moe_balance_layers:
                for _bl in _moe_balance_layers:
                    _c = _bl.step_tokens_per_expert
                    if ddp:
                        dist.all_reduce(_c, op=dist.ReduceOp.SUM)
                    _bl.update_bias(_c)
                    _c.zero_()
            if master is not None:
                master.push()
            step += 1

            if step % args.save_every == 0:
                good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                good_opt = opt_snapshot(optimizers)
                if is_main:
                    # opt=good_opt, and the omission of it is why N7 Stage B's 250-step arms
                    # could not be extended: save_checkpoint has taken an `opt` argument all
                    # along (train.py's `def save_checkpoint(path, model_state, cfg, vocab_id,
                    # opt=None, step=None)`, stored verbatim as ck["opt"]), this site already
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
            if is_main and step == 1 and amp:
                # The 85 GiB launch gate (1e 2026-09-25) needs a step-1 number. allocated, not
                # reserved: reserved includes the caching allocator's held-but-unused pool.
                peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
                runlog(f"step1 peak allocated {peak_gb:.2f} GiB (gate: stop if > 85)")
                print(f"step1 peak allocated {peak_gb:.2f} GiB", flush=True)
        # Epoch-boundary read points for eval (CED code-SFT user order 2026-09-25: score each
        # epoch end, keep the higher HumanEval). Only a FULLY consumed epoch is written: under
        # --max_steps/--stop_after the boundary step is never reached, so no misnamed file.
        # No optimizer, same as the final save: eval read points, not resume sources.
        if is_main and step == (ep + 1) * steps_per_epoch:
            ep_path = args.out + f".epoch{ep + 1}"
            save_checkpoint(ep_path, raw_model.state_dict(), Cfg, ck_vocab, step=step)
            runlog(f"epoch {ep + 1} boundary saved {ep_path} at step {step}")
            print(f"saved {ep_path}", flush=True)
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
