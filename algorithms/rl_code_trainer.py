#!/usr/bin/env python3
"""Code RL trainer: GSPO on executable code problems, raw-continuation prompts.

The math trainer (rlvr_trainer.py) speaks ChatML + \\boxed{} rewards; this one speaks
the SFT-A code distribution: the prompt is a raw code prefix (a `def` signature plus
docstring for call-style, or a module docstring for stdin/stdout) and the model
continues the program. The continuation is executed under isolation and scored
binary by algorithms/code_reward.py.

Differences from the math path, all decided, not inferred:
  - continuation prompts, never format_prompt; a BASE checkpoint is refused
    (score_matrix.classify must read kind sft/rl);
  - reward_fn(code, tests) for call-style rows, reward_fn_stdin(code, cases) for
    stdin rows; model rollouts always run isolated, ALLOW_UNISOLATED=1 raises;
  - advantage is mean-subtracted in the group and NOT divided by std; a constant
    group is exactly zero (see group_advantage);
  - memory design (1e order 2026-09-25): bf16 params, no fp32 master copy. Muon
    keeps fp32 momentum buffers and writes back with stochastic rounding once de's
    train.Muon(stochastic_round=True, momentum_dtype=fp32) lands; generation runs
    on the training model itself in eval()+no_grad, so there is no second model
    copy; only the frozen bf16 KL reference is duplicated (~6 GB at 3.2B).

Heavy deps (torch, train.py) import lazily; this module imports without torch/GPU.

Usage: torchrun --nproc_per_node=8 algorithms/rl_code.py --resume ckpt_sft.pt
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CODE_POOL_DIR = os.path.join(ROOT, "data", "rl", "code_pool")
CKPT_RLCODE = os.path.join(ROOT, "ckpt_rlcode.pt")

#: call-style prompts are short (pool p90 334 tokens); stdin statements are long
#: (p90 831) and their answers longer, so the two modes get different context caps.
MAX_PROMPT_CALL = 512
MAX_PROMPT_STDIN = 1024
#: stdin max_new basis (1e order 2026-09-25): p90 of the 9,504-row stdin pool's
#: reference SOLUTION BODY length is 562 gate-tokenizer tokens (APPS 182 + TACO
#: 9,322, measured 2026-09-25 on digest /data00/home/chenkailun.c/sfta/rl_pool).
#: The builder prepends a ~90-token stdlib header (IMPL_HEADER) to the reference it
#: executes; the model never sees that header in a continuation prompt, so the
#: distribution the bound must cover is the BODY (full-impl p90 would be 653).
MAX_NEW_CALL = 280
MAX_NEW_STDIN = 562

DEGEN_REFUSE_STEPS = 20

try:
    from .code_reward import reward_fn, reward_fn_stdin
    from .rlvr_trainer import group_advantage, seq_logprob
    from .rlvr_generate import generate
except ImportError:
    from code_reward import reward_fn, reward_fn_stdin
    from rlvr_trainer import group_advantage, seq_logprob
    from rlvr_generate import generate


def load_code_pool(pool_dir=CODE_POOL_DIR):
    """All call-style and stdin code rows under pool_dir, excluding quarantines/stats.

    Each returned row is the on-disk shape from scripts/rl_code_pool.py:
      call:  {kind absent/"call", prompt, tests, entry, ...}
      stdin: {kind: "stdin", prompt, cases: [{input, output, rel_tol?, abs_tol?}], ...}
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(pool_dir, "rl_code_*.jsonl"))):
        b = os.path.basename(p)
        if b.endswith("_nondet.jsonl"):
            continue  # quarantined references, never training data
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def program_source(prompt, completion):
    """The executable module a continuation produced: prompt prefix + completion text.

    The prompt is a valid code prefix (def signature + docstring, or a module
    docstring); the model's continuation completes it. The longest line-contiguous
    parseable span is what gets executed, so prose or a trailing fenced block around
    the code does not score (same extractor eval/score_code_exec.py uses). No import
    header is injected: that would execute code the model did not write.
    """
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from score_code_exec import longest_parseable

    text = (prompt or "") + (completion or "")
    return longest_parseable(text)


def score_row(row, source, timeout_call=30, timeout_stdin=10):
    """Binary reward for one reconstructed program against one pool row.

    Returns float in {0.0, 1.0}. call-style rows carry pytest `tests`; stdin rows
    carry `cases`. Both reward entries refuse to run under ALLOW_UNISOLATED=1.
    """
    if not source.strip():
        return 0.0
    if row.get("kind") == "stdin":
        return float(reward_fn_stdin(source, row["cases"], timeout=timeout_stdin))
    return float(reward_fn(source, row["tests"], timeout=timeout_call))


def _sr_bf16_fallback(x, gen=None):
    """Unbiased fp32 -> bf16 stochastic rounding in the magnitude domain.

    Temporary local copy until de's scripts/sr_cast.py lands (same algorithm, exact
    bf16 bit pattern, known-answer gated there): bf16 truncates fp32 to the top 16
    bits; a truncated lower-half remainder rounds UP with probability equal to its
    16-bit fraction, so E[round(x)] = x. Magnitude domain because fp32 is
    sign-magnitude -- adding the bias to the raw signed bit pattern biases negatives.
    Delete this and `from scripts.sr_cast import stochastic_round_bf16` when #de-SR
    merges.
    """
    import torch

    b = x.contiguous().view(torch.int32)
    sign = b & 0x80000000
    mag = b & 0x7FFFFFFF
    lo = mag & 0xFFFF0000
    rem = (mag & 0xFFFF).to(torch.float32)
    if gen is None:
        draw = torch.randint(0, 1 << 16, rem.shape, device=x.device, dtype=torch.int32)
    else:
        draw = torch.randint(0, 1 << 16, rem.shape, device=x.device, dtype=torch.int32,
                             generator=gen)
    up = (draw.to(torch.float32) < rem).to(torch.int32) * 0x10000
    rounded = (sign | (lo + up)).view(torch.float32)
    return rounded.to(torch.bfloat16)


def build_optimizer(params, lr, muon_lr=None):
    """Muon on 2-D params, AdamW on the rest, both holding bf16 params.

    Muon gets stochastic_round=True + fp32 momentum when de's constructor supports
    it; an older train.Muon falls back to its stock bf16 path (the SR writeback then
    waits on the merge, which is the only behavior difference). AdamW params (1-D
    norms/embeddings) are stepped in fp32 and stochastically rounded back by the
    caller via sr_add_, so their small updates survive without a master copy.
    """
    import inspect

    from train import Muon

    matrices = [p for p in params if p.ndim == 2]
    rest = [p for p in params if p.ndim != 2]
    opts = []
    if matrices:
        kw = {"lr": muon_lr or lr, "momentum": 0.95, "weight_decay": 0.0}
        if {"stochastic_round", "momentum_dtype"} <= set(inspect.signature(Muon).parameters):
            import torch
            kw["stochastic_round"] = True
            kw["momentum_dtype"] = torch.float32
        opts.append(Muon(matrices, **kw))
    if rest:
        opts.append(torch.optim.AdamW(rest, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True))
    return opts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="SFT checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4, help="prompts per GPU per step")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--muon_lr", type=float, default=None, help="2-D param lr (default: --lr)")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--pool_dir", default=CODE_POOL_DIR)
    parser.add_argument("--out", default=CKPT_RLCODE)
    args = parser.parse_args()

    # Model rollouts are untrusted and MUST run isolated. ALLOW_UNISOLATED=1 exists for the
    # offline pool builder executing trusted references; a trainer launched from a shell that
    # exported it would otherwise score generated code with rlimits only. Pop it here and let
    # the reward's own guard stay as the second refusal.
    if os.environ.pop("ALLOW_UNISOLATED", None) == "1":
        print("WARNING: ALLOW_UNISOLATED=1 was set in the environment; removed for RL rollouts",
              file=sys.stderr, flush=True)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import datetime

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    sys.path.insert(0, ROOT)
    from scripts.loader import load_checkpoint, load_tokenizer
    from train import RunLog, save_checkpoint

    torch.manual_seed(1337)
    random.seed(1337)
    torch.set_float32_matmul_precision("high")
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
        ddp, rank, world = True, dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
    else:
        ddp, rank, world, local = False, 0, 1, 0
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0
    runlog = RunLog("rlcode") if is_main else print
    amp = device.startswith("cuda")
    if ddp:
        torch.manual_seed(1337 + rank)
        random.seed(1337)

    if not os.path.exists(args.resume):
        print(f"ERROR: checkpoint not found: {args.resume}")
        sys.exit(1)

    model, cfg = load_checkpoint(args.resume, device=device)
    # continuation RL is defined on a ChatML-taught checkpoint: same gating as the
    # math trainer, but here the prompts are raw prefixes the SFT-A code pack taught.
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from score_matrix import classify

    kind = classify(cfg, os.path.basename(args.resume))
    if kind == "base":
        raise SystemExit(
            f"refusing: {args.resume} classifies as BASE. Code RL continues raw code "
            f"prefixes; run the code SFT stage first."
        )
    if is_main:
        print(f"rlcode on a {kind} checkpoint, raw-continuation prompts", flush=True)

    cfg.grad_ckpt = True
    model.grad_ckpt = True
    model = model.to(torch.bfloat16)
    model.train()
    # One frozen bf16 KL reference, the only duplicated weights.
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_core = model
    if ddp:
        model = DDP(model, device_ids=[local], bucket_cap_mb=100, gradient_as_bucket_view=True)

    tok = load_tokenizer(os.path.join(ROOT, "data", "tokenizer.json"), cfg)
    problems = load_code_pool(args.pool_dir)
    if not problems:
        raise SystemExit(f"no code RL rows under {args.pool_dir}")
    optimizers = build_optimizer(list(train_core.parameters()), args.lr, args.muon_lr)

    if is_main:
        n_call = sum(1 for r in problems if r.get("kind") != "stdin")
        print(
            f"RLCODE: {sum(p.numel() for p in train_core.parameters()) / 1e6:.1f}M params | "
            f"{len(problems)} rows ({n_call} call, {len(problems) - n_call} stdin), "
            f"N={args.group_size} batch={args.batch} lr={args.lr} world {world}",
            flush=True,
        )

    def _sr(x):
        if x.device.type != "cuda":
            return x.to(torch.bfloat16)  # CPU smoke: exact cast, SR is GPU-writeback path
        return _sr_bf16_fallback(x)

    tot_groups = tot_degenerate = tot_opt_steps = 0
    tot_truncated = 0  # rollouts that hit max_new with no <eos>; they score 0 like a wrong answer

    for step in range(1, args.steps + 1):
        if ddp:
            random.seed(7337 + step)
        batch = random.sample(problems, args.batch)

        t0 = time.time()
        groups = []
        # GENERATION RUNS ON THE TRAIN MODEL ITSELF (no second copy): eval() + no_grad for
        # sampling, train() again before the forward-with-grad. DDP sees the same module.
        train_core.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            for item in batch:
                is_stdin = item.get("kind") == "stdin"
                max_prompt = MAX_PROMPT_STDIN if is_stdin else MAX_PROMPT_CALL
                max_new = MAX_NEW_STDIN if is_stdin else MAX_NEW_CALL
                prompt_ids = tok.encode(item["prompt"]).ids[-max_prompt:]
                with torch.no_grad():
                    gen_ids_list = generate(
                        train_core if not ddp else model, prompt_ids, args.group_size,
                        max_new, args.temperature, args.top_p, device,
                    )
                rewards = []
                trunc = 0
                for g in gen_ids_list:
                    if len(g) >= max_new:
                        trunc += 1
                    completion = tok.decode(g, skip_special_tokens=True)
                    src = program_source(item["prompt"], completion)
                    rewards.append(score_row(item, src))
                tot_truncated += trunc
                with torch.no_grad():
                    old_lp, _, _ = seq_logprob(
                        train_core if not ddp else model, prompt_ids, gen_ids_list,
                        args.group_size, max_new, ddp, device, amp)
                groups.append((item, prompt_ids, gen_ids_list, rewards, old_lp.detach(),
                               max_new))
        gen_time = time.time() - t0
        train_core.train()

        if ddp:
            dist.barrier()

        keep = torch.tensor(
            [1.0 if 0 < sum(g[3]) < len(g[3]) else 0.0 for g in groups], device=device)
        if ddp:
            dist.all_reduce(keep, op=dist.ReduceOp.MAX)
        kept = [g for g, k in zip(groups, keep.tolist()) if k > 0.5]
        tot_groups += len(groups)
        tot_degenerate += len(groups) - len(kept)
        if not kept:
            if is_main:
                runlog(f"step {step}/{args.steps} all groups degenerate, skipped")
            if step >= DEGEN_REFUSE_STEPS and tot_degenerate == tot_groups:
                raise SystemExit(
                    f"refusing: all {tot_groups} groups degenerate over the first "
                    f"{DEGEN_REFUSE_STEPS} steps; no optimizer step ran. Check the reward "
                    f"fires on THIS pool (isolation available, prompts continue to code).")
            continue

        losses = []
        for item, prompt_ids, gen_ids_list, rewards, old_lp, max_new in kept:
            losses.append(gspo_code_loss(
                model, ref_model, prompt_ids, gen_ids_list, rewards,
                args.group_size, max_new, ddp, device, amp,
                clip_eps=args.clip_eps, kl_beta=args.kl_beta, old_lp=old_lp,
            ))
        loss = torch.stack(losses).mean()
        loss_val = loss.item()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(train_core.parameters(), 1.0)
        if not math.isfinite(gnorm) or not math.isfinite(loss_val):
            if is_main:
                print(f"step {step} NaN, skipped", flush=True)
            continue
        # Muon steps its own 2-D params (SR inside once de's flag is live). The AdamW
        # 1-D params are stepped as fp32 deltas and stochastically rounded back so a
        # 1e-6 update is not absorbed by the bf16 ULP.
        for opt in optimizers:
            if isinstance(opt, torch.optim.AdamW):
                for g in opt.param_groups:
                    for p in g["params"]:
                        if p.grad is None:
                            continue
                        # One fp32 AdamW update from the bf16 param, written back through the
                        # SR cast; moments live in opt.state as fp32, no permanent master.
                        _adam_step_fp32(p, opt.state[p], g, _sr)
            else:
                opt.step()
        tot_opt_steps += 1

        if step % 10 == 0:
            acc = sum(sum(g[3]) for g in groups) / max(sum(len(g[3]) for g in groups), 1)
            if ddp:
                t = torch.tensor([acc, loss_val, gen_time, float(tot_degenerate)], device=device)
                dist.all_reduce(t)
                acc, loss_val, gen_time, degen = (t / world).tolist()
            if is_main:
                runlog(f"step {step}/{args.steps} acc {acc:.3f} loss {loss_val:.4f} "
                       f"gen {gen_time:.0f}s degen {int(tot_degenerate)} trunc {tot_truncated}")

        if step % 200 == 0 or step == args.steps:
            if ddp:
                dist.barrier()
            if is_main:
                sd = {n: p.detach().cpu() for n, p in train_core.state_dict().items()}
                # tied embedding alias, same alias the math RL trainer writes
                if "tok.weight" in sd and "head.weight" not in sd:
                    sd["head.weight"] = sd["tok.weight"]
                save_checkpoint(args.out, sd, cfg, cfg.vocab_id, step=step)
                print(f"saved {args.out} (step {step})", flush=True)
            if ddp:
                dist.barrier()

    if ddp:
        dist.destroy_process_group()
    if tot_opt_steps == 0:
        raise SystemExit(
            f"refusing: {args.steps} step(s) ran and NO optimizer step was applied "
            f"({tot_degenerate}/{tot_groups} groups degenerate).")


def gspo_code_loss(model, ref_model, prompt_ids, gen_ids_list, rewards, group_size,
                   max_new, ddp, device, amp, clip_eps=0.2, kl_beta=0.02, old_lp=None):
    """GSPO loss with mean-only advantages (no std division); same GSPO clip + KL."""
    import torch

    adv = group_advantage(rewards, normalize_std=False).to(device)
    seq_lp, _, _ = seq_logprob(model, prompt_ids, gen_ids_list, group_size, max_new,
                               ddp, device, amp)
    with torch.no_grad():
        ref_lp, _, _ = seq_logprob(ref_model, prompt_ids, gen_ids_list, group_size,
                                   max_new, ddp, device, amp)
    if old_lp is None:
        raise ValueError("gspo_code_loss needs old_lp, the rollout policy's log-probs")
    ratio = torch.exp(seq_lp - old_lp.to(device))
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    d = ref_lp - seq_lp
    kl = torch.exp(d) - d - 1.0
    return -(surr - kl_beta * kl).mean()


def _adam_step_fp32(p, st, group, sr):
    """One AdamW update computed in fp32, written back with stochastic rounding.

    No permanent master copy: moments live in the optimizer state as fp32, the
    fp32 weight is reconstructed from the bf16 param each step. E[bf16 weight after
    write] equals the fp32 result, so 1e-6-scale updates are not rounded to zero.
    """
    import torch

    lr = group["lr"]
    b1, b2 = group["betas"]
    eps = group.get("eps", 1e-8)
    wd = group["weight_decay"]
    w = p.detach().float()
    g = p.grad.detach().float()
    if wd:
        g = g.add(w, alpha=wd)
    if "exp_avg" not in st:
        st["exp_avg"] = torch.zeros_like(w)
        st["exp_avg_sq"] = torch.zeros_like(w)
        st["step"] = torch.tensor(0, dtype=torch.long)
    m, v = st["exp_avg"], st["exp_avg_sq"]
    st["step"] += 1
    t = int(st["step"])
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    w.add_(mh / (vh.sqrt() + eps), alpha=-lr)
    p.data.copy_(sr(w))


if __name__ == "__main__":
    main()
