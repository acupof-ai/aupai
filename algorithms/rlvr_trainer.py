#!/usr/bin/env python3
"""RLVR trainer: GSPO on Chinese math problems with verifiable \\boxed{} rewards.

Per step:
  1. Sample 4 prompts/GPU; generate N=8 responses each (bf16, T=0.8, top-p=0.95)
  2. Extract \\boxed{} answers, normalize, compare with ground truth -> reward {0,1}
  3. GRPO: advantage = (r - mean) / (std + 1e-8) per prompt group;
     loss = -mean(advantage * log_prob(response))
  4. fp32 master-weight AdamW (lr=1e-6); bf16 FP8 training model synced each step

Why fp32 master weights: a 1e-6 AdamW update is far below bf16 ULP (~5e-4 at
0.1), so bf16 params would absorb nothing. Optimizer steps on fp32 master;
both bf16 copies (FP8 train, plain bf16 generation) are synced from it.

Heavy deps (torch, train.py/fla) are imported lazily in main()/gspo_loss(),
so this module is importable without torch/GPU.

Usage: torchrun --nproc_per_node=8 algorithms/rlvr.py --resume ckpt_sft.pt
   or: torchrun --nproc_per_node=8 -m algorithms.rlvr_trainer --resume ckpt_sft.pt
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
RLVR_PATH = os.path.join(ROOT, "data", "rl", "rlvr_math.jsonl")
CKPT_RLVR = os.path.join(ROOT, "ckpt_rlvr.pt")

MAX_PROMPT = 512  # + max_new 512 = 1024 generation context

try:
    from .rlvr_reward import reward_fn
    from .rlvr_generate import generate
    from .rlvr_data import load_problems
except ImportError:
    from rlvr_reward import reward_fn
    from rlvr_generate import generate
    from rlvr_data import load_problems


def seq_logprob(model, prompt_ids, gen_ids_list, group_size, max_new, ddp, device, amp):
    """Length-normalized sequence log-prob per response, plus the padded tensors.

    Returns (seq_lp [G], gen_t [G,L], mask [G,L]). Only tokens the policy actually
    emitted are unmasked — rows are already truncated at their own <eos>.
    """
    import torch
    import torch.nn.functional as F

    plen = len(prompt_ids)
    lens = [len(g) for g in gen_ids_list]
    L = max_new if ddp else max(lens)  # DDP: identical shapes across ranks
    # FP8 _scaled_mm needs the token count (group_size * T) divisible by 16
    q = 16 // math.gcd(group_size, 16)
    L += (-(plen + L)) % q
    gen_t = torch.zeros((group_size, L), dtype=torch.long, device=device)
    mask = torch.zeros((group_size, L), device=device)
    for i, g in enumerate(gen_ids_list):
        n = min(len(g), L)
        gen_t[i, :n] = torch.tensor(g[:n], device=device)
        mask[i, :n] = 1.0
    full = torch.cat([torch.tensor(prompt_ids, device=device).repeat(group_size, 1), gen_t], dim=1)
    # Right-padding is safe: the model is causal (KDA/MLA), so hidden[t] depends
    # only on [:t+1]; pad positions after a response never feed back.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        logits, _ = model(full)
    log_probs = F.log_softmax(logits[:, plen - 1 : -1, :].float(), dim=-1)
    token_lps = log_probs.gather(-1, gen_t.unsqueeze(-1)).squeeze(-1)
    # Per-row length normalization. The previous code divided by mask.sum(1), which
    # was a group constant because every row shared the padded length.
    seq_lp = (token_lps * mask).sum(1) / mask.sum(1).clamp(min=1)
    return seq_lp, gen_t, mask


def gspo_loss(
    model,
    ref_model,
    prompt_ids,
    gen_ids_list,
    rewards,
    group_size,
    max_new,
    ddp,
    device,
    amp,
    clip_eps=0.2,
    kl_beta=0.02,
):
    """GSPO (arXiv 2507.18071): sequence-level importance ratio + sequence-level clip.

    The reward is granted to the whole sequence, so the off-policy correction is
    applied at the same granularity. Token-level ratios (GRPO) accumulate variance
    with response length and are amplified by clipping.

    A KL anchor against the frozen SFT reference keeps 500 steps from drifting —
    the previous run had no anchor and collapsed from 32% to 11%.
    """
    import torch

    r = torch.tensor(rewards, device=device)
    adv = (r - r.mean()) / (r.std() + 1e-8)  # 0 when every reward is equal
    seq_lp, gen_t, mask = seq_logprob(model, prompt_ids, gen_ids_list, group_size, max_new, ddp, device, amp)
    with torch.no_grad():
        ref_lp, _, _ = seq_logprob(ref_model, prompt_ids, gen_ids_list, group_size, max_new, ddp, device, amp)
        old_lp = seq_lp.detach()  # one optimizer step per rollout, so old == current
    ratio = torch.exp(seq_lp - old_lp)  # sequence-level, already length-normalized
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    # k3 estimator: unbiased, non-negative, and unlike (lp - ref_lp) it has a
    # non-zero gradient at theta == theta_old, so the anchor actually pulls.
    d = ref_lp - seq_lp
    kl = torch.exp(d) - d - 1.0
    return -(surr - kl_beta * kl).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="SFT checkpoint (e.g. ckpt_sft.pt)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4, help="prompts per GPU per step")
    parser.add_argument("--group_size", type=int, default=8, help="responses per prompt (N)")
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="low T makes every sample in a group identical -> std=0 -> no gradient",
    )
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip_eps", type=float, default=0.2, help="GSPO sequence-level clip")
    parser.add_argument("--kl_beta", type=float, default=0.02, help="KL anchor to the SFT reference")
    parser.add_argument("--no_fp8", action="store_true")
    parser.add_argument("--data", default=RLVR_PATH, help="problem jsonl (default: rlvr_math.jsonl)")
    parser.add_argument("--out", default=CKPT_RLVR, help="checkpoint to write")
    args = parser.parse_args()

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # Do NOT set FLA_FLASH_KDA=1 — A_log float32 errors (same reason as sft.py).
    import datetime

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from tokenizers import Tokenizer

    sys.path.insert(0, ROOT)
    from train import HybridLM, RunLog, convert_to_fp8_compute

    torch.manual_seed(1337)
    random.seed(1337)
    torch.set_float32_matmul_precision("high")
    # DDP with 1h timeout: generation takes 5-15min/step, default 10min NCCL timeout kills it
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
        ddp, rank, world = True, dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
    else:
        ddp, rank, world, local = False, 0, 1, 0
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0
    runlog = RunLog("rlvr") if is_main else print
    amp = device.startswith("cuda")
    if ddp:
        torch.manual_seed(1337 + rank)  # different generation per rank
        random.seed(1337)  # same prompts on all ranks (re-seeded per step below)

    if not os.path.exists(args.resume):
        print(f"ERROR: checkpoint not found: {args.resume}")
        sys.exit(1)

    # Model: SFT weights -> bf16. Two copies: FP8 for training, plain bf16 for
    # generation (FP8 quantization noise would degrade sampling).
    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    cfg.grad_ckpt = True  # required for stability
    raw_model = HybridLM(cfg).to(device)
    raw_model.load_state_dict(ck["model"])
    raw_model = raw_model.to(torch.bfloat16)
    gen_model = copy.deepcopy(raw_model)
    gen_model.eval()
    for p in gen_model.parameters():
        p.requires_grad = False
    # Frozen SFT reference for the KL anchor — never synced from master.
    ref_model = copy.deepcopy(raw_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    fp8 = not args.no_fp8 and amp
    if fp8:
        convert_to_fp8_compute(raw_model)

    # fp32 master weights: AdamW at lr=1e-6 cannot move bf16 params (ULP ~5e-4)
    master = {n: p.detach().float().clone() for n, p in raw_model.named_parameters()}
    for m in master.values():
        m.requires_grad = True
    opt = torch.optim.AdamW(
        list(master.values()), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True
    )

    model = raw_model
    if ddp:
        # static_graph=False: graph shape varies per step (variable gen lengths)
        model = DDP(raw_model, device_ids=[local], bucket_cap_mb=100, gradient_as_bucket_view=True)
    raw_model.train()  # grad_ckpt only active in train mode; gen_model stays eval

    tok = Tokenizer.from_file(TOK_PATH)
    problems = load_problems(args.data)
    # DDP: all ranks sample same prompts (same random seed per step),
    # generate different responses (different torch seed per rank).
    if is_main:
        print(
            f"RLVR: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M params | "
            f"{len(problems)} problems/rank (world {world}) | N={args.group_size} "
            f"batch={args.batch} lr={args.lr} fp8={fp8}",
            flush=True,
        )

    n_degenerate = 0
    log = {"reward": 0.0, "n": 0, "loss": 0.0, "n_loss": 0, "gnorm": 0.0, "gen": 0.0}

    def save_ckpt(path, step):
        sd = {n: m.detach().cpu() for n, m in master.items()}
        sd["head.weight"] = sd["tok.weight"]  # tied embedding alias
        cfg_clean = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
        torch.save({"model": sd, "cfg": cfg_clean, "step": step}, path)
        # bf16 inference copy: half the size, zero quality loss (inference runs bf16 anyway)
        sd_bf16 = {n: (t.to(torch.bfloat16) if torch.is_tensor(t) and t.is_floating_point() else t) for n, t in sd.items()}
        torch.save({"model": sd_bf16, "cfg": cfg_clean, "step": step}, path.replace(".pt", "_bf16.pt"))

    # Best-ckpt: keep the peak-smoothed-acc state, not just the latest. The 2026-08-27 direct-RL
    # run peaked at step 280, collapsed to 0.339 by step 390, and the step-400 periodic save
    # overwrote the good state -- the peak was unrecoverable.
    best_path = args.out.replace(".pt", "_best.pt")
    best_acc, acc_hist = -1.0, []

    for step in range(1, args.steps + 1):
        if ddp:
            random.seed(1337 + step)  # all ranks sample same prompts
        batch = random.sample(problems, args.batch)

        # --- 1. Generate N responses per prompt (bf16, no grad) ---
        t0 = time.time()
        groups = []  # (prompt_ids, gen_ids_list, rewards)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            for item in batch:
                prompt_ids = tok.encode(f"问：{item['prompt']}\n答：").ids[-MAX_PROMPT:]
                gen_ids_list = generate(
                    gen_model,
                    prompt_ids,
                    args.group_size,
                    args.max_new,
                    args.temperature,
                    args.top_p,
                    device,
                )
                rewards = [
                    reward_fn(tok.decode(g, skip_special_tokens=True), item["answer"]) for g in gen_ids_list
                ]
                groups.append((prompt_ids, gen_ids_list, rewards))
        gen_time = time.time() - t0

        # Sync ranks: generation time varies (different prompt/response lengths),
        # so fast ranks must wait before DDP all-reduce (NCCL timeout is 10min).
        if ddp:
            dist.barrier()

        # --- 2. Drop degenerate groups (all-correct or all-wrong) ---
        # Those have std=0 -> adv=0 -> zero gradient, so a forward/backward on them
        # is pure waste. Which groups are degenerate differs per rank (each samples
        # its own responses), so the keep-decision is all-reduced with MAX: every
        # rank runs the same forward count and DDP stays in lockstep.
        keep = torch.tensor([1.0 if 0 < sum(r) < len(r) else 0.0 for _, _, r in groups], device=device)
        if ddp:
            dist.all_reduce(keep, op=dist.ReduceOp.MAX)
        kept = [g for g, k in zip(groups, keep.tolist()) if k > 0.5]
        n_degenerate += len(groups) - len(kept)
        if not kept:
            # Every prompt this step was trivially solved or hopeless; skipping is
            # safe because all ranks reached the same conclusion from the same tensor.
            if is_main:
                runlog(f"step {step}/{args.steps} all groups degenerate, skipped")
            continue

        # --- 3. GSPO loss (fp32 master gets grads via the bf16/FP8 model) ---
        losses = [
            gspo_loss(
                model,
                ref_model,
                prompt_ids,
                gen_ids_list,
                rewards,
                args.group_size,
                args.max_new,
                ddp,
                device,
                amp,
                clip_eps=args.clip_eps,
                kl_beta=args.kl_beta,
            )
            for prompt_ids, gen_ids_list, rewards in kept
        ]
        loss = torch.stack(losses).mean()
        loss_val = loss.item()

        opt.zero_grad(set_to_none=True)
        raw_model.zero_grad(set_to_none=True)
        loss.backward()

        # bf16 grads -> fp32 master
        for n, p in raw_model.named_parameters():
            if p.grad is not None:
                if master[n].grad is None:
                    master[n].grad = torch.zeros_like(master[n])
                master[n].grad.copy_(p.grad)
        gnorm = torch.nn.utils.clip_grad_norm_(list(master.values()), 1.0)
        if not math.isfinite(gnorm) or not math.isfinite(loss_val):
            opt.zero_grad(set_to_none=True)
            if is_main:
                print(f"step {step}/{args.steps} NaN, skipped", flush=True)
            continue
        opt.step()

        # sync master -> both bf16 copies
        with torch.no_grad():
            for n, p in raw_model.named_parameters():
                p.data.copy_(master[n].data)
            for n, p in gen_model.named_parameters():
                p.data.copy_(master[n].data)

        log["reward"] += sum(sum(r) for _, _, r in groups)
        log["n"] += sum(len(r) for _, _, r in groups)
        log["loss"] += loss_val
        log["n_loss"] += 1
        log["gnorm"] += gnorm
        log["gen"] += gen_time

        # --- 3. Log every 10 steps (globally averaged) ---
        if step % 10 == 0:
            t = torch.tensor(
                [log["reward"], log["n"], log["loss"], log["n_loss"], log["gnorm"], log["gen"]],
                device=device,
            )
            if ddp:
                dist.all_reduce(t)
            if is_main:
                acc = t[0] / max(t[1], 1)
                runlog(
                    f"step {step}/{args.steps} acc {acc:.3f} "
                    f"loss {t[2] / max(t[3], 1):.4f} gnorm {t[4] / max(t[3], 1):.3f} "
                    f"gen {t[5] / 10:.0f}s degen {n_degenerate}"
                )
                # 30-step smoothing: a 10-step window is noise (0.34-0.77 swings), and a collapsing
                # run can still print one good window. Save when the smoothed mean makes a new peak.
                acc_hist.append(acc)
                acc_hist[:] = acc_hist[-3:]
                smoothed = sum(acc_hist) / len(acc_hist)
                if smoothed > best_acc:
                    best_acc = smoothed
                    save_ckpt(best_path, step)
            n_degenerate = 0
            log = {"reward": 0.0, "n": 0, "loss": 0.0, "n_loss": 0, "gnorm": 0.0, "gen": 0.0}

        # --- 4. Save every 200 steps + final ---
        if step % 200 == 0 or step == args.steps:
            if ddp:
                dist.barrier()
            if is_main:
                save_ckpt(args.out, step)
                print(f"saved {args.out} (step {step}) + bf16 copy", flush=True)
            if ddp:
                dist.barrier()

    if is_main:
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
