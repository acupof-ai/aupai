#!/usr/bin/env python3
"""RLVR trainer: GRPO on 218K Chinese math problems with verifiable \\boxed{} rewards.

Per step:
  1. Sample 4 prompts/GPU; generate N=8 responses each (bf16, T=0.8, top-p=0.95)
  2. Extract \\boxed{} answers, normalize, compare with ground truth -> reward {0,1}
  3. GRPO: advantage = (r - mean) / (std + 1e-8) per prompt group;
     loss = -mean(advantage * log_prob(response))
  4. fp32 master-weight AdamW (lr=1e-6); bf16 FP8 training model synced each step

Why fp32 master weights: a 1e-6 AdamW update is far below bf16 ULP (~5e-4 at
0.1), so bf16 params would absorb nothing. Optimizer steps on fp32 master;
both bf16 copies (FP8 train, plain bf16 generation) are synced from it.

Heavy deps (torch, train.py/fla) are imported lazily in main()/grpo_loss(),
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


def grpo_loss(model, prompt_ids, gen_ids_list, rewards, group_size, max_new, ddp, device, amp):
    """One prompt group: group-relative advantage x length-normalized log-prob.
    adv=0 when all rewards are equal (std=0), so the group contributes zero
    loss — never skip a group in DDP (all ranks need identical forward counts)."""
    import torch
    import torch.nn.functional as F

    r = torch.tensor(rewards, device=device)
    adv = (r - r.mean()) / (r.std() + 1e-8)  # 0 when all rewards equal
    plen = len(prompt_ids)
    L = max_new if ddp else max(len(g) for g in gen_ids_list)
    # FP8 _scaled_mm needs token count (group_size*T) divisible by 16
    q = 16 // math.gcd(group_size, 16)  # T must be a multiple of q
    L += (-(plen + L)) % q
    gen_t = torch.zeros((group_size, L), dtype=torch.long, device=device)
    mask = torch.zeros((group_size, L), device=device)
    for i, g in enumerate(gen_ids_list):
        gen_t[i, : len(g)] = torch.tensor(g, device=device)
        mask[i, : len(g)] = 1.0
    full = torch.cat(
        [torch.tensor(prompt_ids, device=device).repeat(group_size, 1), gen_t], dim=1
    )
    # Right-padding is safe: the model is causal (KDA/MLA), so hidden[t]
    # depends only on [:t+1]; pad positions after a response never feed back.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        logits, _ = model(full)
    log_probs = F.log_softmax(logits[:, plen - 1 : -1, :].float(), dim=-1)
    token_lps = log_probs.gather(-1, gen_t.unsqueeze(-1)).squeeze(-1)
    seq_lp = (token_lps * mask).sum(1) / mask.sum(1).clamp(min=1)  # length-normalized
    return -(adv * seq_lp).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="SFT checkpoint (e.g. ckpt_sft.pt)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4, help="prompts per GPU per step")
    parser.add_argument("--group_size", type=int, default=8, help="responses per prompt (N)")
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--no_fp8", action="store_true")
    parser.add_argument("--data", default=RLVR_PATH, help="problem jsonl (default: rlvr_math.jsonl)")
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
        model = DDP(
            raw_model, device_ids=[local], bucket_cap_mb=100, gradient_as_bucket_view=True
        )
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

    log = {"reward": 0.0, "n": 0, "loss": 0.0, "n_loss": 0, "gnorm": 0.0, "gen": 0.0}
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
                    gen_model, prompt_ids, args.group_size,
                    args.max_new, args.temperature, args.top_p, device,
                )
                rewards = [
                    reward_fn(tok.decode(g, skip_special_tokens=True), item["answer"])
                    for g in gen_ids_list
                ]
                groups.append((prompt_ids, gen_ids_list, rewards))
        gen_time = time.time() - t0

        # Sync ranks: generation time varies (different prompt/response lengths),
        # so fast ranks must wait before DDP all-reduce (NCCL timeout is 10min).
        if ddp:
            dist.barrier()

        # --- 2. GRPO loss (fp32 master gets grads via bf16/FP8 model) ---
        # DDP: all ranks must compute the same number of forward/backward
        # passes. Never skip a group — adv=0 when std=0, so loss is 0.
        losses = [
            grpo_loss(
                model, prompt_ids, gen_ids_list, rewards,
                args.group_size, args.max_new, ddp, device, amp,
            )
            for prompt_ids, gen_ids_list, rewards in groups
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
                runlog(
                    f"step {step}/{args.steps} acc {t[0] / max(t[1], 1):.3f} "
                    f"loss {t[2] / max(t[3], 1):.4f} gnorm {t[4] / max(t[3], 1):.3f} "
                    f"gen {t[5] / 10:.0f}s"
                )
            log = {"reward": 0.0, "n": 0, "loss": 0.0, "n_loss": 0, "gnorm": 0.0, "gen": 0.0}

        # --- 4. Save every 200 steps + final ---
        if step % 200 == 0 or step == args.steps:
            if ddp:
                dist.barrier()
            if is_main:
                sd = {n: m.detach().cpu() for n, m in master.items()}
                sd["head.weight"] = sd["tok.weight"]  # tied embedding alias
                torch.save(
                    {
                        "model": sd,
                        "cfg": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
                        "step": step,
                    },
                    CKPT_RLVR,
                )
                # bf16 inference copy: half the size, zero quality loss (inference runs bf16 anyway)
                sd_bf16 = {
                    n: (t.to(torch.bfloat16) if torch.is_tensor(t) and t.is_floating_point() else t)
                    for n, t in sd.items()
                }
                torch.save(
                    {
                        "model": sd_bf16,
                        "cfg": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
                        "step": step,
                    },
                    CKPT_RLVR.replace(".pt", "_bf16.pt"),
                )
                print(f"saved {CKPT_RLVR} (step {step}) + bf16 copy", flush=True)
            if ddp:
                dist.barrier()

    if is_main:
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
