#!/usr/bin/env python3
"""GRPO RL training for math reasoning on GSM8K-zh.
Proper implementation: stored log-probs, KL penalty, clipped surrogate.
Heavy deps (torch, sft/train) are imported lazily so this module is
importable without torch/GPU.
Usage: torchrun --nproc_per_node=N algorithms/rl.py [--steps 200] [--group_size 8]
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # project root (for sft import)

CKPT_SFT = os.path.join(ROOT, "ckpt_sft.pt")
CKPT_RL = os.path.join(ROOT, "ckpt_rl.pt")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
GSM8K_PATH = os.path.join(ROOT, "data", "sft", "gsm8k_zh.json")

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>"]


def forward_logits(model, x):
    """Handle tuple return from train.py HybridLM."""
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def extract_answer(text):
    """Extract numerical answer from generated text."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"答案是[：:]?\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def check_answer(generated, reference):
    """Check answer with reward shaping: partial credit for parseable output."""
    gen_ans = extract_answer(generated)
    if gen_ans is None:
        return 0.0
    ref_clean = re.sub(r"<<[^>]*>>", "", reference)
    ref_nums = re.findall(r"-?[\d,]+(?:\.\d+)?", ref_clean)
    if not ref_nums:
        return 0.0
    ref_ans = ref_nums[-1].replace(",", "")
    try:
        if abs(float(gen_ans) - float(ref_ans)) < 1e-6:
            return 1.0
        # Partial credit: parseable answer in correct format
        if "####" in generated or "答案是" in generated:
            return 0.3
        return 0.1
    except ValueError:
        return 0.0


def generate_with_logprobs(model, tok, prompt_ids, max_new=256, temperature=1.0, device="cuda"):
    """Generate one sequence and store per-token log-probs."""
    import torch

    eos = tok.token_to_id("<|im_end|>")
    x = torch.tensor([prompt_ids], device=device)
    gen_ids = []
    old_lps = []

    for _ in range(max_new):
        with torch.no_grad():
            logits = forward_logits(model, x[:, -1024:])[:, -1] / temperature
            logits[:, tok.get_vocab_size() :] = float("-inf")  # mask padding tokens
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            lp = torch.log(probs.gather(-1, nxt) + 1e-8).squeeze()
            gen_ids.append(nxt.item())
            old_lps.append(lp.item())
            x = torch.cat([x, nxt], dim=1)
            if nxt.item() == eos:
                break

    return gen_ids, old_lps


def grpo_loss(
    model,
    ref_model,
    prompt_ids,
    gen_ids_list,
    old_lps_list,
    advantages,
    device="cuda",
    beta=0.01,
    clip_eps=0.2,
):
    """Proper GRPO: clipped policy gradient + KL penalty.
    Returns None if all groups are uninformative."""
    import torch
    import torch.nn.functional as F

    total_loss = 0.0
    n = 0

    for gen_ids, old_lps, adv in zip(gen_ids_list, old_lps_list, advantages):
        if abs(adv) < 1e-8:
            continue

        full_ids = prompt_ids + gen_ids
        x = torch.tensor([full_ids], device=device)
        gen_t = torch.tensor(gen_ids, device=device)
        old_lps_t = torch.tensor(old_lps, device=device)

        # Policy log-probs (with gradient)
        logits = forward_logits(model, x)
        log_probs = F.log_softmax(logits[:, len(prompt_ids) - 1 : -1], dim=-1)
        token_lps = log_probs.gather(-1, gen_t[None, :, None]).squeeze(-1)

        # Clipped surrogate
        ratio = torch.exp(token_lps - old_lps_t)
        adv_t = torch.tensor(adv, device=device)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
        pg_loss = -torch.min(surr1, surr2).mean()

        # KL penalty (approximate: log π - log π_ref)
        with torch.no_grad():
            ref_logits = forward_logits(ref_model, x)
            ref_log_probs = F.log_softmax(ref_logits[:, len(prompt_ids) - 1 : -1], dim=-1)
            ref_token_lps = ref_log_probs.gather(-1, gen_t[None, :, None]).squeeze(-1)
        kl = (token_lps - ref_token_lps).mean()

        total_loss += pg_loss + beta * kl
        n += 1

    if n == 0:
        return None
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max_new", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from tokenizers import Tokenizer

    from sft import HybridLM

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world, local = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    # Load SFT model
    if not os.path.exists(CKPT_SFT):
        print(f"ERROR: SFT checkpoint not found at {CKPT_SFT}")
        sys.exit(1)

    ck = torch.load(CKPT_SFT, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = HybridLM(cfg).to(device)
    model.load_state_dict(ck["model"])

    # Frozen reference model
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tok = Tokenizer.from_file(TOK_PATH)
    for t in SPECIAL_TOKENS:
        if tok.token_to_id(t) is None:
            tok.add_special_tokens([t])

    if is_main:
        print(
            f"RL: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
            f"steps={args.steps}, group={args.group_size}, lr={args.lr}, T={args.temperature}",
            flush=True,
        )

    # Load GSM8K questions
    if not os.path.exists(GSM8K_PATH):
        print(f"ERROR: GSM8K not found at {GSM8K_PATH}")
        sys.exit(1)

    with open(GSM8K_PATH) as f:
        gsm8k = json.load(f)

    questions = []
    for d in gsm8k:
        if d.get("split") != "train":
            continue
        q = d.get("question_zh") or ""
        a = d.get("answer_zh") or ""
        if q and a:
            a = re.sub(r"<<[^>]*>>", "", a)
            questions.append({"q": q, "a": a})

    if is_main:
        print(f"GSM8K: {len(questions)} questions", flush=True)

    # Shard across ranks
    if ddp:
        questions = questions[rank::world]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    step = 0
    t0 = time.time()

    while step < args.steps:
        batch = random.sample(questions, min(args.batch, len(questions)))

        all_losses = []
        all_correct = 0
        all_total = 0

        for item in batch:
            prompt = f"<|im_start|>user\n{item['q']}<|im_end|>\n<|im_start|>assistant\n"
            prompt_ids = tok.encode(prompt).ids

            # Generate G solutions
            gen_ids_list = []
            old_lps_list = []
            rewards = []

            model.eval()
            for _ in range(args.group_size):
                gen_ids, old_lps = generate_with_logprobs(
                    model, tok, prompt_ids, max_new=args.max_new, temperature=args.temperature, device=device
                )
                text = tok.decode(gen_ids, skip_special_tokens=True)
                reward = check_answer(text, item["a"])
                gen_ids_list.append(gen_ids)
                old_lps_list.append(old_lps)
                rewards.append(reward)

            # Group-relative advantage
            rewards_t = torch.tensor(rewards, device=device)
            mean_r = rewards_t.mean()
            std_r = rewards_t.std()
            if std_r > 1e-8:
                advantages = ((rewards_t - mean_r) / (std_r + 1e-8)).tolist()
            else:
                advantages = [0.0] * len(rewards)

            # Compute GRPO loss
            model.train()
            loss = grpo_loss(
                model,
                ref_model,
                prompt_ids,
                gen_ids_list,
                old_lps_list,
                advantages,
                device=device,
                beta=args.beta,
                clip_eps=args.clip_eps,
            )

            if loss is not None:
                all_losses.append(loss)

            all_correct += sum(rewards)
            all_total += len(rewards)

        if not all_losses:
            if is_main:
                print(f"step {step + 1}/{args.steps} no informative groups, skipping", flush=True)
            step += 1
            continue

        # Average loss across batch
        loss = torch.stack(all_losses).mean()

        optimizer.zero_grad()
        loss.backward()

        # Manual gradient sync for DDP
        if ddp:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        if is_main and step % 10 == 0:
            acc = all_correct / max(all_total, 1)
            dt = time.time() - t0
            print(f"step {step}/{args.steps} acc {acc:.2f} loss {loss.item():.4f} {dt:.0f}s", flush=True)
            t0 = time.time()

    # Save RL checkpoint
    if is_main:
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": vars(cfg),
                "rl_steps": step,
            },
            CKPT_RL,
        )
        print(f"saved {CKPT_RL}", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
