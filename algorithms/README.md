# algorithms/

RL training loops and verifiable-reward utilities for the 200M Chinese LLM.

## Layout

| File | Role | Needs torch to import? |
|---|---|---|
| `rlvr_reward.py` | `\boxed{}` extraction, answer normalization, 0/1 reward | No (stdlib only) |
| `rlvr_generate.py` | Batched top-p autoregressive sampling | No (lazy on call) |
| `rlvr_trainer.py` | RLVR GRPO loop: fp32 master weights, FP8 train + bf16 gen copies, DDP | No (lazy) |
| `rlvr_data.py` | Build/load `data/rl/rlvr_math.jsonl` from raw math datasets | No (stdlib only) |
| `rlvr.py` | Entry point -> `rlvr_trainer.main()` | — |
| `prepare_rlvr.py` | Entry point -> `rlvr_data.main()` | — |
| `rl.py` | GRPO on GSM8K-zh (stored log-probs, clipped surrogate + KL) | No (lazy) |
| `rl_arc.py` | GRPO on ARC-Easy multiple-choice | No (lazy) |
| `rl_ceval.py` | GRPO on C-Eval multiple-choice | No (lazy) |

All paths resolve from the project root (`os.path.dirname` of this directory),
so scripts run from anywhere. Heavy deps (torch, tokenizers, `train`/`sft`
which pull fla/Triton) are imported lazily — `import algorithms` works on a
CPU-only box; torch is only loaded when a training/generation function runs.

## Usage

```bash
# Prepare RLVR data (school_math_r1_zh + gsm8k_zh -> data/rl/rlvr_math.jsonl)
python algorithms/prepare_rlvr.py

# RLVR training (single GPU or DDP)
torchrun --nproc_per_node=8 algorithms/rlvr.py --resume ckpt_sft.pt

# Other GRPO loops
torchrun --nproc_per_node=N algorithms/rl.py        # GSM8K-zh -> ckpt_rl.pt
torchrun --nproc_per_node=N algorithms/rl_arc.py    # ARC-Easy -> ckpt_rl.pt
torchrun --nproc_per_node=N algorithms/rl_ceval.py  # C-Eval -> ckpt_rl.pt
```

Note: `rl_arc.py` and `rl_ceval.py` both write `ckpt_rl.pt` — run them in
separate checkpoints/workdirs if both matter.

## Importing as a library

```python
from algorithms import reward_fn, generate, load_problems, train_rlvr

reward_fn(r"答案是 \boxed{\frac{1}{2}}", r"\dfrac{1}{2}")  # 1.0
problems = load_problems()  # [{prompt, answer, source}, ...]
```

Submodules are also importable directly: `from algorithms.rlvr_reward import normalize_answer`.

## RLVR design notes

- **fp32 master weights**: a 1e-6 AdamW update is far below bf16 ULP (~5e-4 at
  0.1), so the optimizer steps on fp32 master; both bf16 copies (FP8 train,
  plain bf16 generation) sync from it each step.
- **Two model copies**: FP8 for training, plain bf16 for generation — FP8
  quantization noise degrades sampling.
- **DDP**: all ranks sample the same prompts (same seed per step) but generate
  different responses (different torch seed per rank). Never skip a group —
  adv=0 when std=0, so the loss is exactly 0 and all ranks keep identical
  forward/backward counts.
