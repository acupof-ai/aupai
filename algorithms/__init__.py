"""RL algorithms for the 200M Chinese LLM: GRPO/RLVR training loops and
verifiable-reward utilities.

All heavy deps (torch, tokenizers, train.py/fla) are lazy: `import algorithms`
and `from algorithms import reward_fn` work on a CPU-only box. torch is only
imported when a training/generation function is actually called.

Submodules:
  rlvr_reward    — \\boxed{} extraction, answer normalization, 0/1 reward (stdlib only)
  rlvr_generate  — batched top-p autoregressive sampling (lazy torch)
  rlvr_trainer   — RLVR GRPO training loop: fp32 master weights, FP8 train +
                   bf16 generation copies, DDP (lazy torch/train)
  rlvr_data      — build/load data/rl/rlvr_math.jsonl (stdlib only)
"""

__all__ = [
    "reward_fn",
    "normalize_answer",
    "to_number",
    "generate",
    "grpo_loss",
    "load_problems",
    "prepare_rlvr_data",
    "train_rlvr",
]

# name -> (submodule, attribute); attribute defaults to name when only the
# submodule is given.
_LAZY = {
    "reward_fn": "rlvr_reward",
    "normalize_answer": "rlvr_reward",
    "to_number": "rlvr_reward",
    "generate": "rlvr_generate",
    "grpo_loss": "rlvr_trainer",
    "load_problems": "rlvr_data",
    "prepare_rlvr_data": ("rlvr_data", "prepare"),
    "train_rlvr": ("rlvr_trainer", "main"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target if isinstance(target, tuple) else (target, name)
    import importlib

    mod = importlib.import_module(f".{mod_name}", __name__)
    return getattr(mod, attr)
