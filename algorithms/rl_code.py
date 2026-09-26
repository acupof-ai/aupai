#!/usr/bin/env python3
"""Code RL entry point — implementation in rl_code_trainer.py.

Usage: torchrun --nproc_per_node=8 algorithms/rl_code.py --resume ckpt_sft.pt
"""

try:
    from .rl_code_trainer import main
except ImportError:
    from rl_code_trainer import main

if __name__ == "__main__":
    main()
