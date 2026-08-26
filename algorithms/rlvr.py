#!/usr/bin/env python3
"""RLVR entry point — implementation in rlvr_trainer.py.

Usage: torchrun --nproc_per_node=8 algorithms/rlvr.py --resume ckpt_sft.pt
"""

try:
    from .rlvr_trainer import main
except ImportError:
    from rlvr_trainer import main

if __name__ == "__main__":
    main()
