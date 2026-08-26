#!/usr/bin/env python3
"""RLVR data-prep entry point — implementation in rlvr_data.py.

Usage: python algorithms/prepare_rlvr.py
"""

try:
    from .rlvr_data import main
except ImportError:
    from rlvr_data import main

if __name__ == "__main__":
    main()
