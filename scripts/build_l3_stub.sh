#!/bin/bash
# 0e-8: build code_ultra_l3_stub_dc (HumanEval-stub L3), gate vocab, CPU.
# Requires code_ultra_l3_noexec_dc present (left frozen) and the 147 L3 raw parquets.
set -u
cd /work/aupai
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/work/aupai
python3 -u datagen/build_l3_stub.py --workers ${STUB_WORKERS:-24}
