#!/bin/bash
# 3b-24: build the phi-1 CodeExercises holdout + continuation SFT packs from the L3 stub domain. CPU.
# Requires data/corpus/code_ultra_l3_stub_dc and the gate tokenizer; writes data/sft/*.pt + manifest.
set -u
cd /work/aupai
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/work/aupai
python3 -u datagen/build_phi_codeexercises_pack.py --workers ${PHI_WORKERS:-24}
