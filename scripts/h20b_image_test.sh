#!/bin/bash
set -x
echo "PATH=$PATH"
which python3 || true
python3 --version || true
ls -l /usr/local/bin/python* 2>/dev/null || true
find / -maxdepth 6 -type d -name torch 2>/dev/null | head -20
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.device_count()); x=torch.randn(128,128,device="cuda").to(torch.float8_e4m3fn); print("fp8", x.dtype, "finite", torch.isfinite(x.float()).all().item())'
