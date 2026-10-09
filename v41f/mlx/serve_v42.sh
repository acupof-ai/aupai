#!/bin/bash
set -euo pipefail
cd /Users/bytedance/code/aupai
export V42_MOE_BACKEND=q8
exec .venv/bin/python -m v41f.mlx.server --host 127.0.0.1 --port 8731
