"""t57 seam test: does _dynamo.disable on the flash wrapper remove the recompiles?

Acceptance per b0: gap count + seam ms from a trace, never tok/s alone -- a constant tax is
invisible to steady throughput. Before-number comes from THIS run's own OFF arm, not from the
recorded fact, since 3.28% against a 3% gate is thin enough that window variance could flip it.
"""
import os
import sys

ARM = sys.argv[1] if len(sys.argv) > 1 else "off"
sys.argv = ["x"]
sys.path.insert(0, "/work/aupai/scripts")
sys.path.insert(0, "/work/aupai")
import torch  # noqa: E402

import train  # noqa: E402

if ARM == "on":
    _real = train.flash_attn_varlen_func
    train.flash_attn_varlen_func = torch._dynamo.disable(_real)
    print("[seam] flash_attn_varlen_func wrapped in _dynamo.disable", flush=True)
else:
    print("[seam] arm OFF, wrapper untouched", flush=True)

# Hand off to train.py's own main with the recompile log on, so guards and steps share a process.
os.environ["TORCH_LOGS"] = "recompiles"
sys.argv = ["train.py", "--fp8", "--mix", "data/mix_smoke_warmup.json", "--seq", "4096",
            "--batch", "16", "--accum", "2", "--vocab", "32784", "--attn_res_blocks", "0",
            "--attn_every", "4", "--warmup", "5", "--seed", "42", "--name", f"t57_seam_{ARM}"]
train.main()
