#!/usr/bin/env python3
"""Single-GPU GEMM + clock probe: compare GPU0 vs GPU4 sustained bf16 compute."""
import torch, time, sys, argparse, subprocess

p = argparse.ArgumentParser()
p.add_argument("--device", type=int, required=True)
p.add_argument("--n", type=int, default=8192)
p.add_argument("--iters", type=int, default=30)
args = p.parse_args()

torch.cuda.set_device(args.device)
dev = f"cuda:{args.device}"
props = torch.cuda.get_device_properties(dev)
print(f"GPU{args.device} {props.name} sm={props.multi_processor_count} clock={props.clock_rate}kHz")

a = torch.randn(args.n, args.n, dtype=torch.bfloat16, device=dev)
b = torch.randn(args.n, args.n, dtype=torch.bfloat16, device=dev)
for _ in range(5):
    c = a @ b
torch.cuda.synchronize(dev)

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(args.iters):
    c = a @ b
e.record()
torch.cuda.synchronize(dev)
ms = s.elapsed_time(e) / args.iters
tflops = 2 * args.n**3 / (ms / 1000) / 1e12
print(f"GEMM {args.n}x{args.n} bf16: {ms:.2f}ms = {tflops:.1f} TFLOPS")

clock = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
     "--format=csv,noheader,nounits", "-i", str(args.device)]
).decode().strip()
print(f"during load: clock={clock.split(',')[0].strip()}MHz power={clock.split(',')[1].strip()}W temp={clock.split(',')[2].strip()}C")
