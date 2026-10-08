"""Probe: real ChatML generation on the SFT model, accurate mode."""
import sys, time, resource
sys.path.insert(0, '/Users/bytedance/code/aupai')
import mlx.core as mx
from v41f.mlx.cli import build_sft_model

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_000_000

def clear_cache():
    try:
        mx.clear_cache()
    except Exception:
        pass

t0 = time.perf_counter()
model, tok, v42 = build_sft_model()
print(f"[load] {time.perf_counter()-t0:.1f}s layers={model.cfg.n_layers}")

im_end = tok.token_to_id("<|im_end|>")
im_start = tok.token_to_id("<|im_start|>")
print(f"[tokens] im_start={im_start} im_end={im_end}")

user = "写一个 Python 函数判断字符串是否为回文，并给出测试用例。"
chat = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n")
ids = tok.encode(chat).ids
print(f"[prompt] {len(ids)} tokens")

from v41f.mlx.generator import MLXGenerator
gen = MLXGenerator(model, mode="accurate")

t1 = time.perf_counter()
logits = gen.prefill(mx.array([ids], dtype=mx.int32))
clear_cache()
out_ids, steps = [], 0
while steps < 512:
    nxt = int(mx.argmax(logits))
    if nxt == im_end:
        break
    out_ids.append(nxt)
    ids.append(nxt)
    steps += 1
    logits = gen.prefill(mx.array([ids], dtype=mx.int32))
    clear_cache()
    if steps % 24 == 0:
        print(f"  step={steps} rss_peak={rss_gb():.2f}GB")
dt = time.perf_counter() - t1

text = tok.decode(out_ids)
print("===== ASSISTANT =====")
print(text)
print("=====================")
print(f"[gen] new={steps} stop={'im_end' if nxt==im_end else 'max'} "
      f"time={dt:.1f}s tps={steps/dt:.2f}")
