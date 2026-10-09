"""SFT generator test: Engram SSD + KV cache + 4 token decode, vs model.forward."""
import resource, time, sys, runpy
sys.path.insert(0, '.')

# Reuse run_sft.py loading (up to model built)
ns = runpy.run_path('v41f/mlx/run_sft.py', run_name='__loader__')
model = ns['model']
W = ns['W']
mcfg = ns['mcfg']

import mlx.core as mx
from v41f.mlx.generator import MLXGenerator

tokens = mx.array([[1, 5, 9, 13]], dtype=mx.int32)

# Warm up
_ = model.forward(tokens)
gen = MLXGenerator(model)
_ = gen.prefill(tokens)
mx.eval(_)

# Timed comparison
t0 = time.perf_counter()
fwd_logits = model.forward(tokens)
mx.eval(fwd_logits)
fwd_t = (time.perf_counter() - t0) * 1000

gen2 = MLXGenerator(model)
t0 = time.perf_counter()
gen_logits = gen2.prefill(tokens)
mx.eval(gen_logits)
gen_t = (time.perf_counter() - t0) * 1000

fwd_last = fwd_logits[0, -1, :]
diff = mx.abs(fwd_last.astype(mx.float32) - gen_logits[0].astype(mx.float32))
print('\n=== Prefill consistency (model.forward vs generator) ===')
print(f'  model.forward last token: {fwd_t:.1f}ms')
print(f'  generator prefill:        {gen_t:.1f}ms')
print(f'  max abs diff:  {float(mx.max(diff)):.6f}')
print(f'  mean abs diff: {float(mx.mean(diff)):.6f}')
print(f'  argmax fwd:  {int(mx.argmax(fwd_last))}')
print(f'  argmax gen:  {int(mx.argmax(gen_logits[0]))}')

# Generate 4 tokens
print('\n=== Generate 4 tokens ===')
gen3 = MLXGenerator(model)
pl = gen3.prefill(tokens)
mx.eval(pl)
tokens_out = [int(mx.argmax(pl[0]))]
t0 = time.perf_counter()
for i in range(3):
    sl = gen3.decode_step(mx.array([tokens_out[-1]], dtype=mx.int32), pos=4+i)
    mx.eval(sl)
    tokens_out.append(int(mx.argmax(sl[0])))
decode_t = (time.perf_counter() - t0) * 1000
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'  output tokens: {tokens_out}')
print(f'  prefill: {gen_t:.1f}ms, decode: {decode_t:.1f}ms for 3 ({3/(decode_t/1000):.1f} tok/s)')
print(f'  peak RSS: {rss:.0f}MB')

# SSD stats
print('\n=== SSD stats ===')
for L in [1,5,9,13,17,21]:
    s = model.ssd_lookups[L].stats
    print(f'  L{L}: lookups={s.lookups} hits={s.hits} misses={s.misses} bytes={s.bytes_loaded}')

fwd_arg = int(mx.argmax(fwd_last))
gen_arg = int(mx.argmax(gen_logits[0]))
ok = fwd_arg == gen_arg and float(mx.max(diff)) < 1.0
print(f'\n{"PASS" if ok else "FAIL"}: argmax match={fwd_arg==gen_arg}, max_diff={float(mx.max(diff)):.4f}')
sys.exit(0 if ok else 1)
