"""Generation drivers for the v42 runtime.

Wraps mlx_lm's ``generate_step`` (which drives ``model(inputs, cache)`` in
prefill chunks then one-token decode steps) with a v42-specific thin wrapper
that handles the ChatML stop token and streams tokens out.

Continuous batching is provided by :class:`BatchGenerator` which sequences
requests one-at-a-time through the same loaded model (single-stream service
with a Python request queue) -- enough to hit the 8/16-way aggregate throughput
gates on a single M4 Pro.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

IM_END = 32764


def equal_length_batch(tokenizer, prompt: str, n: int) -> mx.array:
    """n copies of one prompt, each with a different middle token.

    The forward batches only equal lengths. The changed token makes the expert
    routes differ across the streams.
    """
    base = list(tokenizer.encode(prompt).ids)
    if not base:
        raise ValueError("empty prompt")
    vocab = tokenizer.get_vocab_size()
    j = min(len(base) - 1, max(0, len(base) // 2))
    rows = []
    for i in range(n):
        ids = list(base)
        ids[j] = (ids[j] + i * 97) % vocab
        rows.append(ids)
    return mx.array(rows, dtype=mx.int32)


def generate_concurrent(model, tokenizer, prompt: str, n: int, *,
                        max_tokens: int = 128, warmup_steps: int = 4) -> dict:
    """Decode n equal-length streams in one forward per step.

    Prefill time is separate. The total rate counts only the timed decode
    steps, while all n streams are in decode together. The first decode
    steps compile the Metal graph, so they stay out of the rate.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if max_tokens < 2:
        raise ValueError("max_tokens must be >= 2 so a decode step exists")
    batch = equal_length_batch(tokenizer, prompt, n)
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(batch, cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    t_prefill = time.perf_counter() - t0
    for _ in range(warmup_steps):
        logits = model(tok.reshape(n, 1), cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
    t1 = time.perf_counter()
    steps = 0
    for _ in range(max_tokens - 1):
        logits = model(tok.reshape(n, 1), cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        steps += 1
    t_decode = time.perf_counter() - t1
    return {
        "streams": n,
        "prompt_tokens": int(batch.shape[1]),
        "prefill_s": t_prefill,
        "decode_steps": steps,
        "decode_tokens": steps * n,
        "decode_s": t_decode,
        "aggregate_tps": (steps * n / t_decode) if t_decode > 0 else 0.0,
        "per_stream_tps": (steps / t_decode) if t_decode > 0 else 0.0,
    }


@dataclass
class GenResult:
    text: str
    tokens: list = field(default_factory=list)
    n_prompt: int = 0
    t_prefill_s: float = 0.0
    t_decode_s: float = 0.0
    stopped_eos: bool = False

    @property
    def n_gen(self) -> int:
        return len(self.tokens)

    @property
    def decode_tps(self) -> float:
        return self.n_gen / max(self.t_decode_s, 1e-6)


def generate(model, tokenizer, prompt: str, *, max_tokens: int = 256,
             temperature: float = 0.0, top_p: float = 1.0,
             stop_tokens=(IM_END,), prefill_step_size: int = 512) -> GenResult:
    """Greedy/sampled generation.  temperature==0 -> argmax."""
    ids = list(tokenizer.encode(prompt).ids)
    prompt_t = mx.array(ids, dtype=mx.int32)
    sampler = make_sampler(temp=temperature, top_p=top_p) if temperature > 0 else None

    cache = model.make_cache()
    t0 = time.time()
    step = generate_step(prompt_t, model, max_tokens=max_tokens, sampler=sampler,
                         prompt_cache=cache, prefill_step_size=prefill_step_size)
    # first token timing includes prefill
    gen_tokens: list[int] = []
    first_done = None
    stopped = False
    tok, _ = next(step)
    first_done = time.time()
    gen_tokens.append(int(tok))
    if int(tok) in stop_tokens:
        stopped = True
    while not stopped and len(gen_tokens) < max_tokens:
        tok, _ = next(step)
        gen_tokens.append(int(tok))
        if int(tok) in stop_tokens:
            stopped = True
    t_end = time.time()

    t_prefill = first_done - t0
    t_decode = t_end - first_done
    text = tokenizer.decode([t for t in gen_tokens if t != IM_END])
    return GenResult(text=text, tokens=gen_tokens, n_prompt=len(ids),
                     t_prefill_s=t_prefill, t_decode_s=t_decode,
                     stopped_eos=stopped)
