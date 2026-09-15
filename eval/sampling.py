"""Shared temperature sampling for base-model continuation evals (stage-2 prereg,
fb order 2026-09-14).

E0/ET/EC checkpoints are scored on HumanEval and MBPP with n=10 samples at
temperature 0.2 per task. To make a paired T-vs-C comparison valid, the two
checkpoints must draw the SAME random choices for the same task_id and sample
index. We force that deterministically: each of the n draws for a task reseeds
torch's CPU/CUDA RNG with a stable hash of (task_id, sample_idx). Two runs over
the same (task_id, si) then walk the identical random stream step-for-step.

Reseed PER SAMPLE, not once per task: T and C are different checkpoints, so
sample 0 almost always emits a different number of tokens and consumes a
different number of multinomial scalars; a single task-level seed leaves sample
1's stream offset diverged between the two runs, and 9 of 10 draws would be
unpaired (3b, pod-reproduced 2026-09-14). Each sample starts from a fresh
seeded state in both runs, and one multinomial per step keeps the streams
aligned regardless of differing logits or lengths.

This is per-task pairing, not HE-to-MBPP pairing (different task spaces).

Decoding stops at eos (tid 1) or max_new; benchmark-specific truncation and
judgement are applied by the caller after sampling, identically for every
sample. No repetition/STOPS early-stop here: T and C must be compared on raw
equal-budget draws, and post-hoc truncation is deterministic given the text.
"""
import contextlib
import hashlib

import torch

EOS_TID = 1


def task_seed(task_id):
    """Stable non-negative int seed from a task id (same across runs/benchmarks)."""
    h = hashlib.sha256(str(task_id).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


@torch.no_grad()
def sample_completions(model, tok, prompt_ids, task_id, n, temperature, max_new,
                       device, seq_window, batched=False):
    """Return n sampled raw decoded strings for one prompt (list of token ids).

    Greedy when temperature<=0 (n identical draws, by construction). Otherwise
    n stochastic draws under a task-seeded RNG so paired T/C runs reproduce the
    same decision sequence.

    batched=False (default, the path E0 already ran): one sequence at a time,
    reseeding the GLOBAL RNG per sample. batched=True decodes all n rows in one
    forward pass but keeps the SAME per-(task_id, si) random stream: each row has
    its own fresh Generator seeded with task_seed("id:si") and draws one
    single-row multinomial per step, so its scalar sequence is the serial one's.
    Finished rows stay in the packed tensor on an EOS filler (rows are causally
    independent, so a filler never changes another row's logits) but stop
    drawing, exactly as the serial loop stops calling multinomial for them.
    """
    dev = torch.device(device)
    base = torch.tensor([prompt_ids], device=dev)
    # bf16 autocast, matching humaneval_gen and the training dtype: without it rms_norm returns
    # fp32 and the CSA2 sliding-window flash kernel (which accepts only fp16/bf16/fp8) asserts.
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else contextlib.nullcontext())
    with ctx:
        if temperature <= 0:
            return [_decode(tok, _greedy(model, base, max_new, seq_window), prompt_ids)] * n
        if batched:
            return _sample_batched(
                model, tok, base, task_id, n, temperature, max_new, seq_window, dev)
        out = []
        for si in range(n):
            seed = task_seed(f"{task_id}:{si}")
            torch.manual_seed(seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            x = base
            for _step in range(max_new):
                logits = model(x[:, -seq_window:])[0][:, -1]
                nxt = torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1)
                if nxt.item() == EOS_TID:
                    break
                x = torch.cat([x, nxt], 1)
            out.append(_decode(tok, x, prompt_ids))
    return out


@torch.no_grad()
def _sample_batched(model, tok, base, task_id, n, temperature, max_new, seq_window, dev):
    """n rows in one forward pass per step, each on its own task-seeded Generator.

    One single-row multinomial per ACTIVE row per step, never one n-row draw from
    a shared generator: a shared batched draw reorders the Philox scalar stream
    across rows and breaks the (task_id, si) pairing contract. Per-row fresh
    generators make row si's scalar stream identical to the serial path's global
    RNG freshly seeded with the same seed (asserted in test_sampling_paired).
    """
    x = base.expand(n, -1).contiguous()
    gens = []
    for si in range(n):
        g = torch.Generator(device=dev) if dev.type == "cuda" else torch.Generator()
        gens.append(g.manual_seed(task_seed(f"{task_id}:{si}")))
    active = [True] * n
    rows = [[] for _ in range(n)]
    for _step in range(max_new):
        if not any(active):
            break
        logits = model(x[:, -seq_window:])[0][:, -1]
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        nxt = torch.full((n, 1), EOS_TID, dtype=torch.long, device=dev)
        for si in range(n):
            if active[si]:
                nxt[si, 0] = torch.multinomial(probs[si:si + 1], 1, generator=gens[si])[0, 0]
        for si in range(n):
            if active[si]:
                tid = int(nxt[si, 0])
                if tid == EOS_TID:
                    active[si] = False  # EOS is appended only as the packed filler, not decoded
                else:
                    rows[si].append(tid)
        x = torch.cat([x, nxt], 1)
    return [tok.decode(ids) for ids in rows]


@torch.no_grad()
def _greedy(model, x, max_new, seq_window):
    for _step in range(max_new):
        logits = model(x[:, -seq_window:])[0][:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        if nxt.item() == EOS_TID:
            break
        x = torch.cat([x, nxt], 1)
    return x


def _decode(tok, x, prompt_ids):
    return tok.decode(x[0, len(prompt_ids):].tolist())
