---
question: "Can flash varlen's unbounded doc-count recompilation be fixed, and is padding the right fix?"
status: recorded
source: "t57 (fb ruling 2026-09-01); facts/efficiency.json#eff.recompile_recurrence_explained; flash_attn 3 cute/interface.py on the pod"
---

# The flash varlen recompile seam: correctness design

fb asked for a correctness design before any padding code. **The answer is that padding is
probably the wrong fix, and the design work found a cheaper one to test first.**

## The mechanism, restated

`eff.recompile_recurrence_explained`: flash's `_flash_attn_fwd` recompiles on
`batch_size == N  # total_mblocks = batch`, with 45 distinct values across ~60 steps spanning
43–116. `batch_size` is the **document count**, from `cu_seqlens_q.shape[0] - 1`
(`interface.py:350`), and it is drawn from a distribution, so the variant set never closes
against `recompile_limit = 64`. Eviction is permanent. Cost: **3.28% measured, 4.56% ceiling.**

## Why `batch_size` reaches a dynamo guard at all

I expected the scheduling heuristic to be the cause. It is not:

| site | use | forces specialisation? |
|---|---|---|
| `:541` `total_mblocks = batch_size * num_head_kv * num_m_blocks` | feeds `num_splits_heuristic` | **No** — guarded by `:544 if num_splits < 1`, and `num_splits` defaults to 1 (`:315`, `:2194`). We do not pass it (`train.py:319`), so the heuristic never runs. |
| `:376` `assert cu_seqlens_k.shape == (batch_size + 1,)` | shape validation | **Yes** |
| `:381` `assert cu_seqlens_q.shape == (batch_size + 1,)` | shape validation | **Yes** |
| `:384` `assert seqused_q is None or seqused_q.shape == (batch_size,)` | shape validation | Yes, if passed |

So the specialisation comes from **argument-validation asserts comparing a tensor shape against a
Python int**, not from anything that affects the kernel. Dynamo must burn `batch_size` into a
guard to prove the assert holds.

That reframes the fix. We are not paying for a scheduling decision; we are paying for
`assert`s that could be `torch._check` or skipped under compile.

## Three candidate fixes, cheapest first

**1. Suppress the wrapper's tracing entirely.** Wrap the flash call in
`torch._dynamo.disable()` or `allow_in_graph`, so the Python validation never enters a graph and
the kernel is called opaquely. No padding, no numerics change, no correctness question about
segment layout. The cost is a graph break at the attention boundary, and the trace already shows
one there — `_flash_attn_fwd_at_406` is itself a **resume** frame, so the break exists today.
**This is the one to test first and it is a one-line change.**

**2. Make the doc count constant by padding `cu_seqlens`.** What fb asked me to design. Its
correctness questions, which I can now answer partly:

- *Does a padded block count require padded `cu_seqlens`?* It is the reverse — `batch_size` is
  **derived from** `cu_seqlens_q.shape[0] - 1`, so padding cu's length *is* the mechanism, not a
  consequence of it. To fix the count you must add entries to cu.
- *Does flash varlen accept zero-length segments?* **Unverified and it is the crux.** Padding cu
  to a fixed length means appending repeated final offsets, i.e. `[..., 8192, 8192, 8192]`, which
  declares zero-length documents. `num_splits_heuristic` has an explicit `total_mblocks == 0`
  guard and the comment at `:259` mentions an "empty-Q early-exit", which suggests degenerate
  segments are contemplated somewhere — but not that a zero-length segment inside a batch is
  safe. This needs a parity test, not a reading.
- *What does the padded tail do to the loss?* Nothing directly: cu only describes attention
  boundaries. But `doc_cu_seqlens` (`train.py:836`) is explicitly documented as rejecting
  zero-length documents, and its docstring records the incident where per-pad-token boundaries
  produced `grid=(2, 78936, 1)` against CUDA's 65535 limit. **Padding cu deliberately
  reintroduces the exact construct that function exists to prevent.**

**3. Raise `recompile_limit`.** Does not work: the distribution is unbounded, so any finite limit
is eventually exceeded. It moves the recurrence later, not away.

## Recommendation

**Test option 1 before designing option 2 further.** It is one line, has no numerics or segment
-layout risk, and if it removes the recompiles the padding question is moot. Option 2 conflicts
with a documented invariant in our own code and would need a zero-length-segment parity test
before a single line is written.

## Acceptance, per b0

Gap count and seam ms read **directly from a trace**, never tok/s alone — a constant tax is
invisible to steady throughput, which is how I originally misread the flat 77K as evidence the
tax did not exist. A fix that trades compile time for a graph break would show no tok/s change
while moving both trace numbers.

## What this does not establish

Whether `_dynamo.disable()` at that call site actually removes the guard, whether the extra graph
break costs more than the 3.28% it saves, and whether flash varlen tolerates zero-length
segments. All three are measurements, none has been made. The 3.28% also caps the whole prize:
even a perfect fix sits at the edge of the 3% ship gate.
