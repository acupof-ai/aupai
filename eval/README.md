# eval/ — Benchmark suite for HybridLM

Unified runner: `run_eval.py`. Every benchmark module also runs standalone
(`python eval/<name>.py`).

## The two that decide anything

The English suite below tracks general ability. Neither of these is in it, and
these are the ones the experiment ledger reports.

| Script | Data | Role |
|---|---|---|
| `math_hard.py` | `data/synthetic/math_hard_eval_1k.jsonl`, 1032 rows | **Metric of record.** `eval/eval_hard.sh <ckpt> [ngpu]` shards it. |
| `math_zh.py` | `data/eval/math_test_500.jsonl`, 500 rows | math-500, saturated at 32.2 / 26.8 / 32.0 / 31.0 — never conclude from it alone. |

math-hard resolves to about ±1 point at a 2–3% pass rate, so test significance
before explaining a gap. A [NUM]-carrying (`--fone`) checkpoint decodes through
`fone.decode_text`; `math_hard.py` switches on `cfg.fone` by itself.

## Benchmarks

| Key | Name | Source | Split | Examples | Metric |
|-----|------|--------|-------|----------|--------|
| `hellaswag` | HellaSwag | `hellaswag` | validation | 10,042 | 4-way continuation log-likelihood |
| `piqa` | PIQA | `piqa` | validation | 1,838 | 2-way continuation log-likelihood |
| `arc-easy` | ARC-Easy | `ai2_arc/ARC-Easy` | test | 2,376 | 3–5 way continuation log-likelihood |
| `arc-challenge` | ARC-Challenge | `ai2_arc/ARC-Challenge` | test | 1,172 | 3–5 way continuation log-likelihood |
| `winogrande` | WinoGrande | `winogrande_xl` | validation | 1,267 | 2-way continuation log-likelihood |
| `boolq` | BoolQ | `boolq` | validation | 3,270 | yes/no continuation log-likelihood |
| `openbookqa` | OpenBookQA | `openbookqa/main` | validation | 500 | 4-way letter log-likelihood |
| `mmlu` | MMLU | `cais/mmlu` (all) | test | 14,042 | 4-way letter log-likelihood, 57 subjects |
| `gsm8k` | GSM8K | `gsm8k/main` | test | 1,319 | greedy generation, exact match |

First run downloads datasets from HuggingFace; subsequent runs use the cache.

## How scoring works

**Multiple-choice (8 benchmarks):** for each option, score the sum of
log-prob of the option tokens given the prompt; pick argmax. In `run_eval.py`
prompt and option are tokenized separately and concatenated; the standalone
`boolq.py`/`openbookqa.py` now use joint tokenization via
`eval.log_likelihood_joint`, the other five still tokenize separately.
MMLU scores bare letter tokens (`"A"`); OpenBookQA scores the letter with a
leading space (`" A"`); the rest score the full option text with a leading space.

**GSM8K:** autoregressive greedy generation (up to 256 new tokens), take the
last number in the output, compare against `#### N`.

## Predictions artifacts and `.REFUSED` sidecars

A generative eval's predictions are the only copy of what a checkpoint produced, so
`scripts/eval_artifacts.open_artifact` refuses to overwrite an existing one and leaves the reason in
a sidecar **beside the artifact** — `data/eval/<name>.jsonl.REFUSED`, never under `runs/`. Looking
for them in `runs/` returns nothing and proves nothing; six exist under `data/eval/` on the pod. The
sidecar is named after the artifact, so several refusals on one artifact name share one file, and a
successful write clears it.

A refusal is the guard working, not a failure. Seven `l1_fewshot` rows in `runs/score_matrix.jsonl`
read `eval_artifacts.ArtifactExists` and were audited as failures twice before anyone opened the
files (`runs/audit_0904/eval_heldout.md` E18, E21, E22) — one of them had a complete 497-row result
on disk. `eval/l1_fewshot.py` now writes an identity header as row 0 of every predictions file
(checkpoint basename and file sha256, vocab_id, tokenizer fingerprint, demos, demo_lang, arm,
rep_stop, UTC), and `score_matrix.metric_l1_fewshot` transcribes an existing artifact only when that
sha matches the checkpoint on disk. No header, or a mismatch, records "artifact unverifiable, not
re-run" rather than re-running or forcing. **Every reader of a predictions file must skip the header
row**, identified by its `_header` key.

## Speed characteristics

The runner is built around one batched log-likelihood scorer:

- bf16 weights, `torch.no_grad()`, `model.eval()`
- All examples pre-tokenized before any forward pass
- All (example, option) pairs flattened into jobs, length-bucketed, scored
  32 sequences per forward pass — one forward per batch instead of one per
  option (~125k options → ~3.9k forward passes for the whole MC suite)
- Right-padding only; causal attention makes pad tokens inert, so no
  attention mask is needed
- Per-benchmark timing printed as each completes

GSM8K is fundamentally different: it is generation-bound, not
log-likelihood-bound. Each of the 1,319 problems decodes up to 256 tokens
autoregressively (batch 16), and the model has no KV cache, so every step
re-feeds the full sequence. It dominates wall time by an order of magnitude.

## Expected iteration time

On the training GPU (H20): **all 8 multiple-choice benchmarks together
< 30 s** (one-time dataset download excluded). GSM8K adds roughly 5–10 min
depending on output lengths — generation, not scoring, is the bottleneck.

## Usage

```bash
# Full suite
python eval/run_eval.py --ckpt ckpt_sft.pt

# Subset
python eval/run_eval.py --ckpt ckpt_sft.pt --benchmarks hellaswag piqa

# Larger scoring batch (more VRAM), CPU fallback
python eval/run_eval.py --ckpt ckpt_sft.pt --batch 64 --device cuda
```

Output:

```
Loaded ckpt_sft.pt: 165.3M params, bf16, device=cuda
  HellaSwag: 45.2% (4.1s)
  ...
Benchmark        Accuracy
────────────────────────
HellaSwag          45.2%
...
────────────────────────
Average            44.9%
```

The average is the unweighted mean over all benchmarks in the run (ARC-Easy
and ARC-Challenge count separately).
