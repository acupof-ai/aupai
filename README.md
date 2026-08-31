# aupai

A 200M-parameter reasoning LLM for coding and math, pretrained on ~30B tokens.
Objective set 2026-08-30; the earlier Chinese-web LLM and its scaling law are retired
(`docs/standards/0830v1_gates.md`). Working rules, layout, and the run book are in
`AGENTS.md`; this file is the short version.

## Model

| | |
|---|---|
| layers | 12: KDA ×9 + gated MLA ×3, d=1024, 206M params |
| position | NoPE — no RoPE, no learned positions; the KDA state carries position |
| attention | full causal over the 4096-token sequence, document-masked (`cu_seqlens`) |
| extras | Attention Residuals on by default; FP8 (e4m3 fwd+bwd); `--fone` number embedding |
| vocabulary | 32,784 slots, frozen 2026-08-29; `data/tokenizer.json` is not in git — copy it from the pod |
| optimizer | Muon for 2D weights, AdamW for embeddings and 1D; `torch.compile` |

Correctness never depends on which attention package is installed: without flash-attn the
SDPA fallback builds the document mask from `cu_seqlens` (~20× slower, refused at startup
unless `--allow_slow_attn`). Measured 2026-08-31 on 7×H20 at the 0.2B point: 76K tok/s/GPU,
MFU 30%.

Interactive parameter/memory calculator: <https://acupof-ai.github.io/aupai/>.

## Quick start

```bash
uv sync
python scripts/harness.py install-hooks        # pre-commit: ruff E9/F, blob guard, manifest, harness check
python scripts/test_arch_compat.py             # CPU: fwd/bwd, checkpoint round-trip, doc-mask known answers
python scripts/harness.py check                # repo invariants, ~5 s; CI runs the same
```

The checkout ships a 2,000-document sample corpus (`data/corpus/sample/`, `data/mix_sample.json`)
that exercises the pipeline end to end. Real corpora are built by
`python datagen/build_corpus.py --domain <d> --source <s>` into `data/corpus/<domain>/`, and a
mix file (`data/mix_scale_*.json`, `data/mix_30b.json`) is the only data path: per-domain
weight, epoch cap, anneal weight. A missing mix is an error, not a fallback.

## Run

Every GPU or corpus job starts through one launcher — it writes the experiment row first,
takes its cards from the controller's allocation (7-card block for training, one lane card
for everything else), detaches with `setsid`, verifies `fa True | doc_mask True` in the
worker log before the job counts as started, and arms a monitor:

```bash
python scripts/harness.py launch <name> --training --hypothesis "..." -- ./run_ddp.sh --mix data/mix_scale_0.2b.json --name <name>
python scripts/harness.py launch <name> -- python3 eval/score_matrix.py --ckpt <ckpt> --json runs/score_matrix.jsonl
python scripts/harness.py launch <name> -- python3 datagen/fetch_corpus.py --source <src> --target_bytes 27e9
```

SFT: `scripts/run_sft.sh <name> <resume_ckpt> <sft.pt>`. RL gate: `eval/math_hard.py --ckpt X --k 8 --temperature 0.8`
opens RL only if pass@8 − pass@1 ≥ 15 pt at the same temperature. Numbers land in
`runs/score_matrix.jsonl` and `facts/*.json`, each with its measurement config; the
pre-registered 30B readout is `docs/lessons/readout_30b_prereg.md` (`eval/readout_30b.py`).

## Commit workflow — one path

1. Work in the shared tree on `main`; stage by path (`git add <file>`), never `-A`/`-a`.
   One concern per commit, message in English.
2. The pre-commit hook (installed above) runs ruff E9/F on staged Python, refuses files over
   5 MB and unlisted `data/` paths, regenerates `data/pod_head_manifest.txt` from the index
   into the same commit, and runs `harness check`. A red hook is a red commit.
3. Push to the pod with `scripts/pod_push.sh <files>` — it refuses uncommitted files and
   ships the manifest; `train.py` refuses to start on a drifted pod.
4. CI on push: ruff, `py_compile`, `test_arch_compat`, `eqcheck`, `holdout`, `harness check`
   and `--selftest`. Push `main` to `origin` only when CI is green.
5. Record every run: `scripts/exp.py start` before, `done` after; tasks live in
   `runs/tasks.jsonl` (`harness task add|done|list`). Status is read from artifacts, never
   from a message.

## Numbers — `--fone`

BPE splits numbers by frequency (1640 → `16|40`). `--fone` gives each number one `[NUM]`
token with a Fourier-encoded value and decodes digits ten-way. The flag changes the data
format everywhere: pack with `datagen/prepare_sft_math.py --fone`; a checkpoint whose flag disagrees
with the pack refuses. `probes/fone_digit_acc.py --ckpt X` scores the digit head.
