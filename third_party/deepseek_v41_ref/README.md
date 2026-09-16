# DeepSeek-V4.1-Flash reference snapshot (vendored, read-only)

Truth source for the v41f faithful reproduction (P0 per-module `allclose`).
Copied verbatim from Hugging Face — **not** GitHub:

  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference

Fetched 2026-09-16. Upstream is public, non-gated, created 2026-09-10. The upstream
LICENSE is MIT; it is preserved here and must travel with these files.

| file | upstream path | sha256 |
|---|---|---|
| `model_ref.py.ref` | `inference/model.py` (61549 B) | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `engram_ref.py.ref` | `inference/engram.py` (8138 B) | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `kernel_ref.py.ref` | `inference/kernel.py` (23790 B) | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `config.json` | `inference/config.json` (1982 B, released shapes) | `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809` |

The `.ref` suffix keeps ruff's pre-commit hook from linting third-party code;
`ref_oracle.py` loads them with `importlib.spec_from_file_location`.

Hugging Face blob oids at fetch (file-level, via the HF tree API):
model `3e7dc222a6352ccc`, engram `a37bc6c6b8cbf2d3`, kernel `6fa7bd5aafbe1bc6`,
config `7a915cc69e21abbc`.

## Why vendored

P0 must compare against a frozen oracle, not a live URL. A re-pull could change bytes,
and CI/laptops must not need network. Do **not** edit these files. Regenerate only by
re-fetching, re-hashing, and updating this table in the same commit. Verify with:

```bash
python third_party/deepseek_v41_ref/verify_ref.py
```

## What P0 can and cannot import

`model_ref.py` imports `kernel` (tilelang SM100 fp8/fp4 kernels) and the vision stack.
The v41f training reproduction does not ship tilelang. P0 stubs `tilelang`, the quant
ops, and `sparse_attn` in `tests/v41f/ref_oracle.py`, then drives only the pure-tensor
math: RMSNorm, Compressor pooling, Indexer einsum+topk, Gate `sqrtsoftplus`, Engram
hash/gate, and a pure-torch re-creation of the `hc_split_sinkhorn` normalization. The
quantized GEMMs and the hand-written sparse attention kernel are numerically validated
against flash-attn/sdpa equivalents, not against tilelang on CPU.
