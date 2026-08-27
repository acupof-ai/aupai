# aupai CLI

A thin, cross-platform harness over the project's Python/bash training tools. One binary,
`aupai`, dispatches every step of the workflow — data, tokenizer, pretrain, eval, SFT, RL,
inference, ops — to the existing script, inherits its stdio, and propagates its exit code.
**Nothing heavy is reimplemented in Rust.** The value is one front door, a `--dry-run` that
prints the exact command, and a `pipeline` command that chains a whole run.

## Build

```sh
cd cli
cargo build --release          # -> cli/target/release/aupai
```

Cross-platform: builds and runs on macOS aarch64 (dev) and linux x86_64 (pod), no
platform-specific deps (only `clap`). `Cargo.lock` is committed; `cli/target/` is gitignored.

Put the binary on your PATH, or run it in place:

```sh
alias aupai=./cli/target/release/aupai
aupai list
```

The binary finds the repo root from its own location (`cli/target/release/aupai` -> repo), and
falls back to walking up from the current directory to the first `pyproject.toml`, so it works
from anywhere in the tree.

## It is a harness, not a reimplementation

Every command shells out. `--dry-run` (global) prints the exact `cd … && …` invocation instead
of running it, so you can see — or copy — what any command does:

```sh
aupai --dry-run eval ckpt_k4.pt 8
# cd /…/aupai && bash scripts/eval_hard.sh ckpt_k4.pt 8
```

`aupai list` prints every command and the script behind it.

| command | shells out to |
|---|---|
| `data`, `mix` | `scripts/data_overview.py`, `scripts/check_mix.py` |
| `tokenizer` | `scripts/build_tokenizer.py` (ByteLevel BPE **+ the 4 chat specials**) |
| `pretokenize` | `scripts/pretokenize.py` |
| `train` | `run_ddp.sh` -> `train.py` |
| `pipeline` | chains the stages below, each its own shell-out |
| `sft`, `rl` | `scripts/run_sft.sh`, `torchrun algorithms/rlvr.py` |
| `prep-sft`, `prep-sft-math`, `prep-rl` | the packing scripts |
| `eval`, `eval-math`, `band`, `select-band` | the eval/analysis scripts |
| `plot`, `dashboard` | `scripts/plot_curves.py` / launch trackio's local UI |
| `nan-probe`, `ckpt-diff`, `exp` | the ops scripts |
| `infer`, `chat`, `serve` | the inference entry points |

## The pipeline: a whole run at a glance

`aupai pipeline` sequences the training stages, each a shell-out to its existing tool, stopping
on the first failure (set-e semantics):

```
tokenizer -> pretokenize -> data -> pretrain -> eval -> sft -> rl
```

```sh
# The single source of truth for what a full run executes — prints every command, runs nothing:
aupai pipeline --dry-run --name mybase --fp8 --attn_res

aupai pipeline --from pretokenize --to eval --name mybase --fp8   # a contiguous slice
aupai pipeline --stages data,pretrain,eval --name mybase          # an explicit subset
```

- `--name` flows to `runs/<name>.log`, `ckpt_<name>.pt`, and the eval target.
- Flags after the known options (`--fp8`, `--attn_res`, …) pass through to the pretrain stage
  (`train.py`).
- `sft` needs `--sft-pt <packed.pt>`; `rl` reads `--rl-data` (default `data/rl/rl_band.jsonl`).
- The default slice (`--from`/`--to` unset) is `tokenizer … eval` — the core pretraining path;
  add `sft`/`rl` with `--to rl` or `--stages`.

## trackio

`aupai train --track` (or `pipeline --track`) turns on trackio logging (local-first,
wandb-API-compatible, SQLite-backed — no login, no server). The CLI's contract with `train.py`:

- env `TRACKIO_PROJECT=<name>` (default `aupai`; override with `--track-project <name>`)
- flag `--track` appended to the `train.py` argv

`train.py`'s `RunLog` reads that flag to init a run and mirror each step's metrics. `aupai
dashboard` launches trackio's local web UI (`uv run python -m trackio show [--project <name>]`);
`aupai dashboard <name> --plot` plots `runs/<name>.log` with `scripts/plot_curves.py` and opens
the PNG.

## Environment

- `AUPAI_CACHE_DIR` — where the pretokenized token caches live (`tokens_<domain>.pt`); consumed
  by the Python side (`train.py`, `pretokenize.py`), not the CLI. Defaults to `/data00`.
- `TRACKIO_PROJECT` — set by `aupai train --track` and read by `aupai dashboard` when `--project`
  is not given.
- `NGPU`, `PORT`, `CUDA_VISIBLE_DEVICES` — honored by the underlying `run_ddp.sh` / eval scripts.
