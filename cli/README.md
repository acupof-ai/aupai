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

Every command shells out. `--dry-run` (global, place it **before** the subcommand) prints the
exact `cd … && …` invocation instead of running it, so you can see — or copy — what any command
does:

```sh
aupai --dry-run eval ckpt_k4.pt --ngpu 8
# cd /…/aupai && bash scripts/eval_hard.sh ckpt_k4.pt 8
```

Backend flags (anything for the underlying `train.py`, `sft_math.py`, `rlvr.py`, …) are passed
**after a literal `--`**, so a stray `--dry-run`/`--name` can never be swallowed into a real launch:

```sh
aupai --dry-run train --name k5 -- --grad_ckpt --max_steps 500
```

`aupai list` prints every command and the script behind it; each command's `--help` names its
backend script so you know where to look for its flags.

| command | shells out to |
|---|---|
| `data`, `mix` | `scripts/data_overview.py`, `scripts/check_mix.py` |
| `tokenizer` | `scripts/build_tokenizer.py` (ByteLevel BPE **+ the 4 chat specials**) |
| `pretokenize` | `scripts/pretokenize.py` (all CPU cores automatically) |
| `train` | `run_ddp.sh` -> `train.py` (verified-best recipe by default, printed before it runs) |
| `pipeline` | chains the stages below, each its own shell-out, with per-stage state |
| `sft`, `rl` | `scripts/run_sft.sh`, `torchrun algorithms/rlvr.py` |
| `prep-sft`, `prep-sft-math`, `prep-rl` | the packing scripts |
| `eval`, `eval-math`, `band`, `select-band` | the eval/analysis scripts |
| `plot`, `dashboard` | `scripts/plot_curves.py` / launch trackio's local UI |
| `nan-probe`, `ckpt-diff`, `exp` | the ops scripts |
| `ckpt list\|best\|clean` | filesystem + `scripts/ckpt_info.py` (no reimplementation) |
| `status` | reads mix/caches/checkpoints/`*.pipeline.json`/`*.log` — read-only |
| `infer`, `chat`, `serve` | the inference entry points |

## `aupai status` — the control-system front door

One read-only, never-blocking dashboard of the whole system (exit 0, `--json` for machines). Run
it first to know the state of everything before deciding the next command:

```sh
aupai status
# DATA         — mix domains with token caches / corpus built, total tokens vs mix target
# CHECKPOINTS  — count of finals + intermediates, best base, total disk
# PIPELINES    — each runs/*.pipeline.json and its furthest-completed stage
# ACTIVE RUN   — the last line of the most-recently-touched runs/*.log
```

## `aupai ckpt` — checkpoint management

```sh
aupai ckpt list            # size, step, params, arch per ckpt_*.pt (reads cfg via scripts/ckpt_info.py)
aupai ckpt best            # the best base (source: runs/experiments.jsonl / EXPERIMENTS.md)
aupai ckpt clean --keep 1  # prune intermediate .stepN/.epN — DRY-RUN by default; --force to delete
```

`clean` never touches a final `ckpt_<name>.pt`, only `.stepN`/`.epN` intermediates, and requires
`--force` (or `--yes`) to actually remove anything.

## The pipeline: a whole run at a glance, with resumable state

`aupai pipeline` sequences the training stages, each a shell-out to its existing tool, stopping
on the first failure (set-e semantics):

```
tokenizer -> pretokenize -> data -> pretrain -> eval -> sft -> rl
```

It persists per-stage state to `runs/<name>.pipeline.json` (status, start/end timestamps, and the
artifact each stage produces), so a run is resumable and inspectable:

```sh
# The single source of truth for what a full run executes — prints every command + the resolved
# recipe, shows what a resume would skip, and touches no files:
aupai --dry-run pipeline --name mybase

aupai pipeline --name mybase                     # run it (best recipe by default)
aupai pipeline --status mybase                   # print the stage table, run nothing (--json too)
aupai pipeline --resume mybase                   # skip done stages / existing artifacts, continue
aupai pipeline --name mybase --force             # rerun every stage, ignoring saved state
aupai pipeline --from pretokenize --to eval --name mybase   # a contiguous slice
aupai pipeline --stages data,pretrain,eval --name mybase    # an explicit subset
```

- `--name` flows to `runs/<name>.log`, `ckpt_<name>.pt`, the eval target, and the state file.
- Backend flags for the pretrain stage go after `--`, e.g. `-- --grad_ckpt`; they override the recipe.
- `sft` needs `--sft-pt <packed.pt>`; `rl` reads `--rl-data` (default `data/rl/rl_band.jsonl`).
- The default slice (`--from`/`--to` unset) is `tokenizer … eval` — the core pretraining path;
  add `sft`/`rl` with `--to rl` or `--stages`.

## Best recipe as the default, fully visible

`aupai train` (and the pipeline's pretrain stage) default to the **verified-best recipe** — the
`ckpt_k4_11b_lr05` run, the only one to hit 51.6% math-500 (`--fp8 --attn_res --attn_res_blocks 4
--warmup 150 --lr_scale 0.5`, sourced from its `runs/k4_11b_lr05.log` cfg line / EXPERIMENTS.md).
Zero input gives a correct run, and the CLI **prints the resolved config before launching** so you
see every effective flag and where it came from:

```sh
aupai train --name k5             # runs the best recipe; prints the resolved-config block first
aupai train --name k5 --profile base   # bare train.py defaults instead
aupai train --name k5 -- --lr_scale 0.3  # any flag you pass after `--` overrides the recipe
```

## trackio

`aupai train --track` (or `pipeline --track`) turns on trackio logging (local-first,
SQLite-backed — no login, no server). The CLI's contract with `train.py`:

- env `TRACKIO_PROJECT=<name>` (default `aupai`; override with `--track-project <name>`)
- flag `--track` appended to the `train.py` argv

`train.py`'s `RunLog` reads that flag to init a run and mirror each step's metrics. `aupai
dashboard` launches trackio's local web UI (`uv run python -m trackio show [--project <name>]`);
`aupai dashboard <name> --plot` plots `runs/<name>.log` with `scripts/plot_curves.py` and opens
the PNG.

## Environment

- `AUPAI_CACHE_DIR` — where the pretokenized token caches live (`tokens_<domain>.pt`); read by
  the Python side (`train.py`, `pretokenize.py`) and by `aupai status` (to check cache readiness).
  Defaults to `/data00`.
- `TRACKIO_PROJECT` — set by `aupai train --track` and read by `aupai dashboard` when `--project`
  is not given.
- `NGPU`, `PORT`, `CUDA_VISIBLE_DEVICES` — honored by the underlying `run_ddp.sh` / eval scripts.

## Runs anywhere (mac + pod), auto-scaled

- **Python launcher auto-detects:** a repo `.venv` (the mac dev box, created by `uv`) runs through
  `uv run python`; otherwise (the pod: system `python3`, no venv) it falls back to plain `python3`.
  No config — just a filesystem check, so the same `aupai <cmd>` works on both.
- **CPU-bound stages use every core automatically:** `data`, `tokenizer`, `pretokenize` are launched
  with `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` set to the detected core count —
  no core-count flag to pass.

## Tests

`cli/e2e.sh` is the end-to-end acceptance gate: it drives the built binary and asserts real behavior
(exit codes, the resolved-config block, the `last=true` dry-run footgun stays closed, fail-fast
checkpoint checks, `status`/`ckpt list`/`pipeline --status`, no `wandb` in the help). Dependency-free
(bash + the binary), runs on mac and pod:

```sh
cli/e2e.sh   # builds if needed; exits non-zero if any assertion fails
```
