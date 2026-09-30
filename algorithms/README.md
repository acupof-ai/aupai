# algorithms/

Two RL training loops and the machinery each needs: verifiable-reward utilities for math
(RLVR), and code execution under isolation for code RL.

## Layout

| File | Role | Needs torch to import? |
|---|---|---|
| `rlvr_reward.py` | `\boxed{}` extraction, answer normalization, 0/1 reward | No (stdlib only) |
| `rlvr_generate.py` | Batched top-p autoregressive sampling | No (lazy on call) |
| `rlvr_trainer.py` | RLVR GSPO loop: fp32 master weights, FP8 train + bf16 gen copies, DDP | No (lazy) |
| `rlvr_data.py` | Build/load `data/rl/rlvr_math.jsonl` from raw math datasets; script entry -> `main()` | No (stdlib only) |
| `rlvr.py` | Entry point -> `rlvr_trainer.main()` | — |
| `code_reward.py` | `(code, tests) -> {0.0, 1.0}`, or raise on a sandbox fault | No (stdlib only) |
| `isolate.py` | Run untrusted code at the best isolation this host offers, and return which | No (stdlib only) |
| `seccomp.py` | seccomp-BPF denylist installed as a `preexec_fn` | No (ctypes only) |
| `rollout.py` | K parallel rollouts, one sandbox and one workdir each | No |
| `rl_code_trainer.py` | Code RL loop: GSPO on executable problems, raw-continuation prompts | No (lazy) |
| `rl_code.py` | Entry point -> `rl_code_trainer.main()` | — |
| `attnres_fused.py` | Fused AttnRes mixing as one autograd node; measured 2.2x slower than eager, flag off | Yes |
| `attnres_triton.py` | Triton kernels for the same mixing, accumulator in registers | Yes |

Tests are in the same directory and run standalone: `test_rlvr_reward_suite.py`,
`test_gspo_ratio.py`, `test_rl_code.py`, `test_rl_code_ced_smoke.py`,
`test_stdin_reward_known.py`. `code_reward.py`, `isolate.py`, `seccomp.py` and
`rollout.py` each carry their own `--selftest`.

All paths resolve from the project root (`os.path.dirname` of this directory),
so scripts run from anywhere. Heavy deps (torch, tokenizers, `train`/`sft`
which pull fla/Triton) are imported lazily — `import algorithms` works on a
CPU-only box; torch is only loaded when a training/generation function runs.

## Usage

```bash
# Prepare RLVR data (school_math_r1_zh + gsm8k_zh -> data/rl/rlvr_math.jsonl)
python algorithms/rlvr_data.py

# RLVR training (single GPU or DDP)
torchrun --nproc_per_node=8 algorithms/rlvr.py --resume ckpt_sft.pt

# Code RL training (pool from scripts/rl_code_pool.py under data/rl/code_pool)
torchrun --nproc_per_node=8 algorithms/rl_code.py --resume ckpt_sft.pt
```

## Importing as a library

```python
from algorithms import reward_fn, generate, load_problems, train_rlvr

reward_fn(r"答案是 \boxed{\frac{1}{2}}", r"\dfrac{1}{2}")  # 1.0
problems = load_problems()  # [{prompt, answer, source}, ...]
```

`algorithms.reward_fn` is the math reward. The code reward is
`algorithms.code_reward.reward_fn`, a different function with a different signature.
Submodules are also importable directly: `from algorithms.rlvr_reward import normalize_answer`.

## Code reward — three states

`code_reward.py` returns a float in `{0.0, 1.0}` or raises. The third state is not a
reward value:

| state | how it is returned |
|---|---|
| PASS | `1.0` — every test passed inside the timeout |
| CANDIDATE-FAILED | `0.0` — the model's code failed, timed out, or produced unreadable output |
| EXECUTION-FACILITY-FAILED | `code_reward.ExecutionFacilityError` is raised — the sandbox failed before the candidate ran |

The facility state raises so no caller can average it into an advantage. It used to be
scored as a candidate failure: on 2026-09-30 every `run_sandboxed` call on the pod returned
`rc 126: setpriv: failed to execute /usr/bin/python3.12: Resource temporarily unavailable`,
which would have handed GRPO eight zeros per group, zero advantage everywhere, and a run
training on noise while holding eight cards. `rl_code_trainer.score_row` lets the exception
propagate on purpose.

`facility_failure(rc, stdout, stderr)` makes the discrimination and requires all three
conditions, because rc alone cannot separate the two — a candidate is free to call
`sys.exit(126)`:

1. rc is one the facility reserves: 97, 98, 126, 127.
2. The first non-empty stderr line starts with a setup tool's marker (`setpriv:`,
   `unshare:`, `mount:`, `chroot:`, `bwrap:`, `sandbox-exec:`, and the rest of
   `_FACILITY_MARKERS`). The facility speaks before the interpreter exists.
3. stdout is empty.

**The residual is resolved toward raising.** A candidate can forge all three and buy
itself an invalidated measurement, never a reward of 1. A real facility fault read as a
candidate failure is the error that silently poisons training.

## Code reward — the two contracts

| | call-style | stdin-style |
|---|---|---|
| functions | `reward_fn(code, tests)`, `score(code, tests)` | `reward_fn_stdin(code, cases)`, `score_stdin(code, cases)` |
| `code` is | a module defining the entry function | a complete program reading `sys.stdin` and printing |
| the second argument | a pytest/unittest file that does `from solution import fn` | `[{"input", "output", "rel_tol"?, "abs_tol"?}]` |
| how it runs | `solution.py` written first, `test_solution.py` second, then `python -m pytest` (or `-m unittest` when pytest is absent) | one process per case, `python solution.py`, case input piped to stdin |
| the signal | the test runner's exit code, plus pytest's summary line | stdout compared to `output` |
| default timeout | 30 s | 10 s |

The write order is the defence against a rollout that rewrites its own tests: `tests` lands
on disk after `code`, so an edit to the test file is overwritten. It is asserted in the
suite.

Stdout comparison for the stdin contract: trailing whitespace is stripped from every line
and leading/trailing blank lines are dropped, then the lines are compared exactly.
Whitespace inside a line is not normalized. Float tolerance applies **only to a case that
declares `rel_tol` or `abs_tol`** — both lines must tokenize to the same count, numeric
tokens are compared with `math.isclose`, every other token verbatim.

`verdict(rc, timed_out, stdout)` scores PASS as rc 0 **and** at least one `passed` in the
summary **and** no failed/error. The exit code stays the authority for failure; the summary
is read only to catch the shapes where rc 0 lies about a pass having happened — `2 skipped
in 0.30s` earns 1.0 for a wrong implementation without the model ever touching the test
file. An unparseable summary scores 0: it is not evidence of a pass.

The reward is binary rather than a pass fraction. Partial credit rewards a rollout that
makes 9 of 10 tests pass by deleting the tenth's assertion. GRPO normalizes within the
group, so a binary reward still separates K rollouts as long as they do not all agree.

`nondeterminism_risk(tests)` names instability patterns in the test source — wall-clock or
date assertions, unseeded random, exact float equality. It detects rather than fixes;
`--roundtrip` reports flagged pairs as a separate bucket.

```bash
python3 algorithms/code_reward.py --selftest
python3 algorithms/code_reward.py --roundtrip <pairs.jsonl>
```

## Isolation

`isolate.run(code=..., argv=..., workdir=, timeout=, level=, stdin_data=, nproc=)` returns
`{level, rc, stdout, stderr, timed_out, isolates}`. `level` is the level actually used;
callers record it rather than assume it. A reward earned under `rlimits_only` and one
earned under a namespace are different measurements, the same reasoning as `vocab_id` on a
checkpoint.

`detect_level()` reads the host, never a config, and returns the first of:

| level | condition | what it gives |
|---|---|---|
| `bwrap` / `nsjail` / `firejail` | the binary is on PATH | none of the three exists on the pod or the Mac (measured 2026-09-02); kept first so adding a host does not mean editing the caller |
| `sandbox_exec` | Linux, euid 0, `unshare` present | chroot into a minimal root, mount/net/pid namespaces, rlimits |
| `seatbelt` | Darwin, `/usr/bin/sandbox-exec` present | denies network and home secrets, allows the workdir |
| `rlimits_only` | nothing else matched | rlimits, timeout, a private tmpdir — no network or filesystem-read isolation |

**`rlimits_only` refuses by default**: `run` raises `isolate.Unisolated` unless
`ALLOW_UNISOLATED=1` is set. Model code at that level could read `~/.ssh` and reach the
network. The level is recorded either way, so a result taken without isolation cannot be
mistaken for one taken with it. `code_reward.score(..., generated=True)` and every reward
entry the rollout path calls refuse to run under `ALLOW_UNISOLATED=1`;
`generated=False` is the offline builder path executing a dataset's own trusted
references.

`SharedNetNotPermitted` is raised when bwrap cannot isolate the network and no shared-net
override was given. `probe(level)` runs every escape attempt at a level and returns
`{axis: bool_isolated}`.

**`nproc` is a machine-wide ceiling, not a per-sandbox budget.** It sets the chroot user's
`RLIMIT_NPROC` under `sandbox_exec` and defaults to 64. `RLIMIT_NPROC` counts the real
uid's tasks machine-wide, every sandbox drops to the same uid 65534, and the pod's host
already carried 121 tasks under that uid on 2026-09-30 — the container's own `ps` shows
none of them. At 64 the `execve` after the uid drop fails with EAGAIN before any code runs
and `print(7)` returns rc 126; the boundary that day sat in (120, 128], and at 512 a fork
loop got 390 children, 121 + 390 = 511. Raising the number borrows headroom from a count
we do not control and lifts the ceiling for every concurrent task on the uid. Pass a value
explicitly if you need one, and record it: results at different caps are not comparable.
Only the fork ceiling changes; chroot, namespaces, uid drop, network and filesystem are
unchanged.

```bash
python3 algorithms/isolate.py --selftest
```

## seccomp

`seccomp.py` installs a hand-written BPF denylist through ctypes and prctl:
`socket`, `socketpair`, `ptrace`, `setsid`. `preexec(denied, errno)` returns a
`preexec_fn` for `subprocess`; `install()` applies the filter to the calling process and is
irreversible. `available()` returns `(ok, why)` and separates ENOSYS (this kernel has no
seccomp) from EINVAL (seccomp is present and rejected a deliberately invalid argument,
which is the answer wanted). `Unsupported` is raised rather than degrading silently.

The denied calls fail with EPERM rather than killing the process with SIGSYS, so every
runtime's existing error handling applies. A denylist is used instead of a whitelist
because a CPython syscall whitelist moves with the libc, the interpreter version and the
imports a test happens to make, and a missing entry kills the process before it runs a
line.

What it adds over the network namespace: it works where a netns is unavailable, it blocks
AF_UNIX and `socketpair` which a netns does not, `ptrace` stops a rollout attaching to a
sibling in the same pid namespace, and `setsid` closes the mechanism for leaving the
process group the runner kills. Defence in depth — the level in a rollout record still says
which isolation was in force.

```bash
python3 algorithms/seccomp.py --selftest
```

## Rollouts

`rollout.rollout(code, tests, ...)` returns one record: `reward, level, isolates, escaped,
rc, timed_out, secs`. `rollout.run_group(variants, tests, k=8, ...)` runs up to k
concurrently and returns `{records, k, levels_agree, wall, serial}`.

Three things it adds over calling `reward_fn` in a loop:

| addition | mechanism |
|---|---|
| rollouts isolated from each other | each gets its own directory — a git worktree at the task's commit when a repo is given, a mkdtemp otherwise |
| comparability stated | `levels_agree` false means the rollouts did not all run at one isolation level, so their rewards are not one group |
| real concurrency | the work is subprocesses, so threads suffice; the suite measures wall clock against the serial sum |

`escaped` scans a rollout's output for the REACHED marker its own test file asserts on. It
catches a variant that tries an escape and says so; it says nothing about code that
exfiltrates quietly. The guarantee is `level`, and the scan is corroboration.

```bash
python3 algorithms/rollout.py --selftest
```

## Code RL

`rl_code_trainer.py` runs GSPO on executable code problems. The prompt is a raw code
prefix — a `def` signature plus docstring for call-style rows, a module docstring for
stdin rows — and the model continues the program. Differences from the math path:

- Continuation prompts, never `format_prompt`. A BASE checkpoint is refused;
  `score_matrix.classify` must read kind `sft` or `rl`.
- `reward_fn(code, tests)` for call-style rows, `reward_fn_stdin(code, cases)` for stdin
  rows. Model rollouts always run isolated and `ALLOW_UNISOLATED=1` raises.
- Advantage is mean-subtracted and **not** divided by std; a constant group is exactly
  zero. `rlvr_trainer.group_advantage(rewards, normalize_std=False)` is the shared
  implementation, called with `normalize_std=True` by the math path.
- bf16 params, no fp32 master copy (1e order 2026-09-25). Muon holds fp32 momentum and
  writes back with stochastic rounding; the 1-D AdamW params step in fp32 and are
  stochastically rounded back through the same `StochasticRounder`. Generation runs on the
  training model itself in `eval()` + `no_grad`, so only the frozen bf16 KL reference is
  duplicated, about 6 GB at 3.2B.

`build_rounder(seed=RL_SR_SEED)` deliberately takes no rank: every DDP rank shares one SR
seed, because DDP synchronizes gradients and not weights.

`load_code_pool(pool_dir)` reads the rows `scripts/rl_code_pool.py` writes under
`data/rl/code_pool`, excluding quarantines and stats. `program_source(prompt, completion)`
executes the longest line-contiguous parseable span of prefix + continuation, the same
extractor `eval/score_code_exec.py` uses; no import header is injected, which would execute
code the model did not write.

`filter_rows_by_solution_len(rows, len_fn, call_cap, stdin_cap)` drops rows whose reference
solution is longer than the mode's `max_new`. Those rows cannot be completed and would
score 0 for a right answer. `len_fn` is injected — the gate tokenizer in training, a
character counter in tests — so the function carries no tokenizer dependency.

## RLVR design notes

- **fp32 master weights**: a 1e-6 AdamW update is far below bf16 ULP (~5e-4 at
  0.1), so the optimizer steps on fp32 master; both bf16 copies (FP8 train,
  plain bf16 generation) sync from it each step.
- **Two model copies**: FP8 for training, plain bf16 for generation — FP8
  quantization noise degrades sampling.
- **DDP**: all ranks sample the same prompts (`random.seed(1337 + step)`) but generate
  different responses (`torch.manual_seed(1337 + rank)`).
- **Degenerate groups are dropped, not stepped on.** An all-correct or all-wrong group has
  std 0, so its advantage is 0 and a forward/backward on it is waste. The keep decision is
  all-reduced with MAX, so every rank runs the same forward count and DDP stays in
  lockstep. A step where every group is degenerate is skipped; a run that stays that way
  refuses rather than exiting 0 with a checkpoint bit-identical to `--resume`.
- **The GSPO ratio is real.** Until 2026-09-05 `old_lp` was `seq_lp.detach()` from the same
  forward, so the importance ratio was identically 1.0 and `--clip_eps` changed nothing at
  any value. `test_gspo_ratio.py` varies `clip_eps` and demands the loss move.
