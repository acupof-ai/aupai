# Card-claim coverage: why the acquire point is not where it looks (de-55)

Measured 2026-09-04. Prompted by `runs/claims` being empty on the pod while lane jobs had held
cards all day.

## The population

11 files load a checkpoint and name `cuda`. **10 have no reference to `card_claim` anywhere:**

`eval/nan_probe.py`, `eval/score_matrix.py`, `probes/fone_digit_acc.py`,
`scripts/eval_heldout.py`, `scripts/eval_heldout_ours.py`, `scripts/n7c_gates.py`,
`scripts/n7c_grad_check.py`, `scripts/test_arch_L32.py`, `scripts/test_arch_compat.py`,
`scripts/test_e2e.py`.

Only `harness launch` acquires (`scripts/harness.py:13949`). So this was never a `score_matrix`
gap: it is every GPU entry point except the one the harness drives. b0 confirmed the card-5
rescore and e1's C11 both ran as `python3 eval/score_matrix.py ...` direct, so no shell wrapper
exists to hold a single acquire.

## Why the shared loader is the wrong place

`scripts/loader.py:44` `load_checkpoint(path, device="cpu", dtype=None, fone_ok=True)` is the
shared helper, and it does build the model from `ck["cfg"]` as AGENTS.md requires (`:53`). It
still cannot be the acquire point, for two independent reasons:

| | |
|---|---|
| coverage | 5 of the 10 call it. All 10 also call `torch.load` directly, 1–4 times each; 4 use no loader helper at all |
| device | its `device` defaults to `"cpu"`, and the value the 10 actually pass is the literal `"cpu"` in 7 of 10, `device` in score_matrix, `DEV` in test_arch_L32, `"cuda:0"` in test_e2e. An acquire inside it fires on CPU loads and claims nothing on the runs that matter |

## The finding that decides the shape

**9 of the 10 never mention `CUDA_VISIBLE_DEVICES` — not read, not set.** Only `test_e2e.py`
does both, and it requires `E2E_GPU` by design.

`card_claim`'s contract is CVD, not `cuda:N`: a claim records the cards the job can SEE, and
`acquire` REFUSES a claim whose cards disagree with the process's CVD (asserted in its own
selftest). So these nine cannot name the card they are using — they inherit it from whoever
launched them. An `acquire` added inside each one would have to be handed a card number the
process does not know, which is why "ten diffs" is fragile rather than merely repetitive: nine
of the ten have nothing correct to pass.

## Options, with their real cost

1. **`claim_my_cards(name)` in `scripts/loader.py` + one call line per entry point.** Reads CVD,
   refuses if unset, acquires for `os.getpid()`, registers an `atexit` release. Still ten call
   sites, but no site needs a card number and the refusal lives in one testable place. Covers
   every future probe by the same line. **Recommended.**
2. **Option B extended to all ten** (each acquires on entry): same site count, and nine sites
   cannot supply the card argument.
3. **Make the nine refuse without CVD set**, leaving `harness launch` as the only way to get a
   card: genuinely one line each, but it is a behaviour change to nine tools people run by hand,
   trading a silent unclaimed card for a refusal in someone's terminal. Controller's call.

`card_held_without_claim` (`harness.py:8498`) already detects the consequence — a card holding
memory that no live claim names — and stays the detector under any of the three. It reports SKIP
off-pod, which is why nobody saw this: the check exists and nothing runs `harness check` on the
pod during a lane job.

## Ruling and the CI hazard it has to survive (6e, 2026-09-04)

Option 1 accepted: `claim_my_cards(name)` in `scripts/loader.py` reading CVD, refusing when
unset, acquiring for `os.getpid()`, `atexit` release; one call at each of the ten; the refusal
message names `CUDA_VISIBLE_DEVICES=N` as the fix. The behaviour change to the nine hand-run
tools is intended — a card job that cannot name its card does not get one.

6e's warning was that an ungated call site makes `test_arch_compat` refuse in CI. Measured, the
shape is worse than that, and it is the same defect one level down:

| file | has a real cpu path | picks its own card | exits without a card |
|---|---|---|---|
| `scripts/test_arch_compat.py` | yes | **yes** | yes |
| `scripts/test_arch_L32.py` | yes | no | yes |
| `scripts/test_e2e.py` | yes | no | no (requires `E2E_GPU`) |
| the other seven | no | `score_matrix` only | no |

`scripts/test_arch_compat.py:49-54`: when `fla` is installed and CVD is **unset**, it reads
`torch.cuda.mem_get_info` across every visible device and lands on the freest one —
`DEV = f"cuda:{...}"`. That is a card taken without CVD and without a claim, chosen by a
free-memory poll, which is the instantaneous-`nvidia-smi` reading AGENTS.md rejects as an
ownership test. So a helper that refuses on unset CVD would refuse **there**, correctly, and the
fix at that site is to set CVD rather than to skip the call.

Consequences for the call sites:

- Three files have a genuine CPU path (`test_arch_compat`, `test_arch_L32`, `test_e2e`). Their
  call goes INSIDE the cuda branch, or CI — which runs `test_arch_compat` on a machine with no
  `fla` and `DEV = "cpu"` — starts refusing.
- The seven with no CPU path can call unconditionally.
- `eval/score_matrix.py` also polls devices; that site needs reading before the call is placed.

## Not measured

Whether the four entry points with no loader helper could adopt `load_checkpoint` at all. That is
a separate refactor and it is not required by any of the three options above.

Whether `test_arch_compat`'s freest-card branch has ever collided with a running job. It is a
latent instance of the same defect and belongs in its own row, not this one.


## The population was 11 because of how it was selected, not because it is 11 (de, 2026-09-04)

Implemented Option 1 and then re-measured the population from the tracked tree rather than from
the earlier hand list. **A file that loads a checkpoint AND puts something on cuda: 48 tracked
`.py`, 41 with a `__main__`.** Ten now call `claim_my_cards`; `harness.py` acquires as before.
Both numbers are FLOORS: the predicate is textual, and the scan's own first version missed
`eval/score_matrix.py`, whose device comes back from a helper returning an f-string.

The predicate is `torch.load|load_checkpoint` AND one of `.cuda()`, an argparse `default="cuda"`,
`dev = "cuda[:N]"`, or `.to("cuda")` — a card actually taken, not the string `cuda` appearing
somewhere. Scanned with `runs/audit_0904/card_claim_population.py`, which is committed so the count can be
re-derived rather than trusted; the looser "names cuda anywhere" predicate returns 50, and the
difference between 50 and 48 is files that mention cuda without taking a card.

So the ten sites in this commit close the ruling as written and cover **10 of 41**. The
uncovered 31 are the same defect: `eval/` has 19 of them (`math_hard`, `mmlu`, `ppl`,
`domain_loss`, `l1_fewshot`, `run_eval`, …), plus `sft.py`, `sft_math.py`, `train.py`,
`chat.py`, `infer.py`, `algorithms/rlvr_trainer.py`, four `scripts/b0_*`/`e1_*` probes and two
`n7c_*` gates.

What this does NOT establish, and the reason it is a note rather than a task: whether all 31
should acquire. `train.py` and `sft*.py` go through `harness launch`, which already claims, so a
second acquire there would refuse its own launcher's claim — the acquire point for a launched job
is the launcher, and adding one inside would be wrong rather than merely redundant. The 19 `eval/`
tools are the live question: `eval_all.sh`/`eval_hard.sh` wrap several of them, so the right unit
may be the wrapper rather than each metric. Deciding that needs a reading of which are ever run
directly, which is a separate measurement.

**Nothing here weakens the ruling; it bounds it.** Option 1's shape — one helper, one line per
entry point, the refusal in one testable place — is what makes 31 more sites cheap if the
controller wants them. The scan is the deciding artifact and it is reproducible: rerun the
predicate above rather than trusting this count.

## The 19 eval/ tools: measured, and the answer is 8

`runs/audit_0904/eval_acquire_unit.py`, 2026-09-04. The population is 20 card-taking eval/ tools
with no claim, not 19 — the earlier figure was read off a hand list.

| bucket | n | acquire goes |
|---|---|---|
| launched directly (a ledger `cmd` row or a doc command block) | 8 | in the tool |
| reached only through `run_eval`'s `import_module` registry | 3 | `run_eval.py`, once |
| reached only through a `.sh` | 3 | the wrapper |
| hook runs or exempts its selftest, never launched with a card | 6 | nowhere |
| no record of ever being run | **0** | — |

The eight: `code_fewshot`, `code_zh`, `domain_loss`, `humaneval_bpb`, `l1_fewshot`, `loop_wrapper`,
`math_hard`, `ppl`.

**Zero deletion candidates, and that is the finding.** The first run of the scan put nine tools in
the never-run bucket and every one of them was reachable by a route the first three predicates
could not see. Two sources fixed it, and both are the shape AGENTS.md already warns about:

- `arc`/`mmlu`/`piqa` are imported by `eval/run_eval.py` through `import_module(f"eval.{name}")`.
  No static analysis sees a runtime loader — the same defect as `vet_programs.py:37`'s glob making
  23 live math generators read as unreferenced. They are library modules; `run_eval` holds the card,
  so an acquire inside each would refuse `run_eval`'s own claim.
- Six more are in the hook's `SELFTEST_FILES` or carry a written `NEEDS_DATA` exemption, i.e. they
  are executed or deliberately excused on every commit that stages them. A selftest takes no card,
  so they are neither dead nor acquire sites.

**A path in backticks is not a command.** The doc reader first counted every backtick span, which
made `docs/lessons/kept_methods.md` a launch and inflated "launched directly" from 8 to 14. The
negative control caught it; command spans now have to start like commands (`python`, `bash`, `./`,
`VAR=`, `torchrun`, `setsid`, `pod`). Both halves are needed: AGENTS.md's entry-point table — the
repo's actual answer to "how do I run this" — puts commands in backticks inside table cells, never
in fenced blocks, so a fenced-only reader misses them entirely.

15 selftest cases, including the bucket partition (`[8, 3, 3, 6, 0]`) so the classification order
is asserted total rather than observed non-overlapping today.

