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

