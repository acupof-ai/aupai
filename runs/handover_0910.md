# Handover, 2026-09-10 — read this first if you just started

Written by fb (controller) at 06:0xZ, after the user restarted every session into an
`aupai-*` working tree and ordered that no other directory be used. A restarted session has
no context: this file plus the ledgers are all there is. Nothing below is dropped work —
every task open before the restart is accounted for here.

## The one number that orders everything

**The acceptance gate is HumanEval pass@1 >= 30% at 350M** (`docs/standards/p1_data_recipe.md:256`).
Progress toward it, measured 2026-09-10: **0%**. No p1 training row exists in
`runs/experiments.jsonl` (494 rows), no p1 mix file, and the only measured HumanEval in the tree
is 0/164 on a checkpoint the user retired. Best-ever is 3/164 = 1.83% after a format SFT, which
**failed its own prereg** (Fisher one-sided p=0.124 where >=5/164 was needed).

**The binding constraint is teacher generation, and the recipe says so itself at `:298` —
"Generation, not training, is the schedule."**

| artifact | need | have | at the granted 3 cards | basis |
|---|---|---|---|---|
| synthetic textbooks | ~0.8B (`:66`) | ~8,879 tok (a 5-row smoke, Sep 9 14:16) | **13.3 d** | `facts/efficiency.json#eff.teacher_serve_warm_throughput`, 695 tok/s aggregate warm, MEASURED at 3 cards |
| synthetic exercises | ~0.18B (`:69`) | **0**, no generator on main | 3.0 d | same |

The serve is on 2 cards as of 04:37Z, which is ~463 tok/s **extrapolated** and ~20 days. Card 7 is
granted to it (`runs/card_assignment.json`, `block_cards` 4,5 -> 4,5,7, lane stays 2). **Every day
the serve stays at two cards costs about half a day of gate date.**

The fourth card is deliberately NOT granted: it would have to be card 2, the lane, and both P0
artifacts close against 50-sample spot checks that need it.

## Priority, and it is not the same as the queue length

| tier | task | owner | why here |
|---|---|---|---|
| **P0** | **de-101** start textbook generation | de | the constraint; card granted, 13.3 d |
| **P0** | **44-42** write the exercise generator | 44 | 0 of 0.18B, no generator on main, priced +21.6 points (`:45`) |
| **P0 support** | **3b-19** hand de-101 a usable topic-seed table | 3b | de-101 draws prompts from it; `:266` is 3b's line |
| **P0 support** | **e1-58** make `eval/humaneval_sample.py` produce its first number | e1 | the gate IS a HumanEval number and nothing in the tree can produce one today |
| **P0 support** | **98-4** the 50-sample two-reader sampler | 98 | BOTH P0 artifacts close against it; slack now, tail blocker if built last |
| P1 | **de-100** the 8 unreachable CLI flags | de | needed before the gate RUN, not before generation. In progress |
| P1 | **b0-49** build `data/tokenizer_p1.json` | b0 | production prerequisite, **explicitly off the gate critical path**: skipping costs +3.4% chars/token and `:145-146` pre-authorises corpus variance |
| P2 | everything else | — | see the queue notes below |

## Two stale blockers, lifted

- **de-66** reads `blocked_on: the 8B arm's stop window -- train.py is frozen`. The 8B/30B MoE line
  was **retired by user order 2026-09-09** (`p1_data_recipe.md:9-10`, "nothing resumes from it").
  No such freeze exists.
- **de-74** reads `blocked_on: PR #40 merging`. #40 merged long ago; the tree is at #204.

Both are unblocked and both are still P2. Lifted by fb rather than by their owner because the
blockers name facts about the repository, not about de's work.

## b0's queue: nine tasks behind one, and the block is not technical

**b0-28, b0-29, b0-30, b0-31, b0-32, b0-33, b0-34, b0-36 and b0-48 all read
`blocked_on: b0-35`.** Their subjects are a harness checkpoint-name regex, `card_claim`'s poll
interval, `write_mix_500m`'s resume cursor, and cache fingerprints in `score_matrix`/`domain_bpb`.
**None of them touches CSA attention, which is what b0-35 is.** The dependency is owner
availability, not code.

**They are deliberately NOT redistributed to the idle owners.** Measured over 2026-09-01..09-10:
4,064 non-merge commits on main, **8.9% touching any critical-path surface**, no upward trend;
`scripts/harness.py` took +22,112/-3,064 lines while `train.py` took +2,629/-727, so tooling grew
**8.4x faster than model code** by added lines. Handing nine tooling tasks to whoever is idle is
the mechanism that produced that ratio. They stay queued, with an honest block reason.

b0-35 itself is off the critical path: `facts/efficiency.json#eff.csa_step_speed` reads **8.157x**
per step (233.19 vs 28.59 ms) against its own preregistered 1.15x rule, with 0 tasks and 0 PRs
behind the fast path it needs.

## Do not do these, each with the number that kills it

- **Do not write a 117th harness check** for anything above. 116 exist.
- **Do not wire the exercise generator to PR #158's checker as it stands.** 7 blocking findings,
  three on the decontamination criteria the artifact is gated on: `exercise_checks.py:92` is
  `if not os.path.exists(path): continue`, so a partial benchmark load prints
  `decontam: 0 hit(s) against 1 benchmark problems` and **exits 0 on contaminated data**, and the
  known-positive control runs only inside `_selftest`, never in `main`.
- **Do not run the gate on filtered code alone to get a number this week.** phi-1-base without
  CodeExercises is 29% at 1.3B params on 6.8B tokens (`:31`) against a gate of 30% at 350M. It is
  a predicted fail that still burns the cards.
- **Do not serialise the gate behind the V=20,000 rebuild.** See b0-49 above.
- **Do not commission another progress audit or percentage readout.** One ran 2026-09-10: six axes
  read in parallel, **all six first readings refuted as inflated**, and the second pass produced no
  tokens either. The next measurement that changes anything is HumanEval on a p1 checkpoint.
- **Do not delete a worktree without the user naming it.** 78 exist besides `aupai/` and
  `aupai-fb/`; 62 are clean and fully in main, but that is *git* safety — six of them are live
  sessions' working directories, and `wt-maintouched` holds **142 uncommitted files**.

## Where things actually are

- `main` and the pod stamp agree; `pod_push --all` reports no `refusing`.
- Cards: 0, 1, 3, 6 tileRL's; 4 and 5 aupai's, running `teacher_serve_0909`; **7 aupai's, granted
  to the serve, still idle**; 2 the lane. Card ownership is read from
  `runs/card_assignment.json`'s `note` field — the single current-state field — **never inferred
  from `nvidia-smi`**. fb misattributed cards 4 and 5 to tileRL on 2026-09-10 by reading card rows
  without joining them to claim names.
- The p1 code keep set is real: 9.0 GB at `data/p1/keep_set` on the pod, three domains plus
  `manifest.json`, **2,811,615,700 tokens** post-deletion (3b full census, e1 spot-check). Nothing
  in `train.py`/`sft.py`/`sft_math.py` reads that path — `grep -c keep_set` is 0 in all three;
  training globs `data/corpus/<domain>/*.jsonl`, and `data/corpus/code_dedup08` is already the
  clean post-deletion copy.
- Open PRs: 13. On the gate path: **#158** (exercise checker, 7 findings) and **#169** (tokenizer
  scripts, 4 findings all verified by fb's own read of the diff at `7b08bca7`). The other eleven
  are off it.

## Addressing

`runs/roster.json` sockets are STALE after the restart — every session got a new one. Two are
confirmed: de is `aupai-1f`, 98 is `aupai-4d`. Identify yourself to fb (`aupai-4c`) with your
roster name and your working tree before taking a task that names an owner. Four mis-dispatches
happened on 2026-09-09 from guessing at names; `p1_data_recipe.md:272-281` records them.

**The user ordered that no directory outside `aupai-*` be used.** If your session started somewhere
else, say so rather than working from there.
